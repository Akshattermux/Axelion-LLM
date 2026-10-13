"""
training/pretrain.py — Axelion causal language-model pretraining.

Usage:
    python -m training.pretrain --size 1b --train-bin data/pretraining/train.bin \\
        --val-bin data/pretraining/validation.bin \\
        --checkpoint-out checkpoints/axelion-base-1b.pt
"""

import argparse
import contextlib
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repository root on Python path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from training.dataset import create_pretrain_dataloader, prepare_binary_dataset


# ---------------------------------------------------------------------------
# Tee — duplicate stdout to a log file
# ---------------------------------------------------------------------------
class _Tee:
    """Writes every .write() call to both the wrapped stream and a file."""

    def __init__(self, stream, fpath: str):
        os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
        self._stream = stream
        self._file = open(fpath, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
def cosine_lr(
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Cosine decay with linear warm-up."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ---------------------------------------------------------------------------
# Autocast context helper
# ---------------------------------------------------------------------------
def _autocast_ctx(device: str, dtype: torch.dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=dtype)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Loss estimator
# ---------------------------------------------------------------------------
def estimate_loss(model, dataloader, ctx, device: str, eval_iters: int = 10) -> float:
    """Average cross-entropy over *eval_iters* random batches."""
    model.eval()
    losses = []
    iterator = iter(dataloader)
    with torch.no_grad():
        for _ in range(eval_iters):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                x, y = next(iterator)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss, _ = model(x, targets=y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# Binary-dataset helper
# ---------------------------------------------------------------------------
def _ensure_binary_dataset(
    text_path: Optional[str],
    bin_path: str,
    tokenizer: AxelionTokenizer,
) -> None:
    if text_path and os.path.exists(text_path):
        prepare_binary_dataset(text_path, bin_path, tokenizer)
    elif not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Missing {bin_path}.  Supply --data-txt or run "
            "python scripts/prepare_data.py --mode pretrain."
        )


# ---------------------------------------------------------------------------
# Checkpoint save helper
# ---------------------------------------------------------------------------
def _save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Main pretraining entry-point
# ---------------------------------------------------------------------------
def pretrain(
    train_bin: str = "data/pretraining/train.bin",
    val_bin: Optional[str] = "data/pretraining/validation.bin",
    data_txt: Optional[str] = None,
    val_txt: Optional[str] = None,
    checkpoint_out: str = "checkpoints/axelion-base-mini.pt",
    resume_from: Optional[str] = None,
    size: str = "mini",
    seq_len: Optional[int] = None,
    batch_size: int = 1,
    grad_accum_steps: int = 16,
    max_iters: int = 10000,
    eval_interval: int = 500,
    eval_iters: int = 10,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
    log_file: Optional[str] = None,
    benchmark_steps: int = 0,
) -> None:
    # ---- reproducibility -------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- log tee ---------------------------------------------------------
    if log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)

    # ---- device / dtype --------------------------------------------------
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = selected_device == "cuda"
    dtype_str = (
        "bfloat16" if on_cuda and torch.cuda.is_bf16_supported()
        else ("float16" if on_cuda else "float32")
    )
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    ctx = _autocast_ctx(selected_device, ptdtype)

    # ---- dataloader knobs ------------------------------------------------
    num_workers = min(4, os.cpu_count() or 1) if on_cuda else 0
    pin_memory = on_cuda

    # ---- tokenizer / config ----------------------------------------------
    tokenizer = AxelionTokenizer()
    preset_map = {
        "micro": ModelConfig.create_micro,
        "mini": ModelConfig.create_mini_gpt,
        "medium": ModelConfig.create_medium,
        "large": ModelConfig.create_large,
        "1b": ModelConfig.create_1b,
    }
    size_key = size.lower()
    if size_key not in preset_map:
        raise ValueError(f"Unknown model size '{size}'. Choose: {list(preset_map)}")
    config = preset_map[size_key](vocab_size=tokenizer.vocab_size)
    if seq_len:
        config.max_seq_len = seq_len

    # ---- binary dataset(s) -----------------------------------------------
    _ensure_binary_dataset(data_txt, train_bin, tokenizer)
    val_available = False
    if val_bin:
        try:
            _ensure_binary_dataset(val_txt, val_bin, tokenizer)
            val_available = True
        except FileNotFoundError:
            print(f"[WARN] val_bin not found ({val_bin}); skipping validation loss.")

    # ---- resume ----------------------------------------------------------
    start_iter = 0
    best_val_loss = float("inf")
    ckpt_payload: Optional[dict] = None

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        print(f"Resuming from {resume_from}")
        ckpt_payload = torch.load(resume_from, map_location="cpu", weights_only=False)
        config = ckpt_payload.get("config", config)

    # ---- model -----------------------------------------------------------
    model = Axelion(config).to(selected_device)
    n_params = model.get_num_params()
    print(
        f"\n{'='*60}\n"
        f"Axelion {size.upper()} pretraining | {n_params/1e6:.2f}M params\n"
        f"Device: {selected_device} | Precision: {dtype_str} | Seed: {seed}\n"
        f"Steps: {max_iters} | Batch: {batch_size} | GradAccum: {grad_accum_steps} | "
        f"Effective batch: {batch_size * grad_accum_steps}\n"
        f"{'='*60}\n"
    )

    # ---- dataloaders -----------------------------------------------------
    train_loader = create_pretrain_dataloader(
        train_bin, config, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = (
        create_pretrain_dataloader(
            val_bin, config, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        if val_available else None
    )

    # ---- optimizer / scaler ---------------------------------------------
    optim_kwargs = {
        "lr": max_lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.95),
    }
    if on_cuda:
        try:
            import bitsandbytes as bnb
            print("[INFO] Using bitsandbytes 8-bit AdamW (saves ~7.1 GB VRAM)")
            optimizer = bnb.optim.AdamW8bit(model.parameters(), **optim_kwargs)
        except Exception:
            try:
                optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
            except Exception:
                optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_str == "float16" and on_cuda))

    if ckpt_payload is not None:
        model.load_checkpoint_weights(ckpt_payload["model"])
        if "optimizer" in ckpt_payload:
            optimizer.load_state_dict(ckpt_payload["optimizer"])
        if "scaler" in ckpt_payload and scaler.is_enabled():
            scaler.load_state_dict(ckpt_payload["scaler"])
        start_iter = ckpt_payload.get("iter_num", -1) + 1
        best_val_loss = ckpt_payload.get("best_val_loss", float("inf"))
        if not math.isfinite(best_val_loss):
            best_val_loss = ckpt_payload.get("val_loss", float("inf"))

    # ---- benchmark mode --------------------------------------------------
    if benchmark_steps > 0:
        _run_benchmark(model, train_loader, optimizer, scaler, ctx, selected_device, benchmark_steps, grad_accum_steps)
        return

    # ---- checkpoint dir must exist before any save ----------------------
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_out))
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = checkpoint_out.replace(".pt", "-best.pt")
    latest_ckpt_path = checkpoint_out.replace(".pt", "-latest.pt")

    # ---- training loop ---------------------------------------------------
    train_iterator = iter(train_loader)
    global_start = time.perf_counter()
    step_start = time.perf_counter()
    tokens_since_last = 0
    tokens_per_step = batch_size * grad_accum_steps * config.max_seq_len
    total_tokens_trained = start_iter * tokens_per_step

    bar = tqdm(
        range(start_iter, max_iters),
        initial=start_iter,
        total=max_iters,
        desc="Pretrain",
        ncols=120,
        unit="step",
    )

    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for iter_num in bar:
        lr = cosine_lr(iter_num, max_iters, max_lr, min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- periodic evaluation -----------------------------------------
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            last_train_loss = estimate_loss(model, train_loader, ctx, selected_device, eval_iters)
            if val_loader is not None:
                last_val_loss = estimate_loss(model, val_loader, ctx, selected_device, eval_iters)
            else:
                last_val_loss = float("nan")

            val_ppl = math.exp(last_val_loss) if (not math.isnan(last_val_loss) and last_val_loss < 20) else float("inf")

            # Always save latest
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "config": config,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "train_loss": last_train_loss,
                "val_loss": last_val_loss,
                "val_ppl": val_ppl,
                "tokens_per_step": tokens_per_step,
            }
            # Update best loss before saving checkpoint metadata.
            cmp_loss = last_val_loss if val_loader is not None else last_train_loss
            improved = cmp_loss < best_val_loss
            if improved:
                best_val_loss = cmp_loss

            ckpt_data["best_val_loss"] = best_val_loss
            _save_checkpoint(latest_ckpt_path, ckpt_data)

            if improved:
                _save_checkpoint(best_ckpt_path, ckpt_data)
                tqdm.write(f"  ✓ New best ({best_val_loss:.4f}) → {best_ckpt_path}")

        # ---- forward / backward pass ------------------------------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                x, y = next(train_iterator)
            x, y = x.to(selected_device), y.to(selected_device)
            with ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ---- throughput / tqdm update -----------------------------------
        tokens_since_last += tokens_per_step
        total_tokens_trained += tokens_per_step
        now = time.perf_counter()
        elapsed_step = max(1e-6, now - step_start)
        tok_per_sec = tokens_since_last / elapsed_step

        # Reset every 25 steps for a rolling window estimate
        if (iter_num + 1) % 25 == 0:
            step_start = now
            tokens_since_last = 0

        remaining_steps = max_iters - iter_num - 1
        eta_sec = remaining_steps / max(1e-6, tok_per_sec / tokens_per_step)
        eta_str = _fmt_time(eta_sec)

        bar.set_postfix(
            TrainL=f"{last_train_loss:.4f}",
            ValL=f"{last_val_loss:.4f}" if not math.isnan(last_val_loss) else "N/A",
            LR=f"{lr:.1e}",
            TokS=f"{tok_per_sec:,.0f}",
            ETA=eta_str,
        )

    total_time = time.perf_counter() - global_start
    best_ppl = math.exp(best_val_loss) if best_val_loss < 20 else float("inf")

    print(f"\n{'='*60}")
    print(f"  Pretraining Summary")
    print(f"{'='*60}")
    print(f"  Total steps         : {max_iters - start_iter:,}")
    print(f"  Best val loss       : {best_val_loss:.4f}")
    print(f"  Best val PPL        : {best_ppl:.2f}")
    print(f"  Total tokens trained: {total_tokens_trained:,}")
    print(f"  Total time          : {_fmt_time(total_time)}")
    print(f"  Best checkpoint     : {best_ckpt_path}")
    print(f"  Latest checkpoint   : {latest_ckpt_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def _run_benchmark(model, train_loader, optimizer, scaler, ctx, device, steps, grad_accum_steps):
    model.train()
    iterator = iter(train_loader)
    tokens = 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(steps):
        for _ in range(grad_accum_steps):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y = next(iterator)
            x, y = x.to(device), y.to(device)
            tokens += x.numel()
            with ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = max(1e-6, time.perf_counter() - t0)
    tps = tokens / elapsed
    print(f"Benchmark  : {tps:,.0f} tok/s  ({steps} optimizer steps)")
    print(f"12h budget : {tps * 12 * 3600:,.0f} tokens")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Axelion causal language-model pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", default="1b", choices=["micro", "mini", "medium", "large", "1b", "1B"],
                        help="Model size preset")
    parser.add_argument("--train-bin", default="data/pretraining/train.bin",
                        help="Tokenized memmap binary for training")
    parser.add_argument("--val-bin", default=None,
                        help="Tokenized memmap binary for validation (optional)")
    parser.add_argument("--data-txt", default=None,
                        help="Raw text file to auto-tokenize into --train-bin")
    parser.add_argument("--val-txt", default=None,
                        help="Raw text file to auto-tokenize into --val-bin")
    parser.add_argument("--checkpoint-out", default="checkpoints/axelion-base-1b.pt",
                        help="Output checkpoint path (best → -best.pt, latest → -latest.pt)")
    parser.add_argument("--resume", default=None, help="Resume from existing checkpoint")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-iters", type=int, default=10000, help="Total optimizer steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--eval-iters", type=int, default=10, help="Batches per eval")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="LR warm-up steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seq-len", type=int, default=None, help="Override context length")
    parser.add_argument("--device", default=None, help="Force device ('cpu' | 'cuda')")
    parser.add_argument("--log-file", default=None, help="Tee stdout to this log file")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Run N benchmark steps to measure throughput then exit")
    args = parser.parse_args()

    pretrain(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        data_txt=args.data_txt,
        val_txt=args.val_txt,
        checkpoint_out=args.checkpoint_out,
        resume_from=args.resume,
        size=args.size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_file=args.log_file,
        benchmark_steps=args.benchmark_steps,
    )
