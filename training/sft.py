"""
training/sft.py — Axelion Supervised Fine-Tuning (SFT).

Usage:
    python -m training.sft \\
        --base-ckpt checkpoints/axelion-base-mini.pt \\
        --output-ckpt checkpoints/axelion-instruct-v1.pt \\
        --train-jsonl data/instruction/train.jsonl \\
        --val-jsonl data/instruction/validation.jsonl \\
        --epochs 3 --lr 3e-5 --log-file logs/sft.log
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from training.dataset import create_sft_dataloader


# ---------------------------------------------------------------------------
# Tee — duplicate stdout to a log file
# ---------------------------------------------------------------------------
class _Tee:
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
# LR schedule helpers
# ---------------------------------------------------------------------------
def _cosine_lr(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


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
# Main SFT entry-point
# ---------------------------------------------------------------------------
def train_sft(
    base_ckpt: str = "checkpoints/axelion-base-mini.pt",
    output_ckpt: str = "checkpoints/axelion-instruct-v1.pt",
    train_jsonl: str = "data/instruction/train.jsonl",
    val_jsonl: Optional[str] = "data/instruction/validation.jsonl",
    epochs: int = 3,
    lr: float = 3e-5,
    min_lr: float = 1e-6,
    batch_size: int = 2,
    grad_accum: int = 4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    # ---- reproducibility -------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- log tee ---------------------------------------------------------
    if log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)

    # ---- device / dtype (detected at runtime, not module-level globals) --
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = selected_device == "cuda"
    dtype_str = (
        "bfloat16" if on_cuda and torch.cuda.is_bf16_supported()
        else ("float16" if on_cuda else "float32")
    )
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    ctx = (
        torch.autocast(device_type="cuda", dtype=ptdtype)
        if on_cuda
        else (torch.autocast(device_type="cpu", dtype=torch.bfloat16) if dtype_str == "bfloat16" else contextlib.nullcontext())
    )

    # ---- dataloader knobs ------------------------------------------------
    num_workers = min(4, os.cpu_count() or 1) if on_cuda else 0
    pin_memory = on_cuda

    # ---- tokenizer -------------------------------------------------------
    tokenizer = AxelionTokenizer()

    # ---- load base checkpoint --------------------------------------------
    if os.path.exists(base_ckpt):
        print(f"Loading base checkpoint: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location=selected_device, weights_only=False)
        cfg = ckpt.get("config", ModelConfig.create_1b(vocab_size=tokenizer.vocab_size))
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
        model = Axelion(cfg).to(selected_device)
        model.load_checkpoint_weights(ckpt["model"])
        print(f"Loaded {model.get_num_params()/1e6:.2f}M param model from checkpoint.")
    else:
        print(f"[WARN] Base checkpoint not found: {base_ckpt}. Initialising 1B model.")
        cfg = ModelConfig.create_1b(vocab_size=tokenizer.vocab_size)
        model = Axelion(cfg).to(selected_device)

    print(
        f"\n{'='*60}\n"
        f"Axelion SFT | {model.get_num_params()/1e6:.2f}M params\n"
        f"Device: {selected_device} | Precision: {dtype_str} | Seed: {seed}\n"
        f"Epochs: {epochs} | Batch: {batch_size} | GradAccum: {grad_accum} | LR: {lr:.2e}\n"
        f"{'='*60}\n"
    )

    # ---- dataloaders -----------------------------------------------------
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"Training JSONL not found: {train_jsonl}")

    train_loader = create_sft_dataloader(
        train_jsonl, tokenizer, cfg.max_seq_len, batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = None
    if val_jsonl and os.path.exists(val_jsonl):
        val_loader = create_sft_dataloader(
            val_jsonl, tokenizer, cfg.max_seq_len, batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        print(f"[WARN] val_jsonl not found ({val_jsonl}); skipping validation.")

    # ---- optimizer / scaler ---------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype_str == "float16" and on_cuda))

    # ---- LR schedule parameters -----------------------------------------
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    print(f"Train samples : {len(train_loader.dataset):,}")
    print(f"Total steps   : {total_steps:,}  (warmup: {warmup_steps})\n")

    # ---- checkpoint paths ------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_ckpt)), exist_ok=True)
    best_ckpt = output_ckpt  # "best" uses the primary output path
    latest_ckpt = output_ckpt.replace(".pt", "-latest.pt")

    best_val_loss = float("inf")
    global_step = 0
    global_start = time.perf_counter()

    # ---- epoch loop ------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=120,
            unit="batch",
        )

        acc_loss = 0.0
        batch_start = time.perf_counter()

        for batch_idx, (x, y) in bar:
            x, y = x.to(selected_device), y.to(selected_device)

            with ctx:
                _, loss, _ = model(x, targets=y, ignore_index=-100)
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            acc_loss += loss.item()

            is_accum_step = (batch_idx + 1) % grad_accum == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)

            if is_accum_step or is_last_batch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                curr_lr = _cosine_lr(global_step, total_steps, lr, min_lr, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr

                elapsed = max(1e-6, time.perf_counter() - batch_start)
                remaining = (total_steps - global_step) * (elapsed / max(1, global_step))
                bar.set_postfix(
                    Loss=f"{acc_loss * grad_accum:.4f}",
                    LR=f"{curr_lr:.1e}",
                    ETA=_fmt_time(remaining),
                )
                acc_loss = 0.0

        # ---- epoch validation --------------------------------------------
        epoch_val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            total_v, count_v = 0.0, 0
            with torch.no_grad():
                for xv, yv in tqdm(val_loader, desc="  Validating", ncols=80, leave=False):
                    xv, yv = xv.to(selected_device), yv.to(selected_device)
                    with ctx:
                        _, v_loss, _ = model(xv, targets=yv, ignore_index=-100)
                    total_v += v_loss.item()
                    count_v += 1
            epoch_val_loss = total_v / max(1, count_v)
            val_ppl = math.exp(epoch_val_loss) if epoch_val_loss < 20 else float("inf")
            tqdm.write(
                f"\nEpoch {epoch+1} | Val Loss: {epoch_val_loss:.4f} | Val PPL: {val_ppl:.2f}"
            )

        # ---- save latest -------------------------------------------------
        ckpt_data = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_loss": epoch_val_loss,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        }
        torch.save(ckpt_data, latest_ckpt)
        tqdm.write(f"  Saved latest → {latest_ckpt}")

        # ---- save best ---------------------------------------------------
        cmp = epoch_val_loss if val_loader is not None else float("inf")
        if cmp < best_val_loss or val_loader is None:
            best_val_loss = cmp
            torch.save(ckpt_data, best_ckpt)
            tqdm.write(f"  ✓ New best ({best_val_loss:.4f}) → {best_ckpt}")

    # ---- summary ---------------------------------------------------------
    total_time = time.perf_counter() - global_start
    best_ppl = math.exp(best_val_loss) if best_val_loss < 20 else float("inf")

    print(f"\n{'='*60}")
    print(f"  SFT Summary")
    print(f"{'='*60}")
    print(f"  Total epochs    : {epochs}")
    print(f"  Best val loss   : {best_val_loss:.4f}")
    print(f"  Best val PPL    : {best_ppl:.2f}")
    print(f"  Total time      : {_fmt_time(total_time)}")
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"  Latest ckpt     : {latest_ckpt}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Axelion Supervised Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-ckpt", default="checkpoints/axelion-base-1b.pt",
                        help="Base pretrained checkpoint to fine-tune")
    parser.add_argument("--output-ckpt", default="checkpoints/axelion-instruct-1b.pt",
                        help="Output checkpoint (best). Latest saved with -latest suffix.")
    parser.add_argument("--train-jsonl", default="data/instruction/train.jsonl",
                        help="Training JSONL (messages format)")
    parser.add_argument("--val-jsonl", default=None,
                        help="Validation JSONL (optional)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR after decay")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Fraction of total steps used for LR warm-up")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="'cpu' | 'cuda'")
    parser.add_argument("--log-file", default=None, help="Tee stdout to this log file")
    args = parser.parse_args()

    train_sft(
        base_ckpt=args.base_ckpt,
        output_ckpt=args.output_ckpt,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_file=args.log_file,
    )
