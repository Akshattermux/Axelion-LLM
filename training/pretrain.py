import argparse
import contextlib
import math
import os
import sys
import time
from typing import Optional

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from training.dataset import create_pretrain_dataloader, prepare_binary_dataset

def cosine_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))

def _autocast_context(device: str, dtype: torch.dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=dtype)
    return contextlib.nullcontext()

def estimate_loss(model, dataloader, ctx, device, eval_iters=10):
    model.eval()
    losses = []
    iterator = iter(dataloader)
    with torch.no_grad():
        for _ in range(eval_iters):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            with ctx:
                _, loss, _ = model(inputs, targets=targets)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))

def _ensure_binary_dataset(text_path: Optional[str], bin_path: str, tokenizer: AxelionTokenizer):
    if text_path and os.path.exists(text_path):
        prepare_binary_dataset(text_path, bin_path, tokenizer)
    elif not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Missing {bin_path}. Provide --data-txt or run "
            "python scripts/prepare_data.py --mode pretrain."
    )

def benchmark_throughput(model, dataloader, optimizer, scaler, ctx, device, steps, grad_accum_steps):
    """Measure optimizer-step throughput before committing to a long run."""
    model.train()
    iterator = iter(dataloader)
    tokens = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(steps):
        for _ in range(grad_accum_steps):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            tokens += inputs.numel()
            with ctx:
                _, loss, _ = model(inputs, targets=targets)
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
    elapsed = max(1e-6, time.perf_counter() - started)
    tokens_per_second = tokens / elapsed
    print(f"Benchmark: {tokens_per_second:,.0f} tokens/sec ({steps} optimizer steps)")
    print(f"12-hour token budget at this rate: {tokens_per_second * 12 * 60 * 60:,.0f} tokens")
    return tokens_per_second

def pretrain(
    train_bin: str = "data/pretraining/train.bin",
    val_bin: str = "data/pretraining/validation.bin",
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
    benchmark_steps: int = 0,
    device: Optional[str] = None,
):
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = "bfloat16" if selected_device == "cuda" and torch.cuda.is_bf16_supported() else ("float16" if selected_device == "cuda" else "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    ctx = _autocast_context(selected_device, ptdtype)
    tokenizer = AxelionTokenizer()

    if size.lower() != "mini":
        print(f"Warning: requested {size.upper()}; the recommended first base run is MINI.")
    preset_map = {"micro": ModelConfig.create_micro, "mini": ModelConfig.create_mini_gpt, "medium": ModelConfig.create_medium, "large": ModelConfig.create_large}
    if size.lower() not in preset_map:
        raise ValueError(f"Unknown model size: {size}")
    config = preset_map[size.lower()](vocab_size=tokenizer.vocab_size)
    if seq_len:
        config.max_seq_len = seq_len
    if size.lower() == "mini":
        config.gradient_checkpointing = True

    _ensure_binary_dataset(data_txt, train_bin, tokenizer)
    _ensure_binary_dataset(val_txt, val_bin, tokenizer)

    start_iter = 0
    best_val_loss = float("inf")

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=selected_device, weights_only=False)
        config = checkpoint.get("config", config)
    model = Axelion(config).to(selected_device)
    train_loader = create_pretrain_dataloader(train_bin, config, batch_size=batch_size)
    val_loader = create_pretrain_dataloader(val_bin, config, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype_str == "float16" and selected_device == "cuda"))

    if resume_from:
        model.load_checkpoint_weights(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        start_iter = checkpoint.get("iter_num", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))

    print(f"Axelion {size.upper()} causal pretraining | {model.get_num_params() / 1e6:.2f}M params | {selected_device} | {dtype_str}")
    if benchmark_steps > 0:
        benchmark_throughput(model, train_loader, optimizer, scaler, ctx, selected_device, benchmark_steps, grad_accum_steps)
        return

    iterator = iter(train_loader)
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    started = time.perf_counter()

    for iter_num in range(start_iter, max_iters):
        lr = cosine_lr(iter_num, max_iters, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            train_loss = estimate_loss(model, train_loader, ctx, selected_device, eval_iters)
            val_loss = estimate_loss(model, val_loader, ctx, selected_device, eval_iters)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"Step {iter_num:6d}/{max_iters} | Train {train_loss:.4f} | Val {val_loss:.4f} | PPL {val_ppl:.2f} | LR {lr:.2e}")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "config": config,
                "iter_num": iter_num,
                "best_val_loss": min(best_val_loss, val_loss),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "tokens_per_step": batch_size * grad_accum_steps * config.max_seq_len,
            }
            torch.save(checkpoint, checkpoint_out)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saved best checkpoint: {checkpoint_out}")

        for _ in range(grad_accum_steps):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(selected_device), targets.to(selected_device)
            with ctx:
                _, loss, _ = model(inputs, targets=targets)
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

        if (iter_num + 1) % 25 == 0:
            elapsed = max(1e-6, time.perf_counter() - started)
            tokens = (iter_num - start_iter + 1) * batch_size * grad_accum_steps * config.max_seq_len
            print(f"Throughput: {tokens / elapsed:,.0f} tokens/sec", flush=True)

    print(f"Pretraining complete. Checkpoint: {checkpoint_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axelion causal language-model pretraining")
    parser.add_argument("--size", default="mini", choices=["micro", "mini", "medium", "large"])
    parser.add_argument("--train-bin", default="data/pretraining/train.bin")
    parser.add_argument("--val-bin", default="data/pretraining/validation.bin")
    parser.add_argument("--data-txt", default=None)
    parser.add_argument("--val-txt", default=None)
    parser.add_argument("--checkpoint-out", default="checkpoints/axelion-base-mini.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=0, help="Measure throughput and exit")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    pretrain(
        train_bin=args.train_bin, val_bin=args.val_bin, data_txt=args.data_txt, val_txt=args.val_txt,
        checkpoint_out=args.checkpoint_out, resume_from=args.resume, size=args.size, seq_len=args.seq_len,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum, max_iters=args.max_iters,
        eval_interval=args.eval_interval, eval_iters=args.eval_iters, max_lr=args.lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps, benchmark_steps=args.benchmark_steps, device=args.device,
    )
import argparse
import contextlib
import math
import os
import sys
import time
from typing import Optional

import torch

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from training.dataset import create_pretrain_dataloader, prepare_binary_dataset


def cosine_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    """Cosine learning rate decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def estimate_loss(model, dataloader, ctx, device, eval_iters=10):
    model.eval()
    losses = torch.zeros(eval_iters)
    iterator = iter(dataloader)
    for k in range(eval_iters):
        try:
            X, Y = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            X, Y = next(iterator)
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            with ctx:
                _, loss, _ = model(X, targets=Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def pretrain(
    train_bin: str = "data/pretraining/train.bin",
    data_txt: Optional[str] = None,
    checkpoint_out: str = "checkpoints/axelion-base-v0.2.pt",
    resume_from: Optional[str] = None,
    size: str = "micro",
    seq_len: Optional[int] = None,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_iters: int = 500,
    eval_interval: int = 50,
    max_lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_steps: int = 50,
    weight_decay: float = 0.1,
    device: Optional[str] = None,
):
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = "bfloat16" if selected_device == "cuda" and torch.cuda.is_bf16_supported() else ("float16" if selected_device == "cuda" else "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]

    print("=" * 60)
    print(f"Axelion Pretraining | Model Preset: {size.upper()} | Device: {selected_device} | Precision: {dtype_str}")
    print("=" * 60)

    ctx = (
        torch.autocast(device_type="cuda", dtype=ptdtype)
        if selected_device == "cuda"
        else (torch.autocast(device_type="cpu", dtype=torch.bfloat16) if dtype_str == "bfloat16" else contextlib.nullcontext())
    )

    tokenizer = AxelionTokenizer()

    # If raw text was passed or binary does not exist, prepare binary dataset
    if data_txt and os.path.exists(data_txt):
        prepare_binary_dataset(data_txt, train_bin, tokenizer)
    elif not os.path.exists(train_bin):
        os.makedirs(os.path.dirname(train_bin), exist_ok=True)
        sample_txt = "data/pretraining/train.txt"
        if not os.path.exists(sample_txt):
            with open(sample_txt, "w", encoding="utf-8") as f:
                f.write("Axelion is a custom high-performance decoder-only transformer language model.\n" * 500)
        prepare_binary_dataset(sample_txt, train_bin, tokenizer)

    # Initialize model configuration
    preset_map = {
        "micro": ModelConfig.create_micro,
        "mini": ModelConfig.create_mini_gpt,
        "medium": ModelConfig.create_medium,
        "large": ModelConfig.create_large,
    }
    cfg_fn = preset_map.get(size.lower(), ModelConfig.create_micro)
    config = cfg_fn(vocab_size=tokenizer.vocab_size)
    if seq_len:
        config.max_seq_len = seq_len

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming pretraining from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=selected_device, weights_only=False)
        config = ckpt.get("config", config)
        model = Axelion(config).to(selected_device)
        model.load_checkpoint_weights(ckpt["model"])
    else:
        print(f"Initializing {size.upper()} Axelion model ({config.dim} dim, {config.n_layers} layers)...")
        model = Axelion(config).to(selected_device)

    print(f"Total Model Parameters: {model.get_num_params() / 1e6:.2f}M | Context: {config.max_seq_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype_str == "float16" and selected_device == "cuda"))

    dataloader = create_pretrain_dataloader(train_bin, config, batch_size=batch_size)
    iterator = iter(dataloader)

    best_val_loss = float("inf")
    t0 = time.time()

    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)

    for iter_num in range(max_iters):
        lr = cosine_lr(iter_num, max_iters, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            val_loss = estimate_loss(model, dataloader, ctx, selected_device)
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"Step {iter_num:4d}/{max_iters} | Train Loss: {val_loss:.4f} | Perplexity: {ppl:.2f} | LR: {lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iter_num": iter_num,
                    "val_loss": best_val_loss,
                }
                torch.save(checkpoint, checkpoint_out)
                print(f" Saved checkpoint: {checkpoint_out}")

        for micro_step in range(grad_accum_steps):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                xb, yb = next(iterator)

            xb, yb = xb.to(selected_device), yb.to(selected_device)

            with ctx:
                logits, loss, _ = model(xb, targets=yb)
                loss = loss / grad_accum_steps

            if selected_device == "cuda" and dtype_str == "float16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if selected_device == "cuda" and dtype_str == "float16":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if iter_num % 25 == 0 and iter_num > 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Iter {iter_num:4d} | Current Batch Loss: {loss.item() * grad_accum_steps:.4f} | Step Time: {dt*1000/25:.1f}ms")

    print(f"\nPretraining Complete! Best checkpoint saved to {checkpoint_out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axelion Pretraining CLI")
    parser.add_argument("--size", type=str, default="micro", choices=["micro", "mini", "medium", "large"], help="Model size preset")
    parser.add_argument("--train-bin", type=str, default="data/pretraining/train.bin", help="Path to tokenized binary file")
    parser.add_argument("--data-txt", type=str, default=None, help="Raw text file to tokenize and train on directly")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/axelion-base-v0.2.pt", help="Output checkpoint file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing checkpoint")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-iters", type=int, default=500, help="Total training iterations")
    parser.add_argument("--eval-interval", type=int, default=50, help="Interval for validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--seq-len", type=int, default=None, help="Override context length")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    pretrain(
        train_bin=args.train_bin,
        data_txt=args.data_txt,
        checkpoint_out=args.checkpoint_out,
        resume_from=args.resume,
        size=args.size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        max_lr=args.lr,
        device=args.device,
    )
