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
