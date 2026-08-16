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

# ----------------- Pretraining Hyperparameters -----------------
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_ITERS = 1000
EVAL_INTERVAL = 100
EVAL_ITERS = 20
MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ("float16" if torch.cuda.is_available() else "float32")
PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]


def cosine_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    """Cosine learning rate decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def estimate_loss(model, dataloader, ctx, eval_iters=EVAL_ITERS):
    model.eval()
    losses = torch.zeros(eval_iters)
    iterator = iter(dataloader)
    for k in range(eval_iters):
        try:
            X, Y = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            X, Y = next(iterator)
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        with torch.no_grad():
            with ctx:
                _, loss, _ = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def pretrain(
    train_bin: str = "data/pretraining/train.bin",
    val_bin: Optional[str] = "data/pretraining/validation.bin",
    checkpoint_out: str = "checkpoints/axelion-base-v0.2.pt",
    resume_from: Optional[str] = None,
):
    print("=" * 60)
    print(f"Axelion Pretraining | Device: {DEVICE} | Precision: {DTYPE}")
    print("=" * 60)

    ctx = (
        torch.autocast(device_type="cuda", dtype=PTDTYPE)
        if DEVICE == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )

    tokenizer = AxelionTokenizer()

    # Ensure train binary exists or create sample
    if not os.path.exists(train_bin):
        os.makedirs(os.path.dirname(train_bin), exist_ok=True)
        sample_txt = "data/pretraining/train.txt"
        if not os.path.exists(sample_txt):
            with open(sample_txt, "w", encoding="utf-8") as f:
                f.write("Axelion is a custom high-performance decoder-only transformer language model.\n" * 200)
        prepare_binary_dataset(sample_txt, train_bin, tokenizer)

    config = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming pretraining from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=DEVICE)
        config = ckpt.get("config", config)
        model = Axelion(config).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        print("Initializing new Axelion model from scratch...")
        model = Axelion(config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16" and DEVICE == "cuda"))

    dataloader = create_pretrain_dataloader(train_bin, config, batch_size=BATCH_SIZE)
    iterator = iter(dataloader)

    best_val_loss = float("inf")
    t0 = time.time()

    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)

    for iter_num in range(MAX_ITERS):
        lr = cosine_lr(iter_num, MAX_ITERS, MAX_LR, MIN_LR, WARMUP_STEPS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            val_loss = estimate_loss(model, dataloader, ctx)
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"Step {iter_num:4d} | Train Loss: {val_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")

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

        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                xb, yb = next(iterator)

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            with ctx:
                logits, loss, _ = model(xb, targets=yb)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if DEVICE == "cuda" and DTYPE == "float16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if DEVICE == "cuda" and DTYPE == "float16":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if iter_num % 50 == 0 and iter_num > 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Iter {iter_num:4d} | Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | Step Time: {dt*1000/50:.1f}ms")


if __name__ == "__main__":
    pretrain()
