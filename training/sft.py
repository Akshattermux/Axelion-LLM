import contextlib
import math
import os
import sys
import time
from typing import Optional

import torch

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from training.dataset import create_sft_dataloader

# ----------------- SFT Hyperparameters -----------------
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
SFT_LR = 3e-5
MIN_LR = 1e-6
EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ("float16" if torch.cuda.is_available() else "float32")
PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]


def train_sft(
    base_checkpoint: str = "checkpoints/axelion-base-v0.1.pt",
    output_checkpoint: str = "checkpoints/axelion-instruct-v0.1.pt",
    train_jsonl: str = "data/instruction/train.jsonl",
    val_jsonl: str = "data/instruction/validation.jsonl",
    epochs: int = EPOCHS,
    lr: float = SFT_LR,
):
    print("=" * 60)
    print(f"Axelion Supervised Fine-Tuning (SFT) | Device: {DEVICE} | Precision: {DTYPE}")
    print("=" * 60)

    ctx = (
        torch.autocast(device_type="cuda", dtype=PTDTYPE)
        if DEVICE == "cuda"
        else (torch.autocast(device_type="cpu", dtype=torch.bfloat16) if DTYPE == "bfloat16" else contextlib.nullcontext())
    )

    tokenizer = AxelionTokenizer()

    # 1. Load Base Checkpoint
    if os.path.exists(base_checkpoint):
        print(f"Loading Base Checkpoint from: {base_checkpoint}")
        ckpt = torch.load(base_checkpoint, map_location=DEVICE, weights_only=False)
        cfg = ckpt.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
        model = Axelion(cfg).to(DEVICE)
        model.load_checkpoint_weights(ckpt["model"])
        print("Loaded base checkpoint successfully.")
    elif os.path.exists("ckpt.pt"):
        print("Loading from root ckpt.pt...")
        ckpt = torch.load("ckpt.pt", map_location=DEVICE, weights_only=False)
        cfg = ckpt.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
        model = Axelion(cfg).to(DEVICE)
        model.load_checkpoint_weights(ckpt["model"])
    else:
        print("No base checkpoint found. Initializing micro model for testing SFT flow...")
        cfg = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)
        model = Axelion(cfg).to(DEVICE)

    # 2. Prepare DataLoaders
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"Training dataset not found: {train_jsonl}")

    train_loader = create_sft_dataloader(train_jsonl, tokenizer, cfg.max_seq_len, BATCH_SIZE, shuffle=True)
    val_loader = (
        create_sft_dataloader(val_jsonl, tokenizer, cfg.max_seq_len, BATCH_SIZE, shuffle=False)
        if os.path.exists(val_jsonl)
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16" and DEVICE == "cuda"))

    total_steps = (len(train_loader) * epochs) // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    print(f"Total training samples: {len(train_loader.dataset)} | Steps: {total_steps} | Peak LR: {lr:.2e}")
    best_val_loss = float("inf")
    step = 0

    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        acc_loss = 0.0

        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            with ctx:
                logits, loss, _ = model(X, targets=Y, ignore_index=-100)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if DEVICE == "cuda" and DTYPE == "float16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            acc_loss += loss.item()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                if DEVICE == "cuda" and DTYPE == "float16":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                # Cosine warmup & decay
                step += 1
                if step < warmup_steps:
                    curr_lr = lr * step / warmup_steps
                else:
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    curr_lr = MIN_LR + 0.5 * (lr - MIN_LR) * (1 + math.cos(math.pi * progress))

                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr

                running_loss += acc_loss
                print(f"Epoch {epoch+1}/{epochs} | Step {step:4d}/{total_steps} | Train Loss: {acc_loss*GRADIENT_ACCUMULATION_STEPS:.4f} | LR: {curr_lr:.2e}", flush=True)
                acc_loss = 0.0

        # Epoch Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for Xv, Yv in val_loader:
                    Xv, Yv = Xv.to(DEVICE), Yv.to(DEVICE)
                    with ctx:
                        _, v_loss, _ = model(Xv, targets=Yv, ignore_index=-100)
                    val_loss += v_loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(1, val_batches)
            ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
            print(f"\n---> Epoch {epoch+1} Complete | Val Loss: {avg_val_loss:.4f} | Perplexity: {ppl:.2f} <---")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_ppl": ppl,
                }
                torch.save(checkpoint, output_checkpoint)
                print(f" Saved Best Instruct Checkpoint: {output_checkpoint}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Axelion SFT Fine-Tuning")
    parser.add_argument("--base-ckpt", type=str, default="checkpoints/axelion-base-v0.1.pt")
    parser.add_argument("--output-ckpt", type=str, default="checkpoints/axelion-instruct-v0.1.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    train_sft(
        base_checkpoint=args.base_ckpt,
        output_checkpoint=args.output_ckpt,
        epochs=args.epochs,
        lr=args.lr,
    )
