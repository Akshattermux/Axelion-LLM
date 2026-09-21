import argparse
import json
import math
import os
import sys
from typing import Optional, Tuple, List

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
from axelion.generation import generate_text
from training.dataset import create_pretrain_dataloader, create_sft_dataloader

BENCHMARK_PROMPTS = [
    "What is Python?",
    "Explain recursion.",
    "What is binary search?",
    "Write a Python function to reverse a string.",
    "What is an operating system?",
    "Explain TCP vs UDP.",
]


def load_fixed_evaluations(eval_dir: str):
    """Load the small, stable prompt suite used for checkpoint comparisons."""
    prompts = []
    if not os.path.isdir(eval_dir):
        return prompts
    for path in sorted(os.listdir(eval_dir)):
        if not path.endswith(".jsonl"):
            continue
        with open(os.path.join(eval_dir, path), "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    prompts.append((item["category"], item["prompt"]))
    return prompts


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[Axelion, ModelConfig, AxelionTokenizer]:
    tokenizer = AxelionTokenizer()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
        model = Axelion(config).to(device)
        model.load_checkpoint_weights(ckpt["model"])
    else:
        print(f"Warning: {checkpoint_path} not found. Using initialized micro model.")
        config = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)
        model = Axelion(config).to(device)
    model.eval()
    return model, config, tokenizer


def evaluate_perplexity_pretrain(model, bin_file: str, config: ModelConfig, device: str = "cpu", max_batches: int = 50):
    if not os.path.exists(bin_file):
        return None, None
    loader = create_pretrain_dataloader(bin_file, config, batch_size=4, shuffle=False)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            if i >= max_batches:
                break
            X, Y = X.to(device), Y.to(device)
            _, loss, _ = model(X, targets=Y)
            total_loss += loss.item()
            count += 1
    if count == 0:
        return None, None
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def evaluate_prompts(model, tokenizer, is_instruct: bool = False, device: str = "cpu"):
    print("\n" + "=" * 60)
    print(f"Evaluation Prompts Benchmark ({'INSTRUCT' if is_instruct else 'BASE'})")
    print("=" * 60)

    for p in BENCHMARK_PROMPTS:
        if is_instruct:
            messages = [
                {"role": "system", "content": "You are Axelion, an AI assistant created by Axionik."},
                {"role": "user", "content": p},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            stop_tokens = [tokenizer.end_id, tokenizer.pad_id]
        else:
            prompt = f"Question: {p}\nAnswer:"
            stop_tokens = [tokenizer.pad_id]

        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop_token_ids=stop_tokens,
            device=device,
        )

        print(f"\n[Prompt]: {p}")
        print(f"[Output]:\n{output.strip()}")
        print("-" * 50)


def evaluate_fixed_suite(model, tokenizer, eval_dir: str, is_instruct: bool, device: str):
    prompts = load_fixed_evaluations(eval_dir)
    if not prompts:
        return
    print("\n" + "=" * 60)
    print(f"Fixed domain suite: {len(prompts)} prompts")
    print("=" * 60)
    for category, prompt_text in prompts:
        if is_instruct:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True)
            stop_tokens = [tokenizer.end_id, tokenizer.pad_id]
        else:
            prompt = f"Question: {prompt_text}\nAnswer:"
            stop_tokens = [tokenizer.pad_id]
        output = generate_text(
            model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=160,
            temperature=0.0, top_p=0.9, stop_token_ids=stop_tokens, device=device,
        )
        print(f"\n[{category}] {prompt_text}\n{output.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Axelion Model Baseline or Instruct")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/axelion-base-v0.1.pt", help="Path to checkpoint")
    parser.add_argument("--val-bin", type=str, default="data/pretraining/validation.bin", help="Validation binary path")
    parser.add_argument("--eval-dir", type=str, default="tests", help="Directory containing fixed JSONL evaluation prompts")
    parser.add_argument("--instruct", action="store_true", help="Run in Instruct mode with chat template")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} on {device}...")
    model, config, tokenizer = load_model(args.checkpoint, device=device)

    total_params = model.get_num_params()
    print(f"Axelion Model Parameters: {total_params / 1e6:.2f}M")
    print(f"Context Length: {config.max_seq_len} | Vocab Size: {config.vocab_size}")

    if os.path.exists(args.val_bin):
        loss, ppl = evaluate_perplexity_pretrain(model, args.val_bin, config, device=device)
        if loss is not None:
            print(f"Validation Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    evaluate_prompts(model, tokenizer, is_instruct=args.instruct, device=device)
    evaluate_fixed_suite(model, tokenizer, args.eval_dir, args.instruct, device)


if __name__ == "__main__":
    main()
