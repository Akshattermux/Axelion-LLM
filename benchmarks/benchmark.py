import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from axelion.generation import generate_text

BENCHMARK_SUITE = {
    "knowledge": [
        "What is an operating system?",
        "Explain TCP vs UDP.",
        "What is the difference between a process and a thread?",
    ],
    "coding": [
        "What is Python?",
        "Write a Python function to reverse a string.",
        "How do you implement binary search in Python?",
    ],
    "reasoning": [
        "Explain recursion.",
        "What is the time complexity of QuickSort in the average case?",
        "Why is KV caching beneficial for autoregressive generation?",
    ],
    "instruction_following": [
        "List three key architectural features of the Axelion Transformer.",
        "Summarize the purpose of RMSNorm in one sentence.",
    ],
}


def run_benchmark(model, tokenizer, is_instruct: bool = False, device: str = "cpu"):
    results = {}
    print(f"\n--- Running Axelion Benchmark Suite (Mode: {'Instruct' if is_instruct else 'Base'}) ---")

    for category, prompts in BENCHMARK_SUITE.items():
        results[category] = []
        print(f"\nCategory: {category.upper()}")
        for prompt in prompts:
            if is_instruct:
                messages = [
                    {"role": "system", "content": "You are Axelion, an AI assistant created by Axionik."},
                    {"role": "user", "content": prompt},
                ]
                formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                stop_ids = [tokenizer.end_id, tokenizer.pad_id]
            else:
                formatted = f"Question: {prompt}\nAnswer:"
                stop_ids = [tokenizer.pad_id]

            t0 = time.time()
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                stop_token_ids=stop_ids,
                device=device,
            )
            elapsed = time.time() - t0

            print(f"  [Q]: {prompt}", flush=True)
            print(f"  [A]: {output.strip()[:180]}...", flush=True)
            print(f"  [Time]: {elapsed:.2f}s\n", flush=True)

            results[category].append({
                "prompt": prompt,
                "output": output.strip(),
                "latency_sec": round(elapsed, 3),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Axelion Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/axelion-base-v0.1.pt")
    parser.add_argument("--output", type=str, default="benchmarks/baseline.json")
    parser.add_argument("--instruct", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AxelionTokenizer()

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
        model = Axelion(config).to(device)
        model.load_checkpoint_weights(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Running initialized test model.")
        config = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)
        model = Axelion(config).to(device)

    model.eval()
    results = run_benchmark(model, tokenizer, is_instruct=args.instruct, device=device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()
