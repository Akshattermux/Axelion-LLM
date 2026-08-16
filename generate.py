"""
Root generation script for Axelion.
Uses `axelion.generation` and `axelion.model`.
"""

import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from axelion.generation import stream_generate, generate_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_candidates = [
        "checkpoints/axelion-instruct-v0.1.pt",
        "checkpoints/axelion-base-v0.1.pt",
        "ckpt.pt",
    ]

    ckpt_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand):
            ckpt_path = cand
            break

    tokenizer = AxelionTokenizer()

    if ckpt_path is None:
        print("No checkpoint found. Initializing micro model for testing.")
        config = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)
        model = Axelion(config).to(device)
    else:
        print(f"Loading model checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
        model = Axelion(config).to(device)
        model.load_checkpoint_weights(checkpoint["model"])

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M | Context: {config.max_seq_len}")

    prompt = "What is Python?"
    is_instruct = "instruct" in (ckpt_path or "")

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

    print(f"\nPrompt:\n{prompt}")
    print("\nAxelion Streaming Output:")
    for chunk in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop_token_ids=stop_ids,
        device=device,
    ):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
