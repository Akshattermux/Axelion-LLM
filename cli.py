"""
Axelion LLM Unified Developer CLI.
Commands:
  python cli.py params [--size large]
  python cli.py train [--size micro] [--data train_data.txt] [--max-iters 100]
  python cli.py sft [--base ckpt.pt] [--data data/instruction/train.jsonl]
  python cli.py serve [--port 8000] [--size micro]
  python cli.py chat [--ckpt ckpt.pt]
  python cli.py agent [--query "..."]
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from axelion.generation import stream_generate


def cmd_params(args):
    tokenizer = AxelionTokenizer()
    configs = {
        "micro": ModelConfig.create_micro(tokenizer.vocab_size),
        "mini": ModelConfig.create_mini_gpt(tokenizer.vocab_size),
        "medium": ModelConfig.create_medium(tokenizer.vocab_size),
        "large": ModelConfig.create_large(tokenizer.vocab_size),
        "3b": ModelConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=4096,
            dim=3072,
            n_layers=32,
            n_heads=24,
            n_kv_heads=8,
            hidden_mult=8.0 / 3.0,
            multiple_of=256,
        ),
        "7b": ModelConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=4096,
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            hidden_mult=8.0 / 3.0,
            multiple_of=256,
        ),
    }

    print("\n" + "=" * 80)
    print("                      AXELION MODEL PARAMETER BREAKDOWN")
    print("=" * 80)
    print(f"{'Preset':<10} | {'Layers':<6} | {'Dim':<5} | {'Q/KV Heads':<10} | {'FFN Dim':<8} | {'Total Params':<14} | {'Billions':<9} | {'FP16 Memory'}")
    print("-" * 80)

    for name, cfg in configs.items():
        total_p = cfg.calculate_params()
        billions = total_p / 1e9
        mem_mb = (total_p * 2) / (1024 * 1024)
        mem_str = f"{mem_mb:.1f} MB" if mem_mb < 1024 else f"{mem_mb / 1024:.2f} GB"
        q_kv = f"{cfg.n_heads}/{cfg.n_kv_heads}"
        print(f"{name.upper():<10} | {cfg.n_layers:<6} | {cfg.dim:<5} | {q_kv:<10} | {cfg.ffn_dim:<8} | {total_p:>12,} | {billions:>7.3f}B | {mem_str}")

    print("=" * 80)
    print("Note: Param counts account for tied embeddings (tok_emb == lm_head), RoPE, and SwiGLU.\n")


def cmd_chat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AxelionTokenizer()
    ckpt_path = args.ckpt

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
        model = Axelion(cfg).to(device)
        model.load_checkpoint_weights(ckpt["model"])
    else:
        print("No checkpoint found. Initializing micro model for interactive test.")
        cfg = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)
        model = Axelion(cfg).to(device)

    print("\nAxelion Chat Ready! Type 'exit' or 'quit' to end.\n" + "-" * 50)
    messages = [{"role": "system", "content": "You are Axelion, an intelligent AI assistant."}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input or user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        print("Axelion: ", end="", flush=True)
        response_chunks = []
        for chunk in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop_token_ids=[tokenizer.end_id, tokenizer.pad_id],
            device=device,
        ):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
        print("\n")
        messages.append({"role": "assistant", "content": "".join(response_chunks)})


def main():
    parser = argparse.ArgumentParser(description="Axelion LLM Command-Line Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # params
    p_params = subparsers.add_parser("params", help="Show exact parameter counts and memory estimates across presets")

    # train
    p_train = subparsers.add_parser("train", help="Run causal pretraining")
    p_train.add_argument("--size", default="mini", choices=["micro", "mini", "medium", "large"])
    p_train.add_argument("--data", default="train_data.txt")
    p_train.add_argument("--bin", default="data/pretraining/train.bin")
    p_train.add_argument("--val-bin", default="data/pretraining/validation.bin")
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--grad-accum", type=int, default=16)
    p_train.add_argument("--max-iters", type=int, default=10000)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--min-lr", type=float, default=3e-5)
    p_train.add_argument("--warmup-steps", type=int, default=500)
    p_train.add_argument("--benchmark-steps", type=int, default=0)
    p_train.add_argument("--device", default=None)

    # sft
    p_sft = subparsers.add_parser("sft", help="Run Supervised Fine-Tuning")
    p_sft.add_argument("--base-ckpt", default="checkpoints/axelion-base-v0.2.pt")
    p_sft.add_argument("--output-ckpt", default="checkpoints/axelion-instruct-v0.1.pt")
    p_sft.add_argument("--data", default="data/instruction/train.jsonl")
    p_sft.add_argument("--epochs", type=int, default=3)
    p_sft.add_argument("--lr", type=float, default=3e-5)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch OpenAI-compatible REST API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--ckpt", default=None)
    p_serve.add_argument("--size", default="micro")

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive terminal chat")
    p_chat.add_argument("--ckpt", default=None)

    # agent
    p_agent = subparsers.add_parser("agent", help="Run agent tool demonstration")
    p_agent.add_argument("--query", default="What is the stock quote for NVDA and what is 128.90 * 10?")

    args = parser.parse_args()

    if args.command == "params":
        cmd_params(args)
    elif args.command == "train":
        from training.pretrain import pretrain
        pretrain(
            train_bin=args.bin,
            val_bin=args.val_bin,
            data_txt=args.data if os.path.exists(args.data) else None,
            size=args.size,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            max_iters=args.max_iters,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            device=args.device,
        )
    elif args.command == "sft":
        from training.sft import train_sft
        train_sft(
            base_checkpoint=args.base_ckpt,
            output_checkpoint=args.output_ckpt,
            train_jsonl=args.data,
            epochs=args.epochs,
            lr=args.lr,
        )
    elif args.command == "serve":
        import uvicorn
        from axelion.server import app, init_model
        init_model(checkpoint_path=args.ckpt, size=args.size)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "agent":
        from agent_demo import main as agent_main
        agent_main()


if __name__ == "__main__":
    main()
