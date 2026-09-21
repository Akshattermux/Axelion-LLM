"""
Axelion Autonomous Agent Demonstration.
Demonstrates how to connect Axelion with external tools (calculator, stock price, lookup)
and execute an autonomous decision loop.
"""

import os
import sys
import torch

# Add current dir to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from axelion.agent import AxelionAgent, ToolRegistry


def build_tools() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(description="Evaluates a mathematical expression and returns the number.")
    def calculate(expression: str) -> str:
        try:
            # Safe evaluation for basic math
            allowed_names = {"__builtins__": None}
            res = eval(expression, allowed_names, {})
            return f"Result: {res}"
        except Exception as e:
            return f"Calculation error: {e}"

    @registry.tool(description="Fetches current stock price and financial status for a ticker symbol (e.g. AAPL, NVDA, TSLA).")
    def get_stock_quote(ticker: str) -> str:
        mock_data = {
            "AAPL": "Apple Inc. (AAPL): $234.50 (+1.2%), Market Cap: $3.58T",
            "NVDA": "NVIDIA Corp (NVDA): $128.90 (+3.4%), Market Cap: $3.16T",
            "TSLA": "Tesla Inc. (TSLA): $248.20 (-0.8%), Market Cap: $790B",
            "MSFT": "Microsoft Corp (MSFT): $448.10 (+0.5%), Market Cap: $3.32T",
        }
        return mock_data.get(ticker.upper(), f"Ticker '{ticker}' quote: $100.00 (Simulated).")

    @registry.tool(description="Searches company database for information about products and roadmap.")
    def lookup_docs(keyword: str) -> str:
        database = {
            "axelion": "Axelion is a decoder-only LLM featuring RoPE, GQA, SwiGLU, and KV caching.",
            "axionik": "Axionik is an AI technology research team advancing efficient open LLM architectures.",
        }
        return database.get(keyword.lower(), f"No direct records found for '{keyword}'.")

    return registry


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AxelionTokenizer()

    ckpt_candidates = [
        "checkpoints/axelion-instruct-v0.1.pt",
        "checkpoints/axelion-base-v0.1.pt",
        "ckpt.pt",
    ]
    ckpt_path = next((c for c in ckpt_candidates if os.path.exists(c)), None)

    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", ModelConfig.create_micro(vocab_size=tokenizer.vocab_size))
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
        model = Axelion(cfg).to(device)
        model.load_checkpoint_weights(ckpt["model"])
    else:
        print("Initializing micro Axelion model for agent demonstration...")
        cfg = ModelConfig.create_micro(vocab_size=tokenizer.vocab_size)
        model = Axelion(cfg).to(device)

    registry = build_tools()
    print("\nRegistered Agent Tools:")
    print(registry.get_tool_descriptions())

    agent = AxelionAgent(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        max_steps=4,
        device=device,
    )

    query = "What is the current stock price of NVDA and what is (128.90 * 10)?"
    result = agent.run(query)

    print("=" * 60)
    print("Agent Run Summary:")
    print(f"Completed: {result['completed']}")
    print(f"Total Tool Steps: {len(result['steps'])}")
    for i, s in enumerate(result["steps"]):
        print(f"  Step {i+1}: called '{s.tool_name}' with {s.tool_args} -> {s.observation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
