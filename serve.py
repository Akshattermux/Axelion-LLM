"""
Axelion API Server Launcher
Run with: python serve.py [--port 8000] [--host 0.0.0.0] [--ckpt checkpoints/axelion-instruct-v0.1.pt] [--size micro]
"""

import argparse
import uvicorn
from axelion.server import app, init_model


def main():
    parser = argparse.ArgumentParser(description="Serve Axelion LLM as an OpenAI-compatible REST API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt) file")
    parser.add_argument("--size", type=str, default="micro", choices=["micro", "mini", "medium", "large"], help="Preset size if no ckpt provided")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on ('cpu', 'cuda')")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Starting Axelion OpenAI-Compatible API Server on http://{args.host}:{args.port}")
    print(f"Endpoints available: /v1/chat/completions, /v1/completions, /v1/models, /health")
    print("=" * 60)

    init_model(checkpoint_path=args.ckpt, size=args.size, device=args.device)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
