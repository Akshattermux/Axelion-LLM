"""
Root entry point for Axelion pretraining.
Forwards CLI arguments to `training/pretrain.py`.
Example:
    python train.py --size micro --max-iters 100
    python train.py --size large --batch-size 2 --grad-accum 8
"""

import argparse
from training.pretrain import pretrain


def main():
    parser = argparse.ArgumentParser(description="Axelion Training Entry Point")
    parser.add_argument("--size", type=str, default="micro", choices=["micro", "mini", "medium", "large"], help="Model size preset")
    parser.add_argument("--train-bin", type=str, default="data/pretraining/train.bin", help="Path to tokenized binary file")
    parser.add_argument("--data-txt", type=str, default="train_data.txt", help="Raw UTF-8 text file; it is tokenized automatically")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/axelion-base-v0.2.pt", help="Output checkpoint file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing checkpoint")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-iters", type=int, default=100, help="Total training iterations")
    parser.add_argument("--eval-interval", type=int, default=25, help="Interval for validation")
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


if __name__ == "__main__":
    main()
