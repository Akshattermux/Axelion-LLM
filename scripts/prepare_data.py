"""
Axelion Data Preparation Script
Prepares pretraining binary token files and validates instruction datasets.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axelion.tokenizer import AxelionTokenizer
from training.dataset import prepare_binary_dataset, SFTDataset


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Axelion")
    parser.add_argument("--mode", choices=["pretrain", "sft", "all"], default="all")
    args = parser.parse_args()

    tokenizer = AxelionTokenizer()

    if args.mode in ["pretrain", "all"]:
        print("\n--- Preparing Pretraining Datasets ---")
        splits = ["train", "validation", "test"]
        for s in splits:
            txt_path = f"data/pretraining/{s}.txt"
            bin_path = f"data/pretraining/{s}.bin"
            if os.path.exists(txt_path):
                prepare_binary_dataset(txt_path, bin_path, tokenizer)
            else:
                print(f"Skipping {txt_path} (not found)")

    if args.mode in ["sft", "all"]:
        print("\n--- Validating Instruction SFT Datasets ---")
        sft_splits = ["train", "validation", "test"]
        for s in sft_splits:
            jsonl_path = f"data/instruction/{s}.jsonl"
            if os.path.exists(jsonl_path):
                ds = SFTDataset(jsonl_path, tokenizer)
                print(f"[OK] {jsonl_path}: {len(ds)} conversational examples verified.")
            else:
                print(f"Skipping {jsonl_path} (not found)")


if __name__ == "__main__":
    main()
