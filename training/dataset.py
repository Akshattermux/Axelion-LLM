"""
training/dataset.py — Axelion dataset utilities.

Classes:
    MemmapDataset          — memory-mapped pretraining dataset
    SFTDataset             — supervised fine-tuning dataset (multi-turn JSONL)

Functions:
    sft_collate_fn                   — padding collator for SFT batches
    create_pretrain_dataloader       — returns a DataLoader over MemmapDataset
    create_sft_dataloader            — returns a DataLoader over SFTDataset
    prepare_binary_dataset           — tokenize a text file into a uint16 binary
    prepare_binary_dataset_chunked   — tokenize a *very large* text file in 64 MB
                                       chunks, appending to the binary output
"""

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from axelion.config import ModelConfig
from axelion.tokenizer import AxelionTokenizer


class MemmapDataset(Dataset):
    """
    High-throughput dataset that loads token IDs from a memory-mapped numpy binary file.
    Efficient for large pretraining corpora with fallback tiling for small corpora.
    """

    def __init__(self, bin_file_path: str, block_size: int):
        self.bin_file_path = bin_file_path
        self.block_size = block_size
        self.data = np.memmap(bin_file_path, dtype=np.uint16, mode="r")
        self._is_short = len(self.data) <= self.block_size + 1

    def __len__(self):
        if self._is_short:
            return max(16, min(256, len(self.data)))
        return max(1, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_short:
            repeats = (self.block_size + 2) // max(1, len(self.data)) + 2
            tiled = np.tile(self.data, repeats)
            offset = (idx * 17) % max(1, len(self.data))
            chunk = tiled[offset : offset + self.block_size + 1].astype(np.int64)
        else:
            chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT) dataset for multi-turn chat conversations.
    Masks user and system prompt tokens with -100 so the cross-entropy loss is
    computed STRICTLY on the assistant's response tokens.
    """

    def __init__(
        self,
        jsonl_file: str,
        tokenizer: AxelionTokenizer,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL dataset file not found: {jsonl_file}")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    self.samples.append(json.loads(line_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        messages = item.get("messages", [])

        input_ids = []
        labels = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            turn_text = f"<|{role}|>\n{content}\n<|end|>\n"
            turn_tokens = self.tokenizer.encode(turn_text).tolist()

            input_ids.extend(turn_tokens)

            if role == "assistant":
                # Mask the turn header "<|assistant|>\n" so loss is on content + <|end|>
                header_tokens = self.tokenizer.encode(f"<|{role}|>\n").tolist()
                turn_labels = [-100] * len(header_tokens) + turn_tokens[len(header_tokens):]
                labels.extend(turn_labels)
            else:
                # Mask system and user inputs completely with -100
                labels.extend([-100] * len(turn_tokens))

        # Truncate to max_seq_len + 1 for next-token shift
        if len(input_ids) > self.max_seq_len + 1:
            input_ids = input_ids[: self.max_seq_len + 1]
            labels = labels[: self.max_seq_len + 1]

        # Shift x and y for causal language modelling
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def sft_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length conversations in a batch to the maximum length in that batch.
    """
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_x = []
    padded_y = []
    for x, y in zip(xs, ys):
        pad_size = max_len - len(x)
        if pad_size > 0:
            padded_x.append(torch.cat([x, torch.full((pad_size,), pad_token_id, dtype=torch.long)]))
            padded_y.append(torch.cat([y, torch.full((pad_size,), ignore_index, dtype=torch.long)]))
        else:
            padded_x.append(x)
            padded_y.append(y)

    return torch.stack(padded_x), torch.stack(padded_y)


def create_pretrain_dataloader(
    bin_file_path: str,
    config: ModelConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Return a DataLoader over a MemmapDataset binary file.

    Args:
        bin_file_path: Path to the uint16 memmap binary.
        config:        ModelConfig (used for ``max_seq_len``).
        batch_size:    Micro-batch size.
        shuffle:       Whether to shuffle samples (default True).
        num_workers:   Dataloader worker processes (default 0 for backward compat).
        pin_memory:    Pin tensors to page-locked memory for faster GPU transfer.
    """
    dataset = MemmapDataset(bin_file_path, config.max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_sft_dataloader(
    jsonl_file: str,
    tokenizer: AxelionTokenizer,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Return a DataLoader over an SFTDataset JSONL file.

    Args:
        jsonl_file:  Path to the JSONL file (one JSON object per line).
        tokenizer:   AxelionTokenizer instance.
        max_seq_len: Maximum sequence length.
        batch_size:  Micro-batch size.
        shuffle:     Whether to shuffle samples (default True).
        num_workers: Dataloader worker processes (default 0 for backward compat).
        pin_memory:  Pin tensors to page-locked memory for faster GPU transfer.
    """
    dataset = SFTDataset(jsonl_file, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=tokenizer.pad_id),
    )


def prepare_binary_dataset(
    text_file_path: str,
    output_bin_path: str,
    tokenizer: Optional[AxelionTokenizer] = None,
) -> None:
    """Tokenize a raw text corpus into a uint16 memmap binary file.

    The entire file is read into memory at once.  For very large files
    (tens of GB) use :func:`prepare_binary_dataset_chunked` instead.
    """
    if tokenizer is None:
        tokenizer = AxelionTokenizer()

    print(f"Tokenizing {text_file_path} -> {output_bin_path}...")
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens_np = tokens.numpy().astype(np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    tokens_np.tofile(output_bin_path)
    print(f"Saved {len(tokens_np):,} tokens to {output_bin_path}")


def prepare_binary_dataset_chunked(
    text_file_path: str,
    output_bin_path: str,
    tokenizer: Optional[AxelionTokenizer] = None,
    chunk_bytes: int = 64 * 1024 * 1024,  # 64 MB per chunk
) -> None:
    """Tokenize a *very large* text file in streaming chunks and append to a
    uint16 binary file.

    This is memory-efficient for corpora that are too large to load at once.
    If ``output_bin_path`` already exists it will be **appended to**, allowing
    incremental builds.

    Args:
        text_file_path: Path to the raw UTF-8 text source.
        output_bin_path: Path to the output binary (created or appended).
        tokenizer: AxelionTokenizer instance (created lazily if None).
        chunk_bytes: Number of bytes to read per chunk (default 64 MB).
    """
    if tokenizer is None:
        tokenizer = AxelionTokenizer()

    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    total_tokens = 0

    print(f"Chunked tokenization: {text_file_path} -> {output_bin_path} ({chunk_bytes // 1024 // 1024} MB chunks)")

    with open(text_file_path, "r", encoding="utf-8", errors="replace") as src, \
         open(output_bin_path, "ab") as dst:

        carry = ""  # leftover text from the previous chunk (partial word/line)
        while True:
            raw = src.read(chunk_bytes)
            if not raw:
                break

            text_chunk = carry + raw

            # Keep last 1 KB as carry to avoid splitting multi-byte characters
            # or BPE tokens at chunk boundaries.
            if len(raw) == chunk_bytes:
                # Not the final chunk — carry the last 1 KB of text
                carry = text_chunk[-1024:]
                text_chunk = text_chunk[:-1024]
            else:
                carry = ""

            tokens = tokenizer.encode(text_chunk)
            tokens_np = tokens.numpy().astype(np.uint16)
            tokens_np.tofile(dst)
            total_tokens += len(tokens_np)
            print(f"  ... {total_tokens:,} tokens written", end="\r", flush=True)

        # Flush any remaining carry text
        if carry:
            tokens = tokenizer.encode(carry)
            tokens_np = tokens.numpy().astype(np.uint16)
            tokens_np.tofile(dst)
            total_tokens += len(tokens_np)

    print(f"\nDone. Saved {total_tokens:,} tokens to {output_bin_path}")
