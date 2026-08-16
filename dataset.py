"""
Root backward-compatibility shim for dataset operations.
Canonical implementations located in `training/dataset.py`.
"""

from training.dataset import (
    MemmapDataset,
    SFTDataset,
    create_pretrain_dataloader,
    create_sft_dataloader,
    prepare_binary_dataset,
)

# Legacy alias
create_dataloader = create_pretrain_dataloader
prepare_tiny_dataset = prepare_binary_dataset

__all__ = [
    "MemmapDataset",
    "SFTDataset",
    "create_pretrain_dataloader",
    "create_sft_dataloader",
    "create_dataloader",
    "prepare_binary_dataset",
    "prepare_tiny_dataset",
]
