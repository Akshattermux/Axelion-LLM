import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizer import CodeTokenizer
from config import ModelConfig

class MemmapDataset(Dataset):
    """
    A Dataset that loads tokens from a memory-mapped numpy array.
    This is highly efficient for very large datasets that don't fit in RAM.
    """
    def __init__(self, bin_file_path: str, block_size: int):
        self.bin_file_path = bin_file_path
        self.block_size = block_size
        
        # We assume tokens are saved as uint16 for efficiency (vocab size < 65536)
        self.data = np.memmap(bin_file_path, dtype=np.uint16, mode='r')
        
    def __len__(self):
        # The number of possible samples we can pull
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # Get a chunk of tokens of length block_size + 1
        chunk = self.data[idx : idx + self.block_size + 1]
        chunk = chunk.astype(np.int64)
        
        # x is the input sequence, y is the target sequence (shifted by 1)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def create_dataloader(bin_file_path: str, config: ModelConfig, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = MemmapDataset(bin_file_path, config.max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def prepare_tiny_dataset(text_file_path: str, output_bin_path: str):
    """
    Helper function to tokenize a text file and save it as a uint16 binary file.
    Useful for creating a small dataset for debugging.
    """
    print(f"Preparing tiny dataset from {text_file_path}...")
    tokenizer = CodeTokenizer()
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    # Save to binary file
    tokens_np = tokens.numpy().astype(np.uint16)
    tokens_np.tofile(output_bin_path)
    print(f"Saved {len(tokens_np)} tokens to {output_bin_path}")

if __name__ == '__main__':
    # Simple test: create a dummy file and process it
    dummy_txt = "dummy.txt"
    dummy_bin = "dummy.bin"
    with open(dummy_txt, "w", encoding="utf-8") as f:
        f.write("def foo():\n    return 'bar'\n" * 100)
        
    prepare_tiny_dataset(dummy_txt, dummy_bin)
    
    config = ModelConfig(max_seq_len=8)
    loader = create_dataloader(dummy_bin, config, batch_size=2)
    x, y = next(iter(loader))
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    
    # cleanup
    os.remove(dummy_txt)
    os.remove(dummy_bin)
