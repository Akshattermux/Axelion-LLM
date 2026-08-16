from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Single source of truth for the decoder-only Axelion Transformer."""

    # Vocabulary / context
    # 50257 (GPT2 base) + 4 special tokens (<|system|>, <|user|>, <|assistant|>, <|end|>)
    vocab_size: int = 50261
    max_seq_len: int = 2048

    # Transformer width/depth
    dim: int = 768
    n_layers: int = 12

    # Attention
    n_heads: int = 12
    n_kv_heads: Optional[int] = 4

    # Feed-forward (SwiGLU)
    hidden_mult: float = 8.0 / 3.0
    multiple_of: int = 256

    # RoPE / normalization
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    # Regularization
    dropout: float = 0.0

    # Memory
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for the RoPE implementation")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        # SwiGLU uses 2 input projections + 1 output projection.
        hidden = int(self.hidden_mult * self.dim)
        if self.multiple_of <= 1 or self.dim < self.multiple_of or hidden % self.multiple_of == 0:
            return hidden
        return ((hidden + self.multiple_of - 1) // self.multiple_of) * self.multiple_of

    # Backward compatibility aliases
    @property
    def block_size(self) -> int:
        return self.max_seq_len

    @property
    def n_layer(self) -> int:
        return self.n_layers

    @property
    def n_head(self) -> int:
        return self.n_heads

    @property
    def n_kv_head(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def n_embd(self) -> int:
        return self.dim

    @classmethod
    def create_micro(cls, vocab_size: int = 50261):
        """Debug / CPU-friendly testing model (~6.5M params)."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=256,
            dim=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            gradient_checkpointing=False,
        )

    @classmethod
    def create_mini_gpt(cls, vocab_size: int = 50261):
        """~114M parameter class with GPT-2-sized vocabulary."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=2048,
            dim=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            hidden_mult=8.0 / 3.0,
            gradient_checkpointing=True,
        )

    @classmethod
    def create_medium(cls, vocab_size: int = 50261):
        """~322M parameter class."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=4096,
            dim=1024,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            hidden_mult=8.0 / 3.0,
            gradient_checkpointing=True,
        )

    @classmethod
    def create_large(cls, vocab_size: int = 50261):
        """~1B parameter class."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=4096,
            dim=2048,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            hidden_mult=8.0 / 3.0,
            gradient_checkpointing=True,
        )
