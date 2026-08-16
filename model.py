"""
Root backward-compatibility shim for Axelion Model.
Canonical implementation located at `axelion/model.py`.
"""

from axelion.model import (
    Axelion,
    TransformerBlock,
    RMSNorm,
    GQAAttention,
    SwiGLU,
    apply_rope,
    precompute_rope,
    CodeTermLLM,
)
from axelion.config import ModelConfig

__all__ = [
    "Axelion",
    "TransformerBlock",
    "RMSNorm",
    "GQAAttention",
    "SwiGLU",
    "apply_rope",
    "precompute_rope",
    "CodeTermLLM",
    "ModelConfig",
]
