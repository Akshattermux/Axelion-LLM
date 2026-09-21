"""
Axelion LLM - A modern, high-performance custom decoder-only Transformer.
Developed by Axionik.
"""

from .config import ModelConfig
from .model import Axelion, TransformerBlock, RMSNorm, GQAAttention, SwiGLU
from .tokenizer import AxelionTokenizer, CodeTokenizer
from .generation import stream_generate, generate_text
from .client import AxelionClient

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "Axelion",
    "TransformerBlock",
    "RMSNorm",
    "GQAAttention",
    "SwiGLU",
    "AxelionTokenizer",
    "CodeTokenizer",
    "stream_generate",
    "generate_text",
    "AxelionClient",
]
