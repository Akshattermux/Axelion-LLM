import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from config import ModelConfig


KVCache = Tuple[torch.Tensor, torch.Tensor]


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device=None,
    dtype=torch.float32,
):
    """Precompute cosine/sine RoPE tables."""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)

    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
):
    """
    x: [B, H, T, head_dim]

    position_offset is essential for KV-cache generation:
    the new token must receive its absolute sequence position.
    """
    T = x.shape[2]
    end = position_offset + T

    if end > cos.shape[0]:
        raise ValueError(
            f"RoPE cache too short: need positions [0, {end}), "
            f"but only {cos.shape[0]} are available."
        )

    cos_t = cos[position_offset:end].to(dtype=x.dtype)
    sin_t = sin[position_offset:end].to(dtype=x.dtype)

    cos_t = cos_t[None, None, :, :]
    sin_t = sin_t[None, None, :, :]

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t

    return torch.stack((out1, out2), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # Compute normalization in fp32 for better numerical stability.
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(dtype=x.dtype) * self.weight


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(
            cfg.dim, cfg.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]

        q = apply_rope(q, cos, sin, position_offset=past_len)
        k = apply_rope(k, cos, sin, position_offset=past_len)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_cache = (k, v)

        # Expand KV heads for GQA.
        k_attn = k.repeat_interleave(self.n_rep, dim=1)
        v_attn = v.repeat_interleave(self.n_rep, dim=1)

        # For the initial prompt, causal masking is required.
        # During one-token cached decoding, the query can only see past + itself.
        is_initial_prefill = kv_cache is None

        out = F.scaled_dot_product_attention(
            q,
            k_attn,
            v_attn,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_initial_prefill,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        hidden = cfg.ffn_dim

        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GQAAttention(cfg)

        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, new_cache = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            kv_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class Axelion(nn.Module):
    """
    Axelion: Custom decoder-only autoregressive Transformer architecture.

    Features:
      - RMSNorm (fp32 normalized for numerical stability)
      - RoPE (with position offset tracking for KV caching)
      - Grouped-Query Attention
      - PyTorch scaled_dot_product_attention
      - SwiGLU
      - Weight tying
      - KV caching
      - Optional gradient checkpointing
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.config = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)

        # Tied to token embeddings after initialization.
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(
            cfg.head_dim,
            cfg.max_seq_len,
            cfg.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

        # Tie weights AFTER initialization.
        self.lm_head.weight = self.tok_emb.weight

        n_params = self.get_num_params()
        print(f"Axelion initialized: {n_params / 1e6:.2f}M parameters")
        print(f"FFN hidden dimension: {cfg.ffn_dim}")
        print(f"Context length: {cfg.max_seq_len}")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
    ):
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, sequence]")

        B, T = idx.shape

        past_len = 0
        if kv_caches is not None:
            past_len = kv_caches[0][0].shape[2]

        if past_len + T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence exceeds max_seq_len={self.config.max_seq_len}"
            )

        x = self.tok_emb(idx)

        new_caches = []

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None

            if self.config.gradient_checkpointing and self.training and cache is None:
                # Checkpoint only the training path without a KV cache.
                def block_fn(hidden):
                    return block(
                        hidden,
                        self.rope_cos,
                        self.rope_sin,
                        None,
                    )[0]

                x = checkpoint.checkpoint(
                    block_fn,
                    x,
                    use_reentrant=False,
                )
                new_caches.append(None)
            else:
                x, new_cache = block(
                    x,
                    self.rope_cos,
                    self.rope_sin,
                    cache,
                )
                new_caches.append(new_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_token: Optional[int] = None,
    ):
        self.eval()

        kv_caches = None

        for _ in range(max_new_tokens):
            if kv_caches is None:
                idx_cond = idx[:, -self.config.max_seq_len:]
            else:
                idx_cond = idx[:, -1:]

            logits, _, kv_caches = self.forward(
                idx_cond,
                kv_caches=kv_caches,
            )

            logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                # Top-p nucleus sampling.
                sorted_probs, sorted_idx = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative = torch.cumsum(sorted_probs, dim=-1)

                remove = cumulative - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0.0)

                denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                sorted_probs = sorted_probs / denom

                sampled = torch.multinomial(sorted_probs, 1)
                next_token = sorted_idx.gather(-1, sampled)

            idx = torch.cat((idx, next_token), dim=1)

            if stop_token is not None:
                if torch.all(next_token == stop_token):
                    break

        return idx


# Backward-compatibility alias
CodeTermLLM = Axelion
