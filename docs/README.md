# Axelion LLM Documentation

This repository contains **Axelion**, a custom decoder-only Large Language Model (LLM) built from scratch using modern Transformer techniques inspired by LLaMA and GPT family models.

## Architecture Overview

Axelion is a decoder-only autoregressive Transformer with:

### 1. Rotary Positional Embeddings (RoPE)
Located in `model.py` (`apply_rope` and `precompute_rope`). RoPE rotates Query and Key vectors in multi-dimensional space based on their sequence positions. Axelion tracks absolute position offsets (`position_offset=past_len`) during KV-cached generation to maintain exact positional embeddings across context growth.

### 2. RMSNorm (Root Mean Square Normalization)
Located in `model.py` (`RMSNorm`). Axelion computes RMSNorm in `float32` for enhanced numerical stability under reduced-precision training (FP16/BF16).

### 3. SwiGLU Feed-Forward Networks
Located in `model.py` (`SwiGLU`). SwiGLU utilizes gated linear projections with SiLU activations. Hidden dimensions are computed cleanly via `ModelConfig.ffn_dim`.

### 4. Grouped-Query Attention (GQA) & SDPA
Located in `model.py` (`GQAAttention`). Query heads share KV heads (`n_heads % n_kv_heads == 0`) to minimize KV-cache memory footprints. PyTorch's native `F.scaled_dot_product_attention` enables optimized fused attention execution.

### 5. Weight Tying
Language model output weights (`lm_head.weight`) are tied to token embedding weights (`tok_emb.weight`) after initialization to conserve parameter size.

## Training & Memory Optimizations

* **Mixed Precision (`torch.autocast`)**: Accelerates training and reduces VRAM usage using BF16 or FP16.
* **Gradient Checkpointing**: Decreases activation memory usage during forward passes.
* **Gradient Accumulation**: Enables larger effective batch sizes on memory-constrained GPUs.
* **Memory-Mapped Datasets**: Loads tokenized binary datasets (`train_data.bin`) via `np.memmap` for efficient streaming.
