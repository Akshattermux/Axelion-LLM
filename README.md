# Axelion LLM

Axelion is a modern, custom decoder-only Transformer language model built from first principles in PyTorch. It incorporates state-of-the-art architectural innovations (RoPE, RMSNorm, SwiGLU, GQA, SDPA, KV Caching, Weight Tying) and provides a clean, modular pipeline for both pretraining and Supervised Fine-Tuning (SFT) with conversational loss masking.

---

## 📁 Repository Structure

```text
Axelion-LLM/
│
├── axelion/                      # Core Model & Engine
│   ├── __init__.py               # Package exports
│   ├── model.py                  # Decoder-only Transformer & attention mechanisms
│   ├── config.py                 # Single source of truth for model configurations
│   ├── tokenizer.py              # TikToken BPE with Chat special tokens & templates
│   └── generation.py             # KV-cache streaming & sampling engine
│
├── training/                     # Training & Evaluation Pipelines
│   ├── pretrain.py               # Autoregressive next-token pretraining loop
│   ├── sft.py                    # Instruction Fine-Tuning with prompt loss masking
│   ├── dataset.py                # MemmapDataset (pretrain) & SFTDataset (chat)
│   └── evaluate.py               # Validation loss, Perplexity (PPL) & prompt test suite
│
├── data/                         # Datasets & Splits
│   ├── pretraining/              # Text & binary tokenized pretraining corpora
│   │   ├── train.txt / train.bin
│   │   ├── validation.txt / validation.bin
│   │   └── test.txt / test.bin
│   ├── instruction/              # Multi-turn JSONL SFT datasets
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── raw/                      # Raw incoming datasets
│   └── processed/                # Cleaned / filtered data
│
├── checkpoints/                  # Saved weights (Base & Instruct)
│   └── axelion-base-v0.1.pt
│
├── benchmarks/                   # Quantitative & qualitative evaluation
│   ├── benchmark.py              # Benchmark runner (Knowledge, Coding, Reasoning)
│   ├── baseline.json             # Base model results
│   └── instruct.json             # Fine-tuned model results
│
├── scripts/                      # Developer tools
│   └── prepare_data.py           # Tokenization & validation CLI
│
├── docs/                         # Extended documentation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🏛️ Architecture Highlights

- **Rotary Positional Embeddings (RoPE)**: Relative position encoding with position offset tracking for KV-cache decoding.
- **RMSNorm**: Root Mean Square Normalization computed in `float32` for numerical stability.
- **Grouped-Query Attention (GQA)**: Interpolates between MHA and MQA to drastically cut KV-cache memory bandwidth.
- **SwiGLU Non-Linearity**: Gated Linear Unit with SiLU activation in the Feed-Forward network.
- **Weight Tying**: Shared weights between token embedding and `lm_head`.
- **Scaled Dot-Product Attention (SDPA)**: PyTorch native fused attention kernel.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets
Tokenize pretraining text files and validate instruction JSONL datasets:
```bash
python scripts/prepare_data.py --mode all
```

### 3. Evaluate Baseline Checkpoint
Establish validation loss and perplexity ($PPL = e^{\text{loss}}$) on the base model:
```bash
python training/evaluate.py --checkpoint checkpoints/axelion-base-v0.1.pt
```

### 4. Continued Pretraining
```bash
python training/pretrain.py
```

### 5. Instruction Fine-Tuning (SFT)
Fine-tune Axelion on multi-turn conversations with prompt masking (loss computed only on `<|assistant|>` tokens):
```bash
python training/sft.py
```

### 6. Interactive Generation & Streaming
```bash
python generate.py
```

### 7. Run Benchmarks (Base vs Instruct)
```bash
python benchmarks/benchmark.py --checkpoint checkpoints/axelion-base-v0.1.pt --output benchmarks/baseline.json
python benchmarks/benchmark.py --checkpoint checkpoints/axelion-instruct-v0.1.pt --output benchmarks/instruct.json --instruct
```
