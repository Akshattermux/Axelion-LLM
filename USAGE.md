# How to Train and Use Axelion

This guide details the complete end-to-end workflow for tokenizing data, configuring model parameters, training the **Axelion** decoder-only Transformer, and running inference for text generation or stock prediction.

---

## 1. Environment Setup

Install the required Python dependencies:

```bash
pip install torch tiktoken yfinance pandas numpy tqdm
```

---

## 2. Prepare Training Data (`train_data.txt`)

Axelion is a domain-agnostic language model. You can train it on general text, programming code, math, or domain-specific text datasets.

* **General Text / Code**: Create `train_data.txt` and populate it with your raw text dataset (e.g. Markdown, Python files, books, or articles).
* **Financial Stock Data**: Run `finance_data.py` to pull historical market prices via `yfinance` and format them into `financial_train_data.txt`:

```bash
python finance_data.py
```

---

## 3. Tokenize Data into Binary Format (`train_data.bin`)

To allow fast streaming memory-mapping during training without overflowing RAM:

1. Open `dataset.py`.
2. Ensure `prepare_tiny_dataset` targets your input text file (e.g., `train_data.txt` or `financial_train_data.txt`):
   ```python
   prepare_tiny_dataset('train_data.txt', 'train_data.bin')
   ```
3. Execute the tokenizer script:
   ```bash
   python dataset.py
   ```
   *This outputs `train_data.bin` containing uint16 BPE token IDs.*

---

## 4. Model Architecture & Preset Selection (`config.py`)

Axelion provides built-in model size presets in `config.py`:

```python
from config import ModelConfig

# Debugging on CPU (~10M parameters, 4 layers, 256 context)
config = ModelConfig.create_micro()

# Mini-GPT (~114M parameters, 12 layers, 2048 context)
config = ModelConfig.create_mini_gpt()

# Medium (~322M parameters, 24 layers, 4096 context)
config = ModelConfig.create_medium()

# Large (~1B parameters, 24 layers, 4096 context)
config = ModelConfig.create_large()
```

### Custom Model Configuration:
You can also initialize `ModelConfig` directly:

```python
config = ModelConfig(
    vocab_size=50257,
    max_seq_len=2048,
    dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    hidden_mult=8.0 / 3.0,
    gradient_checkpointing=True,
)
```

---

## 5. Build a quality-weighted corpus

The first phase is causal language-model pretraining, not chatbot training. The target mixture is defined in `data/pretraining/mixture.json`: 45% programming, 15% computer science, 15% general education, 10% mathematics/reasoning, 7.5% history, and 7.5% social science.

Put reviewed source files under `data/raw/pretraining/<category>/`, then build a cleaned, deduplicated corpus:

```bash
python scripts/prepare_corpus.py --output data/pretraining/train.txt
```

Keep validation material separate. Do not reuse training text for `validation.txt`.

## 6. Prepare train and validation tokens

```bash
python scripts/prepare_data.py --mode pretrain
```

This creates `data/pretraining/train.bin` and `data/pretraining/validation.bin`. Both use causal next-token targets; no assistant/chat loss masking is involved in this phase.

## 7. Measure the hardware first

Run a short Mini benchmark before choosing a 12-hour token budget:

```bash
python train.py --size mini --benchmark-steps 100
```

The output reports tokens/sec and the estimated `12 hours * tokens/sec` budget. Use that measurement to set `--max-iters`; do not guess it in advance.

## 8. Train Axelion Base (`train.py`)

Mini remains the existing 12-layer, 768-dimensional architecture. Its T4-friendly defaults are batch size `1`, gradient accumulation `16`, learning rate `3e-4` to `3e-5`, warmup `500`, validation every `500` steps, and gradient checkpointing enabled.

```bash
python train.py --size mini --data-txt data/pretraining/train.txt --val-bin data/pretraining/validation.bin
```

Checkpoints include model weights, optimizer state, scaler state, iteration, train loss, validation loss, and perplexity. Resume after an interrupted run with:

```bash
python train.py --resume checkpoints/axelion-base-mini.pt --max-iters 10000
```

## 9. Evaluate Base checkpoints

The fixed prompts in `tests/` cover coding, computer science/knowledge, history, and social science:

```bash
python training/evaluate.py --checkpoint checkpoints/axelion-base-mini.pt --val-bin data/pretraining/validation.bin --eval-dir tests
```

## 10. Instruction tuning comes later

Only after the causal base run is evaluated should you run SFT to create Axelion Instruct:

```bash
python training/sft.py --base-ckpt checkpoints/axelion-base-mini.pt --output-ckpt checkpoints/axelion-instruct-mini.pt
```

## 11. Text Generation & Inference

Once the base or instruct checkpoint is generated, run inference:

* **General Autoregressive Text Completion**:
  ```bash
  python generate.py
  ```
* **Stock Market Movement Inference**:
  ```bash
  python stock_predictor.py
  ```
