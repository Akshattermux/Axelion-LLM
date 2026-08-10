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

## 5. Training Axelion (`train.py`)

1. Open `train.py` and set training hyperparameters:
   * `batch_size`: Micro-batch size per step (e.g. `4` or `8`).
   * `gradient_accumulation_steps`: Accumulates gradients across multiple micro-batches to simulate larger batch sizes (e.g. `4`).
   * `max_iters`: Total training steps.
   * `max_lr` / `min_lr`: Warmup and Cosine learning rate decay boundaries.

2. Launch training:
   ```bash
   python train.py
   ```

*Checkpoints are saved automatically to `ckpt.pt` whenever validation loss improves.*

---

## 6. Text Generation & Inference

Once `ckpt.pt` is generated, run inference:

* **General Autoregressive Text Completion**:
  ```bash
  python generate.py
  ```
* **Stock Market Movement Inference**:
  ```bash
  python stock_predictor.py
  ```
