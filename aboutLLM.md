# About Axelion LLM

Axelion is a decoder-only Transformer language model implemented in PyTorch. It uses GPT-2 BPE tokenization, RoPE, RMSNorm, Grouped-Query Attention (GQA), SwiGLU feed-forward layers, weight tying, PyTorch SDPA, and KV caching for generation.

## Exact parameter counts

Counts below are calculated by `ModelConfig.calculate_params()` for the default tokenizer vocabulary of 50,261 tokens. Embeddings and the language-model head are tied, so the vocabulary matrix is counted once. RoPE has no learned parameters.

| Preset | Parameters | Approx. | Context | Layers | Width | Q / KV heads | FFN width | FP16 weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| micro | 16,014,848 | 16.01M | 256 | 4 | 256 | 4 / 2 | 768 | 30.5 MiB |
| mini | 114,117,120 | 114.12M | 2,048 | 12 | 768 | 12 / 4 | 2,048 | 217.9 MiB |
| medium | 322,050,048 | 322.05M | 4,096 | 24 | 1,024 | 16 / 4 | 2,816 | 615.8 MiB |
| large | 1,185,165,312 | 1.19B | 4,096 | 24 | 2,048 | 16 / 4 | 5,632 | 2.21 GiB |

Memory figures are weights only. Training also needs gradients, optimizer state, activations, and workspace memory. A practical AdamW training budget is often several times larger than the weight file.

To print counts from the source of truth:

```bash
python cli.py params
```

## Which preset should I use?

- `micro`: CPU development, API smoke tests, and learning the training loop.
- `mini`: the default serious single-GPU starting point.
- `medium`: use when you have enough GPU memory and a larger, cleaner corpus.
- `large`: deployment or multi-GPU training; do not start here for experimentation.

The included `.pt` files may use an older configuration. Always read the checkpoint's stored `config` when loading weights; the API does this automatically.

## Fastest training path

Install dependencies, then train on any UTF-8 text file. The pretraining script tokenizes the text automatically when the binary dataset does not exist:

```bash
pip install -r requirements.txt
python training/pretrain.py --size micro --data-txt train_data.txt --max-iters 100
```

For a real run, increase `--max-iters`, choose a larger `--size`, and write to a named checkpoint:

```bash
python training/pretrain.py \
  --size mini \
  --data-txt data/pretraining/train.txt \
  --checkpoint-out checkpoints/axelion-base-custom.pt \
  --max-iters 10000
```

Fine-tune conversations with JSONL records containing a `messages` list:

```bash
python training/sft.py \
  --base-ckpt checkpoints/axelion-base-custom.pt \
  --output-ckpt checkpoints/axelion-instruct-custom.pt \
  --epochs 3
```

## Run the production service

```bash
python serve.py --ckpt checkpoints/axelion-instruct-custom.pt --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/dashboard` for the operational dashboard. Use `/health` for liveness, `/ready` for readiness, `/metrics` for request counters, and `/docs` for the OpenAPI contract.

For a deployed service, set an API key before starting it:

```bash
$env:AXELION_API_KEY = "replace-with-a-long-random-secret"
python serve.py --ckpt checkpoints/axelion-instruct-custom.pt
```

Clients must then send `Authorization: Bearer replace-with-a-long-random-secret` to inference routes. Health and readiness remain available to orchestrators.

## Connect an application or agent

The included client has no extra dependency beyond Python's standard library:

```python
from axelion import AxelionClient

client = AxelionClient(
    base_url="http://127.0.0.1:8000",
    api_key="replace-with-a-long-random-secret",
)

reply = client.chat([
    {"role": "user", "content": "Summarize this incident in one sentence."},
])
print(reply["choices"][0]["message"]["content"])

agent_reply = client.agent("Draft a short checklist for deploying this service.")
print(agent_reply["answer"])
```

For in-process tool calling, use `AxelionAgent` and register Python functions with `ToolRegistry` as shown in `agent_demo.py`. The HTTP agent endpoint intentionally does not execute arbitrary remote functions; production applications should register tools in their own trusted process and pass only approved results to the model.

## Important production boundaries

This project provides model inference and a useful service shell. Production operators should still add TLS at the reverse proxy, rate limiting, request logging policy, secret management, checkpoint provenance, model evaluation, and a process supervisor such as Docker, systemd, or Kubernetes.
