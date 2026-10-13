
import torch
from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AxelionTokenizer()

ckpt = torch.load(
    "checkpoints/axelion-micro-best.pt",
    map_location="cpu",
    weights_only=False
)
cfg = ckpt.get("config", ModelConfig.create_micro(tokenizer.vocab_size))

model = Axelion(cfg).to(device)
model.load_checkpoint_weights(ckpt["model"])
model.eval()

prompts = [
    "The history of computers began",
    "Water freezes at",
    "Artificial intelligence is",
    "Once upon a time"
]

print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

with torch.no_grad():
    for prompt in prompts:
        ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
        logits, _, _ = model(ids)
        next_logits = logits[0, -1].float()

        print(f"\nPrompt: {prompt}")
        print("Finite logits:", bool(torch.isfinite(next_logits).all()))
        print("Logit range:", round(next_logits.min().item(), 4),
              "to", round(next_logits.max().item(), 4))

        top = torch.topk(next_logits, k=8)
        for token_id, score in zip(top.indices.tolist(), top.values.tolist()):
            print(f"  {tokenizer.decode([token_id])!r}: {score:.3f}")

print("\nDiagnostic complete.")
