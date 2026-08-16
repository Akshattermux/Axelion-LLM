from typing import Generator, Optional, List
import torch
import torch.nn.functional as F

from .model import Axelion
from .tokenizer import AxelionTokenizer


@torch.no_grad()
def stream_generate(
    model: Axelion,
    tokenizer: AxelionTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_token_ids: Optional[List[int]] = None,
    device: str = "cpu",
) -> Generator[str, None, None]:
    """
    Yields decoded text chunks token-by-token with KV caching for fast streaming responses.
    """
    model.eval()
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.end_id, tokenizer.pad_id]

    idx = tokenizer.encode(prompt).unsqueeze(0).to(device)
    prompt_len = idx.shape[1]
    
    # Ensure initial prompt fits within context window
    if prompt_len > model.config.max_seq_len:
        idx = idx[:, -model.config.max_seq_len:]
        prompt_len = idx.shape[1]

    # Bound max_new_tokens by available context length
    available_tokens = max(1, model.config.max_seq_len - prompt_len)
    max_tokens_to_gen = min(max_new_tokens, available_tokens)
    kv_caches = None

    for _ in range(max_tokens_to_gen):
        # KV cache optimization: feed full sequence once, then only the last token
        idx_cond = idx if kv_caches is None else idx[:, -1:]

        logits, _, kv_caches = model(idx_cond, kv_caches=kv_caches)
        logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # Top-p nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            sampled = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, sampled)

        token_id = next_token.item()
        if token_id in stop_token_ids:
            break

        idx = torch.cat((idx, next_token), dim=1)
        yield tokenizer.decode([token_id])


@torch.no_grad()
def generate_text(
    model: Axelion,
    tokenizer: AxelionTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_token_ids: Optional[List[int]] = None,
    device: str = "cpu",
) -> str:
    """Generate complete text response."""
    chunks = []
    for chunk in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
        device=device,
    ):
        chunks.append(chunk)
    return "".join(chunks)
