from typing import List, Dict, Union, Optional
import tiktoken
import torch

# Dedicated special tokens for conversation formatting
SPECIAL_TOKENS = {
    "<|endoftext|>": 50256,
    "<|system|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|end|>": 50260,
}
INV_SPECIAL_TOKENS = {v: k for k, v in SPECIAL_TOKENS.items()}


class AxelionTokenizer:
    """
    Tokenizer for Axelion LLM.
    Wraps tiktoken GPT-2 BPE with dedicated special tokens for Chat/SFT formatting.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.base_vocab_size = self.enc.n_vocab  # 50257
        self.special_tokens = SPECIAL_TOKENS
        # Vocab accommodates standard GPT2 + extra special tokens
        self.vocab_size = 50261

        self.pad_id = self.special_tokens["<|endoftext|>"]
        self.system_id = self.special_tokens["<|system|>"]
        self.user_id = self.special_tokens["<|user|>"]
        self.assistant_id = self.special_tokens["<|assistant|>"]
        self.end_id = self.special_tokens["<|end|>"]

    def encode(self, text: str, allowed_special: Optional[set] = None) -> torch.Tensor:
        """Encode a string into a 1D Tensor of token IDs."""
        if allowed_special is None:
            allowed_special = set(self.special_tokens.keys())
        tokens = self.enc.encode(text, allowed_special=allowed_special)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: Union[torch.Tensor, List[int], int]) -> str:
        """Decode token IDs back into a readable string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.enc.decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[torch.Tensor]:
        """Encode a batch of strings."""
        return [self.encode(t) for t in texts]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Format a list of message dicts into Axelion chat format:
        <|system|>
        {system_message}
        <|end|>
        <|user|>
        {user_message}
        <|end|>
        <|assistant|>
        {assistant_message}
        <|end|>
        """
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            formatted += f"<|{role}|>\n{content}\n<|end|>\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"

        return formatted


# Backward-compatible alias
CodeTokenizer = AxelionTokenizer


if __name__ == "__main__":
    tokenizer = AxelionTokenizer()
    sample_msgs = [
        {"role": "system", "content": "You are Axelion, an AI assistant."},
        {"role": "user", "content": "Explain binary search."},
    ]
    chat_str = tokenizer.apply_chat_template(sample_msgs, add_generation_prompt=True)
    print("Formatted Chat Template:\n" + chat_str)
    tokens = tokenizer.encode(chat_str)
    print(f"Encoded token IDs (count={len(tokens)}): {tokens.tolist()}")
    print("Decoded back:\n" + tokenizer.decode(tokens))
