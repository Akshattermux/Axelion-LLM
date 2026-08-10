import tiktoken
import torch

class CodeTokenizer:
    def __init__(self, encoding_name="gpt2"):
        # We use tiktoken's gpt2 encoding as a solid baseline for code and English.
        # Alternatively, 'cl100k_base' (GPT-4) could be used for more efficient code tokenization.
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> torch.Tensor:
        """Encode a string into a tensor of token IDs."""
        tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode a tensor of token IDs back into a string."""
        if tokens.dim() > 1:
            tokens = tokens.squeeze()
        return self.enc.decode(tokens.tolist())

    def encode_batch(self, texts: list[str]) -> list[torch.Tensor]:
        """Encode a batch of strings."""
        return [self.encode(t) for t in texts]

if __name__ == '__main__':
    # Simple test
    tokenizer = CodeTokenizer()
    sample_code = "def hello_world():\n    print('hello world!')"
    tokens = tokenizer.encode(sample_code)
    print(f"Encoded shape: {tokens.shape}")
    print(f"Decoded: {tokenizer.decode(tokens)}")
