import os
import torch
from config import ModelConfig
from model import Axelion
from tokenizer import CodeTokenizer
import torch.nn.functional as F

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device='cpu'):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    model.eval()
    
    # encode the prompt
    idx = tokenizer.encode(prompt).unsqueeze(0).to(device) # shape (1, T)
    
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at max_seq_len
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]
        
        # forward the model to get the logits for the index in the sequence
        with torch.no_grad():
            logits, _ = model(idx_cond)
            
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
    # decode the generated tokens
    generated_text = tokenizer.decode(idx[0])
    return generated_text

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'ckpt.pt'
    
    tokenizer = CodeTokenizer()
    
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Creating an untrained micro model for testing.")
        config = ModelConfig.create_micro()
        model = Axelion(config).to(device)
    else:
        print(f"Loading model from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint['config']
        model = Axelion(config).to(device)
        model.load_state_dict(checkpoint['model'])

    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    prompt = "def hello_world():\n"
    print(f"\nPrompt:\n{prompt}")
    print("\nGenerated Output:")
    output = generate(model, tokenizer, prompt, max_new_tokens=50, device=device)
    print(output)
