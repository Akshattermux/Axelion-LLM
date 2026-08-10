import torch
import os
from model import Axelion, ModelConfig
from tokenizer import CodeTokenizer
from finance_data import fetch_stock_data

def predict_stock(ticker: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'ckpt.pt'
    tokenizer = CodeTokenizer()

    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. You must train the model first on financial data.")
        return

    print(f"Loading Financial LLM on {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # We now load the config directly as ModelConfig is in model.py
    config = checkpoint['config'] 
    model = Axelion(config).to(device)
    model.load_state_dict(checkpoint['model'])

    print(f"\nFetching recent data for {ticker}...")
    historical_context = fetch_stock_data(ticker, days=5)
    prompt = historical_context + f"[DATE: TODAY] [OPEN: PENDING] [CLOSE: PENDING] -> [MOVEMENT:"
    
    print("\n--- Model Prompt ---")
    print(prompt)
    
    print("\n--- Model Prediction (Using Fast KV Cache) ---")
    idx = tokenizer.encode(prompt).unsqueeze(0).to(device)
    
    # Use the new KV Cache optimized generate function pulled from NanoLLM
    out_idx = model.generate(idx, max_new_tokens=10, temperature=0.2, top_p=0.9)
    prediction = tokenizer.decode(out_idx[0])
    
    completion = prediction.split("[MOVEMENT:")[-1].strip()
    print(f"Predicted Movement: {completion}")

if __name__ == '__main__':
    predict_stock('AAPL')
