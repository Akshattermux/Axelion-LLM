import os
import math
import time
import torch
from model import Axelion, ModelConfig
from dataset import prepare_tiny_dataset, create_dataloader

batch_size = 4                   
gradient_accumulation_steps = 4  
max_iters = 500                  
eval_interval = 100              
eval_iters = 20                  
max_lr = 1e-3             
min_lr = 1e-4
warmup_steps = 50
weight_decay = 0.1              
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

def cosine_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    """Cosine learning rate decay with warmup (From NanoLLM)."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))

def estimate_loss(model, dataloader, ctx):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    iterator = iter(dataloader)
    
    for k in range(eval_iters):
        try:
            X, Y = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            X, Y = next(iterator)
            
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            with ctx: 
                _, loss, _ = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train():
    print(f"Training on device: {device} | dtype: {dtype}")
    
    ctx = torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=ptdtype) if device != 'cpu' else torch.autocast(device_type='cpu', dtype=torch.bfloat16)

    bin_file = 'train_data.bin'
    if not os.path.exists(bin_file):
        with open('train_data.txt', 'w') as f:
            f.write("[TICKER: AAPL] [PRICE: 150.00] [NEWS: Apple releases new iPhone] -> [PREDICTION: UP]\n" * 500)
        prepare_tiny_dataset('train_data.txt', bin_file)

    config = ModelConfig.create_micro()
    model = Axelion(config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device != 'cpu'))

    dataloader = create_dataloader(bin_file, config, batch_size=batch_size)
    iterator = iter(dataloader)

    best_val_loss = float('inf')
    
    t0 = time.time()
    for iter_num in range(max_iters):
        
        # Apply Cosine LR Schedule
        lr = cosine_lr(iter_num, max_iters, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            loss_val = estimate_loss(model, dataloader, ctx)
            print(f"step {iter_num}: train loss {loss_val:.4f} | lr {lr:.2e}")
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, 'ckpt.pt')

        for micro_step in range(gradient_accumulation_steps):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                xb, yb = next(iterator)
                
            xb, yb = xb.to(device), yb.to(device)

            with ctx:
                logits, loss, _ = model(xb, yb)
                loss = loss / gradient_accumulation_steps

            if device != 'cpu' and dtype == 'float16':
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if device != 'cpu' and dtype == 'float16':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)

        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, time {dt*1000:.2f}ms")

if __name__ == '__main__':
    train()
