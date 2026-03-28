import math, random, numpy as np, torch

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def move_to_device(x, device):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, dict): return {k: move_to_device(v, device) for k,v in x.items()}
    if isinstance(x, list): return [move_to_device(v, device) for v in x]
    return x

def cosine_lr(step,total,warmup,base_lr,min_lr):
    if step < warmup: return base_lr * (step+1)/max(1,warmup)
    p=(step-warmup)/max(1,total-warmup)
    return min_lr + (base_lr-min_lr)*0.5*(1+math.cos(math.pi*p))
