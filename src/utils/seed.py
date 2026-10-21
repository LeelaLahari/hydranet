import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True hurts throughput a lot on T4
    torch.backends.cudnn.benchmark = True
