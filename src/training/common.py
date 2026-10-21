"""Shared training utilities: AMP loop, early stopping, checkpointing."""
import time, torch
from pathlib import Path

class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.wait = float("inf"), 0
    def step(self, loss):
        if loss < self.best - self.min_delta:
            self.best, self.wait = loss, 0
            return True
        self.wait += 1
        return False
    @property
    def stop(self):
        return self.wait >= self.patience

def run_epoch(model, loader, opt, loss_fn, device, scaler=None, clip=None):
    model.train(); tot, n = 0.0, 0
    for batch in loader:
        x = [b.to(device) for b in batch]
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = loss_fn(model, x)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            if clip: scaler.unscale_(opt)
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        tot += loss.item() * x[0].size(0); n += x[0].size(0)
    return tot / n

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval(); tot, n = 0.0, 0
    for batch in loader:
        x = [b.to(device) for b in batch]
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            tot += loss_fn(model, x).item() * x[0].size(0)
        n += x[0].size(0)
    return tot / n

def checkpoint(model, path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **kwargs}, path)
