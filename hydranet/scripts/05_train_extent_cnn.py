"""Train FloodCNN; compare against HAND-threshold baseline on spatial val basins."""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.utils.io import load_config, append_csv
from src.utils.seed import set_seed
from src.models.flood_cnn import FloodCNN
from src.training.common import run_epoch, eval_epoch, EarlyStopping, checkpoint
from src.evaluation.metrics import iou, f1, contingency

class PatchDS(Dataset):
    def __init__(self, path): self.d = np.load(path)
    def __len__(self): return len(self.d["Y"])
    def __getitem__(self, i):
        return torch.from_numpy(self.d["X"][i]), torch.from_numpy(self.d["Y"][i])[None]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(); cfg = load_config(); set_seed(cfg["seed"]); C = cfg["cnn"]
    epochs = a.epochs or C["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trl = DataLoader(PatchDS("data/processed/extent_train.npz"), C["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    vad = PatchDS("data/processed/extent_val.npz")
    model = FloodCNN(C["in_channels"], C["base_channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"])
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    lossf = lambda m, x: nn.functional.binary_cross_entropy_with_logits(m(x[0]), x[1])
    stop = EarlyStopping(C["patience"])
    for ep in range(epochs):
        tl = run_epoch(model, trl, opt, lossf, device, scaler)
        vl = eval_epoch(model, DataLoader(vad, 256), lossf, device)
        append_csv({"epoch": ep, "train_loss": tl, "val_loss": vl}, "reports/cnn_log.csv")
        print(f"ep {ep:03d} train {tl:.4f} val {vl:.4f}")
        if stop.step(vl):
            checkpoint(model, "checkpoints/cnn_best.pt", config=cfg)
            if stop.stop: break
    # benchmark: CNN vs naive HAND<threshold baseline on val
    model.load_state_dict(torch.load("checkpoints/cnn_best.pt")["model"])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(vad.d["X"]).to(device))).cpu().numpy().squeeze(1)
    pred, obs = probs > 0.5, vad.d["Y"] > 0.5
    m_cnn = dict(model="flood_cnn", **{k: float(v) for k, v in contingency(pred, obs).items() if not isinstance(v, np.integer)},
                 IoU=iou(pred, obs), F1=f1(pred, obs))
    # baseline: HAND channel (index 3) below per-patch median
    hand = vad.d["X"][:, 3]
    base = hand < np.median(hand, axis=(1, 2), keepdims=True)
    m_base = dict(model="hand_baseline", **{k: float(v) for k, v in contingency(base, obs).items() if not isinstance(v, np.integer)},
                  IoU=iou(base, obs), F1=f1(base, obs))
    import pandas as pd
    pd.DataFrame([m_base, m_cnn]).to_csv("reports/extent_benchmark.csv", index=False)
    print(pd.DataFrame([m_base, m_cnn]))

if __name__ == "__main__":
    main()
