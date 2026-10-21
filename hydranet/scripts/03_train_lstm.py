"""Train regional quantile LSTM with MC-dropout; spatial val basins only."""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.utils.io import load_config, append_csv
from src.utils.seed import set_seed
from src.models.lstm import RegionalLSTM, pinball_loss
from src.training.common import run_epoch, eval_epoch, EarlyStopping, checkpoint
from src.training.ddp import init_ddp, wrap_ddp, cleanup_ddp

class SeqDS(Dataset):
    def __init__(self, npz):
        self.data = list(npz.values())
    def __len__(self): return sum(len(d["y"]) for d in self.data)
    def __getitem__(self, i):
        for d in self.data:
            if i < len(d["y"]):
                return (torch.from_numpy(d["Xd"][i]), torch.from_numpy(d["Xs"][i]),
                        torch.tensor(d["y"][i]), torch.tensor(d["basin"]))
            i -= len(d["y"])

def loss_fn(model, x):
    pred = model(x[0], x[1], x[2])
    return pinball_loss(pred, x[2], model.module.quantiles if hasattr(model, "module") else model.quantiles)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", action="store_true")
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--ddp", action="store_true")
    a = ap.parse_args(); cfg = load_config(); set_seed(cfg["seed"])
    L = cfg["lstm"]; epochs = a.epochs or L["epochs"]
    is_ddp, world, rank = init_ddp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = dict(np.load("data/processed/lstm_train.npz", allow_pickle=True))
    va = dict(np.load("data/processed/lstm_val.npz", allow_pickle=True))
    n_dyn = tr[list(tr)[0]]["Xd"].shape[-1]; n_static = tr[list(tr)[0]]["Xs"].shape[-1]
    model = RegionalLSTM(n_dyn, n_static, len(tr) + len(va), L["emb_dim"], L["hidden_size"],
                         L["num_layers"], L["dropout"], tuple(L["quantiles"])).to(device)
    model = wrap_ddp(model, is_ddp and a.ddp)
    opt = torch.optim.AdamW(model.parameters(), lr=L["lr"], weight_decay=L["weight_decay"])
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    trl, val = DataLoader(SeqDS(tr), L["batch_size"], shuffle=True, num_workers=2, pin_memory=True), SeqDS(va)
    stop = EarlyStopping(L["patience"])
    for ep in range(epochs):
        tl = run_epoch(model, trl, opt, loss_fn, device, scaler, L["grad_clip"])
        vl = eval_epoch(model, DataLoader(val, 512), loss_fn, device)
        append_csv({"epoch": ep, "train_loss": tl, "val_loss": vl}, "reports/lstm_log.csv")
        if rank == 0: print(f"ep {ep:03d} train {tl:.4f} val {vl:.4f}")
        if stop.step(vl):
            checkpoint(model, "checkpoints/lstm_best.pt", config=cfg)
            if stop.stop: break
    cleanup_ddp(is_ddp)
    print("saved checkpoints/lstm_best.pt")

if __name__ == "__main__":
    main()
