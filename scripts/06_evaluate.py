"""Final evaluation: runoff NSE/KGE/PBIAS per val basin + MC-dropout reliability."""
import json, numpy as np, pandas as pd, torch
from src.utils.io import load_config
from src.models.lstm import RegionalLSTM, mc_predict
from src.evaluation.metrics import nse, kge, pbias, reliability_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = load_config(); L = cfg["lstm"]
meta = json.load(open("data/processed/lstm_meta.json"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
va = dict(np.load("data/processed/lstm_val.npz", allow_pickle=True))
sample = va[list(va)[0]]
model = RegionalLSTM(sample["Xd"].shape[-1], sample["Xs"].shape[-1], len(meta["basin_index"]),
                     L["emb_dim"], L["hidden_size"], L["num_layers"], L["dropout"], tuple(L["quantiles"])).to(device)
model.load_state_dict(torch.load("checkpoints/lstm_best.pt")["model"])
sd_y = meta["norm"]["sd_y"]; mu_y = meta["norm"]["mu_y"]
rows = []
for bid, d in va.items():
    q50, qstd, _ = mc_predict(model, torch.from_numpy(d["Xd"]).to(device),
                              torch.from_numpy(np.tile(d["Xs"], (len(d["y"]), 1))).to(device),
                              torch.full((len(d["y"]),), d["basin"], dtype=torch.long, device=device),
                              L["mc_samples"])
    q_sim = q50.cpu().numpy() * sd_y + mu_y
    q_std = qstd.cpu().numpy() * sd_y
    q_obs = d["y"] * sd_y + mu_y
    rows.append({"basin": bid, "NSE": nse(q_obs, q_sim), "KGE": kge(q_obs, q_sim),
                 "PBIAS": pbias(q_obs, q_sim),
                 "coverage68": float(np.mean((q_obs >= q_sim - q_std) & (q_obs <= q_sim + q_std)))})
df = pd.DataFrame(rows); df.to_csv("reports/runoff_metrics.csv", index=False)
print(df.describe())
cov, ef, of = reliability_curve(q_obs, q_sim, q_std)
plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "k--")
if len(ef): plt.scatter(ef, of, c="tab:blue")
plt.xlabel("expected"); plt.ylabel("observed"); plt.title("Reliability (last val basin)")
plt.tight_layout(); plt.savefig("reports/figures/reliability.png", dpi=150)
print("wrote reports/runoff_metrics.csv + reports/figures/reliability.png")
