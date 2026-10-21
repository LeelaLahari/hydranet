"""Build normalized LSTM sequence dataset with SPATIAL train/val basin split."""
import argparse, numpy as np, pandas as pd
from src.utils.io import load_config, save_json
from src.utils.seed import set_seed
from src.features.sequences import build_basin_frame, make_sequences, Normalizer

def load_basins(cfg, toy):
    from src.data.toy import make_toy_basins
    if toy:
        return make_toy_basins(n=cfg["data"]["n_basins"], seed=0)
    from src.data.camels import read_forcing, read_streamflow, read_attributes, list_basins
    root = "data/raw/camels"
    attrs = read_attributes(f"{root}/camels_attributes_v2.0/camels_attributes_v2.0/camels_clim.txt")
    basins = {}
    for bid in list_basins(root)[: cfg["data"]["n_basins"]]:
        f = read_forcing(f"{root}/basin_mean_forcing/daymet/{bid}.txt")
        q = read_streamflow(f"{root}/usgs_streamflow/{bid}.txt")
        df = f.join(q, how="inner")
        if len(df) > cfg["data"]["seq_length"] * 2:
            basins[bid] = df
    return basins, attrs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", action="store_true")
    a = ap.parse_args(); cfg = load_config(); set_seed(cfg["seed"])
    basins, attrs = load_basins(cfg, a.toy)
    ids = sorted(basins)
    rng = np.random.default_rng(cfg["seed"])
    val_ids = set(rng.choice(ids, int(cfg["data"]["val_basin_frac"] * len(ids)), replace=False).tolist())
    seq_len = cfg["data"]["seq_length"]; horizon = cfg["data"]["target_horizon"]
    feats = cfg["data"]["static_features"]
    data = {"train": [], "val": []}
    for bid in ids:
        df = build_basin_frame(basins[bid])
        df = df.sort_index()
        srow = attrs.loc[bid, feats] if bid in attrs.index else pd.Series(0.0, index=feats)
        Xd, Xs, y, t = make_sequences(df, seq_len, horizon, srow.astype(float))
        split = "val" if bid in val_ids else "train"
        data[split].append(dict(id=bid, Xd=Xd, Xs=Xs, y=y, t=t.astype(str)))
    # fit normalizer on train only
    Xd_tr = np.concatenate([d["Xd"] for d in data["train"]])
    Xs_tr = np.concatenate([d["Xs"] for d in data["train"]])
    y_tr = np.concatenate([d["y"] for d in data["train"]])
    nz = Normalizer().fit(Xd_tr, Xs_tr, y_tr)
    basin_index = {bid: i for i, bid in enumerate(ids)}
    for split in ("train", "val"):
        out = {d["id"]: dict(Xd=nz.transform(d["Xd"], d["Xs"], d["y"])[0].astype(np.float32),
                             Xs=nz.transform(d["Xd"], d["Xs"], d["y"])[1][0].astype(np.float32),
                             y=nz.transform(d["Xd"], d["Xs"], d["y"])[2].astype(np.float32),
                             t=d["t"], basin=basin_index[d["id"]]) for d in data[split]}
        np.savez_compressed(f"data/processed/lstm_{split}.npz", **out)
    save_json({"basin_index": basin_index,
               "norm": {"mu_x": nz.mu_x.tolist(), "sd_x": nz.sd_x.tolist(),
                        "mu_s": nz.mu_s.tolist(), "sd_s": nz.sd_s.tolist(),
                        "mu_y": float(nz.mu_y), "sd_y": float(nz.sd_y)},
               "val_basins": sorted(val_ids)}, "data/processed/lstm_meta.json")
    print(f"train basins={len(data['train'])} val basins={len(data['val'])} (spatial holdout)")

if __name__ == "__main__":
    main()
