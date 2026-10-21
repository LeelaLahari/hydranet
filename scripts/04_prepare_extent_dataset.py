"""Stage 3 dataset: terrain+forcing stacks -> HAND-threshold pseudo extent labels.

Honest limitation (README): labels are baseline-derived; replace with Sentinel-1
observed extents (GEE) for real supervision. Benchmark vs HAND baseline stays.
"""
import argparse, numpy as np
from src.utils.io import load_config
from src.utils.seed import set_seed
from src.data.dem import get_dem
from src.features.terrain import derive_all
from src.evaluation.baselines import hand_quantile_levels, hand_threshold_extent

def norm(a):
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", action="store_true")
    a = ap.parse_args(); cfg = load_config(); set_seed(cfg["seed"])
    P = cfg["extent_data"]; size = cfg["cnn"]["patch_size"]
    npz_tr = dict(np.load("data/processed/lstm_train.npz", allow_pickle=True))
    npz_va = dict(np.load("data/processed/lstm_val.npz", allow_pickle=True))
    rng = np.random.default_rng(cfg["seed"])
    def build(npz, name):
        X, Y, B = [], [], []
        for bid in npz:
            dem = get_dem(bid, toy=a.toy, seed=abs(hash(bid)) % 9999)
            t = derive_all(dem)
            hand, stack_static = t["hand"], np.stack([norm(t["dem"]), norm(t["slope"]),
                                                      norm(t["twi"]), norm(hand)])
            levels = hand_quantile_levels(hand, tuple(P["hand_quantiles"]))
            run = np.abs(npz[bid]["y"]).mean()  # event magnitude proxy from LSTM stage
            for k in range(P["patches_per_basin"]):
                q = P["hand_quantiles"][k % len(P["hand_quantiles"])]
                level = levels[q] * (0.7 + 0.6 * rng.random()) * (0.5 + run)
                label = hand_threshold_extent(hand, level)
                y_ = np.clip(label + rng.normal(0, cfg["cnn"]["label_noise"], label.shape), 0, 1)
                forc = np.stack([np.full_like(hand, run), np.full_like(hand, q)])
                X.append(np.concatenate([stack_static, forc])[..., :size, :size] if False
                         else np.concatenate([stack_static, forc])[:, :size, :size])
                Y.append(y_[:size, :size]); B.append(bid)
        np.savez_compressed(f"data/processed/extent_{name}.npz",
                            X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32),
                            basins=np.asarray(B))
        print(name, np.asarray(X).shape)
    build(npz_tr, "train"); build(npz_va, "val")

if __name__ == "__main__":
    main()
