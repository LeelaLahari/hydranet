"""Build LSTM training sequences + normalize with basin-wise stats.

Spatial holdout: validation basins are ENTIRELY unseen (tests regional
generalization, the thing the JD cares about).
"""
import numpy as np
import pandas as pd

DYN = ["prcp(mm/day)", "tmean(C)", "pet(C)"]

def build_basin_frame(basin_df: pd.DataFrame) -> pd.DataFrame:
    df = basin_df.copy()
    if "tmean(C)" not in df:
        df["tmean(C)"] = (df["tmax(C)"] + df["tmin(C)"]) / 2
    if "pet(C)" not in df:
        df["pet(C)"] = np.clip(df["tmean(C)"] - 5, 0, None)
    return df

def make_sequences(df, seq_len, horizon, statics_row, dyn_cols=DYN):
    Xd, Xs, y, t = [], [], [], []
    dyn = df[dyn_cols].to_numpy(np.float32)
    q = df["q"].to_numpy(np.float32)
    s = statics_row.to_numpy(np.float32)
    for i in range(len(df) - seq_len - horizon):
        Xd.append(dyn[i: i + seq_len])
        Xs.append(s)
        y.append(q[i + seq_len + horizon - 1])
        t.append(df.index[i + seq_len + horizon - 1])
    return (np.asarray(Xd), np.asarray(Xs), np.asarray(y), np.asarray(t))

class Normalizer:
    """Per-feature standardization fitted on TRAIN basins only."""
    def __init__(self):
        self.mu_x = None; self.sd_x = None
        self.mu_s = None; self.sd_s = None
        self.mu_y = None; self.sd_y = None

    def fit(self, Xd, Xs, y):
        self.mu_x, self.sd_x = Xd.reshape(-1, Xd.shape[-1]).mean(0), Xd.reshape(-1, Xd.shape[-1]).std(0) + 1e-6
        self.mu_s, self.sd_s = Xs.mean(0), Xs.std(0) + 1e-6
        self.mu_y, self.sd_y = y.mean(), y.std() + 1e-6
        return self

    def transform(self, Xd, Xs, y):
        return ((Xd - self.mu_x) / self.sd_x,
                (Xs - self.mu_s) / self.sd_s,
                (y - self.mu_y) / self.sd_y)
