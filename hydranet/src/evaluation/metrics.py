"""Hydrological + flood-extent metrics. Pure numpy, unit-tested."""
import numpy as np

# ---- Runoff ----
def nse(q_obs, q_sim):
    q_obs, q_sim = np.asarray(q_obs, float), np.asarray(q_sim, float)
    ss_res = np.sum((q_obs - q_sim) ** 2)
    ss_tot = np.sum((q_obs - np.mean(q_obs)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)

def kge(q_obs, q_sim):
    q_obs, q_sim = np.asarray(q_obs, float), np.asarray(q_sim, float)
    r = np.corrcoef(q_obs, q_sim)[0, 1]
    a = np.std(q_sim) / (np.std(q_obs) + 1e-12)
    b = np.mean(q_sim) / (np.mean(q_obs) + 1e-12)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)

def pbias(q_obs, q_sim):
    q_obs, q_sim = np.asarray(q_obs, float), np.asarray(q_sim, float)
    return 100 * np.sum(q_sim - q_obs) / (np.sum(q_obs) + 1e-12)

# ---- Flood extents ----
def _bin(a): return np.asarray(a) > 0.5

def iou(a, b):
    a, b = _bin(a), _bin(b)
    inter = np.logical_and(a, b).sum()
    return inter / (np.logical_or(a, b).sum() + 1e-12)

def f1(a, b):
    a, b = _bin(a), _bin(b)
    tp = np.logical_and(a, b).sum(); fp = np.logical_and(a, ~b).sum(); fn = np.logical_and(~a, b).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-12)

def contingency(a, b):
    """a=forecast, b=observed -> POD, FAR, CSI."""
    a, b = _bin(a), _bin(b)
    tp = np.logical_and(a, b).sum(); fp = np.logical_and(a, ~b).sum()
    fn = np.logical_and(~a, b).sum()
    pod = tp / (tp + fn + 1e-12)
    far = fp / (tp + fp + 1e-12)
    csi = tp / (tp + fp + fn + 1e-12)
    return dict(POD=pod, FAR=far, CSI=csi, TP=tp, FP=fp, FN=fn)

# ---- Probabilistic (MC-dropout quantiles) ----
def reliability_curve(q_obs, q_mean, q_std, n_bins=10):
    """Check: does the 68% band cover ~68% of obs? Returns coverage + bins."""
    q_obs, q_mean, q_std = map(lambda x: np.asarray(x, float), (q_obs, q_mean, q_std))
    lo, hi = q_mean - q_std, q_mean + q_std
    coverage = np.mean((q_obs >= lo) & (q_obs <= hi))
    z = (q_obs - q_mean) / (q_std + 1e-12)
    edges = np.linspace(-3, 3, n_bins + 1)
    obs_freq, exp_freq = [], []
    for i in range(n_bins):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if m.sum() > 5:
            from scipy.stats import norm
            obs_freq.append(np.mean(q_obs[m] <= q_mean[m]))
            exp_freq.append(np.mean(norm.cdf(z[m])))
    return coverage, np.array(exp_freq), np.array(obs_freq)
