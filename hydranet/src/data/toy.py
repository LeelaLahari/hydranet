"""Synthetic dataset for a 10-minute end-to-end smoke test (no downloads)."""
import numpy as np, pandas as pd

def make_toy_basins(n=12, start="1990-01-01", end="2010-09-30", seed=0):
    """Rainfall -> routed discharge with basin-specific response + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    basins = {}
    for b in range(n):
        aridity = rng.uniform(0.6, 1.6)
        p = np.clip(rng.gamma(0.5, 4.0, len(dates)) * aridity, 0, None)
        # simple non-linear routing: two reservoirs
        q = np.zeros(len(dates)); s1 = 0.0; s2 = 0.0
        k1, k2 = rng.uniform(2, 5), rng.uniform(5, 15)
        for t in range(len(dates)):
            s1 += p[t] - s1 / k1; s2 += s1 / k1 - s2 / k2; q[t] = s2 / k2
        q *= rng.uniform(0.5, 1.5)
        tmean = 10 + 12 * np.sin(2 * np.pi * dates.dayofyear / 365) + rng.normal(0, 2, len(dates))
        basins[f"{b:08d}"] = pd.DataFrame({
            "prcp(mm/day)": p, "tmean(C)": tmean,
            "pet(C)": np.clip(tmean - 5, 0, None), "q": q + rng.normal(0, 0.01, len(dates)),
        }, index=dates)
    statics = pd.DataFrame({
        "elev_mean": rng.uniform(100, 2000, n), "slope_mean": rng.uniform(5, 30, n),
        "p_mean": rng.uniform(1, 5, n), "pet_mean": rng.uniform(2, 5, n),
        "aridity": rng.uniform(0.6, 1.6, n), "frac_forest": rng.uniform(0, 0.9, n),
        "soil_depth_pelletier": rng.uniform(0.5, 3, n), "gvf_max": rng.uniform(0.3, 1, n),
        "gauge_lat": rng.uniform(35, 45, n), "gauge_lon": rng.uniform(-110, -90, n),
    }, index=[f"{b:08d}" for b in range(n)])
    return basins, statics

def make_toy_dem(n=64, seed=1):
    """Fractal-ish DEM with a valley network (hand-made structure, not pure noise)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)[None, :]; y = np.linspace(0, 1, n)[:, None]
    dem = 50 + 80 * y + 30 * np.sin(3 * np.pi * x) * (1 - y)
    dem += np.cumsum(rng.normal(0, 0.8, (n, 1)), axis=0)  # ridge drift
    dem += rng.normal(0, 1.0, (n, n))
    return dem
