"""Terrain derivatives from DEM: cleaning, slope, TWI, HAND.

'Elevation model refining' (JD): despike + fill before derivatives.
D8 flow routing implemented in numpy/scipy — fine for 64x64-1km patches;
swap for pysheds/taudem at production scale.
"""
import numpy as np
from scipy import ndimage

def clean_dem(dem: np.ndarray, spike_thresh: float = 15.0) -> np.ndarray:
    """Remove impulse spikes (LiDAR-style artifacts) and fill pits iteratively."""
    d = dem.copy()
    med = ndimage.median_filter(d, size=3)
    spikes = np.abs(d - med) > spike_thresh
    d[spikes] = med[spikes]
    # pit filling: raise each cell to min of neighbor max, repeat until stable
    filled = d.copy()
    for _ in range(25):
        nb_max = ndimage.maximum_filter(filled, size=3)
        new = np.maximum(filled, np.minimum(filled, nb_max))
        # proper fill: only raise cells below all neighbors (local pits)
        local_min = ndimage.minimum_filter(filled, size=3) >= filled
        new[local_min] = nb_max[local_min]
        if np.allclose(new, filled):
            break
        filled = new
    return filled

def slope_deg(dem: np.ndarray, cell=30.0) -> np.ndarray:
    gy, gx = np.gradient(dem, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

def d8_flow(dem: np.ndarray):
    """Return (flow_dir_index 0..7, accumulation). Uses simple steepest descent."""
    n, m = dem.shape
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    dist = [np.sqrt(2),1,np.sqrt(2),1,1,np.sqrt(2),1,np.sqrt(2)]
    fd = np.zeros((n, m), dtype=int)
    for i in range(1, n-1):
        for j in range(1, m-1):
            best, bd = -1, 0.0
            for k, (di, dj) in enumerate(dirs):
                drop = (dem[i, j] - dem[i+di, j+dj]) / dist[k]
                if drop > bd:
                    bd, best = drop, k
            fd[i, j] = best
    # accumulation via repeated downslope transfer (topological relaxation)
    acc = np.ones((n, m))
    order = np.dstack(np.unravel_index(np.argsort(dem.ravel()), dem.shape))[0][::-1]
    for i, j in order:
        k = fd[i, j]
        if k >= 0:
            di, dj = dirs[k]
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < m:
                acc[ni, nj] += acc[i, j]
    return fd, acc

def twi(dem: np.ndarray, cell=30.0) -> np.ndarray:
    _, acc = d8_flow(dem)
    s = np.clip(slope_deg(dem, cell), 0.1, None)
    a = acc * cell ** 2  # specific catchment area (m)
    return np.log((a + 1) / np.tan(np.radians(s)))

def hand(dem: np.ndarray, acc_thresh=200.0, cell=30.0) -> np.ndarray:
    """Height Above Nearest Drainage. Drainage = cells with flow accumulation > thresh."""
    _, acc = d8_flow(dem)
    drainage = acc * cell ** 2 > acc_thresh
    # distance + elevation transform approximated by masked minimum over growing radius
    if drainage.sum() == 0:
        drainage[np.unravel_index(np.argmin(dem), dem.shape)] = True
    drain_elev = np.where(drainage, dem, np.inf)
    filled = drain_elev.copy()
    for _ in range(64):  # propagate min elevation outward until stable
        new = ndimage.minimum_filter(filled, size=3, mode="nearest")
        filled = np.minimum(filled, np.where(drainage, np.inf, new))
        filled[drainage] = dem[drainage]
    return dem - np.where(np.isinf(filled), dem, filled)

def derive_all(dem_raw: np.ndarray, cell=30.0) -> dict:
    dem = clean_dem(dem_raw)
    return {"dem": dem, "slope": slope_deg(dem, cell),
            "twi": twi(dem, cell), "hand": hand(dem, cell=cell)}
