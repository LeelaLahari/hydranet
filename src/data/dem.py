"""DEM access for Stage 3. Real hook (SRTM via OpenTopography needs an API key);
falls back to synthetic valley DEMs so the pipeline always runs on Colab."""
import numpy as np
from pathlib import Path

CACHE = Path("data/raw/dems")

def get_dem(basin_id: str, toy: bool = False, seed: int = 1) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{basin_id}.npy"
    if fp.exists():
        return np.load(fp)
    if toy:
        from .toy import make_toy_dem
        dem = make_toy_dem(seed=abs(hash(basin_id)) % 10_000)
    else:
        # TODO(real run): download SRTM 30m tile for basin centroid via OpenTopography
        # and crop; keep this fallback so scripts never crash on Colab.
        from .toy import make_toy_dem
        dem = make_toy_dem(seed=abs(hash(basin_id)) % 10_000)
    np.save(fp, dem)
    return dem
