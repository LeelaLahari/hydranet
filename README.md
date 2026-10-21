# HydraNet — Regional Flood Hazard Prediction with LSTM + CNN

End-to-end prototype of a regional-scale flood hazard pipeline, built for a single GPU
(Google Colab free tier, T4). Mirrors a production flood-model stack:

```
GPM/Daymet precipitation ──► [LSTM] rainfall–runoff + uncertainty (quantile heads, MC-dropout)
                                   │
DEM + statics (slope/TWI/HAND) ────┤
                                   ▼
              [CNN] flood extent / depth-binned map (64x64 patches)
                                   │
              Validation vs HAND baseline + Sentinel-1 observed extents (optional, GEE)
```

## Why this design (maps to the job description)
| JD requirement | Where it lives |
|---|---|
| LSTM / CNN, PyTorch, model training & validation | `src/models/`, `scripts/03,05` |
| Rainfall–runoff processes, calibration vs observed data | `scripts/02,03`, NSE/KGE/PBIAS in `src/evaluation/metrics.py` |
| Elevation model refining, LiDAR cleaning (advantageous) | `src/features/terrain.py` (slope/TWI/HAND + DEM artifact removal: fill + despike) |
| Spatial downscaling, feature extraction | LSTM stage + CNN extent stage, `scripts/04` |
| Statistical evaluation of flood footprints (POD/FAR/CSI) | `src/evaluation/metrics.py`, `scripts/06` |
| Reproducible workflows, code repos, pipelines | Hydra configs, seeds, tests, deterministic colab notebooks |
| HPC/Unix awareness | Xarray/Zarr-ready loaders, AMP, single-node DDP hook (`src/training/ddp.py`) |
| Responsible AI / explainability | MC-dropout UQ, reliability diagrams, SHAP-ready feature tables |

## Repo layout
```
configs/            Hydra-style YAML configs (single source of truth)
src/
  data/             CAMELS/Caravan loaders, GPM helpers
  features/         terrain derivatives, scaling, sequence builders
  models/           RegionalLSTM (quantile + MC-dropout), FloodCNN
  training/         trainers, losses (pinball), DDP hook
  evaluation/       NSE/KGE/PBIAS, IoU/POD/FAR/CSI, reliability
  utils/            seeding, Colab/Drive helpers, logging
scripts/            00_setup ... 06_evaluate (run in order)
notebooks/          Colab notebooks that call the scripts cell-by-cell
tests/              pytest (metrics sanity)
```

## Quickstart (Google Colab, free tier)
1. Upload this repo to GitHub (or Drive) and open `notebooks/01_data_setup.ipynb` in Colab.
2. Enable GPU: Runtime -> Change runtime type -> T4 GPU.
3. Run cells top to bottom. Data + checkpoints persist to `/content/drive/MyDrive/hydranet`.
4. Use the provided toy-data generator (`--toy`) to smoke-test the full pipeline in ~10 min
   without downloading anything, then switch to real CAMELS data.

Local:
```bash
pip install -r requirements.txt
python scripts/00_setup.py --toy        # smoke test end-to-end
python scripts/01_download_camels.py    # real data (~1 GB)
python scripts/02_prepare_lstm_dataset.py
python scripts/03_train_lstm.py
python scripts/04_prepare_extent_dataset.py --toy
python scripts/05_train_extent_cnn.py
python scripts/06_evaluate.py
pytest tests/
```

## Data
| Use | Dataset | Access |
|---|---|---|
| Rainfall–runoff (Stage 1/2) | CAMELS-US (or Caravan) | `scripts/01_download_camels.py` (NCAR public zip) |
| Observed flood extents (validation) | Global Flood Database / Sentinel-1 | optional, via GEE cell in notebook 03 |
| DEM (Stage 3) | SRTM/Copernicus patch per basin | `src/data/dem.py` (OpenTopography-free fallback: synthetic + real hook) |

## Known limitations (be honest in interviews)
- Stage-3 CNN labels come from a HAND-threshold baseline (documented, not hydraulic simulation);
  swap in GEE Sentinel-1 extents for real supervision — notebook 03 has the cell.
- CAMELS forcing units/flags must be verified per release; loader is tolerant but check headers.
- Single-node DDP hook included but untested outside SLURM; AMP path is the Colab default.

## License / attribution
Code: MIT. Data: CAMELS (NCAR), GPM (NASA), Sentinel (ESA/EE) — cite accordingly.
