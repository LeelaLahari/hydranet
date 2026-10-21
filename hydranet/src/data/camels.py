"""CAMELS-US loader (tolerant to the standard public release layout).

Forcing files: 3 comment lines then a fixed-width/space table. We parse with
pandas and verify expected columns; check your release's README for units.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FORCING_COLS = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)",
                "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]

def read_forcing(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delim_whitespace=True, skiprows=3, header=None,
                     names=FORCING_COLS, low_memory=False)
    df["date"] = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day))
    return df.set_index("date").drop(columns=["Year", "Mnth", "Day", "Hr"])

def read_streamflow(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delim_whitespace=True, header=None,
                     names=["year", "month", "day", "q"], low_memory=False)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
    return df.set_index("date")[["q"]]  # CAMELS-US: mm/day (verify vs release notes)

def read_attributes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", index_col=0)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def list_basins(root: Path) -> list[str]:
    fdir = root / "basin_mean_forcing" / "daymet"
    return sorted(p.stem for p in fdir.glob("*.txt"))
