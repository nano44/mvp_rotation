from __future__ import annotations
import numpy as np, pandas as pd, random, os
from typing import Iterable

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def month_ends(start: str, end: str) -> pd.DatetimeIndex:
    idx = pd.date_range(start, end, freq="ME")
    return idx

def zscore_cross_sectional(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    return (x - mu) / (sd if sd > 1e-12 else 1.0)

def ann_to_monthly_vol(vol_ann: float) -> float:
    # convert annualized vol to monthly std (approx)
    return vol_ann / np.sqrt(12)

def ensure_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
