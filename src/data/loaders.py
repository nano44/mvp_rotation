from __future__ import annotations
import numpy as np, pandas as pd
from . import loaders
from src.utils.common import month_ends, ann_to_monthly_vol, set_seed

SECTORS = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]

def load_synthetic_sector_returns(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """
    Creates synthetic monthly total returns for 11 sectors + a benchmark weight series.
    Not realistic, but good enough to prove the pipeline works.
    Returns: DataFrame indexed by month-end, columns = sectors, values = decimal returns.
    """
    set_seed(seed)
    dates = month_ends(start, end)
    n = len(dates); S = len(SECTORS)

    # baseline drift & vol per sector (randomized a bit)
    mu_month = np.random.normal(0.005, 0.002, size=S)  # ~0.5% avg monthly
    vol_month = np.full(S, ann_to_monthly_vol(0.18)) * np.random.uniform(0.8, 1.2, size=S)

    # factor structure: a common market + sector idiosyncratic
    market = np.random.normal(0.004, ann_to_monthly_vol(0.16), size=n)
    eps = np.random.normal(0, 1, size=(n, S))
    corr_structure = np.eye(S) * 0.6 + (1 - np.eye(S)) * 0.4  # modest correlation
    L = np.linalg.cholesky(corr_structure)
    shocks = eps @ L.T

    rets = np.zeros((n, S))
    for t in range(n):
        rets[t, :] = mu_month + market[t] + shocks[t, :] * vol_month

    df = pd.DataFrame(rets, index=dates, columns=SECTORS)
    return df.clip(-0.3, 0.3)  # clamp extremes

def load_benchmark_weights(start: str, end: str) -> pd.DataFrame:
    dates = month_ends(start, end)
    # simple equal-weight benchmark by default (we’ll replace with real weights later)
    S = len(SECTORS)
    w = np.full(S, 1.0 / S)
    return pd.DataFrame([w]*len(dates), index=dates, columns=SECTORS)
