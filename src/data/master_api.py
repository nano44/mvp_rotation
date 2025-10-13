# src/data/master_api.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List, Optional

def _assets(root: str) -> pd.DataFrame:
    path = os.path.join(root, "assets", "assets.parquet")
    df = pd.read_parquet(path)
    return df.astype({"asset_id": "int32"})

def _asset_id_to_ticker_map(root: str) -> dict:
    a = _assets(root)
    return dict(zip(a["asset_id"].tolist(), a["ticker"].tolist()))

def _ticker_to_asset_id_map(root: str) -> dict:
    a = _assets(root)
    return dict(zip(a["ticker"].tolist(), a["asset_id"].tolist()))

# ---------- RETURNS ----------
def load_returns_wide(root: str, sectors: List[str]) -> pd.DataFrame:
    """Wide returns (dates × sectors), columns ordered by 'sectors' list."""
    t = pd.read_parquet(os.path.join(root, "prices", "returns_m", "returns_m.parquet"))
    id2tkr = _asset_id_to_ticker_map(root)
    R = t.pivot(index="month_end", columns="asset_id", values="r_m").sort_index()
    R.columns = [id2tkr.get(int(c), f"id{int(c)}") for c in R.columns]
    keep = [c for c in sectors if c in R.columns]
    return R[keep].astype("float64")

# ---------- BENCH WEIGHTS ----------
def load_benchmark_weights_wide(root: str, sectors: List[str]) -> pd.DataFrame:
    t = pd.read_parquet(os.path.join(root, "portfolio", "bench_weights_m", "bench_weights_m.parquet"))
    id2tkr = _asset_id_to_ticker_map(root)
    W = t.pivot(index="month_end", columns="asset_id", values="w_bench").sort_index()
    W.columns = [id2tkr.get(int(c), f"id{int(c)}") for c in W.columns]
    keep = [c for c in sectors if c in W.columns]
    return W[keep].astype("float64")

# ---------- FEATURES PANEL ----------
def load_features_panel(root: str, feature_names: List[str], version: str, sectors: List[str]) -> pd.DataFrame:
    """
    Return a DataFrame indexed by (date, sector_ticker) with columns = feature_names (wide).
    """
    path = os.path.join(root, "features", "monthly", "features_monthly.parquet")
    F = pd.read_parquet(path)
    if feature_names:
        F = F[F["feature"].isin(feature_names)]
    if "version" in F.columns and version:
        F = F[F["version"] == version]

    id2tkr = _asset_id_to_ticker_map(root)
    F["sector"] = F["asset_id"].map(id2tkr)
    F = F[F["sector"].isin(sectors)]

    P = F.pivot_table(index=["month_end", "sector"], columns="feature", values="value")
    cols = [c for c in feature_names if c in P.columns] if feature_names else list(P.columns)
    P = P.reindex(columns=cols).sort_index()
    return P

# ---------- COVARIANCE ----------
def load_cov(root: str, asof: pd.Timestamp, sectors: List[str]) -> Optional[np.ndarray]:
    """
    Load a covariance matrix (len(sectors) × len(sectors)) for 'asof'. Return None if not found.
    """
    path = os.path.join(root, "risk", "cov_m", "cov_m.parquet")
    cov = pd.read_parquet(path)
    sub = cov[cov["month_end"] == asof]
    if sub.empty:
        return None
    M = sub.pivot(index="asset_i", columns="asset_j", values="cov_ij").sort_index().values
    return M.astype("float64")