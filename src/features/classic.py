from __future__ import annotations
import numpy as np, pandas as pd
from src.utils.common import zscore_cross_sectional

def _momentum(df_ret: pd.DataFrame, lookback: int, skip: int = 1) -> pd.DataFrame:
    # sum of returns over last L months, skipping the most recent "skip" to avoid look-ahead
    return (1 + df_ret.shift(skip)).rolling(lookback).apply(lambda r: np.prod(r) - 1.0, raw=False)

def _volatility(df_ret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return df_ret.rolling(lookback).std()

def make_classic_features(df_ret: pd.DataFrame) -> pd.DataFrame:
    mom_12_1 = _momentum(df_ret, 12, skip=1)
    mom_6_1  = _momentum(df_ret, 6,  skip=1)
    mom_3_1  = _momentum(df_ret, 3,  skip=1)
    vol_6    = _volatility(df_ret, 6)
    vol_12   = _volatility(df_ret, 12)

    # stack features; then z-score cross-sectionally each month
    feats = pd.concat({
        "mom_12_1": mom_12_1,
        "mom_6_1":  mom_6_1,
        "mom_3_1":  mom_3_1,
        "vol_6":    vol_6,
        "vol_12":   vol_12,
    }, axis=1)  # columns become a MultiIndex

    # tidy to wide with one row per (date, sector)
    feats = feats.swaplevel(0,1,axis=1).sort_index(axis=1)

    # cross-sectional z-score per feature each date
    out = {}
    for date, row in feats.iterrows():
        df = row.unstack()  # sectors x features
        df_z = df.apply(zscore_cross_sectional, axis=0)
        out[date] = df_z

    X = pd.concat(out, axis=0).reset_index()
    X.columns = ["date","sector"] + list(df_z.columns)
    X = X.set_index(["date","sector"])
    return X
