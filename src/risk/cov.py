from __future__ import annotations
import numpy as np, pandas as pd

def estimate_cov(df_ret: pd.DataFrame, asof: pd.Timestamp, lookback: int = 60, shrink: float = 0.1) -> np.ndarray:
    """
    Rolling covariance (monthly returns), simple diagonal shrinkage toward var*I.
    """
    end_idx = df_ret.index.get_loc(asof)
    start_idx = max(0, end_idx - lookback + 1)
    window = df_ret.iloc[start_idx:end_idx+1]
    S = window.cov().values
    var = np.mean(np.diag(S))
    I = np.eye(S.shape[0])
    Sigma = (1 - shrink) * S + shrink * var * I
    return Sigma
