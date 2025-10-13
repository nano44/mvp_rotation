from __future__ import annotations
import numpy as np, pandas as pd
from src.utils.common import month_ends, ann_to_monthly_vol, set_seed

SECTORS = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]

def load_synthetic_sector_returns(start: str, end: str, seed: int = 42, sectors: list[str] = SECTORS) -> pd.DataFrame:
    """
    Creates synthetic monthly total returns for the given sectors.
    Returns: DataFrame indexed by month-end, columns = sectors, values = decimal returns.
    """
    set_seed(seed)
    dates = month_ends(start, end)
    n = len(dates); S = len(sectors)

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

    df = pd.DataFrame(rets, index=dates, columns=sectors)
    return df.clip(-0.3, 0.3)  # clamp extremes

def load_benchmark_weights(start: str, end: str, sectors: list[str] = SECTORS) -> pd.DataFrame:
    dates = month_ends(start, end)
    # simple equal-weight benchmark by default (we’ll replace with real weights later)
    S = len(sectors)
    w = np.full(S, 1.0 / S)
    return pd.DataFrame([w]*len(dates), index=dates, columns=sectors)

def _read_one_csv(path: str) -> pd.Series:
    """
    Robust CSV reader that supports common schemas:
    - Yahoo-style: columns "Date","Adj Close" (or variants)
    - Lowercase variants: "date","adj_close"
    - Generic: pick the last numeric column if needed.
    Returns a Series indexed by datetime with an adjusted/close price.
    """
    import pandas as pd
    df = pd.read_csv(path)
    # choose date column
    date_cols = [c for c in df.columns if c.lower() in ("date","dt","timestamp")]
    date_col = date_cols[0] if date_cols else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # choose price column
    cand = [c for c in df.columns if c.lower().replace(" ", "") in (
        "adjclose","adjustedclose","adjusted_close","close","adjclose*")]
    price_col = cand[0] if cand else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][-1]
    return df[price_col].astype(float)

def load_sector_returns_from_csv(folder: str, sectors: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Load monthly returns for each sector ticker from CSV files located in `folder`.
    Expects one file per ticker: e.g., data/raw/XLY.csv with Date + Adj Close (or Close).
    Returns monthly decimal returns indexed by month-end, with rows containing any NA dropped.
    """
    import os
    import pandas as pd
    from src.utils.common import month_ends

    series = []
    for ticker in sectors:
        path = os.path.join(folder, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV: {path}")
        s = _read_one_csv(path)
        s_me = s.resample("ME").last()
        r = s_me.pct_change().rename(ticker)
        series.append(r)
    df = pd.concat(series, axis=1)
    df = df.loc[start:end].dropna(how="all")
    idx = month_ends(start, end)
    df = df.reindex(idx)
    df = df.dropna(how="any")
    return df

def load_sector_returns_from_wide_csv(file_path: str, sectors: list[str], start: str, end: str, date_col: str = "Date") -> pd.DataFrame:
    """
    Load monthly returns from a single 'wide' CSV where each sector ticker is a column.
    - file_path: path to CSV (e.g., data/raw/sector_etf_prices_monthly.csv)
    - sectors: list of tickers to keep (must match column names, e.g., ["XLY","XLP",...])
    - date_col: name of the date column (default "Date")
    Returns a DataFrame of monthly decimal returns indexed by month-end.
    """
    import pandas as pd
    from src.utils.common import month_ends

    df = pd.read_csv(file_path)

    # detect date column if not present
    if date_col not in df.columns:
        cand = [c for c in df.columns if c.lower() in ("date","dt","timestamp")]
        if not cand:
            raise ValueError(f"Could not find a date column in {file_path}; available cols: {list(df.columns)}")
        date_col = cand[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # keep only requested sector columns that exist in the file
    keep = [c for c in sectors if c in df.columns]
    missing = [c for c in sectors if c not in df.columns]
    if not keep:
        raise ValueError(f"None of the requested sectors found in {file_path}. Requested={sectors}, available={list(df.columns)}")
    if missing:
        print(f"[WARN] Missing tickers in wide CSV (will be dropped): {missing}")

    prices = df[keep].astype(float)

    # ensure month-end frequency and compute monthly returns
    prices_me = prices.resample("ME").last()
    rets = prices_me.pct_change()

    # align to requested date range and canonical month-end index
    idx = month_ends(start, end)
    rets = rets.reindex(idx)

    # drop any rows with NA across the selected sectors (early months with NA will fall out)
    rets = rets.dropna(how="any")
    return rets
