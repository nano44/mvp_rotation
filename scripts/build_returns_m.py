# scripts/build_returns_m.py
from __future__ import annotations
import os, re, yaml
import pandas as pd
import numpy as np
from src.utils.common import month_ends

# Optional: fetch proxy price series if not present in the CSV
def _fetch_proxy_series(ticker: str, start: str = "1998-01-01") -> pd.Series | None:
    try:
        import yfinance as yf  # lazy import
    except Exception:
        print(f"[WARN] yfinance not available; cannot fetch proxy {ticker}.")
        return None
    try:
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"[WARN] Could not download proxy {ticker}.")
            return None
        s = df["Adj Close"].resample("ME").last()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        s.name = ticker
        return s
    except Exception as e:
        print(f"[WARN] Proxy download failed for {ticker}: {e}")
        return None

# -----------------------------------------------------------------------------
# Configurable constants / defaults
# -----------------------------------------------------------------------------
DEFAULT_CSV_PATH = "data/raw/sector_etf_prices_monthly.csv"
SECTORS_ALL = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]
# Proxies used only when native sector column is missing
PROXIES = {
    "XLRE": ["VNQ"],
    "XLC":  ["IYZ", "VOX"],
}

MIN_START = "2000-01-31"

# -----------------------------------------------------------------------------
# Helpers to normalize CSV (wide/long) and detect levels vs returns
# -----------------------------------------------------------------------------

def _find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in {"date","dt","timestamp","month_end","as_of","period"}:
            return c
    # fallback to first column
    return df.columns[0]


def _wide_from_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long schema (date,ticker,price) into wide with tickers as columns."""
    c_date = _find_date_col(df)
    c_tkr = next((c for c in df.columns if str(c).lower() in {"ticker","sect","symbol"}), None)
    if c_tkr is None:
        raise ValueError("Long format must include a 'ticker' column (ticker/sect/symbol)")
    # choose a price/level column
    cand = [c for c in df.columns if re.search(r"(?i)adj|close|price|px|level|tri|index", str(c))]
    if not cand:
        raise ValueError("Long format must include a price/level column (adj_close/close/price/px/level/tri/index)")
    c_val = cand[0]
    temp = (df[[c_date, c_tkr, c_val]].rename(columns={c_date: "date", c_tkr: "ticker", c_val: "px"}))
    temp["date"] = pd.to_datetime(temp["date"]).dt.to_period("M").dt.to_timestamp("M")
    wide = temp.pivot_table(index="date", columns="ticker", values="px", aggfunc="last").sort_index()
    return wide


def _to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide month-end DataFrame from either wide or long input."""
    lower = {c.lower() for c in df.columns}
    if {"ticker"}.intersection(lower):
        w = _wide_from_long(df)
    else:
        c_date = _find_date_col(df)
        w = df.rename(columns={c_date: "date"}).set_index("date")
        w.index = pd.to_datetime(w.index).to_period("M").to_timestamp("M")
        # keep symbol-like columns (letters only) or non-all-null columns
        keep = [c for c in w.columns if re.match(r"^[A-Za-z]{2,5}$", str(c))]
        if not keep:
            keep = [c for c in w.columns if not w[c].isna().all()]
        w = w[keep]
    return w.sort_index()


def _apply_proxies(wide_px: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    W = wide_px.copy()
    for tgt, cands in PROXIES.items():
        if tgt not in sectors:
            continue
        # Ensure target column exists (so we can fill into it)
        if tgt not in W.columns:
            W[tgt] = np.nan
        # Ensure candidate proxy columns exist; if missing, try to fetch via yfinance
        for c in cands:
            if c not in W.columns:
                s = _fetch_proxy_series(c)
                if s is not None:
                    W = W.join(s, how="outer")
        # Fill target's NaNs from proxies in priority order
        for c in cands:
            if c in W.columns:
                W[tgt] = W[tgt].combine_first(W[c])
    # keep only requested sectors that exist after proxy fill
    keep = [s for s in sectors if s in W.columns]
    return W[keep]


def _returns_from_wide(wide_px: pd.DataFrame) -> pd.DataFrame:
    """Detect whether values look like returns; otherwise compute pct_change of levels."""
    # If values look like returns (bound reasonably and centered), use directly
    med = wide_px.median().median()
    maxabs = wide_px.abs().max().max()
    if -0.5 < med < 0.5 and maxabs < 1.5:
        print("[INFO] Detected returns-like values; using CSV as returns.")
        R = wide_px.copy()
    else:
        print("[INFO] Computing returns via pct_change of levels from CSV.")
        R = wide_px.pct_change(fill_method=None)
    R = R.dropna(how="any")
    R.index.name = "month_end"
    return R

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg_path: str = "config/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    sectors_cfg = cfg.get("universe", {}).get("sectors", SECTORS_ALL)
    csv_path = (
        cfg.get("data", {}).get("csv_folder")
        or cfg.get("data", {}).get("csv_path")
        or DEFAULT_CSV_PATH
    )
    # If a directory is passed, try the default filename inside it
    if os.path.isdir(csv_path):
        candidate = os.path.join(csv_path, os.path.basename(DEFAULT_CSV_PATH))
        if os.path.isfile(candidate):
            csv_path = candidate

    out_dir = "data/master/prices/returns_m"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # 1) Read and normalize to wide month-end prices/returns
    raw = pd.read_csv(csv_path)
    wide = _to_wide(raw)
    wide = _apply_proxies(wide, sectors_cfg)
    wide = wide.asfreq("ME", method="pad")  # ensure month-end index (use 'ME' to avoid deprecation)

    # 2) Convert to returns if needed
    rets_wide = _returns_from_wide(wide)

    # 3) Reindex to config date window and ensure sector order
    idx = month_ends(cfg["dates"]["start"], cfg["dates"]["end"]) if "dates" in cfg else rets_wide.index
    rets_wide = rets_wide.reindex(idx)
    rets_wide.index = pd.DatetimeIndex(rets_wide.index, name="month_end")
    keep = [c for c in sectors_cfg if c in rets_wide.columns]
    missing = [c for c in sectors_cfg if c not in rets_wide.columns]
    if missing:
        print(f"[WARN] Missing sectors in CSV (after proxies): {missing}")
    rets_wide = rets_wide[keep]
    rets_wide = rets_wide.dropna(how="any")

    # 4) Write tall parquet: month_end, asset_id, r_m (float32)
    id_map = {t: i for i, t in enumerate(sectors_cfg)}
    tall = (
        rets_wide.reset_index()
        .melt(id_vars=["month_end"], var_name="ticker", value_name="r_m")
    )
    tall["asset_id"] = tall["ticker"].map(id_map).astype("Int64")
    tall = tall.dropna(subset=["asset_id"]).astype({"asset_id": "int32", "r_m": "float32"})
    tall = tall[["month_end", "asset_id", "r_m"]]

    out_path = f"{out_dir}/returns_m.parquet"
    tall.to_parquet(out_path, index=False)
    print(
        f"[OK] returns_m.parquet: {tall['month_end'].min().date()} → {tall['month_end'].max().date()}, "
        f"{tall['asset_id'].nunique()} assets, {len(tall)} rows → {out_path}"
    )


if __name__ == "__main__":
    main()