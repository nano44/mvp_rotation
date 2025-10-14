# scripts/build_labels_forward.py
from __future__ import annotations
import os
import pandas as pd

OUT = "data/master/labels/forward_returns/fwd_excess_1m.parquet"
RET_PATH = "data/master/prices/returns_m/returns_m.parquet"
BENCH_PATH = "data/master/benchmark/bench_weights_m.parquet"  # <- new SPX benchmark
ASSETS_PATH = "data/master/assets/assets.parquet"

SECTORS = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]


def load_returns_wide(path: str) -> pd.DataFrame:
    """Return a wide DataFrame (dates × tickers) of monthly sector returns.
    Supports these cases:
      1) tall with {month_end, asset_id, r_m}
      2) tall with {month_end, ticker, ret}
      3) already-wide with index=month_end and sector columns
    """
    R = pd.read_parquet(path)

    # Case 1: tall with asset_id
    if {"month_end", "asset_id", "r_m"}.issubset(R.columns):
        Rw = R.pivot(index="month_end", columns="asset_id", values="r_m").sort_index()
        # Map asset_id -> ticker using assets.parquet
        if os.path.exists(ASSETS_PATH):
            A = pd.read_parquet(ASSETS_PATH)
            if {"asset_id", "ticker"}.issubset(A.columns):
                id2tkr = dict(zip(A["asset_id"].astype(int), A["ticker"].astype(str)))
                Rw = Rw.rename(columns=id2tkr)
        # Keep only our 11 sectors
        Rw = Rw.reindex(columns=[c for c in SECTORS if c in Rw.columns])
        return Rw

    # Case 2: tall with ticker
    if {"month_end", "ticker", "ret"}.issubset(R.columns):
        Rw = R.pivot(index="month_end", columns="ticker", values="ret").sort_index()
        Rw = Rw.reindex(columns=[c for c in SECTORS if c in Rw.columns])
        return Rw

    # Case 3: assume already-wide
    if "month_end" in R.columns:
        Rw = R.set_index("month_end").sort_index()
    else:
        Rw = R.sort_index()
    Rw = Rw.reindex(columns=[c for c in SECTORS if c in Rw.columns])
    return Rw


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # 1) Load sector returns (wide) and SPX benchmark weights (wide)
    Rw = load_returns_wide(RET_PATH)
    Wb = pd.read_parquet(BENCH_PATH).sort_index()

    # 2) Align on dates and columns (tickers)
    Rw, Wb = Rw.align(Wb, join="inner", axis=0)
    Rw = Rw.reindex(columns=SECTORS)
    Wb = Wb.reindex(columns=SECTORS)

    # 3) Compute forward 1m returns
    R_fwd  = Rw.shift(-1)                    # sector return at t+1
    Rb_fwd = (Wb * Rw).sum(axis=1).shift(-1) # benchmark return at t+1

    # 4) Forward excess per sector
    Y = (R_fwd.sub(Rb_fwd, axis=0)).dropna(how="all")

    # 5) Stack to tall labels; include asset_id if mapping exists
    tall = (
        Y.stack()
         .rename("fwd_excess_1m")
         .rename_axis(index=["month_end", "ticker"]).reset_index()
    )

    if os.path.exists(ASSETS_PATH):
        A = pd.read_parquet(ASSETS_PATH)
        if {"ticker", "asset_id"}.issubset(A.columns):
            tkr2id = dict(zip(A["ticker"].astype(str), A["asset_id"].astype("int32")))
            tall["asset_id"] = tall["ticker"].map(tkr2id).astype("Int32")
            if tall["asset_id"].notna().all():
                tall = tall[["month_end", "asset_id", "fwd_excess_1m"]]

    tall.to_parquet(OUT, index=False)

    # Diagnostics: dispersion should be clearly > 0
    mean_abs = float(Y.abs().mean(axis=1).mean()) if not Y.empty else float("nan")
    print(f"[OK] fwd_excess_1m.parquet: dates={Y.shape[0]}, assets={Y.shape[1]}, rows={len(tall)} → {OUT}")
    print(f"[INFO] per-date mean(|excess|) mean(abs): {mean_abs:.6f}")
    if not Y.empty:
        print(f"[INFO] first/last label dates: {Y.index.min().date()} → {Y.index.max().date()}")


if __name__ == "__main__":
    main()