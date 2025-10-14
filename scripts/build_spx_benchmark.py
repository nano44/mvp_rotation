# scripts/build_spx_benchmark.py
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np
import yaml

DEFAULT_SECTORS = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]

def load_sectors_from_config(cfg_path: str = "config/default.yaml") -> list[str]:
    try:
        cfg = yaml.safe_load(open(cfg_path))
        secs = cfg.get("universe", {}).get("sectors", DEFAULT_SECTORS)
        if not isinstance(secs, list) or len(secs) != 11:
            return DEFAULT_SECTORS
        return secs
    except Exception:
        return DEFAULT_SECTORS

def main():
    SECTORS = load_sectors_from_config()

    RAW = "data/raw/spx_sector_weights_monthly.csv"
    OUT = "data/master/benchmark/bench_weights_m.parquet"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    if not os.path.exists(RAW):
        raise FileNotFoundError(f"Missing raw weights CSV: {RAW}")

    # 1) load raw CSV
    df = pd.read_csv(RAW, parse_dates=["date"])
    req = {"date","sector","weight"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{RAW} missing columns: {sorted(missing)}")

    # 2) clean + basic types
    df["sector"] = df["sector"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="raise")
    if (df["weight"] < 0).any() or (df["weight"] > 1).any():
        raise ValueError("Weights must be decimals within [0,1].")

    # 3) sector whitelist
    bad = set(df["sector"].unique()) - set(SECTORS)
    if bad:
        raise ValueError(f"Unknown sectors in CSV: {sorted(bad)}\nExpected one of {SECTORS}")

    # 4) per-date checks: exactly 11 rows and sums ~ 1.0 (tolerate tiny rounding; renormalize)
    cnt = df.groupby("date")["sector"].count()
    if not (cnt.min() == 11 and cnt.max() == 11):
        raise ValueError("Each date must have exactly 11 sector rows.")

    sums = df.groupby("date")["weight"].sum()
    dev = (sums - 1.0).abs()
    tol = 5e-4  # allow up to 0.05% rounding drift
    if (dev > tol).any():
        bad = sums[dev > tol]
        raise ValueError(
            "Weights per date must sum to 1.0; deviations too large: "
            + str({d.strftime('%Y-%m-%d'): float(v) for d, v in bad.items()})
        )

    # Renormalize within each date to enforce exact 1.0 sums
    df["weight"] = df["weight"] / df.groupby("date")["weight"].transform("sum")

    # 5) pivot to wide, fixed column order, standard index name
    W = (df.pivot(index="date", columns="sector", values="weight")
            .sort_index()
            .reindex(columns=SECTORS))
    W.index.name = "month_end"

    # 6) optional: report overlap with returns (if present)
    ret_path = "data/master/prices/returns_m/returns_m.parquet"
    if os.path.exists(ret_path):
        R = pd.read_parquet(ret_path)["month_end"].drop_duplicates().sort_values()
        inter = pd.Index(R).intersection(W.index)
        print(f"[INFO] overlap with returns: {len(inter)} months "
              f"({inter.min().date()} → {inter.max().date()})")

    # 7) write parquet
    W.to_parquet(OUT, index=True)
    print(f"[OK] wrote {OUT} | rows={len(W)} | {W.index.min().date()}→{W.index.max().date()}")
    # 8) quick echo of last date diff vs raw to help diagnose staleness
    print("[INFO] last-date sum:", float(W.iloc[-1].sum()),
          "| first 5 cols:", W.iloc[-1].iloc[:5].round(4).to_dict())

if __name__ == "__main__":
    main()