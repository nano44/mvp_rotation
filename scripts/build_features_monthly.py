# scripts/build_features_monthly.py
import os
import yaml
import numpy as np
import pandas as pd

FEATURE_VERSION = "v1.0.0"  # bump if you change definitions

def _load_returns_tall(path="data/master/prices/returns_m/returns_m.parquet") -> pd.DataFrame:
    """Tall table: month_end, asset_id, r_m"""
    df = pd.read_parquet(path)
    # basic sanity
    need = {"month_end", "asset_id", "r_m"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"returns_m missing columns: {missing}")
    return df

def _pivot_returns_wide(rets_tall: pd.DataFrame) -> pd.DataFrame:
    """Wide R: index=month_end, columns=asset_id (0..10), values=r_m (float)."""
    R = rets_tall.pivot(index="month_end", columns="asset_id", values="r_m").sort_index()
    # ensure numeric column labels and float dtype
    R.columns = R.columns.astype(int)
    return R.astype(float)

def _momentum(R: pd.DataFrame, lookback: int, skip: int = 1) -> pd.DataFrame:
    """
    Past-returns momentum: product(1+r) over last 'lookback' months, skipping the most recent 'skip' month.
    For example 12_1 uses months t-12..t-1 (skip current), then minus 1.
    """
    return (1.0 + R).shift(skip).rolling(lookback, min_periods=lookback).apply(lambda x: np.prod(x) - 1.0, raw=False)

def _volatility(R: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Rolling std of monthly returns using full window ending at t."""
    return R.rolling(lookback, min_periods=lookback).std()

def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each row across assets: (x - mean)/std. If std≈0, return 0.
    df: index=dates, columns=asset_id
    """
    def z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        return (x - mu) / (sd if sd > 1e-12 else 1.0)
    return df.apply(z, axis=1)

def main(cfg_path: str = "config/default.yaml"):
    # --- load returns (wide) ---
    rets_tall = _load_returns_tall()
    R = _pivot_returns_wide(rets_tall)  # dates × 11

    # --- compute raw feature panels (wide) ---
    mom_12_1 = _momentum(R, 12, skip=1)
    mom_6_1  = _momentum(R,  6, skip=1)
    mom_3_1  = _momentum(R,  3, skip=1)
    vol_12   = _volatility(R, 12)
    vol_6    = _volatility(R,  6)

    # --- keep only dates where *all* features are fully available for *all* assets ---
    masks = [
        mom_12_1.notna().all(axis=1),
        mom_6_1.notna().all(axis=1),
        mom_3_1.notna().all(axis=1),
        vol_12.notna().all(axis=1),
        vol_6.notna().all(axis=1),
    ]
    valid_dates = masks[0]
    for m in masks[1:]:
        valid_dates = valid_dates & m
    if valid_dates.sum() == 0:
        raise ValueError("No dates with full features across all assets. Check your returns and lookbacks.")
    mom_12_1 = mom_12_1.loc[valid_dates]
    mom_6_1  = mom_6_1.loc[valid_dates]
    mom_3_1  = mom_3_1.loc[valid_dates]
    vol_12   = vol_12.loc[valid_dates]
    vol_6    = vol_6.loc[valid_dates]

    # --- cross-sectional z-score (per date across assets) ---
    Z = {
        "mom_12_1": _cross_sectional_zscore(mom_12_1),
        "mom_6_1":  _cross_sectional_zscore(mom_6_1),
        "mom_3_1":  _cross_sectional_zscore(mom_3_1),
        "vol_12":   _cross_sectional_zscore(vol_12),
        "vol_6":    _cross_sectional_zscore(vol_6),
    }

    # --- stack to tall table: (month_end, asset_id, feature, value, version) ---
    parts = []
    for fname, panel in Z.items():
        tall = (
            panel.reset_index()
                 .melt(id_vars=["month_end"], var_name="asset_id", value_name="value")
                 .astype({"asset_id":"int32", "value":"float32"})
        )
        tall["feature"] = fname
        parts.append(tall)
    feats_tall = pd.concat(parts, axis=0, ignore_index=True)

    # reorder columns and add version
    feats_tall["version"] = FEATURE_VERSION
    feats_tall = feats_tall[["month_end", "asset_id", "feature", "value", "version"]]

    # --- write parquet ---
    out_dir = "data/master/features/monthly"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "features_monthly.parquet")
    feats_tall.to_parquet(out_path, index=False)

    n_dates = feats_tall["month_end"].nunique()
    n_assets = feats_tall["asset_id"].nunique()
    n_features = feats_tall["feature"].nunique()
    print(f"[OK] features_monthly.parquet: dates={n_dates}, assets={n_assets}, features={n_features}, "
          f"rows={len(feats_tall)} → {out_path}")

    # quick sanity: z-scored per-date should have mean≈0, std≈1 for each feature
    sanity = []
    for fname, panel in Z.items():
        m = panel.mean(axis=1).abs().mean()
        s = (panel.std(axis=1, ddof=0) - 1.0).abs().mean()
        sanity.append((fname, float(m), float(s)))
    print("[INFO] per-date z-score sanity (mean(|μ|), mean(|σ-1|)):", sanity)

if __name__ == "__main__":
    main()