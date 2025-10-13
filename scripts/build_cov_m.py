# scripts/build_cov_m.py
"""
Build monthly risk table: 60-month rolling covariance (with simple diagonal shrinkage)
for the 11 sector monthly returns.

Output schema (tall):
  month_end: timestamp
  asset_i: int32
  asset_j: int32
  cov_ij:  float32
"""
import os
import numpy as np
import pandas as pd

LOOKBACK = 36          # months
SHRINK  = 0.10         # shrinkage toward avg variance * I

def _load_returns_wide() -> pd.DataFrame:
    """Wide matrix R: index=month_end, columns=asset_id (0..10), values=r_m, float64."""
    t = pd.read_parquet("data/master/prices/returns_m/returns_m.parquet")
    R = t.pivot(index="month_end", columns="asset_id", values="r_m").sort_index()
    R.columns = R.columns.astype(int)
    return R.astype("float64")

def _shrink_cov(S: np.ndarray, shrink: float) -> np.ndarray:
    """Diagonal shrinkage toward avg variance * I."""
    S = np.array(S, dtype=np.float64, copy=False)
    var = float(np.mean(np.diag(S))) if S.size else 0.0
    n = S.shape[0]
    return (1.0 - shrink) * S + shrink * (var if var > 0 else 0.0) * np.eye(n, dtype=np.float64)

def main():
    out_dir = "data/master/risk/cov_m"
    os.makedirs(out_dir, exist_ok=True)

    R = _load_returns_wide()
    # keep only rows with all 11 assets present
    R = R.dropna(how="any")

    dates = R.index.to_list()
    n = R.shape[1]
    rows = []

    # start at the first index where we have LOOKBACK months available
    for t in range(LOOKBACK - 1, len(dates)):
        end_dt = dates[t]
        start_idx = t - LOOKBACK + 1
        window = R.iloc[start_idx:t+1]
        if window.isna().any().any():
            # skip if any NA in the 60-month window
            continue
        # sample covariance (pandas uses ddof=1 by default)
        S = window.cov().values
        # shrinkage
        Sigma = _shrink_cov(S, SHRINK)

        # append tall rows for this date
        for i in range(n):
            for j in range(n):
                rows.append((end_dt, i, j, float(Sigma[i, j])))

    cov_tall = pd.DataFrame(rows, columns=["month_end", "asset_i", "asset_j", "cov_ij"])
    cov_tall = cov_tall.astype({"asset_i": "int32", "asset_j": "int32", "cov_ij": "float32"})

    out_path = os.path.join(out_dir, "cov_m.parquet")
    cov_tall.to_parquet(out_path, index=False)

    # reporting
    n_months = cov_tall["month_end"].nunique()
    print(f"[OK] cov_m.parquet: months={n_months}, matrix={n}×{n}, rows={len(cov_tall)} → {out_path}")
    if n_months:
        print("[INFO] first/last:", str(cov_tall['month_end'].min().date()), "→", str(cov_tall['month_end'].max().date()))
    # simple PSD sanity (allow tiny negative eigenvals due to numerics)
    try:
        sample_dt = cov_tall["month_end"].max()
        M = cov_tall[cov_tall["month_end"] == sample_dt].pivot(index="asset_i", columns="asset_j", values="cov_ij").values.astype("float64")
        eig = np.linalg.eigvalsh(M)
        print("[INFO] eig(min,max) at last month:", float(eig.min()), float(eig.max()))
    except Exception as e:
        print("[WARN] PSD check skipped:", e)

if __name__ == "__main__":
    main()