# scripts/build_labels_forward.py
import os
import numpy as np
import pandas as pd

def _load_returns_wide() -> pd.DataFrame:
    """Return wide matrix R: index=month_end, columns=asset_id, values=r_m (float64)."""
    t = pd.read_parquet("data/master/prices/returns_m/returns_m.parquet")
    R = t.pivot(index="month_end", columns="asset_id", values="r_m").sort_index()
    R.columns = R.columns.astype(int)
    return R.astype("float64")

def _load_bench_wide() -> pd.DataFrame:
    """Return wide matrix W: index=month_end, columns=asset_id, values=w_bench (float64)."""
    t = pd.read_parquet("data/master/portfolio/bench_weights_m/bench_weights_m.parquet")
    W = t.pivot(index="month_end", columns="asset_id", values="w_bench").sort_index()
    W.columns = W.columns.astype(int)
    return W.astype("float64")

def main():
    out_dir = "data/master/labels/forward_returns"
    os.makedirs(out_dir, exist_ok=True)

    R = _load_returns_wide()   # returns (monthly)
    W = _load_bench_wide()     # benchmark weights

    # Align dates (use intersection), ensure full rows
    idx = R.index.intersection(W.index)
    R = R.loc[idx]
    W = W.loc[idx]

    # Compute benchmark return per month: B[t] = sum_i W[t,i]*R[t,i]
    B = (W * R).sum(axis=1)  # Series indexed by month_end

    # Forward excess labels for each asset: y[t] = (R[t+1] - B[t+1])
    # Shift -1 so that label “belongs” to as-of date t (predicting next month)
    Y = (R.sub(B, axis=0)).shift(-1)

    # Drop last month (no t+1) and any rows with NaN
    Y = Y.dropna(how="any")

    # Stack to tall format
    tall = (
        Y.reset_index()
         .melt(id_vars=["month_end"], var_name="asset_id", value_name="fwd_excess_1m")
         .astype({"asset_id": "int32", "fwd_excess_1m": "float32"})
         .sort_values(["month_end","asset_id"])
    )

    out_path = os.path.join(out_dir, "fwd_excess_1m.parquet")
    tall.to_parquet(out_path, index=False)

    print(f"[OK] fwd_excess_1m.parquet: dates={tall['month_end'].nunique()}, "
          f"assets={tall['asset_id'].nunique()}, rows={len(tall)} → {out_path}")

    # Quick sanity: mean of labels across assets per date should be near 0 (it’s “excess” vs benchmark)
    Yw = Y.copy()
    mu = Yw.mean(axis=1)
    print("[INFO] per-date mean(|excess|) mean(abs):", float(mu.abs().mean()))
    print("[INFO] first/last label dates:", str(tall['month_end'].min().date()), "→", str(tall['month_end'].max().date()))

if __name__ == "__main__":
    main()