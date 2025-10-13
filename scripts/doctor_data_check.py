# scripts/doctor_data_check.py
import numpy as np
import pandas as pd

def main():
    # 1) assets
    assets = pd.read_parquet("data/master/assets/assets.parquet")
    assert assets["asset_id"].is_unique and len(assets) == 11
    print("[OK] assets:", len(assets), "rows")

    # 2) calendar
    cal = pd.read_parquet("data/master/calendar/month_ends.parquet")
    print("[OK] calendar:", len(cal), "rows", cal["ds"].min().date(), "→", cal["ds"].max().date())

    # 3) returns (tall)
    rets = pd.read_parquet("data/master/prices/returns_m/returns_m.parquet")
    cnt = rets.groupby("month_end")["asset_id"].nunique()
    assert (cnt == 11).all(), "Some months lack all 11 sectors"
    assert rets["r_m"].notna().all(), "NaNs in r_m"
    print("[OK] returns_m:", rets["month_end"].min().date(), "→", rets["month_end"].max().date(), "| rows:", len(rets))

    # 4) benchmark weights and sums ≈ 1.0 (use consistent dtype)
    bench = pd.read_parquet("data/master/portfolio/bench_weights_m/bench_weights_m.parquet")

    # Wide panels, consistent dtype
    R = rets.pivot(index="month_end", columns="asset_id", values="r_m").sort_index().astype("float64")
    W = bench.pivot(index="month_end", columns="asset_id", values="w_bench").sort_index().astype("float64")

    # per-month weight sums
    row_sums = W.sum(axis=1)
    ok = np.allclose(row_sums.values, 1.0, rtol=1e-7, atol=1e-8)
    print(f"[{'OK' if ok else 'WARN'}] benchmark sum≈1.0 per month?", ok)

    # 5) overlap & self-consistency (mean vs weighted sum)
    W_aligned = W.reindex(R.index).dropna()
    R_aligned = R.loc[W_aligned.index]
    Rb1 = R_aligned.mean(axis=1)                   # equal-weight average
    Rb2 = (W_aligned * R_aligned).sum(axis=1)      # weights·returns
    bench_ok = np.allclose(Rb1.values, Rb2.values, rtol=1e-7, atol=1e-8)
    print(f"[{'OK' if bench_ok else 'WARN'}] equal-weight benchmark self-consistency:", bench_ok)

    print("All checks done.")

if __name__ == "__main__":
    main()