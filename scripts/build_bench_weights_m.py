# scripts/build_bench_weights_m.py
import os, yaml, pandas as pd
from src.utils.common import month_ends

def main(cfg_path="config/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    sectors = cfg["universe"]["sectors"]
    n = len(sectors)

    idx = month_ends(cfg["dates"]["start"], cfg["dates"]["end"])
    rows = [(dt, aid, 1.0/n) for dt in idx for aid in range(n)]
    out = pd.DataFrame(rows, columns=["month_end","asset_id","w_bench"]).astype({"asset_id":"int32","w_bench":"float32"})

    out_dir = "data/master/portfolio/bench_weights_m"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/bench_weights_m.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[OK] bench_weights_m.parquet: {len(idx)} months × {n} assets → {len(out)} rows → {out_path}")

if __name__ == "__main__":
    main()