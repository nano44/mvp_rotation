# scripts/build_assets.py
import os, yaml, pandas as pd

def main(cfg_path="config/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    sectors = cfg["universe"]["sectors"]
    out_dir = "data/master/assets"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        "asset_id": range(len(sectors)),
        "ticker": sectors,
        "currency": ["USD"] * len(sectors),
        "listing_dt": pd.NaT,
        "delisting_dt": pd.NaT,
        "active": [True] * len(sectors),
    })
    out_path = f"{out_dir}/assets.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] assets.parquet written: {df.shape[0]} rows → {out_path}")

if __name__ == "__main__":
    main()