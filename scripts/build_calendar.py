# scripts/build_calendar.py
import os, yaml, pandas as pd
from src.utils.common import month_ends

def main(cfg_path="config/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]
    out_dir = "data/master/calendar"
    os.makedirs(out_dir, exist_ok=True)

    idx = month_ends(start, end)
    df = pd.DataFrame({"ds": idx, "is_trading_day": True})
    out_path = f"{out_dir}/month_ends.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] month_ends.parquet: {len(df)} rows {df['ds'].min().date()} → {df['ds'].max().date()} → {out_path}")

if __name__ == "__main__":
    main()