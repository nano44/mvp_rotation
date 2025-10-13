# scripts/build_returns_m.py
import os, yaml, pandas as pd
from src.utils.common import month_ends

def _read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date","dt","timestamp")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def main(cfg_path="config/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    sectors = cfg["universe"]["sectors"]
    csv_path = cfg["data"]["csv_folder"]

    out_dir = "data/master/prices/returns_m"
    os.makedirs(out_dir, exist_ok=True)

    df = _read_wide_csv(csv_path)
    keep = [c for c in sectors if c in df.columns]
    missing = [c for c in sectors if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in wide CSV (ignored): {missing}")

    prices = df[keep].astype(float)
    prices_me = prices.resample("ME").last()
    rets = prices_me.pct_change()

    idx = month_ends(cfg["dates"]["start"], cfg["dates"]["end"])
    rets = rets.reindex(idx).dropna(how="any")

    id_map = {t:i for i,t in enumerate(sectors)}
    tall = (rets.reset_index()
                .melt(id_vars=["index"], var_name="ticker", value_name="r_m")
                .rename(columns={"index":"month_end"}))
    tall["asset_id"] = tall["ticker"].map(id_map).astype("Int64")
    tall = tall.dropna(subset=["asset_id"]).astype({"asset_id":"int32","r_m":"float32"})
    tall = tall[["month_end","asset_id","r_m"]]

    out_path = f"{out_dir}/returns_m.parquet"
    tall.to_parquet(out_path, index=False)
    print(f"[OK] returns_m.parquet: {tall['month_end'].min().date()} → {tall['month_end'].max().date()}, "
          f"{tall['asset_id'].nunique()} assets, {len(tall)} rows → {out_path}")

if __name__ == "__main__":
    main()