# scripts/build_graph_features.py
"""
Graph-lite features from monthly sector returns.

For each month t (with a 36m history window), we:
  1) Compute the 36m sector correlation matrix Corr(t).
  2) Build a weighted graph where edge_ij = max(0, Corr_ij(t)) and diagonal=0.
  3) Compute per-sector metrics:
       - g_deg_wpos_36m: weighted degree (sum of positive-corr edges)
       - g_eigvec_wpos_36m: eigenvector centrality (weighted)
       - g_cluster_wpos_36m: weighted clustering coefficient
     Plus a simple cross-section return feature:
       - cs_absdev_1m: |r_i(t) - mean_j r_j(t)|
  4) Cross-sectionally z-score each feature at t (so they’re comparable across dates).
  5) Append to features table (version kept as 'v1.0.0' so you don't have to touch config).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

VERSION = "v1.0.0"   # keep same version so config doesn't need changing
LOOKBACK = 36        # months for rolling correlation window

MASTER_ROOT = "data/master"

def _zscore_cs(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    return (x - mu) / (sd if sd > 1e-12 else 1.0)

def _load_returns_wide() -> pd.DataFrame:
    """Wide monthly returns: index=month_end (Timestamp), columns=tickers (e.g., XLY...), float64."""
    # returns
    t = pd.read_parquet(os.path.join(MASTER_ROOT, "prices", "returns_m", "returns_m.parquet"))
    # assets (for ticker labels)
    a = pd.read_parquet(os.path.join(MASTER_ROOT, "assets", "assets.parquet"))
    id2tkr = dict(zip(a["asset_id"].astype(int), a["ticker"]))
    R = t.pivot(index="month_end", columns="asset_id", values="r_m").sort_index()
    R.columns = [id2tkr.get(int(c), f"id{int(c)}") for c in R.columns]
    return R.astype("float64")

def _ticker_to_id() -> dict:
    a = pd.read_parquet(os.path.join(MASTER_ROOT, "assets", "assets.parquet"))
    return dict(zip(a["ticker"], a["asset_id"].astype(int)))

def main():
    out_path = os.path.join(MASTER_ROOT, "features", "monthly", "features_monthly.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    R = _load_returns_wide()
    tickers = list(R.columns)
    tkr2id = _ticker_to_id()

    rows = []  # (month_end, asset_id, feature, value, version)

    dates = R.index.to_list()
    # start when we have LOOKBACK months of history
    for t in range(LOOKBACK - 1, len(dates)):
        asof = dates[t]
        window = R.iloc[t - LOOKBACK + 1 : t + 1]  # 36 rows
        if window.isna().any().any():
            continue

        # 1) 36m correlation
        C = window.corr().fillna(0.0).values.astype("float64")
        # clip negatives to 0 (we only keep positive correlations for the graph)
        W = np.clip(C, 0.0, 1.0)
        np.fill_diagonal(W, 0.0)

        # 2) build graph
        G = nx.Graph()
        G.add_nodes_from(tickers)
        n = len(tickers)
        for i in range(n):
            for j in range(i + 1, n):
                w = float(W[i, j])
                if w > 0.0:
                    G.add_edge(tickers[i], tickers[j], weight=w)

        # 3) graph metrics (weighted)
        # weighted degree ("strength")
        deg_w = {n: 0.0 for n in tickers}
        for n_ in tickers:
            deg_w[n_] = float(G.degree(n_, weight="weight"))

        # eigenvector centrality (weighted). If disconnected/degenerate, fill zeros.
        try:
            eig_w = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            eig_w = {n_: 0.0 for n_ in tickers}

        # weighted clustering coefficient (Onnela-style in networkx)
        try:
            cluster_w = nx.clustering(G, weight="weight")
        except Exception:
            cluster_w = {n_: 0.0 for n_ in tickers}

        # 4) one-month cross-sectional absolute deviation
        r_t = R.iloc[t]  # returns at month t
        absdev = (r_t - r_t.mean()).abs().to_dict()

        # assemble features for this date (as Series) then z-score cross-sectionally
        df_feats = pd.DataFrame({
            "g_deg_wpos_36m": pd.Series(deg_w),
            "g_eigvec_wpos_36m": pd.Series(eig_w),
            "g_cluster_wpos_36m": pd.Series(cluster_w),
            "cs_absdev_1m": pd.Series(absdev),
        })  # index = tickers

        # z-score per feature across tickers at this date
        df_feats = df_feats.apply(_zscore_cs, axis=0)

        # push to rows (map ticker -> asset_id)
        for tkr, featvals in df_feats.iterrows():
            asset_id = int(tkr2id[tkr])
            for feat_name, val in featvals.items():
                rows.append((asof, asset_id, feat_name, float(val), VERSION))

    # tall to DataFrame
    new_feats = pd.DataFrame(rows, columns=["month_end", "asset_id", "feature", "value", "version"])
    new_feats = new_feats.astype({"asset_id": "int32", "value": "float32", "version": "string"})
    new_feats = new_feats.sort_values(["month_end", "asset_id", "feature"])

    # 5) append to existing features_monthly.parquet (if exists)
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        # ensure old has 'version' column
        if "version" not in old.columns:
            old["version"] = VERSION
        combined = pd.concat([old, new_feats], ignore_index=True)
        combined = combined.drop_duplicates(subset=["month_end", "asset_id", "feature", "version"], keep="last")
        combined = combined.sort_values(["month_end", "asset_id", "feature"])
        combined.to_parquet(out_path, index=False)
        print(f"[OK] Appended graph-lite features → {out_path}")
        print(f"[INFO] total rows: {len(combined)} | dates: {combined['month_end'].nunique()} | features: {combined['feature'].nunique()}")
    else:
        new_feats.to_parquet(out_path, index=False)
        print(f"[OK] Created features file with graph-lite features → {out_path}")
        print(f"[INFO] rows: {len(new_feats)} | dates: {new_feats['month_end'].nunique()} | features: {new_feats['feature'].nunique()}")

if __name__ == "__main__":
    main()