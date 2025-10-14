from __future__ import annotations
import numpy as np, pandas as pd, yaml
import os
from dataclasses import dataclass
from src.utils.common import set_seed
from src.data.loaders import load_synthetic_sector_returns, load_benchmark_weights, load_sector_returns_from_csv
from src.features.classic import make_classic_features
from src.risk.cov import estimate_cov
from src.portfolio.optimizer import solve_overlay

from src.data import master_api

from sklearn.linear_model import Ridge

@dataclass
class Results:
    pnl: pd.Series
    pnl_net: pd.Series
    turnover: pd.Series
    te_ann: pd.Series
    weights: dict  # date -> np.ndarray

def run_backtest(config_path: str) -> Results:
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg["seed"])
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]

    # 1) load returns & benchmark weights based on data source
    data_cfg    = cfg.get("data", {})
    data_source = data_cfg.get("source", "csv")
    master_root = data_cfg.get("master_root", "data/master")

    # Universe from config (robust; no global dependency)
    sectors = cfg.get("universe", {}).get("sectors")
    if not sectors:
        try:
            from src.data.loaders import SECTORS as DEFAULT_SECTORS
            sectors = DEFAULT_SECTORS
        except Exception:
            raise ValueError("Config 'universe.sectors' is missing and no default SECTORS found.")
    
    if data_source == "master":
        # load from master database
        df_ret = master_api.load_returns_wide(master_root, sectors)
        df_wb  = master_api.load_benchmark_weights_wide(master_root, sectors)
        X      = master_api.load_features_panel(
                    master_root,
                    cfg.get("features", {}).get("names", []),
                    cfg.get("features", {}).get("version", "v1.0.0"),
                    sectors
                )
        X.index.set_names(["date","sector"], inplace=True)
        X = X.fillna(0.0)
    elif data_source == "csv":
        # CSV wide-file path you already use
        from src.data.loaders import load_sector_returns_from_csv, load_benchmark_weights
        df_ret = load_sector_returns_from_csv(data_cfg.get("csv_folder"), sectors, start, end)
        df_wb  = load_benchmark_weights(start, end)
        X      = make_classic_features(df_ret)
    else:
        # synthetic fallback
        from src.data.loaders import load_synthetic_sector_returns, load_benchmark_weights
        df_ret = load_synthetic_sector_returns(start, end, cfg["seed"])
        df_wb  = load_benchmark_weights(start, end)
        X      = make_classic_features(df_ret)

    # 2) build feature panel (classic)
    if data_source != "master":
        X = make_classic_features(df_ret)  # MultiIndex (date, sector)

    # 3) training/evaluation windows
    warmup = cfg["evaluation"]["warmup_months"]
    dates = df_ret.index

    pnl, pnl_net = [], []
    turnover = []
    te_ann = []
    weights = {}
    preds = {}  # date -> pd.Series of predicted sector excess returns
    db_cov_used = 0
    fb_cov_used = 0
    w_prev = df_wb.iloc[0].reindex(sectors).values  # start from benchmark

    cost_bps = cfg["costs"]["one_way_bps"]
    te_target = cfg["portfolio"]["te_target_annual"]
    to_cap    = cfg["portfolio"]["turnover_cap"]
    lam       = cfg["portfolio"]["smoothing_lambda"]

    for t in range(warmup, len(dates)-1):
        asof = dates[t]
        nxt  = dates[t+1]

        # slice features up to asof
        Xi = X.loc[X.index.get_level_values(0) <= asof]

        # build training set from dates with complete features and available next-month returns
        X_train = []
        y_train = []
        unique_dates = Xi.index.get_level_values(0).unique()
        candidate_dates = unique_dates[:-1]  # exclude asof itself
        for d in candidate_dates:
            # features at date d (sectors x features), enforce sector order
            row_df = Xi.loc[d].reindex(sectors)
            # skip if any NaNs in this date's feature block
            if row_df.isna().values.any():
                continue
            next_date = d + pd.offsets.MonthEnd(1)
            if next_date not in df_ret.index:
                continue

            # next month returns (excess vs benchmark)
            r_next = df_ret.loc[d + pd.offsets.MonthEnd(1)].reindex(sectors)
            r_bmk  = (df_wb.loc[d + pd.offsets.MonthEnd(1)].reindex(sectors) * r_next).sum()
            y = r_next - r_bmk

            row = Xi.loc[d]                  # Xi is X filtered up to asof
            row = row.reindex(sectors)       # ensure consistent sector order
            X_train.append(row.values)
            y_train.append(y.reindex(sectors).values)
        if len(X_train) == 0:
            # not enough clean data yet; skip this month
            continue

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        # ---- explicit mean-impute + diagnostics to kill any remaining NaNs ----
        # force numeric dtype
        X_train = X_train.astype(float, copy=False)
        y_train = y_train.astype(float, copy=False)

        # report NaNs before impute
        n_nan_cells = int(np.isnan(X_train).sum())
        if n_nan_cells:
            print(f"[DEBUG] asof={asof.date()}  BEFORE IMPUTE: NaN cells in X_train = {n_nan_cells}")

        # column-wise mean (ignore NaNs); if a column is all-NaN, fallback to 0.0
        col_means = np.nanmean(X_train, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)

        # fill NaNs with column means
        nan_idx = np.where(np.isnan(X_train))
        if nan_idx[0].size:
            X_train[nan_idx] = col_means[nan_idx[1]]

        # final assert & log
        if np.isnan(X_train).any():
            raise RuntimeError(f"NaNs remained in X_train after impute at {asof.date()}")
        # ---- end impute block ----

        # fit forecaster and predict mu_hat for asof (next month)
        # select Ridge alpha on a small validation slice (last 24 months ≈ 24*len(sectors) rows)
        alphas = [0.1, 1.0, 10.0]
        best_alpha, best_score = 1.0, -1e9
        rows_per_month = len(sectors)
        if len(X_train) > 24 * rows_per_month:
            split = len(X_train) - 24 * rows_per_month
            X_tr, y_tr = X_train[:split], y_train[:split]
            X_va, y_va = X_train[split:], y_train[split:]
            for a in alphas:
                m = Ridge(alpha=a)
                m.fit(X_tr, y_tr)
                preds_va = m.predict(X_va)
                # correlation as a simple, scale-free score
                v = np.corrcoef(preds_va, y_va)[0, 1]
                if np.isnan(v):
                    v = -1.0
                if v > best_score:
                    best_score, best_alpha = v, a
        else:
            best_alpha = 1.0
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        row_t = X.loc[asof].reindex(sectors).fillna(0.0).values
        if np.isnan(row_t).any():
            raise RuntimeError(f"NaNs in as-of features at {asof.date()} after fillna(0.0)")
        mu_hat = model.predict(row_t)
        # cross-sectional standardize and clip for stable overlay behavior
        mu_hat = (mu_hat - mu_hat.mean()) / (mu_hat.std() + 1e-12)
        mu_hat = np.clip(mu_hat, -3.0, 3.0)

        # store predictions for auditing
        preds[asof] = pd.Series(mu_hat, index=sectors)

        # risk model at t (Hybrid: try DB covariance first in master mode)
        if data_source == "master":
            Sigma = master_api.load_cov(master_root, asof, sectors)
            if Sigma is None:
                fb = cfg.get("risk", {}).get("fallback_lookback", 36)
                Sigma = estimate_cov(df_ret, asof, lookback=fb, shrink=0.1)
                fb_cov_used += 1
            else:
                db_cov_used += 1
        else:
            Sigma = estimate_cov(df_ret, asof, lookback=60, shrink=0.1)

        # crisis flag (synthetic proxy: high cross-sec vol)
        crisis = df_ret.loc[asof].std() > 0.10

        # optimize overlay weights
        w_bench = df_wb.loc[asof].reindex(sectors).values
        w = solve_overlay(mu_hat, Sigma, w_bench, w_prev,
                          te_target, to_cap, lam, crisis)

        # compute turnover cost (one-way bps)
        one_way = 0.5 * np.sum(np.abs(w - w_prev)) * 2
        cost = (cost_bps / 10000.0) * one_way

        # realize pnl next month (active return vs benchmark)
        r_next  = df_ret.loc[nxt].reindex(sectors).values
        r_bmk   = df_wb.loc[nxt].reindex(sectors).values @ r_next
        r_port = w @ r_next
        active = r_port - r_bmk
        active_net = active - cost

        # realized TE report (uses Sigma at asof)
        te = ((w - w_bench) @ Sigma @ (w - w_bench))**0.5 * (12**0.5)

        pnl.append(pd.Series(active, index=[nxt]))        # active gross
        pnl_net.append(pd.Series(active_net, index=[nxt]))# active net
        turnover.append(pd.Series(one_way, index=[nxt]))
        te_ann.append(pd.Series(te, index=[nxt]))
        weights[asof] = w
        w_prev = w

    pnl = pd.concat(pnl) ; pnl_net = pd.concat(pnl_net)
    turnover = pd.concat(turnover) ; te_ann = pd.concat(te_ann)

    # ---- save monthly predictions to CSV ----
    try:
        os.makedirs("output", exist_ok=True)
        preds_df = pd.DataFrame.from_dict(preds, orient="index", columns=sectors).sort_index()
        # ensure datetime index
        preds_df.index = pd.to_datetime(preds_df.index)
        preds_df.to_csv("output/preds.csv")
        print(f"[INFO] Saved preds to output/preds.csv ({preds_df.shape[0]} rows × {preds_df.shape[1]} sectors).")
    except Exception as e:
        print(f"[WARN] Could not save preds.csv: {e}")

    # ---- risk source usage summary ----
    print(f"[INFO] Risk source usage — DB: {db_cov_used} months, Fallback: {fb_cov_used} months.")

    # ---- save monthly weights to CSV ----
    try:
        os.makedirs("output", exist_ok=True)
        weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=sectors).sort_index()
        # ensure datetime index
        weights_df.index = pd.to_datetime(weights_df.index)
        weights_df.to_csv("output/weights.csv")
        print(f"[INFO] Saved weights to output/weights.csv ({weights_df.shape[0]} rows × {weights_df.shape[1]} sectors).")
    except Exception as e:
        print(f"[WARN] Could not save weights.csv: {e}")
    # ---- end weights export ----

    return Results(pnl=pnl, pnl_net=pnl_net, turnover=turnover, te_ann=te_ann, weights=weights)
