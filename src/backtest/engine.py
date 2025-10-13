from __future__ import annotations
import numpy as np, pandas as pd, yaml
from dataclasses import dataclass
from src.utils.common import set_seed
from src.data.loaders import load_synthetic_sector_returns, load_benchmark_weights, SECTORS
from src.features.classic import make_classic_features
from src.risk.cov import estimate_cov
from src.portfolio.optimizer import solve_overlay

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

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

    # 1) load synthetic returns & benchmark weights
    df_ret = load_synthetic_sector_returns(start, end, cfg["seed"])
    df_wb  = load_benchmark_weights(start, end)

    # 2) build feature panel (classic)
    X = make_classic_features(df_ret)  # MultiIndex (date, sector)

    # 3) training/evaluation windows
    warmup = cfg["evaluation"]["warmup_months"]
    dates = df_ret.index
    model = make_pipeline(SimpleImputer(strategy="mean"), Ridge(alpha=1.0))

    pnl, pnl_net = [], []
    turnover = []
    te_ann = []
    weights = {}
    w_prev = df_wb.iloc[0].values  # start from benchmark

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
            row_df = Xi.loc[d].reindex(SECTORS)
            # skip if any NaNs in this date's feature block
            if row_df.isna().values.any():
                continue
            next_date = d + pd.offsets.MonthEnd(1)
            if next_date not in df_ret.index:
                continue

            # next month returns (excess vs benchmark)
            r_next = df_ret.loc[next_date]
            r_bmk  = (df_wb.loc[next_date] * df_ret.loc[next_date]).sum()
            y = r_next - r_bmk

            X_train.append(row_df.values)                 # shape (11, n_feat)
            y_train.append(y.reindex(SECTORS).values)     # shape (11,)

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
        model.fit(X_train, y_train)
        row_t = X.loc[asof].reindex(SECTORS).values  # features at t

        # impute current as-of feature matrix defensively (same column means as training if available)
        if np.isnan(row_t).any():
            # compute means per feature from the already-imputed training matrix
            train_means = X_train.mean(axis=0)
            row_nan = np.isnan(row_t)
            if row_nan.any():
                # broadcast means to sector rows
                row_t[row_nan] = train_means[np.where(row_nan)[1]]

        mu_hat = model.predict(row_t)

        # risk model at t
        Sigma = estimate_cov(df_ret, asof, lookback=60, shrink=0.1)

        # crisis flag (synthetic proxy: high cross-sec vol)
        crisis = df_ret.loc[asof].std() > 0.10

        # optimize overlay weights
        w_bench = df_wb.loc[asof].values
        w = solve_overlay(mu_hat, Sigma, w_bench, w_prev,
                          te_target, to_cap, lam, crisis)

        # compute turnover cost (one-way bps)
        one_way = 0.5 * np.sum(np.abs(w - w_prev)) * 2
        cost = (cost_bps / 10000.0) * one_way

        # realize pnl next month (active return vs benchmark)
        r_next = df_ret.loc[nxt].values
        r_bmk  = df_wb.loc[nxt].values @ r_next
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

    return Results(pnl=pnl, pnl_net=pnl_net, turnover=turnover, te_ann=te_ann, weights=weights)
