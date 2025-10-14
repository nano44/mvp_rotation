from __future__ import annotations
import numpy as np, pandas as pd, yaml
import os
from dataclasses import dataclass
from src.utils.common import set_seed, month_ends
from src.features.classic import make_classic_features
from src.risk.cov import estimate_cov
from src.portfolio.optimizer import solve_overlay

from src.data import master_api


from sklearn.linear_model import Ridge

# ---- Alpha selection helpers (Spearman rank IC on a 24m validation window) ----
from typing import List, Tuple
try:
    from scipy.stats import spearmanr  # optional; safe fallback below if unavailable
except Exception:
    spearmanr = None
import numpy as _np

def _rank_spearman(x: _np.ndarray, y: _np.ndarray) -> float:
    """Spearman rank correlation with a safe fallback if scipy is unavailable."""
    x = _np.asarray(x, float)
    y = _np.asarray(y, float)
    if x.size != y.size or x.size == 0:
        return _np.nan
    if spearmanr is not None:
        v, _ = spearmanr(x, y)
        return float(v)
    # Fallback: rank via argsort and compute Pearson on ranks
    rx = _np.argsort(_np.argsort(x))
    ry = _np.argsort(_np.argsort(y))
    sx = (rx - rx.mean()) / (rx.std(ddof=1) if rx.std(ddof=1) > 0 else 1.0)
    sy = (ry - ry.mean()) / (ry.std(ddof=1) if ry.std(ddof=1) > 0 else 1.0)
    return float(_np.mean(sx * sy))

def _select_alpha(
    X_list: List[_np.ndarray],
    y_list: List[_np.ndarray],
    alphas: List[float],
    val_k: int = 24,
) -> Tuple[float, dict, float]:
    """Choose Ridge alpha using mean Spearman IC over the last `val_k` months.
    X_list: list of (S, F) arrays per month; y_list: list of (S,) arrays per month.
    Returns (best_alpha, {alpha: mean_IC}, best_ic).
    """
    n = len(X_list)
    if n == 0:
        return 1.0, {}, float('nan')
    k = min(val_k, max(1, n // 5))  # ensure some validation, cap at val_k
    if n <= k + 2:
        return 1.0, {}, float('nan')

    # Split: train months [0 : n-k), validation months [n-k : n)
    X_tr = _np.vstack(X_list[: n - k])
    y_tr = _np.hstack(y_list[: n - k])

    ic_by_alpha = {}
    best_alpha, best_ic = 1.0, -1e9
    for a in alphas:
        m = Ridge(alpha=a)
        m.fit(X_tr, y_tr)
        ics = []
        for d in range(n - k, n):
            pred_d = m.predict(X_list[d])  # (S,)
            ic = _rank_spearman(pred_d, y_list[d])
            if _np.isfinite(ic):
                ics.append(ic)
        score = float(_np.nanmean(ics)) if len(ics) else _np.nan
        ic_by_alpha[a] = score
        if _np.isfinite(score) and score > best_ic:
            best_ic, best_alpha = score, a
    return best_alpha, ic_by_alpha, best_ic

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
    # --- recency weighting: print configured half-life ---
    hl_cfg = int(cfg.get("model", {}).get("half_life_months", 0))
    print(f"[INFO] Recency weighting (half-life months): {hl_cfg if hl_cfg > 0 else 'OFF'}")
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]

    # ---- mtime guard: warn if benchmark/labels are stale vs raw CSV or returns ----
    def _mtime(path: str):
        try:
            return os.path.getmtime(path)
        except Exception:
            return None

    raw_csv      = "data/raw/spx_sector_weights_monthly.csv"
    bench_parq   = "data/master/benchmark/bench_weights_m.parquet"
    labels_parq  = "data/master/labels/forward_returns/fwd_excess_1m.parquet"
    returns_parq = "data/master/prices/returns_m/returns_m.parquet"

    mt_raw     = _mtime(raw_csv)
    mt_bench   = _mtime(bench_parq)
    mt_labels  = _mtime(labels_parq)
    mt_returns = _mtime(returns_parq)

    if mt_raw and mt_bench and mt_raw > mt_bench:
        print("[WARN] Benchmark parquet is older than raw SPX CSV — run: python -m scripts.build_spx_benchmark")

    if mt_bench and mt_labels and mt_bench > mt_labels:
        print("[WARN] Labels parquet is older than benchmark — run: python -m scripts.build_labels_forward")
    if mt_returns and mt_labels and mt_returns > mt_labels:
        print("[WARN] Labels parquet may be stale vs returns — run: python -m scripts.build_labels_forward")

    # 1) load REAL returns (tall parquet → wide) & benchmark weights (canonical path)
    data_cfg    = cfg.get("data", {})
    data_source = data_cfg.get("source", "csv")  # retained only for downstream risk branch
    master_root = data_cfg.get("master_root", "data/master")

    # Universe from config (robust; no global dependency)
    sectors = cfg.get("universe", {}).get("sectors")
    if not sectors:
        try:
            from src.data.loaders import SECTORS as DEFAULT_SECTORS
            sectors = DEFAULT_SECTORS
        except Exception:
            raise ValueError("Config 'universe.sectors' is missing and no default SECTORS found.")

    # returns: tall schema (month_end, asset_id, r_m) → wide DataFrame (cols=sectors)
    R = pd.read_parquet("data/master/prices/returns_m/returns_m.parquet")
    id_to_sector = {i: s for i, s in enumerate(sectors)}
    if "sector" not in R.columns:
        R["sector"] = R["asset_id"].map(id_to_sector)
    df_ret = (
        R.pivot_table(index="month_end", columns="sector", values="r_m", aggfunc="last")
         .sort_index()
         .reindex(columns=sectors)
    )

    # restrict to config date window and drop rows with any missing sector
    idx = pd.Index(month_ends(start, end), name="month_end")
    df_ret = df_ret.reindex(idx).dropna(how="any")

    # benchmark (wide parquet expected): align to returns index/columns and normalize rows to 1
    df_wb = pd.read_parquet("data/master/benchmark/bench_weights_m.parquet")
    if not isinstance(df_wb.index, pd.DatetimeIndex):
        df_wb.index = pd.to_datetime(df_wb.index)
    df_wb.index = df_wb.index.to_period("M").to_timestamp("M")
    df_wb.index.name = "month_end"
    df_wb = (df_wb.reindex(df_ret.index)
                  .reindex(columns=sectors)
                  .ffill())
    df_wb = df_wb.div(df_wb.sum(axis=1).replace(0, np.nan), axis=0)

    # build classic features on the aligned returns
    X = make_classic_features(df_ret)
    X.index.set_names(["date","sector"], inplace=True)
    X = X.fillna(0.0)

    # Optionally join graph features if present (degree/eigvec/clustering + cs_absdev)
    graph_paths = [
        "data/master/features/graph/graph_features.parquet",
        "data/master/features/monthly/features_monthly.parquet",
    ]
    joined_graph = False
    for _p in graph_paths:
        try:
            if not os.path.exists(_p):
                continue
            Xg = pd.read_parquet(_p)
            # ensure MultiIndex with names (date, sector)
            if not isinstance(Xg.index, pd.MultiIndex):
                Xg.index = pd.MultiIndex.from_frame(Xg.index.to_frame(index=False))
            Xg.index.set_names(["date", "sector"], inplace=True)
            # keep only graph-lite columns
            keep_cols = [c for c in Xg.columns if str(c).startswith("g_") or str(c)=="cs_absdev_1m"]
            if not keep_cols:
                continue
            X = X.join(Xg[keep_cols], how="left").fillna(0.0)
            print(f"[INFO] Joined graph features from {_p}: X shape → {X.shape}")
            joined_graph = True
            break
        except Exception as e:
            print(f"[WARN] Failed reading graph features at {_p}: {e}")
    if not joined_graph:
        print("[WARN] Graph features not joined: no feature file found in expected locations")

    # quick sanity
    print(f"[INFO] Engine data span: {df_ret.index.min().date()} → {df_ret.index.max().date()} | months={len(df_ret)}")


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
    gate_count = 0
    w_prev = df_wb.iloc[0].reindex(sectors).values  # start from benchmark
    diags = []  # per-month diagnostics rows
    # Enforce sector availability on initial weights (pre-inception XLRE/XLC = 0)
    inception = {"XLRE": pd.Timestamp("2016-09-30"),
                 "XLC":  pd.Timestamp("2018-09-30")}
    first_date = df_wb.index[0]
    avail0 = np.array([(first_date >= inception.get(s, pd.Timestamp.min)) for s in sectors], dtype=bool)
    w_prev = np.where(avail0, w_prev, 0.0)
    if w_prev.sum() > 0:
        w_prev = w_prev / w_prev.sum()

    cost_bps = cfg["costs"]["one_way_bps"]
    te_target = cfg["portfolio"]["te_target_annual"]
    to_cap    = cfg["portfolio"]["turnover_cap"]
    lam       = cfg["portfolio"]["smoothing_lambda"]

    # log chosen Ridge alpha per month (for reporting)
    alpha_chosen = {}

    for t in range(warmup, len(dates)-1):
        asof = dates[t]
        nxt  = dates[t+1]

        # slice features up to asof
        Xi = X.loc[X.index.get_level_values(0) <= asof]

        # Assemble per-month design matrices (keep per-month for IC-on-month validation)
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        unique_dates = Xi.index.get_level_values(0).unique()
        if len(unique_dates) < 2:
            continue  # need at least one month for y (next) beyond features month

        for d in unique_dates[:-1]:  # exclude asof itself
            next_date = d + pd.offsets.MonthEnd(1)
            if next_date not in df_ret.index:
                continue

            # next month returns (excess vs benchmark)
            r_next = df_ret.loc[next_date].reindex(sectors)
            r_bmk  = (df_wb.loc[next_date].reindex(sectors) * r_next).sum()
            y = (r_next - r_bmk).values  # (S,)

            # features at date d (sectors × features) in config order
            row_df = Xi.loc[d].reindex(sectors)
            if row_df.isna().values.any():
                continue  # skip months with incomplete features
            Xd = row_df.values  # (S, F)

            X_list.append(Xd)
            y_list.append(y)

        if not X_list:
            continue

        # Choose alpha on last 24 months by mean Spearman IC
        alpha_grid = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
        best_alpha, _ic_map, _best_ic = _select_alpha(X_list, y_list, alpha_grid, val_k=24)
        sign_gate = 1.0 if (_best_ic is None or not np.isfinite(_best_ic) or _best_ic >= 0) else -1.0

        # --- Recency weighting for final fit (exponential decay by month) ---
        hl = int(cfg.get("model", {}).get("half_life_months", 0))
        if hl and hl > 0:
            ages = np.arange(len(X_list))[::-1]  # 0 = most recent month at the end
            w_months = np.exp(-np.log(2) * ages / float(hl))
        else:
            w_months = np.ones(len(X_list), dtype=float)
        w_recent = float(w_months[-1])
        w_oldest = float(w_months[0])
        w_ratio  = float(w_recent / (w_oldest if w_oldest > 0 else 1.0))

        # Fit final model on ALL available training months up to asof using best_alpha
        X_train = np.vstack(X_list).astype(float, copy=False)
        y_train = np.hstack(y_list).astype(float, copy=False)

        # Safety impute (should be no-ops due to skip-above); keep for robustness
        if np.isnan(X_train).any():
            col_means = np.nanmean(X_train, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            nan_idx = np.where(np.isnan(X_train))
            if nan_idx[0].size:
                X_train[nan_idx] = col_means[nan_idx[1]]
            if np.isnan(X_train).any():
                raise RuntimeError(f"NaNs remained in X_train after impute at {asof.date()}")

        model = Ridge(alpha=best_alpha)
        # expand month weights to observation weights (repeat per sector)
        obs_w = np.repeat(w_months, repeats=len(sectors))
        model.fit(X_train, y_train, sample_weight=obs_w)
        alpha_chosen[asof] = float(best_alpha)

        # Predict mu_hat for asof (next month) and apply sign gate
        row_t = X.loc[asof].reindex(sectors).fillna(0.0).values  # (S, F)
        mu_hat = sign_gate * model.predict(row_t)  # (S,)

        # Availability mask (no signal before sector inception)
        inception = {"XLRE": pd.Timestamp("2016-09-30"),
                     "XLC":  pd.Timestamp("2018-09-30")}
        avail_mask = np.array([(asof >= inception.get(s, pd.Timestamp.min)) for s in sectors], dtype=bool)

        # Rank transform among available sectors only (more stable than levels)
        mu_ser = pd.Series(mu_hat, index=sectors)
        if avail_mask.any():
            mu_av = mu_ser[avail_mask]
            ranks = mu_av.rank(method="average")
            mu_av = (ranks - ranks.mean()) / (ranks.std(ddof=0) + 1e-12)
            mu_ser.loc[avail_mask] = mu_av.clip(-2.0, 2.0)
        mu_ser.loc[~avail_mask] = 0.0
        mu_hat = mu_ser.values
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

        # --- Dispersion gate (defensive): lower TE, raise smoothing when opportunity is low ---
        # 12m average of cross-sectional std of sector returns
        j0 = max(0, t-11)
        disp_12 = df_ret.iloc[j0:t+1].std(axis=1).mean()
        # rolling median of historical dispersion up to asof
        hist_disp = df_ret.iloc[:t+1].std(axis=1)
        disp_med = float(hist_disp.median()) if len(hist_disp) else float(disp_12)
        low_disp = np.isfinite(disp_12) and np.isfinite(disp_med) and (disp_12 < disp_med)

        te_eff = te_target * (0.5 if low_disp else 1.0)
        lam_eff = max(lam, 0.7) if low_disp else lam
        if low_disp:
            gate_count += 1

        # optimize overlay weights
        w_bench = df_wb.loc[asof].reindex(sectors).values
        w = solve_overlay(mu_hat, Sigma, w_bench, w_prev,
                          te_eff, to_cap, lam_eff, crisis)
        # Enforce availability in portfolio weights and renormalize
        w = np.where(avail_mask, w, 0.0)
        s = w.sum()
        if s <= 1e-12:
            # Fallback: use available portion of benchmark
            wb_avail = np.where(avail_mask, w_bench, 0.0)
            s2 = wb_avail.sum()
            w = wb_avail / (s2 if s2 > 0 else 1.0)
        else:
            w = w / s

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

        # record diagnostics for this month
        diags.append({
            "date": asof,
            "best_alpha": float(best_alpha),
            "best_ic": float(_best_ic) if (_best_ic is not None and np.isfinite(_best_ic)) else np.nan,
            "sign_gate": float(sign_gate),
            "hl_months": int(cfg.get("model", {}).get("half_life_months", 0)),
            "w_recent": w_recent,
            "w_oldest": w_oldest,
            "w_ratio": w_ratio,
            "disp_12": float(disp_12) if np.isfinite(disp_12) else np.nan,
            "disp_med": float(disp_med) if np.isfinite(disp_med) else np.nan,
            "low_disp": bool(low_disp),
            "te_eff": float(te_eff),
            "lam_eff": float(lam_eff),
            "realized_te": float(te),
            "turnover": float(one_way),
            "active_gross": float(active),
            "active_net": float(active_net),
        })
        pnl.append(pd.Series(active, index=[nxt]))        # active gross
        pnl_net.append(pd.Series(active_net, index=[nxt]))# active net
        turnover.append(pd.Series(one_way, index=[nxt]))
        te_ann.append(pd.Series(te, index=[nxt]))
        weights[asof] = w
        w_prev = w

    pnl = pd.concat(pnl) ; pnl_net = pd.concat(pnl_net)
    turnover = pd.concat(turnover) ; te_ann = pd.concat(te_ann)

    # ---- save diagnostics CSV ----
    try:
        os.makedirs("output", exist_ok=True)
        diag_df = pd.DataFrame(diags)
        if not diag_df.empty:
            diag_df["date"] = pd.to_datetime(diag_df["date"])  # ensure datetime indexable
            diag_df = diag_df.set_index("date").sort_index()
            diag_df.to_csv("output/diagnostics.csv")
            print(f"[INFO] Saved diagnostics to output/diagnostics.csv ({diag_df.shape[0]} rows).")
        else:
            print("[WARN] No diagnostics rows to save.")
    except Exception as e:
        print(f"[WARN] Could not save diagnostics.csv: {e}")

    # ---- save chosen alphas per month for diagnostics/reporting ----
    try:
        os.makedirs("output", exist_ok=True)
        s_alpha = pd.Series(alpha_chosen)
        s_alpha.index = pd.to_datetime(s_alpha.index)
        s_alpha.sort_index().to_csv("output/chosen_alpha.csv")
        print("[INFO] Saved chosen Ridge alphas to output/chosen_alpha.csv")
    except Exception as e:
        print(f"[WARN] Could not write chosen_alpha.csv: {e}")

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
    print(f"[INFO] Dispersion gate active in {gate_count} of {len(dates)-1-warmup} months.")

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
