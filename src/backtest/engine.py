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
from scipy.stats import spearmanr

@dataclass
class Results:
    pnl: pd.Series
    pnl_net: pd.Series
    turnover: pd.Series
    te_ann: pd.Series
    weights: dict  # date -> np.ndarray

# ----------------- PHASE 1 HELPERS -----------------
def _half_life_weights(n: int, hl: int) -> np.ndarray:
    """Month-level exp decay weights ending at the most recent month (index n-1). Mean-normalized."""
    if hl is None or hl <= 0 or n <= 0:
        return np.ones(n, dtype=float)
    lags = (np.arange(n) - (n - 1)).astype(float)  # 0 for most recent
    w = np.exp(-np.log(2.0) * np.abs(lags) / float(hl))
    return w * (n / w.sum())  # mean ≈ 1

def _winsorize_in_window(y: np.ndarray, pct: float) -> np.ndarray:
    """Clip y to [p,1-p] quantiles. Works on 1D; caller reshapes."""
    if pct <= 0:
        return y
    lo, hi = np.nanpercentile(y, [100*pct, 100*(1 - pct)])
    return np.clip(y, lo, hi)

def _rank_ladder(x: np.ndarray) -> np.ndarray:
    """Ranks 1..N → centered ladder (e.g., N=11 → [-5..+5])."""
    s = pd.Series(x).rank(method="first")
    N = len(s)
    return (s - (N + 1) / 2.0).to_numpy()

def _xsec_zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    return (x - mu) / (sd if sd > 0 else 1.0)

# Numerically safe Ridge solver for small/ill-conditioned problems
def _ridge_solve(sumA: np.ndarray, sumb: np.ndarray, alpha: float) -> np.ndarray:
    """Solve (sumA + alpha I) beta = sumb with a stable fallback."""
    F = sumA.shape[0]
    I = np.eye(F)
    lam = float(alpha)
    try:
        return np.linalg.solve(sumA + lam * I, sumb)
    except np.linalg.LinAlgError:
        # tiny jitter + least squares fallback
        return np.linalg.lstsq(sumA + (lam + 1e-8) * I, sumb, rcond=None)[0]

def _decayed_stats(values: list[float], hl: int) -> tuple[float,float,float]:
    """Weighted mean, median, std with exp decay (oldest→newest)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return -np.inf, -np.inf, np.inf
    w = _half_life_weights(n, hl)
    w /= w.sum()
    mean = np.nansum(w * v)
    var = np.nansum(w * (v - mean)**2)
    std = float(np.sqrt(var))
    order = np.argsort(v)
    cdf = np.cumsum(w[order])
    median = v[order][np.searchsorted(cdf, 0.5)]
    return float(mean), float(median), std
# --------------- END HELPERS -----------------------

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
        # Align benchmark to returns months (defensive: ensures we use the canonical master parquet span)
        df_wb = df_wb.reindex(df_ret.index).dropna(how="all")
        if df_wb.empty:
            raise RuntimeError("[FATAL] Benchmark weights empty after aligning to returns. "
                               "Rebuild with: python -m scripts.build_spx_benchmark")
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

    # ---- FAST PRECOMPUTE: month arrays & sufficient statistics ----
    sectors = sectors  # local alias to emphasize fixed order
    all_feat_dates = X.index.get_level_values(0).unique().tolist()
    all_feat_dates.sort()
    Xs, Ys, valid_dates = [], [], []  # Xs: (T,N,F), Ys: (T,N)
    for d in all_feat_dates:
        nxt = d + pd.offsets.MonthEnd(1)
        if nxt not in df_ret.index:
            continue
        try:
            row_df = X.loc[d].reindex(sectors)
        except KeyError:
            continue
        if row_df.isna().values.any():
            row_df = row_df.fillna(0.0)
        r_next = df_ret.loc[nxt].reindex(sectors)
        r_bmk  = (df_wb.loc[nxt].reindex(sectors) * r_next).sum()
        y_vec = (r_next - r_bmk).values
        Xs.append(row_df.values.astype(float))
        Ys.append(y_vec.astype(float))
        valid_dates.append(d)

    if len(Xs) == 0:
        raise RuntimeError("No valid (features, next-month returns) pairs to train on.")

    Xs = np.stack(Xs)   # T × N × F
    Ys = np.stack(Ys)   # T × N
    T, N, F = Xs.shape

    # Per-month sufficient stats: A_t = X_t^T X_t (F×F), b_t = X_t^T y_t (F)
    A_blocks = np.einsum('tnf,tng->tfg', Xs, Xs)  # (T,F,F)
    b_blocks = np.einsum('tnf,tn->tf',   Xs, Ys)  # (T,F)
    date_to_idx = {d: i for i, d in enumerate(valid_dates)}
    # ---- END FAST PRECOMPUTE ----

    # 3) training/evaluation windows
    warmup = cfg["evaluation"]["warmup_months"]
    dates = df_ret.index

    # ---- run-span diagnostics (data mode & spans) ----
    try:
        ret_min = pd.to_datetime(df_ret.index.min()).date()
        ret_max = pd.to_datetime(df_ret.index.max()).date()
        bmk_min = pd.to_datetime(df_wb.index.min()).date()
        bmk_max = pd.to_datetime(df_wb.index.max()).date()
        print(f"[INFO] Data source: {data_source} | sectors={len(sectors)}")
        print(f"[INFO] Returns span:   {ret_min} → {ret_max} | n={len(df_ret)}")
        print(f"[INFO] Benchmark span: {bmk_min} → {bmk_max} | n={len(df_wb)}")
        if isinstance(X.index, pd.MultiIndex):
            xd = X.index.get_level_values(0)
            x_min = pd.to_datetime(xd.min()).date()
            x_max = pd.to_datetime(xd.max()).date()
            print(f"[INFO] Features span:  {x_min} → {x_max} | months={xd.nunique()} | cols={X.shape[1]}")
        else:
            print(f"[INFO] Features shape: {X.shape}")
        print(f"[INFO] Eval windows: warmup={warmup}, purge={cfg['evaluation'].get('purge_months',0)}, embargo={cfg['evaluation'].get('embargo_months',0)}")
        expected_oos = max(0, len(dates) - warmup - 1)
        print(f"[INFO] Expected OOS months (approx): {expected_oos}")
    except Exception as e:
        print(f"[WARN] run-span prints failed: {e}")

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

    # ---- model knobs (Phase 1) ----
    M = cfg.get("model", {})
    R = cfg.get("runtime", {})
    train_L         = int(M.get("train_lookback_months", 84))
    min_L           = int(M.get("min_lookback_months", 36))
    hl_fit          = int(M.get("half_life_months", 0))
    alpha_grid      = list(M.get("alpha_grid", [0.1, 1.0, 10.0]))
    V_tail          = int(M.get("selector_tail_months", 24))
    hl_sel          = int(M.get("selector_half_life_months", max(hl_fit, 12)))
    sel_metric      = M.get("selector_metric", "spearman_ic_median")
    d_alpha_min     = float(M.get("alpha_improvement_min", 0.0))
    alpha_beta      = float(M.get("alpha_ema_beta", 0.0))  # EMA on log-alpha
    tie_breaker     = M.get("alpha_tie_breaker", "smallest_alpha")
    pred_transform  = M.get("prediction_transform", "zscore")
    win_pct         = float(M.get("label_winsor_pct", 0.0))
    purge_m         = int(cfg.get("evaluation", {}).get("purge_months", 1))
    embargo_m       = int(cfg.get("evaluation", {}).get("embargo_months", 1))
    turnover_eps    = float(R.get("turnover_eps", 0.005))

    diagnostics_rows = []
    state = {"alpha_prev": None, "last_weights": None}

    for t in range(warmup, len(dates)-1):
        asof = dates[t]
        nxt  = dates[t+1]

        # ---- FAST PHASE 1: windowed + decayed training using sufficient stats ----
        # Map as-of date to index in our cached arrays
        if asof not in date_to_idx:
            continue
        i_asof = date_to_idx[asof]

        # Determine rolling lookback length at this as-of
        L_t = min(train_L, i_asof)  # months strictly before asof
        if L_t < min_L:
            continue

        # Train window indices [start_win..end_win] = [i_asof-L_t .. i_asof-1]
        start_win = i_asof - L_t
        end_win   = i_asof - 1

        # Tail months for α-selection live in this window (last V_tail months)
        tail_start = max(start_win, end_win - V_tail + 1)
        tail_idx = np.arange(tail_start, end_win + 1, dtype=int)

        def _fit_stats_over(idx_arr: np.ndarray, hl: int, use_winsor: bool) -> tuple[np.ndarray, np.ndarray]:
            """Return (sumA, sumb) for the given month indices with half-life weights.
            Optionally winsorize labels within the window before building sumb.
            Applies embargo by dropping last `embargo_m` months from idx_arr."""
            if idx_arr.size == 0:
                return None, None
            # Apply embargo: drop last `embargo_m` months from training
            if embargo_m > 0 and idx_arr.size > embargo_m:
                idx_arr = idx_arr[:-embargo_m]
            if idx_arr.size == 0:
                return None, None
            m = idx_arr.size
            w = _half_life_weights(m, hl)
            # Weighted sum of per-month Gram matrices
            sumA = np.tensordot(w, A_blocks[idx_arr], axes=(0, 0))  # (F,F)
            # Weighted sum for b: optionally winsorize y within the window
            if use_winsor and win_pct > 0.0:
                Y_blk = Ys[idx_arr]  # (m,N)
                lo, hi = np.percentile(Y_blk, [100*win_pct, 100*(1-win_pct)])
                Yc = np.clip(Y_blk, lo, hi)
                # per-month b_k = X_k^T y_k
                b_blk = np.einsum('mnf,mn->mf', Xs[idx_arr], Yc)  # (m,F)
                sumb = w @ b_blk  # (F,)
            else:
                sumb = np.tensordot(w, b_blocks[idx_arr], axes=(0, 0))  # (F,)
            return sumA, sumb

        # α-selection: compute monthly ICs on the tail and score by decayed median
        scores = []  # (alpha, mean_ic, median_ic, ic_ir)
        for a in alpha_grid:
            ics = []
            for m_idx in tail_idx:
                # Train window for month m: [max(start_win, m_idx-L_t) .. m_idx-1]
                start_m = max(start_win, m_idx - L_t)
                end_m   = m_idx - 1
                if end_m < start_m:
                    continue
                train_idx = np.arange(start_m, end_m + 1, dtype=int)
                sumA, sumb = _fit_stats_over(train_idx, hl_fit, use_winsor=True)
                if sumA is None:
                    continue
                # Closed-form Ridge: (A + αI)β = b
                beta = _ridge_solve(sumA, sumb, a)
                # Predict at month m and compute IC with realized next-month y
                yhat_m = Xs[m_idx] @ beta  # (N,)
                ic = spearmanr(yhat_m, Ys[m_idx], nan_policy="omit").correlation
                if np.isfinite(ic):
                    ics.append(float(ic))
            mean_ic, med_ic, std_ic = _decayed_stats(ics, hl_sel)
            ic_ir = (mean_ic / std_ic) if np.isfinite(std_ic) and std_ic > 1e-8 else 0.0
            scores.append((a, mean_ic, med_ic, ic_ir))

        if not scores:
            best_alpha = alpha_grid[len(alpha_grid)//2]
            mean_star = median_star = icir_star = 0.0
        else:
            keyfun = (lambda s: s[2]) if sel_metric == "spearman_ic_median" else (lambda s: s[1])
            scores.sort(key=keyfun, reverse=True)
            best_alpha, mean_star, median_star, icir_star = scores[0]
            # stability guard vs previous alpha
            alpha_prev = state.get("alpha_prev", best_alpha)
            prev_row = next((s for s in scores if s[0] == alpha_prev), None)
            prev_metric = keyfun(prev_row) if prev_row else -np.inf
            if (keyfun(scores[0]) - prev_metric) < d_alpha_min:
                best_alpha = alpha_prev
            # EMA on log-alpha (optional)
            if alpha_beta and alpha_prev is not None and alpha_prev > 0 and best_alpha > 0:
                best_alpha = float(np.exp(alpha_beta*np.log(alpha_prev) + (1 - alpha_beta)*np.log(best_alpha)))
            # tie-breaker among near-bests
            if tie_breaker == "smallest_alpha":
                best_val = keyfun((best_alpha, mean_star, median_star, icir_star))
                near = [row for row in scores if (best_val - keyfun(row)) <= turnover_eps]
                if near:
                    best_alpha = min(r[0] for r in near)

        # Final fit on full window [start_win..end_win] with HL weights (no embargo for final fit)
        train_idx_full = np.arange(start_win, end_win + 1, dtype=int)
        sumA, sumb = _fit_stats_over(train_idx_full, hl_fit, use_winsor=True)
        if sumA is None:
            continue
        beta = _ridge_solve(sumA, sumb, best_alpha)

        # Predict at as-of (features at month i_asof)
        mu_raw = Xs[i_asof] @ beta  # (N,)
        if pred_transform.lower() == "rank":
            mu_t = _rank_ladder(mu_raw)
        else:
            mu_t = _xsec_zscore(mu_raw)
            mu_t = np.clip(mu_t, -3.0, 3.0)

        # store predictions (transformed) for auditing
        preds[asof] = pd.Series(mu_t, index=sectors)
        # ---- END FAST PHASE 1 ----

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
        w = solve_overlay(mu_t, Sigma, w_bench, w_prev,
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

        diagnostics_rows.append({
            "date": asof.date(),
            "L_t": L_t,
            "HL": hl_fit,
            "alpha": float(best_alpha),
            "selector": sel_metric,
            "tail_months": V_tail,
            "risk_source": ("db" if data_source == "master" and Sigma is not None else "fb"),
            "turnover_one_way": one_way,
            "te_ann": te,
        })
        state["alpha_prev"] = float(best_alpha)
        state["last_weights"] = w.copy()

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

    # ---- save diagnostics per month ----
    try:
        if diagnostics_rows:
            diag_df = pd.DataFrame(diagnostics_rows)
            os.makedirs("output", exist_ok=True)
            diag_df.to_csv("output/diagnostics.csv", index=False)
            print(f"[INFO] Saved diagnostics to output/diagnostics.csv ({len(diag_df)} rows).")
    except Exception as e:
        print(f"[WARN] Could not save diagnostics.csv: {e}")

    return Results(pnl=pnl, pnl_net=pnl_net, turnover=turnover, te_ann=te_ann, weights=weights)
