from __future__ import annotations

import math
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from src.data import master_api
from src.data.loaders import (
    load_benchmark_weights,
    load_sector_returns_from_csv,
    load_synthetic_sector_returns,
)
from src.features.classic import make_classic_features
from src.portfolio.optimizer import solve_overlay
from src.risk.cov import estimate_cov
from src.utils.common import set_seed


def _exp_weight(age_months: int, half_life: Optional[float]) -> float:
    if not half_life or half_life <= 0:
        return 1.0
    return 0.5 ** (age_months / half_life)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    mean = w.mean()
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(w)
    return w / mean


def _impute_mean_inplace(X: np.ndarray) -> None:
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_idx = np.where(np.isnan(X))
    if nan_idx[0].size:
        X[nan_idx] = col_means[nan_idx[1]]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_weights = np.cumsum(w_sorted)
    cutoff = 0.5 * w_sorted.sum()
    idx = np.searchsorted(cum_weights, cutoff)
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


def _weighted_ic_ir(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.astype(float)
    if np.allclose(w.sum(), 0):
        w = np.ones_like(w)
    w = w / w.sum()
    mean = np.sum(w * values)
    var = np.sum(w * (values - mean) ** 2)
    std = math.sqrt(max(var, 1e-12))
    return float(mean / std)


def _spearman_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    pred_series = pd.Series(pred)
    actual_series = pd.Series(actual)
    if pred_series.nunique() <= 1 or actual_series.nunique() <= 1:
        return float("nan")
    return float(pred_series.rank().corr(actual_series.rank(), method="spearman"))


def _apply_prediction_transform(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rank":
        series = pd.Series(values)
        ranks = series.rank(method="average").to_numpy()
        ranks -= 0.5 * (len(values) + 1)
        return ranks
    # default: z-score
    mean = values.mean()
    std = values.std()
    std = std if std > 1e-12 else 1.0
    z = (values - mean) / std
    return np.clip(z, -3.0, 3.0)


def _prepare_design(
    month_indices: Sequence[int],
    ref_date: pd.Timestamp,
    monthly_features: Sequence[np.ndarray],
    monthly_targets: Sequence[np.ndarray],
    month_dates: Sequence[pd.Timestamp],
    half_life: Optional[float],
    label_winsor_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not month_indices:
        raise ValueError("No month indices provided for training design.")

    blocks: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    weights: List[float] = []

    for idx in month_indices:
        X_block = monthly_features[idx]
        y_block = monthly_targets[idx]
        date = month_dates[idx]
        age = int((ref_date.to_period("M") - date.to_period("M")).n)
        w = _exp_weight(age, half_life)
        blocks.append(X_block)
        labels.append(y_block)
        weights.extend([w] * len(y_block))

    X = np.vstack(blocks).astype(float)
    y = np.hstack(labels).astype(float)
    sample_weights = _normalize_weights(np.asarray(weights, dtype=float))

    if label_winsor_pct and label_winsor_pct > 0:
        lo = np.nanpercentile(y, 100 * label_winsor_pct)
        hi = np.nanpercentile(y, 100 * (1 - label_winsor_pct))
        if np.isfinite(lo) and np.isfinite(hi):
            y = np.clip(y, lo, hi)

    _impute_mean_inplace(X)
    if np.isnan(X).any():
        raise RuntimeError("NaNs remained in X after imputation")

    return X, y, sample_weights


def _select_alpha(
    alpha_grid: Sequence[float],
    tail_indices: Sequence[int],
    monthly_features: Sequence[np.ndarray],
    monthly_targets: Sequence[np.ndarray],
    month_dates: Sequence[pd.Timestamp],
    half_life: Optional[float],
    selector_half_life: Optional[float],
    label_winsor_pct: float,
    purge_months: int,
    embargo_months: int,
    selector_metric: str,
    min_lookback: int,
) -> Tuple[Optional[float], float, float, Dict[float, Dict[str, float]]]:
    if not tail_indices:
        return None, float("nan"), float("nan"), {}

    metrics: Dict[float, Dict[str, float]] = {}
    tail_dates = [month_dates[idx] for idx in tail_indices]
    last_tail_date = tail_dates[-1]

    best_alpha: Optional[float] = None
    best_score = -np.inf
    best_ic_ir = -np.inf

    for alpha in alpha_grid:
        ic_values: List[float] = []
        ic_weights: List[float] = []

        for eval_idx in tail_indices:
            eval_date = month_dates[eval_idx]
            cutoff = eval_idx - purge_months
            if cutoff <= 0 or cutoff < min_lookback:
                continue
            train_indices = list(range(0, cutoff))
            if embargo_months:
                cutoff_future = min(len(month_dates), eval_idx + 1 + embargo_months)
                # embargo months are simply ignored because training uses history only
                _ = cutoff_future  # placeholder to highlight the parameter

            try:
                X_tr, y_tr, w_tr = _prepare_design(
                    train_indices,
                    eval_date,
                    monthly_features,
                    monthly_targets,
                    month_dates,
                    half_life,
                    label_winsor_pct,
                )
            except ValueError:
                continue

            model = Ridge(alpha=alpha)
            model.fit(X_tr, y_tr, sample_weight=w_tr)

            preds = model.predict(monthly_features[eval_idx])
            actual = monthly_targets[eval_idx]
            ic = _spearman_ic(preds, actual)
            if not np.isfinite(ic):
                continue

            age = int((last_tail_date.to_period("M") - eval_date.to_period("M")).n)
            weight = _exp_weight(age, selector_half_life)
            ic_values.append(ic)
            ic_weights.append(weight)

        if not ic_values:
            metrics[alpha] = {"selector": float("nan"), "ic_ir": float("nan"), "n_tail": 0}
            continue

        ic_arr = np.asarray(ic_values, dtype=float)
        weight_arr = np.asarray(ic_weights, dtype=float)
        if np.allclose(weight_arr.sum(), 0):
            weight_arr = np.ones_like(weight_arr)

        if selector_metric == "spearman_ic_median":
            selector_score = _weighted_median(ic_arr, weight_arr)
        else:
            selector_score = float(np.average(ic_arr, weights=weight_arr))
        ic_ir = _weighted_ic_ir(ic_arr, weight_arr)

        metrics[alpha] = {
            "selector": selector_score,
            "ic_ir": ic_ir,
            "n_tail": float(len(ic_values)),
        }

        better = False
        if selector_score > best_score:
            better = True
        elif np.isclose(selector_score, best_score):
            if ic_ir > best_ic_ir:
                better = True
            elif np.isclose(ic_ir, best_ic_ir):
                better = best_alpha is None or alpha < best_alpha

        if better:
            best_alpha = alpha
            best_score = selector_score
            best_ic_ir = ic_ir

    if best_alpha is None:
        best_score = float("nan")
        best_ic_ir = float("nan")

    return best_alpha, best_score, best_ic_ir, metrics

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
        _verify_master_artifacts(master_root)
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
        df_ret = load_sector_returns_from_csv(data_cfg.get("csv_folder"), sectors, start, end)
        df_wb  = load_benchmark_weights(start, end)
        X      = make_classic_features(df_ret)
    else:
        # synthetic fallback
        df_ret = load_synthetic_sector_returns(start, end, cfg["seed"])
        df_wb  = load_benchmark_weights(start, end)
        X      = make_classic_features(df_ret)

    # 2) build feature panel (classic)
    if data_source != "master":
        X = make_classic_features(df_ret)  # MultiIndex (date, sector)

    # 3) training/evaluation windows
    eval_cfg = cfg.get("evaluation", {})
    warmup = eval_cfg.get("warmup_months", 24)
    purge_months = eval_cfg.get("purge_months", 1)
    embargo_months = eval_cfg.get("embargo_months", 1)
    dates = df_ret.index

    model_cfg = cfg.get("model", {})
    lookback = model_cfg.get("train_lookback_months")
    min_lookback = model_cfg.get("min_lookback_months", 36)
    half_life = model_cfg.get("half_life_months")
    alpha_grid = model_cfg.get("alpha_grid", [0.1, 1.0, 10.0])
    selector_tail_months = model_cfg.get("selector_tail_months", 24)
    selector_half_life = model_cfg.get("selector_half_life_months", 12)
    selector_metric = model_cfg.get("selector_metric", "spearman_ic_median")
    alpha_improvement_min = model_cfg.get("alpha_improvement_min", 0.0)
    alpha_ema_beta = model_cfg.get("alpha_ema_beta")
    prediction_transform = model_cfg.get("prediction_transform", "zscore")
    label_winsor_pct = model_cfg.get("label_winsor_pct", 0.0)

    runtime_cfg = cfg.get("runtime", {})
    turnover_eps = runtime_cfg.get("turnover_eps", 0.0)

    pnl, pnl_net = [], []
    turnover = []
    te_ann = []
    weights = {}
    preds = {}  # date -> pd.Series of predicted sector excess returns
    diagnostics_rows: List[Dict[str, object]] = []
    alpha_history: List[float] = []
    alpha_guard_blocks = 0
    alpha_switches = 0
    first_prediction_date: Optional[pd.Timestamp] = None
    skipped_due_history = 0
    db_cov_used = 0
    fb_cov_used = 0
    w_prev = df_wb.iloc[0].reindex(sectors).values  # start from benchmark

    cost_bps = cfg["costs"]["one_way_bps"]
    te_target = cfg["portfolio"]["te_target_annual"]
    to_cap    = cfg["portfolio"]["turnover_cap"]
    lam       = cfg["portfolio"]["smoothing_lambda"]

    alpha_state: Optional[float] = None
    alpha_selector_state: Optional[float] = None
    alpha_selector_icir_state: Optional[float] = None

    for t in range(warmup, len(dates) - 1):
        asof = dates[t]
        nxt = dates[t + 1]

        Xi = X.loc[X.index.get_level_values(0) <= asof]
        unique_dates = Xi.index.get_level_values(0).unique()
        candidate_dates = unique_dates[:-1]
        if lookback:
            candidate_dates = candidate_dates[-lookback:]

        monthly_features: List[np.ndarray] = []
        monthly_targets: List[np.ndarray] = []
        month_dates: List[pd.Timestamp] = []

        for d in candidate_dates:
            row_df = Xi.loc[d].reindex(sectors)
            if row_df.isna().values.any():
                continue
            next_date = d + pd.offsets.MonthEnd(1)
            if next_date not in df_ret.index or next_date not in df_wb.index:
                continue
            r_next = df_ret.loc[next_date].reindex(sectors)
            r_bmk = (df_wb.loc[next_date].reindex(sectors) * r_next).sum()
            y = (r_next - r_bmk).reindex(sectors)
            monthly_features.append(row_df.values.astype(float))
            monthly_targets.append(y.values.astype(float))
            month_dates.append(pd.Timestamp(d))

        if len(month_dates) < min_lookback:
            skipped_due_history += 1
            continue

        tail_len = selector_tail_months or len(month_dates)
        tail_len = min(tail_len, len(month_dates))
        tail_indices = (
            list(range(len(month_dates) - tail_len, len(month_dates)))
            if tail_len > 0
            else []
        )

        alpha_candidate, selector_score_candidate, selector_ic_ir_candidate, _alpha_metrics = _select_alpha(
            alpha_grid,
            tail_indices,
            monthly_features,
            monthly_targets,
            month_dates,
            half_life,
            selector_half_life,
            label_winsor_pct,
            purge_months,
            embargo_months,
            selector_metric,
            min_lookback,
        )

        selector_score_used = alpha_selector_state if alpha_selector_state is not None else float("nan")
        selector_ic_ir_used = (
            alpha_selector_icir_state if alpha_selector_icir_state is not None else float("nan")
        )
        switched = False

        alpha_improvement = float("nan")
        guard_blocked = False

        if alpha_state is None:
            if alpha_candidate is not None and np.isfinite(selector_score_candidate):
                alpha_state = float(alpha_candidate)
                selector_score_used = selector_score_candidate
                selector_ic_ir_used = selector_ic_ir_candidate
                switched = True
            else:
                default_alpha = float(alpha_grid[0]) if alpha_grid else 1.0
                alpha_state = default_alpha
        else:
            if alpha_candidate is not None and np.isfinite(selector_score_candidate):
                prev_score = alpha_selector_state if alpha_selector_state is not None else -np.inf
                alpha_improvement = selector_score_candidate - prev_score
                if alpha_improvement >= alpha_improvement_min:
                    new_alpha = float(alpha_candidate)
                    if alpha_ema_beta is not None:
                        beta = max(0.0, min(1.0, float(alpha_ema_beta)))
                        log_prev = math.log(max(alpha_state, 1e-12))
                        log_new = math.log(max(new_alpha, 1e-12))
                        alpha_state = math.exp(beta * log_prev + (1 - beta) * log_new)
                    else:
                        alpha_state = new_alpha
                    selector_score_used = selector_score_candidate
                    selector_ic_ir_used = selector_ic_ir_candidate
                    switched = True
                else:
                    guard_blocked = np.isfinite(alpha_improvement)

        if switched:
            alpha_selector_state = selector_score_candidate if np.isfinite(selector_score_candidate) else alpha_selector_state
            alpha_selector_icir_state = (
                selector_ic_ir_candidate if np.isfinite(selector_ic_ir_candidate) else alpha_selector_icir_state
            )
            alpha_switches += 1
        elif guard_blocked:
            alpha_guard_blocks += 1

        alpha_used = alpha_state
        if np.isfinite(alpha_used):
            alpha_history.append(float(alpha_used))

        train_indices = list(range(len(month_dates)))
        X_train, y_train, sample_weights = _prepare_design(
            train_indices,
            pd.Timestamp(asof),
            monthly_features,
            monthly_targets,
            month_dates,
            half_life,
            label_winsor_pct,
        )

        model = Ridge(alpha=alpha_used)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        row_t = X.loc[asof].reindex(sectors).fillna(0.0).values.astype(float)
        if np.isnan(row_t).any():
            raise RuntimeError(f"NaNs in as-of features at {asof.date()} after fillna(0.0)")
        mu_hat = model.predict(row_t)
        mu_hat = _apply_prediction_transform(mu_hat, prediction_transform)

        preds[asof] = pd.Series(mu_hat, index=sectors)

        if first_prediction_date is None:
            first_prediction_date = pd.Timestamp(nxt)

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

        crisis = df_ret.loc[asof].std() > 0.10

        w_bench = df_wb.loc[asof].reindex(sectors).values
        w = solve_overlay(
            mu_hat,
            Sigma,
            w_bench,
            w_prev,
            te_target,
            to_cap,
            lam,
            crisis,
            turnover_eps,
        )

        delta_cost = w - w_prev
        if turnover_eps > 0:
            delta_cost = delta_cost.copy()
            delta_cost[np.abs(delta_cost) < turnover_eps] = 0.0
        one_way = 0.5 * np.sum(np.abs(delta_cost)) * 2
        cost = (cost_bps / 10000.0) * one_way

        r_next = df_ret.loc[nxt].reindex(sectors).values
        r_bmk = df_wb.loc[nxt].reindex(sectors).values @ r_next
        r_port = w @ r_next
        active = r_port - r_bmk
        active_net = active - cost

        te = ((w - w_bench) @ Sigma @ (w - w_bench)) ** 0.5 * (12 ** 0.5)

        pnl.append(pd.Series(active, index=[nxt]))
        pnl_net.append(pd.Series(active_net, index=[nxt]))
        turnover.append(pd.Series(one_way, index=[nxt]))
        te_ann.append(pd.Series(te, index=[nxt]))
        weights[asof] = w

        realized_active = r_next - r_bmk
        pred_series = preds[asof]
        realized_ic = _spearman_ic(pred_series.values, realized_active)
        top3 = pred_series.nlargest(3).index
        bottom3 = pred_series.nsmallest(3).index
        top3_hit = float((pd.Series(realized_active, index=sectors).loc[top3] > 0).mean())
        top_minus_bottom = float(
            pd.Series(realized_active, index=sectors).loc[top3].mean()
            - pd.Series(realized_active, index=sectors).loc[bottom3].mean()
        )

        diagnostics_rows.append(
            {
                "date": pd.Timestamp(nxt),
                "asof": pd.Timestamp(asof),
                "alpha_used": float(alpha_used),
                "alpha_candidate": float(alpha_candidate) if alpha_candidate is not None else float("nan"),
                "alpha_switched": bool(switched),
                "alpha_improvement": alpha_improvement,
                "alpha_guard_blocked": bool(guard_blocked),
                "selector_score_used": selector_score_used,
                "selector_score_candidate": selector_score_candidate,
                "selector_ic_ir_used": selector_ic_ir_used,
                "selector_ic_ir_candidate": selector_ic_ir_candidate,
                "selector_tail_count": len(tail_indices),
                "train_months": len(month_dates),
                "train_observations": len(month_dates) * len(sectors),
                "half_life_months": half_life,
                "label_winsor_pct": label_winsor_pct,
                "prediction_transform": prediction_transform,
                "realized_ic": realized_ic,
                "top3_hit_rate": top3_hit,
                "top_minus_bottom": top_minus_bottom,
                "turnover_one_way": one_way,
                "cost_bps": cost_bps,
                "te_annualized": te,
            }
        )

        w_prev = w

    pnl = pd.concat(pnl) if pnl else pd.Series(dtype=float)
    pnl_net = pd.concat(pnl_net) if pnl_net else pd.Series(dtype=float)
    turnover = pd.concat(turnover) if turnover else pd.Series(dtype=float)
    te_ann = pd.concat(te_ann) if te_ann else pd.Series(dtype=float)

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

    total_eval_months = max(0, len(dates) - 1 - warmup)
    realized_months = len(pnl)
    if total_eval_months:
        coverage_pct = 100.0 * realized_months / total_eval_months
        print(
            "[INFO] Evaluation coverage: "
            f"{realized_months}/{total_eval_months} months ({coverage_pct:0.1f}%)."
        )
    if skipped_due_history:
        print(
            "[INFO] Skipped "
            f"{skipped_due_history} evaluation months because <{min_lookback} clean history was available."
        )
    if first_prediction_date is not None:
        print(f"[INFO] First prediction delivered for {first_prediction_date.date()}.")
    if alpha_history:
        alpha_arr = np.asarray(alpha_history)
        median_alpha = float(np.median(alpha_arr))
        print(
            f"[INFO] Alpha usage: median={median_alpha:0.4f}, "
            f"switches={alpha_switches}, guard blocks={alpha_guard_blocks}."
        )

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

    # ---- save diagnostics ----
    if diagnostics_rows:
        try:
            os.makedirs("output", exist_ok=True)
            diagnostics_df = pd.DataFrame(diagnostics_rows)
            diagnostics_df.sort_values("date", inplace=True)
            diagnostics_df.to_csv("output/diagnostics.csv", index=False)
            print(
                "[INFO] Saved diagnostics to output/diagnostics.csv "
                f"({diagnostics_df.shape[0]} rows)."
            )
        except Exception as e:
            print(f"[WARN] Could not save diagnostics.csv: {e}")

    return Results(pnl=pnl, pnl_net=pnl_net, turnover=turnover, te_ann=te_ann, weights=weights)


def _verify_master_artifacts(root: str) -> None:
    """Ensure the master data tree contains the minimum artifacts we rely on."""

    returns_path = os.path.join(root, "prices", "returns_m", "returns_m.parquet")
    assets_path = os.path.join(root, "assets", "assets.parquet")
    features_path = os.path.join(root, "features", "monthly", "features_monthly.parquet")
    bench_wide = os.path.join(root, "benchmark", "bench_weights_m.parquet")
    bench_tall = os.path.join(root, "portfolio", "bench_weights_m", "bench_weights_m.parquet")

    missing: List[Tuple[str, str, str]] = []

    if not os.path.exists(returns_path):
        missing.append(
            ("returns", returns_path, "python -m scripts.build_returns_m")
        )
    if not os.path.exists(assets_path):
        missing.append(
            ("assets", assets_path, "python -m scripts.build_assets")
        )
    if not os.path.exists(features_path):
        missing.append(
            (
                "features",
                features_path,
                "python -m scripts.build_features_monthly && python -m scripts.build_graph_features",
            )
        )
    if not (os.path.exists(bench_wide) or os.path.exists(bench_tall)):
        missing.append(
            (
                "benchmark",
                bench_wide,
                "python -m scripts.build_spx_benchmark (or python -m scripts.build_bench_weights_m)",
            )
        )

    if missing:
        lines = [
            "Master data root is missing required artifacts:",
        ]
        for kind, path, cmd in missing:
            lines.append(f"  - {kind}: {path}\n      build with: {cmd}")
        msg = "\n".join(lines)
        raise FileNotFoundError(textwrap.dedent(msg))

