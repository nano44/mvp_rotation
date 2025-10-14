# scripts/report.py
"""
Generate basic diagnostics for the latest backtest:
- Equity curve of cumulative active (net) return
- Rolling 24m Information Ratio (IR)
- Realized TE(ann) per month (DB covariance if available; fallback 36m)
- Turnover per month (reconstructed from weights.csv)
- Weights heatmap (dates × sectors)

Outputs:
  output/report_equity_curve.png
  output/report_rolling_ir_24m.png
  output/report_te_vs_target.png
  output/report_turnover.png
  output/report_weights_heatmap.png
  output/report_summary.csv
Optional inputs (read automatically if present):
  config/default.yaml
  output/active_net.csv
  output/weights.csv
  data/master/... (for covariances & benchmark weights)
"""

from __future__ import annotations

import os
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
def _next_month_date(rtn_index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    """Return the next available month-end in `rtn_index` after date d (same index as returns).
    If none exists, return None.
    """
    if d not in rtn_index:
        # find insertion position
        pos = rtn_index.searchsorted(pd.Timestamp(d))
    else:
        pos = rtn_index.get_loc(pd.Timestamp(d)) + 1
    if isinstance(pos, (list, np.ndarray)):
        pos = int(np.atleast_1d(pos)[0])
    if pos >= len(rtn_index):
        return None
    return rtn_index[pos]

def compute_weight_ic_and_spread(weights: pd.DataFrame, sectors: list[str]) -> tuple[pd.Series, pd.Series]:
    """
    For each month t, compute:
      • Weight IC: Spearman rank corr between active weights a_t = w_t - w_b,t
        and *next-month* sector returns r_{t+1} (cross-section).
      • Paper Top3–Bottom3 spread formed by ranking sectors by a_t and
        taking equal-weight top3 minus bottom3 on r_{t+1}.
    Returns two aligned Series indexed by t.
    """
    w = weights.sort_index()
    R, Wb = _load_returns_and_bench(sectors)

    out_ic: dict[pd.Timestamp, float] = {}
    out_spread: dict[pd.Timestamp, float] = {}

    for d, row in w.iterrows():
        d = pd.Timestamp(d)
        # next month available in returns
        dn = _next_month_date(R.index, d)
        if dn is None:
            continue

        # active weights at t
        wb = (Wb.loc[d] if d in Wb.index else pd.Series(1.0/len(sectors), index=sectors))
        a = (row.reindex(sectors).astype(float) - wb.reindex(sectors).astype(float)).values

        # realized next-month sector returns
        r_next = R.loc[dn].reindex(sectors).astype(float).values
        if np.isnan(a).any() or np.isnan(r_next).any():
            continue

        # Spearman IC across sectors
        ic = spearmanr(a, r_next, nan_policy="omit")[0]
        out_ic[d] = float(ic) if np.isfinite(ic) else np.nan

        # Top3–Bottom3 spread (equal-weight, paper)
        order = np.argsort(a)
        bot_idx = order[:3]
        top_idx = order[-3:]
        spread = float(np.nanmean(r_next[top_idx]) - np.nanmean(r_next[bot_idx]))
        out_spread[d] = spread

    s_ic = pd.Series(out_ic).sort_index(); s_ic.name = "weight_ic"
    s_sp = pd.Series(out_spread).sort_index(); s_sp.name = "weight_spread_3_3"
    return s_ic, s_sp


# --- New helpers for weight-vs-score correlation ---
def compute_weight_vs_score_corr(
    weights: pd.DataFrame, preds: pd.DataFrame, sectors: list[str]
) -> pd.Series:
    """
    For each date in the intersection of weights and preds, compute Spearman correlation
    across sectors between that date's portfolio weights and prediction scores.
    Returns a Series indexed by date, named 'weight_vs_score_ic'.
    """
    idx = weights.index.intersection(preds.index)
    out = {}
    for d in idx:
        w = weights.loc[d, sectors].astype(float)
        p = preds.loc[d, sectors].astype(float)
        ic = spearmanr(w.values, p.values, nan_policy="omit")[0]
        out[d] = float(ic) if np.isfinite(ic) else np.nan
    s = pd.Series(out).sort_index()
    s.name = "weight_vs_score_ic"
    return s


def plot_weight_vs_score_corr(series: pd.Series, outfile: str):
    plt.figure(figsize=(10, 3.5))
    plt.plot(series.index, series.values)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Spearman corr (weights ↔ scores) by month")
    plt.xlabel("Date")
    plt.ylabel("Spearman corr")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()

def plot_weight_ic(ic: pd.Series, outfile: str):
    plt.figure(figsize=(10, 3.5))
    plt.plot(ic.index, ic.values)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Spearman IC by month (weights ↔ next-mo returns)")
    plt.xlabel("Date"); plt.ylabel("IC")
    plt.tight_layout(); plt.savefig(outfile, dpi=120); plt.close()

def plot_spread_cum(spread: pd.Series, outfile: str):
    cum = spread.cumsum()
    plt.figure(figsize=(10, 3.5))
    plt.plot(cum.index, cum.values)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Cumulative Top3–Bottom3 spread (by weights, paper)")
    plt.xlabel("Date"); plt.ylabel("Cumulative spread")
    plt.tight_layout(); plt.savefig(outfile, dpi=120); plt.close()

# Internal helpers from your repo
from src.data import master_api
from src.risk.cov import estimate_cov


def _ensure_output_dir():
    os.makedirs("output", exist_ok=True)


def _read_config(cfg_path: str = "config/default.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _load_active_net() -> pd.Series:
    # output/active_net.csv has a single column of net active returns
    s = pd.read_csv("output/active_net.csv", index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "active_net"
    return s.sort_index()


def _load_weights() -> pd.DataFrame:
    df = pd.read_csv("output/weights.csv", index_col=0, parse_dates=True)
    return df.sort_index()


def _load_returns_and_bench(sectors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide returns & benchmark weights from master DB (dates × sectors)."""
    R = master_api.load_returns_wide("data/master", sectors)
    Wb = master_api.load_benchmark_weights_wide("data/master", sectors)
    # Ensure the exact sector order/columns
    R = R.reindex(columns=sectors)
    Wb = Wb.reindex(columns=sectors)
    return R, Wb


def info_ratio(series_monthly: pd.Series) -> float:
    mu = series_monthly.mean()
    sd = series_monthly.std()
    return float(mu / sd * math.sqrt(12)) if sd > 0 else float("nan")


def rolling_ir(series_monthly: pd.Series, window: int = 24) -> pd.Series:
    def _ir(x):
        x = pd.Series(x)
        sd = x.std()
        return (x.mean() / sd) * math.sqrt(12) if sd > 0 else np.nan

    return series_monthly.rolling(window).apply(_ir, raw=False)


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Reconstruct one-way turnover as sum(abs(delta w)) per month."""
    w = weights.sort_index()
    dw = w.diff()  # month-over-month change
    tr = dw.abs().sum(axis=1)  # sum over sectors
    tr.name = "turnover_one_way"
    # First row has NaN (no previous weights); drop it
    return tr.iloc[1:]


def compute_te_series(
    weights: pd.DataFrame,
    sectors: list[str],
    target_te: float | None,
    lookback_fb: int = 36,
) -> pd.Series:
    """
    Realized TE(ann) per month using covariance at as-of:
      - Try DB covariance (data/master/risk/cov_m.parquet)
      - Fallback to 36m rolling covariance from returns
    """
    weights = weights.sort_index()
    dates = weights.index
    # Load returns & benchmark weights once
    R, Wb = _load_returns_and_bench(sectors)

    te_vals = {}
    for d in dates:
        w = weights.loc[d].reindex(sectors).values
        wb = Wb.loc[d].reindex(sectors).values if d in Wb.index else np.full(len(sectors), 1.0 / len(sectors))
        a = w - wb

        # Try DB covariance first
        Sigma = master_api.load_cov("data/master", d, sectors)
        if Sigma is None:
            # Fallback: 36m rolling cov up to as-of
            Sigma = estimate_cov(R, pd.Timestamp(d), lookback=lookback_fb, shrink=0.1)

        te_m = float(np.sqrt(max(0.0, a @ Sigma @ a)))  # monthly TE
        te_ann = te_m * math.sqrt(12)
        te_vals[d] = te_ann

    te = pd.Series(te_vals).sort_index()
    te.name = "te_ann"
    return te


def plot_equity_curve(active_net: pd.Series, outfile: str):
    eq = (1.0 + active_net).cumprod()
    plt.figure(figsize=(10, 4))
    plt.plot(eq.index, eq.values)
    plt.title("Cumulative Active (Net) — Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Multiple")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()


def plot_rolling_ir(active_net: pd.Series, outfile: str, window: int = 24):
    roll = rolling_ir(active_net, window=window)
    plt.figure(figsize=(10, 4))
    plt.plot(roll.index, roll.values)
    plt.title(f"Rolling {window}m Information Ratio (Net)")
    plt.xlabel("Date")
    plt.ylabel("IR")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()


def plot_te_vs_target(te: pd.Series, target_te: float | None, outfile: str):
    plt.figure(figsize=(10, 4))
    plt.plot(te.index, te.values, label="Realized TE (ann)")
    if target_te is not None:
        plt.axhline(y=target_te, linestyle="--", label=f"Target TE (ann) = {target_te:.3f}")
    plt.title("Tracking Error — Realized vs Target")
    plt.xlabel("Date")
    plt.ylabel("TE (annualized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()


def plot_turnover(turnover: pd.Series, outfile: str):
    plt.figure(figsize=(10, 3.5))
    plt.bar(turnover.index, turnover.values, width=20)  # width ~20 days
    plt.title("One-way Turnover per Month (sum |Δw|)")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()


def plot_weights_heatmap(weights: pd.DataFrame, outfile: str):
    # Normalize order
    w = weights.sort_index()
    plt.figure(figsize=(10, 6))
    plt.imshow(w.values, aspect="auto", interpolation="nearest")
    plt.title("Weights Heatmap (rows = dates, cols = sectors)")
    plt.xlabel("Sectors")
    plt.ylabel("Dates")
    plt.xticks(range(len(w.columns)), w.columns, rotation=45, ha="right")
    plt.yticks([])  # too many dates; omit labels
    plt.colorbar(label="Weight")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()


# ---- Prediction‑level diagnostics: IC, Top‑3 hit‑rate, 3–3 spread ----

def _load_preds_wide(path: str = "output/preds.csv") -> pd.DataFrame:
    """Load preds written by the engine (wide: dates × sectors).
    Supports either index-as-date or an explicit 'date' column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # If the first column was actually not the date, fall back
        if not np.issubdtype(df.index.dtype, np.datetime64):
            raise Exception("index not datetime")
        df.index.name = "date"
        return df.sort_index()
    except Exception:
        df = pd.read_csv(path)
        dcol = None
        for c in ("date", "month_end", "asof"):
            if c in df.columns:
                dcol = c
                break
        if dcol is None:
            raise ValueError("preds.csv has no date column")
        df[dcol] = pd.to_datetime(df[dcol])
        df = df.rename(columns={dcol: "date"}).set_index("date").sort_index()
        return df


def _decayed_mean_std(x: pd.Series | np.ndarray, hl: int = 12) -> tuple[float, float]:
    v = pd.Series(x).astype(float).dropna().values
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    # newest gets largest weight
    lags = np.arange(n) - (n - 1)
    w = np.exp(-np.log(2.0) * np.abs(lags) / max(1, hl))
    w = w / w.sum()
    mu = float((w * v).sum())
    sd = float(np.sqrt(max(1e-12, (w * (v - mu) ** 2).sum())))
    return mu, sd


def _try_load_preds() -> pd.DataFrame | None:
    """
    Try to load preds.csv using _load_preds_wide, return DataFrame or None if not found or error.
    """
    try:
        return _load_preds_wide("output/preds.csv")
    except Exception:
        return None


def compute_pred_metrics(sectors: list[str]) -> None:
    """Compute monthly Spearman IC, Top‑3 hit‑rate, and 3–3 spread using
    engine predictions (t) vs realized next‑month excess sector returns (t+1).
    Artifacts written under output/: pred_metrics.csv, report_ic_line.png, report_spread_cum.png
    and fields appended to report_summary.csv.
    """
    # Load inputs
    preds_w = _load_preds_wide("output/preds.csv")  # dates × sectors
    # Use master DB to compute realized forward excess returns
    R = master_api.load_returns_wide("data/master", sectors)        # dates × sectors
    Wb = master_api.load_benchmark_weights_wide("data/master", sectors)  # dates × sectors

    # Align columns just in case
    sectors = [s for s in sectors if s in preds_w.columns and s in R.columns and s in Wb.columns]
    preds_w = preds_w.reindex(columns=sectors)
    R = R.reindex(columns=sectors)
    Wb = Wb.reindex(columns=sectors)

    rows = []
    for d in preds_w.index:
        nxt = d + pd.offsets.MonthEnd(1)
        if nxt not in R.index or nxt not in Wb.index:
            continue
        p = preds_w.loc[d, sectors]
        y = R.loc[nxt, sectors]
        y_bmk = float((Wb.loc[nxt, sectors] * y).sum())
        y_ex = y - y_bmk  # vector of sector excess returns at t+1

        # Spearman IC between predicted and realized excess x‑sec
        ic = p.rank().corr(y_ex.rank(), method="spearman")

        # Top‑3 hit‑rate (set overlap of predicted Top‑3 vs realized Top‑3)
        top3_pred = set(p.sort_values(ascending=False).index[:3])
        top3_real = set(y_ex.sort_values(ascending=False).index[:3])
        hit3 = len(top3_pred & top3_real) / 3.0

        # Paper spread using predicted ranking: Top3(y_ex) − Bottom3(y_ex)
        srt = p.sort_values(ascending=False).index
        spread = float(y_ex.loc[srt[:3]].mean() - y_ex.loc[srt[-3:]].mean())

        rows.append({"date": d, "ic": ic, "hit3": hit3, "spread_3_3": spread})

    if not rows:
        print("[REPORT] pred-metrics: no overlapping dates — skipped")
        return

    M = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    M.to_csv("output/pred_metrics.csv", index=False)

    # Decayed aggregates (HL=12)
    ic_mu, ic_sd = _decayed_mean_std(M["ic"], hl=12)
    ic_ir = (ic_mu / ic_sd) if (ic_sd and ic_sd > 1e-12) else float("nan")
    hit_mu, _ = _decayed_mean_std(M["hit3"], hl=12)
    spr_mu, _ = _decayed_mean_std(M["spread_3_3"], hl=12)

    # Plots
    plt.figure(figsize=(10, 3))
    plt.plot(M["date"], M["ic"], lw=1.2)
    plt.axhline(0, ls="--", lw=0.8, color="gray")
    plt.title("Spearman IC by month (pred vs realized excess)")
    plt.tight_layout()
    plt.savefig("output/report_ic_line.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(M["date"], M["spread_3_3"].cumsum(), lw=1.2)
    plt.axhline(0, ls="--", lw=0.8, color="gray")
    plt.title("Cumulative Top3–Bottom3 spread (paper)")
    plt.tight_layout()
    plt.savefig("output/report_spread_cum.png", dpi=120)
    plt.close()

    # Append aggregates into report_summary.csv
    summ_path = "output/report_summary.csv"
    summ = pd.DataFrame([
        {
            "pred_ic_mean_hl12": round(float(ic_mu), 4) if np.isfinite(ic_mu) else np.nan,
            "pred_ic_ir_hl12": round(float(ic_ir), 2) if np.isfinite(ic_ir) else np.nan,
            "pred_hit3_mean": round(float(hit_mu), 3) if np.isfinite(hit_mu) else np.nan,
            "pred_spread_3_3_mean": round(float(spr_mu), 4) if np.isfinite(spr_mu) else np.nan,
        }
    ])
    try:
        base = pd.read_csv(summ_path)
        out = pd.concat([base, summ], axis=1)
    except FileNotFoundError:
        out = summ
    out.to_csv(summ_path, index=False)

    print("[REPORT] Pred metrics saved → output/pred_metrics.csv, output/report_ic_line.png, output/report_spread_cum.png")
    print(
        f"[REPORT] Decayed IC mean (HL=12): {ic_mu:.4f} | IC-IR: {ic_ir:.2f} | Hit3: {hit_mu:.3f} | Spread mean: {spr_mu:.4f}"
    )


def main():
    _ensure_output_dir()
    cfg = _read_config("config/default.yaml")
    sectors = cfg.get("universe", {}).get("sectors")
    if not sectors:
        # Fallback: read from weights.csv columns
        wtmp = _load_weights()
        sectors = list(wtmp.columns)

    # Load artifacts
    active_net = _load_active_net()
    weights = _load_weights()

    # Compute derived series
    ir_full = info_ratio(active_net)
    te_target = cfg.get("portfolio", {}).get("te_target_annual", None)
    te_series = compute_te_series(weights, sectors, te_target)
    turnover = compute_turnover(weights)

    # Weight-based diagnostics (active weights vs next-month returns)
    w_ic, w_spread = compute_weight_ic_and_spread(weights, sectors)

    # Decayed stats for weight diagnostics (HL=12)
    w_ic_mu, w_ic_sd = _decayed_mean_std(w_ic, hl=12)
    w_ic_ir = (w_ic_mu / w_ic_sd) if (w_ic_sd and w_ic_sd > 1e-12) else float("nan")
    w_sp_mu, _ = _decayed_mean_std(w_spread, hl=12)

    # Persist diagnostics
    pd.DataFrame({"weight_ic": w_ic, "weight_spread_3_3": w_spread}).to_csv(
        "output/weight_metrics.csv"
    )

    # Plots for the new diagnostics
    plot_weight_ic(w_ic, "output/report_weight_ic_line.png")
    plot_spread_cum(w_spread, "output/report_spread_cum_weights.png")

    # Compute weights-vs-score correlation if preds available
    preds_w = _try_load_preds()
    if preds_w is not None:
        # align columns
        common = [s for s in sectors if s in preds_w.columns]
        preds_w = preds_w.reindex(columns=common)
        w_vs = compute_weight_vs_score_corr(weights.reindex(columns=common), preds_w, common)
        w_vs.to_csv("output/weight_vs_score.csv")
        plot_weight_vs_score_corr(w_vs, "output/report_weight_vs_score_corr.png")
        w_vs_mu, _ = _decayed_mean_std(w_vs, hl=12)
    else:
        w_vs = None
        w_vs_mu = float("nan")

    # Plots
    plot_equity_curve(active_net, "output/report_equity_curve.png")
    plot_rolling_ir(active_net, "output/report_rolling_ir_24m.png", window=24)
    plot_te_vs_target(te_series, te_target, "output/report_te_vs_target.png")
    plot_turnover(turnover, "output/report_turnover.png")
    plot_weights_heatmap(weights, "output/report_weights_heatmap.png")

    # Prediction‑level diagnostics (IC / hit‑rate / spread)
    try:
        compute_pred_metrics(sectors)
    except Exception as e:
        print("[WARN] pred-metrics failed:", e)

    # Summary CSV, merged with previous if exists, and add new decayed fields
    summary_dict = {
        "IR_net_full": ir_full,
        "TE_ann_mean": te_series.mean(),
        "TE_ann_median": te_series.median(),
        "Turnover_mean": turnover.mean(),
        "Obs_active": len(active_net),
        "Weight_IC_mean": w_ic.mean(),
        "Weight_spread_3_3_mean": w_spread.mean(),
        "Weight_IC_mean_HL12": round(float(w_ic_mu), 4) if np.isfinite(w_ic_mu) else np.nan,
        "Weight_IC_IR_HL12": round(float(w_ic_ir), 2) if np.isfinite(w_ic_ir) else np.nan,
        "Weight_spread_3_3_mean_HL12": round(float(w_sp_mu), 4) if np.isfinite(w_sp_mu) else np.nan,
        "Weight_vs_score_ic_mean_HL12": round(float(w_vs_mu), 4) if np.isfinite(w_vs_mu) else np.nan,
    }
    try:
        base = pd.read_csv("output/report_summary.csv")
        out = pd.concat([base, pd.DataFrame([summary_dict])], axis=1)
    except FileNotFoundError:
        out = pd.DataFrame([summary_dict])
    out.to_csv("output/report_summary.csv", index=False)

    print("[REPORT] Saved:")
    print("  - output/report_equity_curve.png")
    print("  - output/report_rolling_ir_24m.png")
    print("  - output/report_te_vs_target.png")
    print("  - output/report_turnover.png")
    print("  - output/report_weights_heatmap.png")
    print("  - output/report_weight_ic_line.png")
    print("  - output/report_spread_cum_weights.png")
    print("  - output/weight_metrics.csv")
    if preds_w is not None:
        print("  - output/report_weight_vs_score_corr.png")
        print("  - output/weight_vs_score.csv")
    print("  - output/report_summary.csv")
    print(
        f"IR net (full): {ir_full:.2f} | TE mean: {te_series.mean():.2%} | Turnover mean: {turnover.mean():.2%} | "
        f"Weight IC HL12 mean: {w_ic_mu:.4f} | IC‑IR: {w_ic_ir:.2f}"
    )


if __name__ == "__main__":
    main()