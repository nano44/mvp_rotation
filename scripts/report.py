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

# === Stronger stats helpers: Block-IR and HAC t-stat ===

def _block_ir(series: "pd.Series", block: int = 12):
    """Compute non-overlapping block IRs over monthly returns.
    Returns dict with mean/median and count of blocks.
    IR per block = mean(block)/std(block) * sqrt(12).
    """
    x = pd.Series(series).dropna().astype(float).values
    n = len(x)
    if n < block:
        return {"ir_mean": np.nan, "ir_median": np.nan, "blocks": 0}
    m = n // block
    X = x[: m * block].reshape(m, block)
    sd = X.std(axis=1, ddof=1)
    ir_blocks = np.where(sd > 0, (X.mean(axis=1) / sd) * np.sqrt(12.0), np.nan)
    return {
        "ir_mean": float(np.nanmean(ir_blocks)) if np.isfinite(ir_blocks).any() else np.nan,
        "ir_median": float(np.nanmedian(ir_blocks)) if np.isfinite(ir_blocks).any() else np.nan,
        "blocks": int(m),
    }


def _hac_tstat_mean(series: "pd.Series", max_lag: int = 6):
    """Newey–West (HAC) t-stat for the mean of a monthly-return series.
    Bartlett weights; variance of sqrt(n)*mean = S; var(mean) = S/n; t = sqrt(n)*mean / sqrt(S).
    """
    x = pd.Series(series).dropna().astype(float).values
    n = len(x)
    if n == 0:
        return np.nan
    mu = x.mean()
    xc = x - mu
    gamma0 = (xc @ xc) / n  # γ_0
    S = gamma0
    L = min(max_lag, n - 1) if n > 1 else 0
    for l in range(1, L + 1):
        cov = (xc[l:] @ xc[:-l]) / n
        w = 1.0 - l / (L + 1.0)  # Bartlett weight
        S += 2.0 * w * cov
    if S <= 0:
        return np.nan
    t = (np.sqrt(n) * mu) / np.sqrt(S)
    return float(t)

# === Bootstrap helpers: Moving-Block Bootstrap (MBB) for CI ===

def _annualized_ir(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(12.0)) if sd > 0 else np.nan


def _mbb_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices of length n with block_len.
    We sample starting points uniformly and wrap around modulo n.
    """
    starts = rng.integers(0, n, size=max(1, int(np.ceil(n / block_len))))
    idx = []
    for s in starts:
        block = (np.arange(block_len) + s) % n
        idx.append(block)
    idx = np.concatenate(idx)[:n]
    return idx


def _mbb_ci(x: "pd.Series", stat_fn, block_len: int = 6, B: int = 2000, alpha: float = 0.10, seed: int | None = None):
    """Moving-block bootstrap CI for a statistic of a monthly-return series.
    Returns (lo, hi) quantiles for (1-alpha) CI.
    """
    s = pd.Series(x).dropna().astype(float).values
    n = len(s)
    if n < 3:
        return (np.nan, np.nan)
    b = int(max(2, min(block_len, n)))
    B = int(max(200, B))  # safety
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for bidx in range(B):
        idx = _mbb_indices(n, b, rng)
        stats[bidx] = stat_fn(s[idx])
    lo = float(np.nanpercentile(stats, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi

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


# === New helpers: drawdown & alpha diagnostics ===
def compute_drawdown(active_net: pd.Series) -> pd.Series:
    eq = (1.0 + active_net).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0  # negative or 0
    dd.name = "drawdown"
    return dd


def plot_drawdown(active_net: pd.Series, outfile: str):
    dd = compute_drawdown(active_net)
    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd.values)
    plt.title("Active Drawdown (Net)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()


def plot_alpha_diagnostics(outfile_hist: str, outfile_line: str):
    path = os.path.join("output", "chosen_alpha.csv")
    if not os.path.exists(path):
        print("[REPORT] chosen_alpha.csv not found — skipping alpha plots")
        return
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = "ridge_alpha"

    # histogram
    plt.figure(figsize=(6, 4))
    try:
        # bins chosen to separate our {0.1, 1.0, 10.0}
        plt.hist(s.values, bins=[0.05, 0.15, 0.5, 1.5, 5, 15], align="left", rwidth=0.8)
    except Exception:
        plt.hist(s.values, bins=10)
    plt.title("Chosen Ridge α — Histogram")
    plt.xlabel("alpha bins")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outfile_hist, dpi=120)
    plt.close()

    # line plot
    plt.figure(figsize=(10, 4))
    plt.plot(s.index, s.values)
    plt.title("Chosen Ridge α by Month")
    plt.xlabel("Date")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(outfile_line, dpi=120)
    plt.close()


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

    # Plots
    plot_equity_curve(active_net, "output/report_equity_curve.png")
    plot_rolling_ir(active_net, "output/report_rolling_ir_24m.png", window=24)
    plot_te_vs_target(te_series, te_target, "output/report_te_vs_target.png")
    plot_turnover(turnover, "output/report_turnover.png")
    plot_weights_heatmap(weights, "output/report_weights_heatmap.png")

    # New: drawdown & alpha diagnostics
    plot_drawdown(active_net, "output/report_active_drawdown.png")
    plot_alpha_diagnostics("output/report_alpha_hist.png", "output/report_alpha_line.png")

    # Compute max drawdown for summary
    dd = compute_drawdown(active_net)
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    # Summary CSV (extended)
    alpha_count = None
    if os.path.exists("output/chosen_alpha.csv"):
        try:
            alpha_count = len(pd.read_csv("output/chosen_alpha.csv"))
        except Exception:
            alpha_count = None

    summary = pd.DataFrame([
        {
            "IR_net_full": ir_full,
            "TE_ann_mean": te_series.mean(),
            "TE_ann_median": te_series.median(),
            "Turnover_mean": turnover.mean(),
            "MaxDD_active": max_dd,
            "Alpha_records": alpha_count,
            "Obs_active": len(active_net),
        }
    ])
    summary.to_csv("output/report_summary.csv", index=False)

    print("[REPORT] Saved:")
    print("  - output/report_equity_curve.png")
    print("  - output/report_rolling_ir_24m.png")
    print("  - output/report_te_vs_target.png")
    print("  - output/report_turnover.png")
    print("  - output/report_weights_heatmap.png")
    print("  - output/report_summary.csv")
    print("  - output/report_active_drawdown.png")
    print("  - output/report_alpha_hist.png")
    print("  - output/report_alpha_line.png")
    print(f"IR net (full): {ir_full:.2f} | TE mean: {te_series.mean():.2%} | Turnover mean: {turnover.mean():.2%}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    # === Stronger stats: Block-IR(12m) and HAC t-stat ===
    try:
        # robust read of output/active_net.csv (date in col0, value in col1)
        _p = "output/active_net.csv"
        try:
            df_active = pd.read_csv(_p)
            if df_active.shape[1] >= 2:
                dates = pd.to_datetime(df_active.iloc[:, 0])
                vals = pd.to_numeric(df_active.iloc[:, 1], errors="coerce")
                active_net = pd.Series(vals.values, index=dates).sort_index()
            else:
                raise ValueError("active_net.csv has too few columns")
        except Exception:
            # fallback to header-less
            df_active = pd.read_csv(_p, header=None)
            dates = pd.to_datetime(df_active.iloc[:, 0])
            vals = pd.to_numeric(df_active.iloc[:, 1], errors="coerce")
            active_net = pd.Series(vals.values, index=dates).sort_index()

        blk12 = _block_ir(active_net, block=12)
        hac6  = _hac_tstat_mean(active_net, max_lag=6)
        hac12 = _hac_tstat_mean(active_net, max_lag=12)

        # Bootstrap CI (moving-block) for IR and mean active
        # Make bootstrap reproducible: seed from config (fallback 42), override via env BOOT_SEED
        _B = int(os.getenv("BOOT_B", "2000"))
        _seed_cfg = None
        try:
            _seed_cfg = _read_config("config/default.yaml").get("seed", None)
        except Exception:
            _seed_cfg = None
        _SEED = int(os.getenv("BOOT_SEED", str(_seed_cfg) if _seed_cfg is not None else "42"))

        ir_lo, ir_hi = _mbb_ci(active_net, _annualized_ir, block_len=6, B=_B, alpha=0.10, seed=_SEED)
        mean_lo, mean_hi = _mbb_ci(active_net, lambda a: float(np.mean(a)), block_len=6, B=_B, alpha=0.10, seed=_SEED)

        row = {
            "BlockIR12_mean": blk12["ir_mean"],
            "BlockIR12_median": blk12["ir_median"],
            "BlockIR12_count": blk12["blocks"],
            "HAC_t_lag6": hac6,
            "HAC_t_lag12": hac12,
            "IR_boot_ci90_lo": ir_lo,
            "IR_boot_ci90_hi": ir_hi,
            "Mean_boot_ci90_lo": mean_lo,
            "Mean_boot_ci90_hi": mean_hi,
        }

        summary_path = "output/report_summary.csv"
        if os.path.exists(summary_path):
            try:
                S = pd.read_csv(summary_path)
                if len(S) == 0:
                    S = pd.DataFrame([row])
                else:
                    for k, v in row.items():
                        S[k] = [v]
            except Exception:
                S = pd.DataFrame([row])
        else:
            S = pd.DataFrame([row])

        S.to_csv(summary_path, index=False)
        print(
            f"[REPORT] BlockIR12 mean={row['BlockIR12_mean']:.3f}, median={row['BlockIR12_median']:.3f}, n={row['BlockIR12_count']} | "
            f"HAC t-stat (L=6)={row['HAC_t_lag6']:.2f}, (L=12)={row['HAC_t_lag12']:.2f} | "
            f"IR 90% CI=({row['IR_boot_ci90_lo']:.2f},{row['IR_boot_ci90_hi']:.2f}), Mean 90% CI=({row['Mean_boot_ci90_lo']:.4f},{row['Mean_boot_ci90_hi']:.4f}) | "
            f"boot B={_B}, seed={_SEED}"
        )
    except Exception as _e:
        print(f"[WARN] Stronger stats computation skipped: {_e}")