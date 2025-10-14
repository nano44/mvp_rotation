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

    # Summary CSV
    summary = pd.DataFrame(
        [
            {
                "IR_net_full": ir_full,
                "TE_ann_mean": te_series.mean(),
                "TE_ann_median": te_series.median(),
                "Turnover_mean": turnover.mean(),
                "Obs_active": len(active_net),
            }
        ]
    )
    summary.to_csv("output/report_summary.csv", index=False)

    print("[REPORT] Saved:")
    print("  - output/report_equity_curve.png")
    print("  - output/report_rolling_ir_24m.png")
    print("  - output/report_te_vs_target.png")
    print("  - output/report_turnover.png")
    print("  - output/report_weights_heatmap.png")
    print("  - output/report_summary.csv")
    print(f"IR net (full): {ir_full:.2f} | TE mean: {te_series.mean():.2%} | Turnover mean: {turnover.mean():.2%}")


if __name__ == "__main__":
    main()