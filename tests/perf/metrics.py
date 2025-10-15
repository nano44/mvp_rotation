"""
Metric computation helpers for performance regression harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd

from scripts import report as report_mod
from tests.perf.bootstrap import bootstrap_metric_ci

Tier = Literal["smoke", "regression", "full"]


@dataclass
class MetricBundle:
    core: Dict[str, float]
    distributions: Dict[str, Iterable[float]]
    metadata: Dict[str, str]


def _load_series(path: Path) -> pd.Series:
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def compute_metrics(output_dir: Path, tier: Tier = "smoke") -> Dict[str, float]:
    """
    Aggregate all scalar metrics needed for diffing/threshold checks.
    """
    active_net_path = output_dir / "active_net.csv"
    weights_path = output_dir / "weights.csv"

    active_net = _load_series(active_net_path)
    weights = _load_frame(weights_path)

    ir_net = report_mod.info_ratio(active_net)
    rolling_ir = report_mod.rolling_ir(active_net, window=24)
    equity_curve = (1.0 + active_net).cumprod()
    max_dd = float(equity_curve.div(equity_curve.cummax()).min() - 1.0)
    vol_ann = float(active_net.std() * np.sqrt(12))
    turnover = report_mod.compute_turnover(weights)
    te_series = report_mod.compute_te_series(weights, list(weights.columns), target_te=None)

    metrics: Dict[str, float] = {
        "ir_net": float(ir_net),
        "ir_net_var": float(np.nanvar(rolling_ir.dropna())),
        "te_ann_mean": float(np.nanmean(te_series)),
        "te_ann_std": float(np.nanstd(te_series)),
        "turnover_mean": float(np.nanmean(turnover)),
        "turnover_std": float(np.nanstd(turnover)),
        "max_drawdown": max_dd,
        "vol_ann": vol_ann,
        "rolling_ir_var": float(np.nanvar(rolling_ir.dropna())),
        "observations": int(active_net.count()),
    }

    # Optional artifacts
    txn_path = output_dir / "txn_costs.csv"
    if txn_path.exists():
        txn = _load_series(txn_path)
        metrics["txn_cost_mean_bps"] = float(txn.mean() * 1e4)
        metrics["txn_cost_std_bps"] = float(txn.std() * 1e4)
    else:
        metrics["txn_cost_mean_bps"] = float("nan")
        metrics["txn_cost_std_bps"] = float("nan")

    signals_path = output_dir / "signals.csv"
    if signals_path.exists():
        signals = _load_series(signals_path)
        realized = active_net.reindex(signals.index).shift(-1)
        metrics["ic_spearman"] = float(realized.corr(signals, method="spearman"))
        metrics["hit_rate"] = float((realized * np.sign(signals) > 0).mean())
    else:
        metrics["ic_spearman"] = float("nan")
        metrics["hit_rate"] = float("nan")

    # Sector exposures (mean weights)
    for sector, exposure in weights.mean().items():
        metrics[f"exposure_{sector}"] = float(exposure)

    if tier in ("regression", "full"):
        metrics["ir_ci_lower"], metrics["ir_ci_upper"] = bootstrap_metric_ci(
            active_net, metric="ir", block_length=6, n_draws=1000, seed=1234
        )
        metrics["te_ci_lower"], metrics["te_ci_upper"] = bootstrap_metric_ci(
            te_series, metric="mean", block_length=6, n_draws=1000, seed=1234
        )
        metrics["turnover_ci_lower"], metrics["turnover_ci_upper"] = bootstrap_metric_ci(
            turnover, metric="mean", block_length=6, n_draws=1000, seed=1234
        )

    if tier == "full":
        walk_ir = []
        for start in range(0, len(active_net) - 35):
            window = active_net.iloc[start : start + 36]
            walk_ir.append(report_mod.info_ratio(window))
        metrics["ir_walkforward_std"] = float(np.nanstd(walk_ir)) if walk_ir else float("nan")

    return metrics

