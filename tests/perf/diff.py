"""
Compute and annotate metric deltas between baseline and candidate runs.
"""

from __future__ import annotations

from typing import Dict, Literal

import math

from tests.perf.bootstrap import attach_ci_to_diff

Tier = Literal["smoke", "regression", "full"]


def diff_metrics(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    tier: Tier,
) -> Dict[str, Dict[str, float]]:
    diff: Dict[str, Dict[str, float]] = {}
    for key, base_val in baseline.items():
        cand_val = candidate.get(key)
        if cand_val is None:
            continue
        if _both_nan(base_val, cand_val):
            continue
        delta = cand_val - base_val
        delta_pct = float("nan")
        if base_val not in (0.0, float("nan")) and not math.isnan(base_val):
            delta_pct = delta / base_val
        diff[key] = {
            "baseline": base_val,
            "candidate": cand_val,
            "delta": delta,
            "delta_pct": delta_pct,
        }

    if tier in ("regression", "full"):
        diff = attach_ci_to_diff(diff)

    return diff


def _both_nan(a: float, b: float) -> bool:
    return math.isnan(a) and math.isnan(b)


def evaluate_thresholds(diff: Dict[str, Dict[str, float]], tier: Tier) -> Dict[str, object]:
    """
    Apply tier-specific pass/fail gates. Returns dict with status/failure list.
    """
    failures = []
    gates = {
        "ir_net": {"min_delta": -0.05},
        "te_ann_mean": {"max_delta": 0.01},
        "max_drawdown": {"max_delta": 0.02},
        "turnover_mean": {"max_delta": 0.02 if tier != "smoke" else 0.01},
    }

    for metric, rule in gates.items():
        payload = diff.get(metric)
        if not payload:
            continue
        delta = payload["delta"]
        if "min_delta" in rule and delta < rule["min_delta"]:
            failures.append(f"{metric} delta {delta:.4f} < {rule['min_delta']:.4f}")
        if "max_delta" in rule and delta > rule["max_delta"]:
            failures.append(f"{metric} delta {delta:.4f} > {rule['max_delta']:.4f}")

    status = "pass" if not failures else "fail"
    return {
        "tier": tier,
        "status": status,
        "failures": failures,
    }

