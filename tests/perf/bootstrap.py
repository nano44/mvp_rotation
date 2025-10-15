"""
Block bootstrap utilities for deterministic confidence interval estimation.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def moving_block_bootstrap(
    series: Iterable[float],
    block_length: int,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """
    Moving-block bootstrap to preserve serial correlation in time-series.
    Returns array with shape (n_draws, len(series)).
    """
    values = np.asarray(list(series), dtype=float)
    n_obs = len(values)
    if n_obs == 0 or block_length <= 0 or n_draws <= 0:
        return np.empty((0, 0))

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_obs / block_length)
    idx = np.arange(n_obs)

    draws = []
    for _ in range(n_draws):
        sample_idx = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, n_obs - block_length + 1))
            sample_idx.extend(idx[start : start + block_length])
        sample_idx = np.asarray(sample_idx[:n_obs], dtype=int)
        draws.append(values[sample_idx])

    return np.stack(draws, axis=0)


def _metric_from_sample(sample: np.ndarray, metric: str) -> float:
    series = pd.Series(sample)
    if metric == "ir":
        mu = series.mean()
        sigma = series.std()
        return float(mu / sigma * math.sqrt(12)) if sigma > 0 else float("nan")
    if metric == "mean":
        return float(series.mean())
    if metric == "median":
        return float(series.median())
    raise ValueError(f"Unsupported metric '{metric}'")


def bootstrap_metric_ci(
    series: Iterable[float],
    metric: str,
    block_length: int,
    n_draws: int,
    seed: int,
) -> Tuple[float, float]:
    resamples = moving_block_bootstrap(series, block_length, n_draws, seed)
    if resamples.size == 0:
        return float("nan"), float("nan")
    stats = np.array([_metric_from_sample(sample, metric) for sample in resamples])
    lower, upper = np.nanpercentile(stats, [2.5, 97.5])
    return float(lower), float(upper)


def attach_ci_to_diff(diff: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Injects a formatted CI string if `ci_lower`/`ci_upper` are present.
    """
    for key, payload in diff.items():
        lower = payload.get("ci_lower")
        upper = payload.get("ci_upper")
        if lower is not None and upper is not None:
            payload["ci_95"] = f"[{lower:.4f}, {upper:.4f}]"
    return diff

