from pathlib import Path

import numpy as np
import pytest

from tests.perf.metrics import compute_metrics


SAMPLE_OUTPUT = Path(__file__).resolve().parent / "data" / "sample_output"


@pytest.mark.parametrize("tier", ["smoke", "regression"])
def test_compute_metrics_sample_output(tier):
    metrics = compute_metrics(SAMPLE_OUTPUT, tier=tier)

    # Core metrics should be finite
    for key in ["ir_net", "te_ann_mean", "turnover_mean", "max_drawdown", "vol_ann"]:
        assert key in metrics
        assert np.isfinite(metrics[key])

    # Regression tier must include bootstrap intervals
    if tier == "regression":
        assert metrics["ir_ci_lower"] < metrics["ir_ci_upper"]
        assert metrics["te_ci_lower"] < metrics["te_ci_upper"]
        assert metrics["turnover_ci_lower"] < metrics["turnover_ci_upper"]

    # Exposures should be recorded for each sector column
    exposures = {k: v for k, v in metrics.items() if k.startswith("exposure_")}
    assert exposures
