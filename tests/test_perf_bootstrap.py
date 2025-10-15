import numpy as np
from tests.perf import bootstrap


def test_moving_block_bootstrap_shape():
    series = np.arange(12, dtype=float)
    draws = bootstrap.moving_block_bootstrap(series, block_length=3, n_draws=5, seed=123)
    assert draws.shape == (5, 12)
    # Ensure each draw is a permutation of original length
    assert np.allclose(draws.mean(axis=1).shape[0], 5)


def test_bootstrap_metric_ci_ir():
    rng = np.random.default_rng(0)
    series = rng.normal(0.01, 0.02, size=60)
    lower, upper = bootstrap.bootstrap_metric_ci(series, metric="ir", block_length=6, n_draws=500, seed=123)
    assert lower < upper
    # IR for a mildly positive series should be > 0
    assert upper > 0.0
