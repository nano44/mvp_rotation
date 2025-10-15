from tests.perf.diff import diff_metrics, evaluate_thresholds


def test_diff_metrics_and_thresholds_pass():
    baseline = {"ir_net": 0.35, "te_ann_mean": 0.05, "max_drawdown": -0.10, "turnover_mean": 0.15}
    candidate = {"ir_net": 0.36, "te_ann_mean": 0.055, "max_drawdown": -0.11, "turnover_mean": 0.16}
    diff = diff_metrics(baseline, candidate, tier="regression")
    eval_result = evaluate_thresholds(diff, tier="regression")
    assert eval_result["status"] == "pass"
    assert not eval_result["failures"]


def test_diff_metrics_and_thresholds_fail():
    baseline = {"ir_net": 0.35, "te_ann_mean": 0.05, "max_drawdown": -0.10, "turnover_mean": 0.15}
    candidate = {"ir_net": 0.20, "te_ann_mean": 0.07, "max_drawdown": -0.05, "turnover_mean": 0.20}
    diff = diff_metrics(baseline, candidate, tier="smoke")
    eval_result = evaluate_thresholds(diff, tier="smoke")
    assert eval_result["status"] == "fail"
    assert eval_result["failures"]
