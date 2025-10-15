from tests.perf.reporting import assemble_markdown_report, write_pr_comment_payload


def test_assemble_markdown_report_contains_metrics():
    diff = {
        "ir_net": {"baseline": 0.35, "candidate": 0.37, "delta": 0.02, "delta_pct": 0.057},
        "te_ann_mean": {"baseline": 0.05, "candidate": 0.055, "delta": 0.005, "delta_pct": 0.1},
    }
    md = assemble_markdown_report(diff, {}, {}, tier="smoke")
    assert "Strategy Performance Comparison" in md
    assert "| ir_net | 0.3500 | 0.3700 | 0.0200 | 0.0570 |" in md


def test_write_pr_comment_payload_reports_status():
    diff = {
        "ir_net": {"baseline": 0.35, "candidate": 0.37, "delta": 0.02, "delta_pct": 0.057},
    }
    evaluation = {"tier": "smoke", "status": "pass", "failures": []}
    comment = write_pr_comment_payload(diff, evaluation)
    assert "Performance Check (smoke)" in comment
    assert "✅" in comment
