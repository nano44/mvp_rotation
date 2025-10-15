#!/usr/bin/env python3
"""
CLI orchestrator for strategy performance regression checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.perf.metrics import compute_metrics
from tests.perf.diff import diff_metrics, evaluate_thresholds
from tests.perf.reporting import assemble_markdown_report, write_pr_comment_payload

DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
CACHE_ROOT = REPO_ROOT / ".cache" / "perf"


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 900,
):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=True, timeout=timeout)


def prepare_checkout(ref: str) -> Path:
    """
    Creates a throwaway checkout of the repo at the requested ref by cloning
    into the local cache directory. Avoids git worktree (blocked in sandbox).
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    checkout_dir = CACHE_ROOT / f"checkout_{ref.replace('/', '_')}"
    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    run_cmd(["git", "clone", "--no-checkout", ".", str(checkout_dir)], cwd=REPO_ROOT)
    run_cmd(["git", "checkout", ref], cwd=checkout_dir)
    shared_data = REPO_ROOT / "data"
    target_data = checkout_dir / "data"
    if shared_data.exists():
        if target_data.exists():
            if target_data.is_symlink() or target_data.is_file():
                target_data.unlink()
            else:
                shutil.rmtree(target_data)
        try:
            os.symlink(shared_data, target_data)
        except OSError:
            shutil.copytree(shared_data, target_data)
    return checkout_dir


def cleanup_checkout(path: Path):
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        pass


def run_pipeline(repo_dir: Path, config_path: Path, tier: str, seed: int) -> Path:
    """
    Run data prep, backtest, and report generation for a given repo snapshot.
    """
    mpl_cache_dir = repo_dir / ".cache" / "mpl"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "PYTHONHASHSEED": "0",
        "NUMPY_SEED": str(seed),
        "PYTHONPATH": str(repo_dir / "src") + os.pathsep + str(repo_dir),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TIER": tier,
        "MPLCONFIGDIR": str(mpl_cache_dir),
    }
    output_dir = repo_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_cmd(["python", "scripts/build_assets.py", "--config", str(config_path)], cwd=repo_dir, env=env, timeout=600)
    run_cmd(["python", "scripts/run_backtest.py", str(config_path)], cwd=repo_dir, env=env, timeout=900)
    run_cmd(["python", "scripts/report.py"], cwd=repo_dir, env=env, timeout=600)
    return output_dir


def cache_artifacts(source: Path, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def load_cached_metrics(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Strategy performance regression harness.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ref", default="origin/main", help="Baseline commit/branch.")
    parser.add_argument("--cmp", default="HEAD", help="Candidate commit (default HEAD).")
    parser.add_argument("--tier", choices=["smoke", "regression", "full"], default="smoke")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse cached baseline metrics.")
    parser.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    seed = 1234

    baseline_cache_dir = CACHE_ROOT / args.ref.replace("/", "_")
    baseline_metrics_path = baseline_cache_dir / "metrics.json"
    cmp_dir = artifacts_root / "candidate"
    cmp_metrics_path = cmp_dir / "metrics.json"

    baseline_metrics = None if args.no_cache else load_cached_metrics(baseline_metrics_path)

    if baseline_metrics is None:
        checkout = prepare_checkout(args.ref)
        try:
            output = run_pipeline(checkout, Path(args.config), args.tier, seed)
            cache_artifacts(output, baseline_cache_dir / "output")
            baseline_metrics = compute_metrics(baseline_cache_dir / "output", tier=args.tier)
            baseline_metrics_path.write_text(json.dumps(baseline_metrics, indent=2))
        finally:
            cleanup_checkout(checkout)

    output_cmp = run_pipeline(REPO_ROOT, Path(args.config), args.tier, seed)
    cache_artifacts(output_cmp, cmp_dir / "output")
    candidate_metrics = compute_metrics(cmp_dir / "output", tier=args.tier)
    cmp_metrics_path.write_text(json.dumps(candidate_metrics, indent=2))

    diff = diff_metrics(baseline_metrics, candidate_metrics, tier=args.tier)
    evaluation = evaluate_thresholds(diff, tier=args.tier)

    report_md = assemble_markdown_report(diff, baseline_metrics, candidate_metrics, tier=args.tier)
    (artifacts_root / "report.md").write_text(report_md, encoding="utf-8")
    (artifacts_root / "summary.json").write_text(json.dumps(evaluation, indent=2))
    pr_comment = write_pr_comment_payload(diff, evaluation)
    (artifacts_root / "pr_comment.md").write_text(pr_comment, encoding="utf-8")

    print(json.dumps(evaluation, indent=2))
    if evaluation["status"] == "fail":
        print("[perf] Performance regression detected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
