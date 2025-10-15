# Strategy Performance Testing Runbook

## Purpose
Ensure every code change is evaluated for performance, risk, and stability regressions before merge.

## Local Workflow
- From repo root run: `python tests/run_perf.py --tier smoke`
- Compare against another baseline: `python tests/run_perf.py --tier regression --ref <commit>`
- Force fresh baseline (skip cache): add `--no-cache`
- Artifacts land in `artifacts/` (metrics JSON, Markdown report, PR comment payload)

## Gates & Overrides
- Fails if IR_net drops by more than 0.05, TE_ann increases by more than 1pp (0.01), drawdown worsens by >2pp, or turnover rises >1pp (smoke) / >2pp (others)
- Instability warnings come from rolling IR variance, walk-forward IR, turnover drift
- Override protocol: only when IC improves with bootstrap p < 0.10 and risk metrics remain within limits. Document rationale in PR and label `perf:override`

## Tiers
- `smoke`: last 24 months, no bootstrap; runtime target ≤ 7 min
- `regression`: full history, bootstrap n=1000, includes stability diagnostics; runtime ~15–20 min
- `full`: same as regression + walk-forward analysis; runtime ~35 min; runs on main merges

## Data & Determinism
- Seeds: `PYTHONHASHSEED=0`, `NUMPY_SEED=1234`, single-thread BLAS (OMP/OPENBLAS/MKL = 1)
- Data snapshots expected under `data/snapshots/<id>` (update `tests/perf/config.yaml` when rotating)
- If external endpoints fail, harness continues using cached artifacts and annotates missing metrics

## Troubleshooting
- Missing `weights.csv` or `active_net.csv`: check `scripts/run_backtest.py` to ensure outputs are written
- CI cache mismatch: run `python tests/run_perf.py --no-cache` locally, push updated artifacts to prime cache on main
- Non-deterministic output: confirm packages respect seed settings; otherwise pin versions in `requirements.txt`

## Escalation
- Persistent failures → sync with Quant Research lead, attach `artifacts/` bundle
- For overrides, mention `docs/RUNBOOK.md` and tag `@perf-guardians` in PR
