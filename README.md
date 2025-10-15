# mvp_rotation

Monthly sector-rotation MVP: engineered features → ridge forecaster → TE/turnover-aware overlay with deterministic backtest and reporting harness.

## Environment

- Create a virtualenv (e.g. `python -m venv .venv`) and activate it.
- Install dependencies: `pip install -r requirements.txt`. This now includes `pytest` for the unit/performance tests.
- Data snapshot: `data/master/` currently starts in **2018-07**. Earlier IR figures (≈0.45) assumed a longer history; restore the older snapshot if you need full 2005–2024 coverage.

```
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Backtest & Reports

```
# Generate required master tables (only if snapshot missing or refreshed)
PYTHONPATH=src:. python scripts/build_returns_m.py --config config/default.yaml
PYTHONPATH=src:. python scripts/build_bench_weights_m.py --config config/default.yaml
PYTHONPATH=src:. python scripts/build_features_monthly.py
PYTHONPATH=src:. python scripts/build_graph_features.py
PYTHONPATH=src:. python scripts/build_cov_m.py

# Core backtest (writes output/active_net.csv, weights.csv, etc.)
python scripts/run_backtest.py config/default.yaml

# Generate plots/summary (PNG + CSV in output/)
python scripts/report.py
```

Typical smoke run with the current snapshot produces roughly:
```
IR gross: ~0.31 | IR net: ~0.29
Avg turnover: ~22% | Avg TE(ann): ~5.7%
```

## Performance Regression Harness

`tests/run_perf.py` compares `HEAD` vs a baseline reference using identical data/seeds, caches artifacts, and enforces metric gates.

```
# Smoke tier (default): reuses cache if available
python tests/run_perf.py --tier smoke

# Force fresh baseline cache
python tests/run_perf.py --tier regression --ref origin/main --no-cache
```

Artifacts land in `artifacts/`:
- `report.md`: human summary with delta table and links to plots.
- `summary.json`: machine-readable status (`pass` / `fail` + reasons).
- `pr_comment.md`: reusable PR comment payload.

## CI Workflow

GitHub Actions workflow `.github/workflows/perf.yml` runs automatically:
- Pull requests: smoke tier vs PR base, posts summary table as sticky comment.
- Pushes to `main`: full tier (bootstrap + walk-forward) vs previous commit, uploads artifacts.
- Both jobs restore `.cache/perf` for faster baselines and enforce deterministic seeds (`PYTHONHASHSEED=0`, `NUMPY_SEED=1234`).

## Testing

- Unit/perf tests: `python -m pytest tests -q` (uses synthetic fixtures under `tests/data/`).
- Smoke performance check: `python tests/run_perf.py --tier smoke`.
- Update `.gitignore` already includes `.cache/` and `artifacts/`; clean them before committing if desired.

## Project Layout

```
mvp_rotation/
  config/                      # YAML configs
  data/
    master/                    # reproducible snapshot (parquet outputs)
    raw/                       # source data (ignored)
  docs/                        # documentation (RUNBOOK.md, etc.)
  scripts/                     # ETL, backtest, reporting scripts
  src/                         # strategy source (engine, data loaders, features, risk, optimizer)
  tests/                       # unit + performance harness utilities
  output/                      # backtest artifacts (ignored)
  artifacts/                   # perf harness outputs (ignored)
  .cache/                      # cached baselines (ignored)
```

## Troubleshooting

- **PyArrow “Repetition level histogram size mismatch”**: regenerate parquet tables via the build scripts above or reinstall `pyarrow` if the file is corrupt.
- **Missing baseline data**: ensure `data/master/` contains the expected snapshot; rerun builders or fetch archived copy.
- **Matplotlib cache warnings**: handled automatically by the harness (`MPLCONFIGDIR` redirected under `.cache/`).
- **Non-determinism**: confirm seeds exported (`NUMPY_SEED`, `PYTHONHASHSEED`) and single-thread BLAS env vars set, especially in CI.

## Git Tips

```
git status
git add -A
git commit -m "Describe your change"
git log --oneline -n 5
```

Keep feature branches in sync with `main`. After branching or refreshing data, rerun the build scripts listed above to align the snapshot. Continuous performance monitoring via `tests/run_perf.py` is expected before requesting review.
