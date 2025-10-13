# mvp_rotation

Option-A sector rotation MVP: features → ridge forecaster → TE/turnover-aware overlay.

## Quickstart

```bash
# 1) activate the virtualenv
source .venv/bin/activate

# 2) run the backtest (from the project root)
python -m scripts.run_backtest
You should see something like:

IR gross: 0.47 | IR net: 0.45
Avg turnover: 24.16% | Avg TE(ann): 3.82%

Project layout

mvp_rotation/
  config/                 # YAML config(s)
  data/
    raw/                  # raw inputs (ignored by git)
    interim/              # intermediate files (ignored)
    features/             # engineered features (ignored)
  src/
    data/                 # loaders
    features/             # classic features
    risk/                 # covariance model
    portfolio/            # optimizer
    backtest/             # engine
    utils/                # helpers
  scripts/                # run_backtest.py, doctor.py
  output/                 # results (ignored)
  .venv/                  # virtual environment (ignored)

Common commands

# run the backtest
python -m scripts.run_backtest

# basic health check
python -m scripts.doctor

Git basics (for this repo)

# stage and commit changes
git add -A
git commit -m "Describe your change"

# view recent commits
git log --oneline -n 5

