# System Status — Sector Rotation MVP
_Last updated: 2025-10-13_

## 0) TL;DR
- ✅ End-to-end pipeline running on **real monthly sector price data** (wide CSV).
- ✅ Builds **classic features** (momentum & volatility, x-section z-scored), trains **Ridge** monthly, predicts **next-month sector excess returns**.
- ✅ Long-only overlay vs equal-weight benchmark with **TE target**, **turnover cap**, and **smoothing**.
- ✅ Backtest prints **IR gross / IR net / Avg TE / Avg turnover**, saves `output/active_net.csv` and **weights** to `output/weights.csv`.
- ❗Not yet: graph/GIN features, official S&P500 sector weights, purge/embargo, reporting charts.

---

## 1) Data & Universe
- **Primary source (in use):** Single wide CSV with **Date** and one column per sector (e.g., `XLY, XLP, …, XLC`).  
  - File path set in config (see §7): `data/raw/sector_etf_prices_monthly.csv`.
  - Loader: `load_sector_returns_from_wide_csv(...)` (resamples to **month-end**, computes monthly returns).
- **Alternative source (also supported):** Folder with one CSV per ticker (`XLY.csv`, `XLK.csv`, …).  
  - Loader: `load_sector_returns_from_csv(folder, sectors, start, end)`.
- **Synthetic fallback:** `load_synthetic_sector_returns(...)` for quick runs.

**Universe:** 11 SPDR sectors (from config):
["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]

---

## 2) Feature Engineering (implemented)
- **Momentum**: `mom_12_1`, `mom_6_1`, `mom_3_1` (skip last month to avoid look-ahead).
- **Volatility**: `vol_12`, `vol_6`.  
- **Full-window only** (`min_periods=lookback`) → avoids partial-window NaNs.
- **Cross-section z-scores per feature per month** (standardizes across the 11 sectors).

_Source:_ `src/features/classic.py`

---

## 3) Forecasting Model (implemented)
- **Estimator:** `sklearn` **Pipeline** = `SimpleImputer(strategy="mean")` → `Ridge(alpha=1.0)`.
- **Target:** next-month **excess** sector returns vs benchmark (sector return – benchmark return).
- **Training window:** expanding, using months strictly **before** `asof`.
- **NaN safety:**  
  - Full windows in features, drop dates with NaNs, plus imputation in the pipeline (and extra defensive guards in code).
  
_Source:_ `src/backtest/engine.py`

---

## 4) Risk Model (implemented)
- **Covariance:** 60-month rolling monthly covariance.
- **Shrinkage:** 10% diagonal shrink toward average variance.

_Source:_ `src/risk/cov.py`

---

## 5) Portfolio Construction (implemented)
- **Overlay:** optimize **long-only** portfolio vs benchmark (sum=1, no shorts).
- **Controls:**
  - **TE target (annual):** `te_target_annual: 0.055` (5.5%).  
    - Internal scaling via bisection; realized TE may undershoot after constraints.
  - **Turnover cap:** `turnover_cap: 0.35` (35% one-way/month).
  - **Smoothing (EMA on active):** `smoothing_lambda: 0.30`.  
    - **Crisis fallback** (if cross-sec vol high): `λ=0.10`.
- **Costs:** `2 bps` one-way applied to turnover.

_Source:_ `src/portfolio/optimizer.py`, `src/backtest/engine.py`

---

## 6) Backtest Mechanics (implemented)
- **Frequency:** monthly (month-ends via `ME` index).
- **Warmup:** `60` months (configurable).  
- **Loop:** for each `asof`:
  1) Build features up to `asof`.
  2) Train Ridge; predict next-month per-sector excess returns.
  3) Estimate covariance; solve overlay with TE/turnover/smoothing.
  4) Realize next-month active return vs benchmark; subtract costs.
  5) Record TE(ann), turnover, weights.

**Outputs:**
- **Console summary:** IR gross, IR net, Avg turnover, Avg TE(ann).
- **CSV:** `output/active_net.csv` (net active series), **`output/weights.csv`** (monthly weights).

_Source:_ `src/backtest/engine.py`

---

## 7) Configuration (implemented)
_File:_ `config/default.yaml` (key parts)
```yaml
seed: 42
dates:
  start: "2005-01-31"
  end:   "2024-12-31"

data:
  source: "csv"                      # "csv" or "synthetic"
  csv_folder: "data/raw/sector_etf_prices_monthly.csv"  # file OR folder

universe:
  sectors: ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU","XLRE","XLC"]

portfolio:
  te_target_annual: 0.055
  turnover_cap: 0.35
  smoothing_lambda: 0.30
  crisis_vix_threshold: 35.0  # placeholder; we use internal cross-sec vol proxy

costs:
  one_way_bps: 2.0

evaluation:
  warmup_months: 60
  purge_months: 1   # not yet enforced
  embargo_months: 1 # not yet enforced
8) Current Behavior & Metrics
Data source (from runtime logs):
[INFO] Data source: CSV (wide file) → data/raw/sector_etf_prices_monthly.csv
[INFO] Dataset: 78 months × 11 sectors | range 2018-07-31 → 2024-12-31
Recent run metrics (real data):
IR gross: ~0.32 | IR net: ~0.31
Avg turnover: ~23.56%
Avg TE(ann): ~4.82%
(Note: with warmup=60, you have ~18–19 out-of-sample months; consider warmup=36 to increase live evaluations.)
Artifacts created:
output/active_net.csv (net active series)
output/weights.csv (17 rows × 11 sectors in your last run)
9) What’s NOT Implemented Yet (intentionally)
❌ Graph-smoothed features (corr-based smoothing).
❌ GNN-lite embeddings / message passing.
❌ Official S&P 500 sector weights benchmark (currently equal-weight).
❌ Purging / embargo around train/test splits.
❌ Reporting (plots: equity curve, drawdown, rolling IR, weight heatmap).
❌ Predictions dump (preds.csv) for per-month alpha audit.
❌ Doctor script to pre-flight environment.
10) How to Run & Verify
# (1) Activate venv and run
source .venv/bin/activate
python -m scripts.run_backtest

# (2) Confirm data source in logs
# [INFO] Data source: CSV (wide file) → ...

# (3) Inspect outputs
head output/active_net.csv
head output/weights.csv
To switch data:
CSV (wide file): set data.source: "csv" and data.csv_folder: "data/raw/sector_etf_prices_monthly.csv".
Synthetic: set data.source: "synthetic".
11) Recommended Next Steps (small, high-impact)
Lower warmup to 36 for real data → more live months.
Dump predictions each month (output/preds.csv) for audit (very helpful for debugging/feature tweaks).
Graph-smoothed momentum (cheap 80/20 lift):
Build 6m correlation matrix across sectors.
X' = (1-α)·X + α·ÂX (α≈0.3–0.5; Â = normalized adj matrix).
Concatenate to classic features and re-run.
Reporting: small matplotlib script for equity curve, rolling 12m IR, drawdown, turnover hist, weight heatmap.
Benchmark: add S&P500 sector weight history (CSV) and switch overlay benchmark.
12) Repo State
GitHub: nano44/mvp_rotation
Main: initial commit pushed (466caf4), tag v0.1-baseline.
Branch: feature/real-sector-data pushed; ready for PRs.
Credential helper: osxkeychain with a working PAT.
13) Open Questions / Decisions
What warmup do we prefer for this data window? (36 vs 60 months)
Equal-weight benchmark vs. official cap-weighted sector weights?
Which graph smoothing α and lookback (α ∈ [0.3, 0.5], L=6m?)
When to add purge/embargo (before any performance claims)?