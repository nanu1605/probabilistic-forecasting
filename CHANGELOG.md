# Changelog

All notable changes to this project, one line per phase. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); commits follow Conventional Commits.

## 2025-06-18 — Report, final documentation, and polish

- Wrote `report/report.md` — paper-style writeup covering data, methods, results,
  and limitations (~3500 words).
- Finalized README with results table, calibration plot, and quickstart.
- `make reproduce` verified end to end from clean state.

## 2025-06-17 — Calibration experiment writeup

- Wrote `experiments/calibration_experiment.md` with the full analysis: miscalibration
  diagnosis, recalibration attempt, oracle experiment, sharpness tradeoff, and
  honest limitations.
- The key finding: recalibration works when val ≈ test but breaks under seasonal
  non-stationarity. Oracle fit-on-test confirms the method itself is correct
  (ECE 0.090 → 0.020).

## 2025-06-16 — Post-hoc recalibration (the extension)

- Implemented isotonic recalibration (Kuleshov 2018) fitted on validation-set quantiles.
- Applied to DeepAR test predictions. Recalibration improved validation ECE but did
  *not* improve test ECE — distribution shift between seasons.
- Ran oracle experiment (fit on test) to verify the method works in principle.
- Generated `calibration_comparison.png` — the before/after plot.
- Full results table across all models compiled to `metrics/`.

## 2025-06-15 — Calibration analysis

- Implemented calibration curve computation: predicted vs observed coverage at
  10 confidence levels.
- Computed per-horizon calibration: ECE degrades from 0.03 at h+1 to 0.17 at h+24.
- Generated `calibration_before.png`. DeepAR is systematically overconfident —
  90% intervals cover only 74% of true values.

## 2025-06-14 — DeepAR reproduction (5 seeds)

- GluonTS DeepAR estimator with Student-t output distribution.
- Trained with 5 random seeds, reported mean ± std for all metrics.
- DeepAR beats baselines on CRPS (43.6 vs 47.4 for ARIMA) but is worst-calibrated.
- All runs logged to MLflow with full provenance (data hash, git SHA, config).

## 2025-06-13 — Classical baselines

- Implemented Seasonal Naive (repeat last day), ARIMA (pmdarima auto_arima),
  and ETS (statsmodels Holt-Winters).
- Evaluation engine: MAE, RMSE, CRPS (properscoring), coverage at 50/80/90%,
  ECE, Winkler score.
- All models emit 100 sample paths through a shared `BaseForecaster` interface,
  so metrics are computed by identical code.
- Known-answer tests for CRPS and coverage to verify metric correctness.

## 2025-06-12 — Data pipeline and EDA

- UCI Beijing Air Quality dataset (Aotizhongxin station, hourly PM2.5).
- Preprocessing: forward-fill ≤6h gaps, cyclical time features, lag features,
  rolling statistics. Synthetic fallback generator if UCI is unreachable.
- Strict temporal split: train (2013–mid 2016), val (mid 2016–end 2016),
  test (Jan–Feb 2017). Leakage assertions in tests.
- Pandera schema validation. DVC-tracked data pipeline.
- EDA notebook with seasonality decomposition, distribution analysis,
  missing-value heatmap.

## 2025-06-11 — Project setup

- Repository scaffold: `pyproject.toml`, `uv`, `ruff.toml`, `pytest`, Makefile.
- Config loader (`pydantic-settings`) for `configs/config.yaml` and
  `configs/model_params.yaml`.
- DVC init, MLflow local backend, CI workflow.