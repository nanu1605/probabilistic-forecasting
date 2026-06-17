# Changelog

All notable changes to this project, one line per phase. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); commits follow Conventional Commits.

## [Unreleased]

- Phase 0: project scaffold — pyproject (Python 3.12), ruff, pydantic-settings config, module tree, Makefile, DVC init, test framework (6 tests pass).
  - GPU verified: RTX 5060 Ti (Blackwell sm_120) runs torch 2.12.0+cu130; GPU matmul confirmed → DeepAR will train on GPU at full 50 epochs (no CPU fallback needed).
- Phase 1: data pipeline + EDA — download (UCI #501 → synthetic fallback), preprocess (continuous hourly index, gap-aware imputation, backward-only features), strict temporal split, pandera schema, DVC pipeline (download→preprocess→split), EDA notebook (5 plots). 22 tests pass.
  - **Real UCI data used** (`SOURCE=uci`): station Aotizhongxin, 35064 hourly rows, raw PM2.5 missingness 2.64%. Splits: train 29232 / val 4416 / test 1416. Long gaps (>6h) left NaN and excluded from eval.
- Phase 2: classical baselines + metrics engine — Seasonal Naive, ARIMA (auto_arima + SARIMAX walk-forward), ETS (ETSModel, native simulate). Shared rolling-origin harness (59 non-overlapping 24h windows) + uniform 100-sample representation for all models. Metrics: MAE/RMSE/CRPS/coverage/ECE/Winkler. MLflow (sqlite backend). 37 tests pass.
  - Test results (CRPS, lower=better): seasonal_naive 68.4, **arima 47.4**, ets 47.5 — both classical models beat naive. cov@90: naive 0.79 / arima 0.80 / ets 0.88. auto_arima order (0,1,1). All 3 models ok (none failed to converge).
- Phase 3: DeepAR reproduction — GluonTS DeepAREstimator (StudentT, ctx 168, 50 epochs, GPU), same BaseForecaster interface + rolling harness as baselines. 5 seeds [42,123,456,789,1337], mean±std, MLflow per-seed runs w/ provenance metadata. 41 tests pass.
  - **DeepAR is the best model: CRPS 43.6 ± 1.7** (vs naive 68.4, arima 47.4, ets 47.5). MAE 59.7, RMSE 99.5.
  - **But miscalibrated / overconfident**: coverage@50/80/90 = 0.36/0.62/0.74 (all below nominal), ECE 0.125 — intervals too narrow. This is the diagnosis Phase 4 visualizes and Phase 5 recalibration fixes.
  - Batched per-seed predict (one multi-series predict call over 59 window-contexts) replaced per-window predict → ~50x faster eval. GPU determinism test pinned to CPU. ~6 min/seed training on RTX 5060 Ti.
- Phase 4: calibration analysis (diagnosis) — calibration.py (curve data, per-horizon, ECE; reuses metrics engine) + plotting/calibration.py (curve, comparison money-shot, per-horizon grid). Post-processes saved DeepAR forecasts (no retrain). 45 tests pass.
  - **Diagnosis:** DeepAR under-covers at every level (0.9→0.80, 0.5→0.39, …) → consistent overconfidence, ECE 0.087 (best seed). Per-horizon ECE worsens monotonically: h+1 0.03 → h+6 0.06 → h+12 0.12 → h+24 0.17 — miscalibration grows with horizon. Plots: calibration_before.png, calibration_per_horizon.png.
- Phase 5: post-hoc recalibration (the extension) — Kuleshov isotonic recalibration (PIT→empirical-CDF map, resample via F⁻¹(R⁻¹(u))) + variance-scaling fallback; leakage-safe (transform takes no labels). Self-contained retrain (seed 1337) → val+test forecasts. Full results table (5 models). calibration_comparison.png (money shot), calibration_after.png. 50 tests pass.
  - **Honest finding (the interesting result):** recalibration fit on val does NOT improve test calibration — test ECE 0.090→0.098, cov@90 0.80→0.80. Root cause isolated: **val→test distribution shift**. DeepAR is already well-calibrated in-distribution (val ECE **0.014**) so the val-fit map is ≈identity; the test miscalibration (ECE 0.090, Jan–Feb 2017 winter) cannot be learned from the calibrated val period. The recalibration METHOD is proven correct by an oracle (leakage) upper bound: fitting on test drops test ECE 0.090→**0.020**. Practitioner lesson + spec §13/limitations (non-stationarity, val≈test assumption). Drives the Phase 6 writeup.
