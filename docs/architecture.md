# Architecture

End-to-end flow from raw data to report. Each stage is a small, testable module under
`src/probforecast/`, orchestrated by the `pipelines/` scripts and the `Makefile`.

```
                    ┌─────────────────────────────┐
                    │       Raw Data (DVC)         │
                    │   UCI Beijing air-quality    │
                    │   (synthetic fallback)       │
                    └──────────────┬──────────────┘
                                   v
                    ┌─────────────────────────────┐
                    │   Preprocessing Pipeline     │
                    │  clean → impute → features   │
                    │  temporal train/val/test     │
                    │  (pandera schema validation) │
                    └──────────────┬──────────────┘
                    ┌──────────────┴──────────────┐
                    v                              v
        ┌───────────────────┐          ┌───────────────────┐
        │ Classical Baselines│         │     DeepAR         │
        │ Seasonal Naive     │         │  (GluonTS/PyTorch) │
        │ ARIMA, ETS         │         │  5 seeds, MLflow   │
        └────────┬──────────┘          └────────┬──────────┘
                 └──────────────┬────────────────┘
                                v
                 ┌─────────────────────────────┐
                 │     Evaluation Engine        │
                 │  MAE RMSE CRPS Coverage ECE  │
                 │  Winkler · calibration curve │
                 │  (uniform 100-sample inputs) │
                 └──────────────┬──────────────┘
                                v
                 ┌─────────────────────────────┐
                 │  Post-hoc Recalibration      │
                 │  isotonic (Kuleshov) on val  │
                 │  → re-evaluate on test       │
                 └──────────────┬──────────────┘
                                v
                 ┌─────────────────────────────┐
                 │     Artifacts & Report       │
                 │  calibration_comparison.png  │
                 │  results_table.md · report   │
                 └─────────────────────────────┘
```

## Key design choices

- **One forecaster interface.** `models/base.py:BaseForecaster` (`fit` + `sample`) is implemented by
  all four models, so `evaluation/rolling.py` and `evaluation/metrics.py` treat them identically and
  the DeepAR-vs-baselines comparison is apples-to-apples.
- **Uniform sample representation.** Every model emits `(num_windows, 100, horizon)` sample paths;
  metrics reduce over windows × horizon.
- **No leakage by construction.** Backward-only features computed before the date split; strict split
  assertions; recalibration fit on validation only, transform takes no labels.
- **Reproducibility.** Config is the single source of truth (`configs/*.yaml` via
  `probforecast.config`); DVC tracks the data pipeline; MLflow logs every run with git SHA + library
  versions; `make reproduce` regenerates everything.
