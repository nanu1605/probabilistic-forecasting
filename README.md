# Probabilistic Time-Series Forecasting with Calibrated Uncertainty

Reproduce **DeepAR** on Beijing air-quality data, evaluate against classical baselines with
probabilistic metrics, then diagnose and fix miscalibration via **post-hoc recalibration**.

> 🚧 Under construction — built phase by phase. See `CHANGELOG.md` for progress.

## Quickstart

```bash
make setup       # install deps + DVC init
make reproduce   # data -> baselines -> deepar -> calibration -> recalibrate -> figures
make test        # run test suite
```

Full README (results, calibration comparison plot, design decisions) lands in Phase 7.
