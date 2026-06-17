"""Calibration analysis tests: diagonal for perfect, direction of mis-calibration, per-horizon."""

from __future__ import annotations

import numpy as np

from probforecast.evaluation.calibration import (
    compute_calibration_data,
    compute_per_horizon_calibration,
)
from probforecast.evaluation.metrics import compute_ece

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _make(scale: float, n_windows: int = 200, n_samples: int = 500, horizon: int = 24):
    """Forecast samples ~ N(0, scale); actuals ~ N(0,1). scale=1 -> calibrated."""
    rng = np.random.default_rng(0)
    actuals = rng.normal(0, 1, size=(n_windows, horizon))
    samples = rng.normal(0, scale, size=(n_windows, n_samples, horizon))
    return samples, actuals


def test_perfectly_calibrated_on_diagonal():
    samples, actuals = _make(scale=1.0)
    data = compute_calibration_data(samples, actuals, LEVELS)
    for predicted, observed in data:
        assert abs(observed - predicted) < 0.05
    assert compute_ece(dict(data)) < 0.03


def test_overconfident_below_diagonal():
    # Too-narrow intervals -> observed coverage < predicted.
    samples, actuals = _make(scale=0.3)
    data = compute_calibration_data(samples, actuals, LEVELS)
    high = [(p, o) for p, o in data if p >= 0.5]
    assert all(o < p for p, o in high)


def test_underconfident_above_diagonal():
    # Too-wide intervals -> observed coverage > predicted.
    samples, actuals = _make(scale=3.0)
    data = compute_calibration_data(samples, actuals, LEVELS)
    mid = [(p, o) for p, o in data if 0.2 <= p <= 0.8]
    assert all(o > p for p, o in mid)


def test_per_horizon_returns_valid_curves():
    samples, actuals = _make(scale=1.0)
    ph = compute_per_horizon_calibration(samples, actuals, LEVELS, [1, 6, 12, 24])
    assert set(ph) == {1, 6, 12, 24}
    for curve in ph.values():
        assert len(curve) == len(LEVELS)
        assert all(0.0 <= o <= 1.0 for _, o in curve)
