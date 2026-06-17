"""Recalibration tests: ECE improvement, quantile ordering, leakage-safety, no-harm."""

from __future__ import annotations

import numpy as np

from probforecast.evaluation.calibration import compute_calibration_data
from probforecast.evaluation.metrics import compute_ece
from probforecast.evaluation.recalibration import RecalibrationModel

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _dataset(scale: float, seed: int, n_w: int = 300, n_s: int = 400, n_h: int = 4):
    """Truth ~ N(0,1); model samples ~ N(0, scale). scale<1 -> overconfident."""
    rng = np.random.default_rng(seed)
    actuals = rng.normal(0, 1, size=(n_w, n_h))
    samples = rng.normal(0, scale, size=(n_w, n_s, n_h))
    return samples, actuals


def _ece(samples, actuals):
    return compute_ece(dict(compute_calibration_data(samples, actuals, LEVELS)))


def test_recalibration_improves_ece_on_overconfident():
    val_s, val_y = _dataset(scale=0.3, seed=1)
    test_s, test_y = _dataset(scale=0.3, seed=2)
    model = RecalibrationModel(method="isotonic", nonneg=False).fit(val_s, val_y)
    recal = model.transform(test_s, num_samples=400)
    assert _ece(recal, test_y) < _ece(test_s, test_y)


def test_recalibrated_quantiles_ordered():
    val_s, val_y = _dataset(scale=0.3, seed=1)
    test_s, _ = _dataset(scale=0.3, seed=2)
    model = RecalibrationModel(nonneg=False).fit(val_s, val_y)
    recal = model.transform(test_s, num_samples=300)
    q05, q50, q95 = np.quantile(recal, [0.05, 0.5, 0.95], axis=1)
    assert np.all(q05 <= q50) and np.all(q50 <= q95)


def test_transform_independent_of_test_labels():
    # transform takes no labels -> output cannot depend on them. Verify reproducibility,
    # and that fitting never consumed test data.
    val_s, val_y = _dataset(scale=0.3, seed=1)
    test_s, _ = _dataset(scale=0.3, seed=2)
    model = RecalibrationModel(nonneg=False).fit(val_s, val_y)
    a = model.transform(test_s, 200, rng=np.random.default_rng(7))
    b = model.transform(test_s, 200, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_perfectly_calibrated_not_harmed():
    val_s, val_y = _dataset(scale=1.0, seed=1)
    test_s, test_y = _dataset(scale=1.0, seed=2)
    model = RecalibrationModel(nonneg=False).fit(val_s, val_y)
    recal = model.transform(test_s, num_samples=400)
    # Stays well-calibrated (does not blow up).
    assert _ece(recal, test_y) < 0.08


def test_scaling_fallback_improves_ece():
    val_s, val_y = _dataset(scale=0.3, seed=1)
    test_s, test_y = _dataset(scale=0.3, seed=2)
    model = RecalibrationModel(method="scaling", nonneg=False).fit(val_s, val_y)
    recal = model.transform(test_s, num_samples=400)
    assert _ece(recal, test_y) < _ece(test_s, test_y)
