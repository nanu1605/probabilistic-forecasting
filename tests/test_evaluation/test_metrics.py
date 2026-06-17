"""Known-answer tests for the metrics engine."""

from __future__ import annotations

import numpy as np

from probforecast.evaluation.metrics import (
    compute_coverage,
    compute_ece,
    crps_ensemble,
    crps_gaussian,
    summarize_samples,
    winkler_score,
)


def test_crps_perfect_forecast_is_zero():
    y = np.array([3.0, 5.0, 7.0])
    samples = np.repeat(y[:, None], 100, axis=1)  # every sample == truth
    assert crps_ensemble(y, samples) == 0.0


def test_crps_gaussian_analytic_value():
    # CRPS(N(0,1), 0) = sqrt(2/pi) - 1/sqrt(pi) ~= 0.233695.
    expected = np.sqrt(2 / np.pi) - 1 / np.sqrt(np.pi)
    assert np.isclose(crps_gaussian(0.0, 0.0, 1.0), expected, atol=1e-6)


def test_coverage_always_inside_is_one():
    y = np.array([1.0, 2.0, 3.0])
    assert compute_coverage(y, y - 1, y + 1) == 1.0


def test_coverage_never_inside_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert compute_coverage(y, y + 10, y + 20) == 0.0


def test_ece_perfectly_calibrated_is_zero():
    curve = {0.1: 0.1, 0.5: 0.5, 0.9: 0.9}
    assert compute_ece(curve) == 0.0


def test_ece_known_gap():
    curve = {0.5: 0.4, 0.9: 1.0}  # gaps 0.1 and 0.1
    assert np.isclose(compute_ece(curve), 0.1)


def test_winkler_equals_width_when_inside():
    y = np.array([0.0, 0.0])
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    # All truths inside -> score == mean width.
    assert np.isclose(winkler_score(y, lower, upper, 0.9), np.mean(upper - lower))


def test_winkler_penalizes_miss():
    y = np.array([5.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])
    inside = winkler_score(np.array([0.0]), lower, upper, 0.9)
    missed = winkler_score(y, lower, upper, 0.9)
    assert missed > inside


def test_summarize_samples_keys_and_shapes():
    rng = np.random.default_rng(0)
    actuals = rng.normal(size=(5, 24))
    samples = actuals[:, None, :] + rng.normal(scale=0.01, size=(5, 100, 24))
    out = summarize_samples(samples, actuals, [0.1, 0.5, 0.9])
    for key in ["mae", "rmse", "crps", "ece", "coverage@50", "coverage@90", "width@90"]:
        assert key in out
    # Tight samples around truth -> small CRPS/MAE.
    assert out["crps"] < 0.1
    assert isinstance(out["calibration_curve"], dict)
