"""Point + probabilistic forecasting metrics.

All models in this project emit **sample paths**, so the canonical input is an array of
forecast samples shaped ``(num_samples, horizon)`` per window (or stacked across windows).
Metrics reduce over the sample and horizon axes. ``crps_gaussian`` is also provided (and
known-answer tested) for the parametric dual-dispatch described in spec §7.

Array conventions
-----------------
- ``y_true``: observations, shape ``(..., horizon)`` or flattened ``(N,)``.
- ``samples``: forecast samples, shape ``(..., num_samples)`` with the **ensemble on the last
  axis** (matches ``properscoring.crps_ensemble``).
"""

from __future__ import annotations

import numpy as np
import properscoring as ps

ArrayLike = np.ndarray


# --------------------------------------------------------------------------- point metrics
def compute_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff**2)))


def point_from_samples(samples: ArrayLike, *, sample_axis: int = -1) -> np.ndarray:
    """Median point forecast across the sample axis."""
    return np.median(np.asarray(samples), axis=sample_axis)


# --------------------------------------------------------------------------- CRPS
def crps_ensemble(y_true: ArrayLike, samples: ArrayLike) -> float:
    """Mean CRPS for sample-based forecasts. ``samples`` ensemble on the last axis."""
    return float(np.mean(ps.crps_ensemble(np.asarray(y_true), np.asarray(samples))))


def crps_gaussian(y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> float:
    """Mean CRPS for a Gaussian predictive distribution (parametric dual-dispatch)."""
    return float(np.mean(ps.crps_gaussian(np.asarray(y_true), np.asarray(mu), np.asarray(sigma))))


# --------------------------------------------------------------------------- quantiles / intervals
def quantiles_from_samples(
    samples: ArrayLike, levels: list[float], *, sample_axis: int = -1
) -> np.ndarray:
    """Empirical quantiles at ``levels``; returns shape ``(len(levels), *reduced)``."""
    return np.quantile(np.asarray(samples), levels, axis=sample_axis)


def central_interval(
    samples: ArrayLike, coverage: float, *, sample_axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bounds of the central interval covering ``coverage`` of the mass."""
    lo_q = (1.0 - coverage) / 2.0
    hi_q = (1.0 + coverage) / 2.0
    lower = np.quantile(np.asarray(samples), lo_q, axis=sample_axis)
    upper = np.quantile(np.asarray(samples), hi_q, axis=sample_axis)
    return lower, upper


def compute_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of observations within [lower, upper]."""
    y = np.asarray(y_true)
    inside = (y >= np.asarray(lower)) & (y <= np.asarray(upper))
    return float(np.mean(inside))


def winkler_score(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, coverage: float) -> float:
    """Mean Winkler interval score for a central ``coverage`` interval.

    Width plus a miss penalty of ``(2/α) * distance_outside`` where ``α = 1 - coverage``.
    Rewards sharp intervals that still contain the truth.
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    alpha = 1.0 - coverage
    width = hi - lo
    below = lo - y
    above = y - hi
    penalty = np.where(y < lo, (2.0 / alpha) * below, 0.0)
    penalty += np.where(y > hi, (2.0 / alpha) * above, 0.0)
    return float(np.mean(width + penalty))


# --------------------------------------------------------------------------- calibration
def compute_calibration_curve(
    y_true: ArrayLike, samples: ArrayLike, levels: list[float], *, sample_axis: int = -1
) -> dict[float, float]:
    """Observed coverage of the central α interval for each α in ``levels``.

    ``samples`` ensemble on ``sample_axis``; ``y_true`` broadcasts against the reduced shape.
    Returns ``{predicted_coverage: observed_coverage}``.
    """
    y = np.asarray(y_true)
    out: dict[float, float] = {}
    for alpha in levels:
        lower, upper = central_interval(samples, alpha, sample_axis=sample_axis)
        out[float(alpha)] = compute_coverage(y, lower, upper)
    return out


def compute_ece(calibration_curve: dict[float, float]) -> float:
    """Expected Calibration Error: mean |observed - predicted| across levels."""
    if not calibration_curve:
        return float("nan")
    gaps = [abs(obs - pred) for pred, obs in calibration_curve.items()]
    return float(np.mean(gaps))


def mean_interval_width(samples: ArrayLike, coverage: float, *, sample_axis: int = -1) -> float:
    """Mean width of the central ``coverage`` interval (sharpness)."""
    lower, upper = central_interval(samples, coverage, sample_axis=sample_axis)
    return float(np.mean(upper - lower))


def summarize_samples(
    samples: ArrayLike,
    actuals: ArrayLike,
    coverage_levels: list[float],
    *,
    headline_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, float | dict]:
    """Compute the full metric suite from windowed sample forecasts.

    Args:
        samples: shape ``(num_windows, num_samples, horizon)``.
        actuals: shape ``(num_windows, horizon)``.
        coverage_levels: levels for the calibration curve / ECE.
        headline_levels: coverage levels reported individually (e.g. 50/80/90%).
    """
    samples = np.asarray(samples, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    # Move ensemble to the last axis and flatten windows*horizon.
    whs = np.moveaxis(samples, 1, -1)  # (W, H, S)
    n_w, n_h, n_s = whs.shape
    flat_samples = whs.reshape(n_w * n_h, n_s)
    flat_actuals = actuals.reshape(n_w * n_h)

    point = np.median(flat_samples, axis=-1)
    curve = compute_calibration_curve(flat_actuals, flat_samples, coverage_levels)

    out: dict[str, float | dict] = {
        "mae": compute_mae(flat_actuals, point),
        "rmse": compute_rmse(flat_actuals, point),
        "crps": crps_ensemble(flat_actuals, flat_samples),
        "ece": compute_ece(curve),
        "calibration_curve": {str(k): v for k, v in curve.items()},
    }
    for lvl in headline_levels:
        lower, upper = central_interval(flat_samples, lvl)
        tag = int(round(lvl * 100))
        out[f"coverage@{tag}"] = compute_coverage(flat_actuals, lower, upper)
        out[f"winkler@{tag}"] = winkler_score(flat_actuals, lower, upper, lvl)
        out[f"width@{tag}"] = float(np.mean(upper - lower))
    return out


__all__ = [
    "compute_mae",
    "compute_rmse",
    "point_from_samples",
    "crps_ensemble",
    "crps_gaussian",
    "quantiles_from_samples",
    "central_interval",
    "compute_coverage",
    "winkler_score",
    "compute_calibration_curve",
    "compute_ece",
    "mean_interval_width",
    "summarize_samples",
]
