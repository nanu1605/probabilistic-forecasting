"""Post-hoc recalibration of sample-based probabilistic forecasts.

Primary method is Kuleshov et al. (2018) isotonic recalibration. On a held-out calibration
set we measure how the model's predictive CDF maps to observed frequencies, fit a monotonic
correction ``R``, and at test time resample each forecast through the inverse map so the
recalibrated predictive distribution is (marginally) calibrated.

The model **never sees test labels**: :meth:`RecalibrationModel.fit` takes only the calibration
(validation) samples + actuals; :meth:`transform` takes only test samples.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

_GRID = np.linspace(0.0, 1.0, 1001)


def _flatten(samples: np.ndarray) -> np.ndarray:
    """(W, S, H) -> (W*H, S): ensemble on the last axis."""
    whs = np.moveaxis(np.asarray(samples, dtype=float), 1, -1)
    n_w, n_h, n_s = whs.shape
    return whs.reshape(n_w * n_h, n_s)


class RecalibrationModel:
    """Marginal post-hoc recalibration. method ∈ {"isotonic", "scaling"}."""

    def __init__(self, method: str = "isotonic", seed: int = 0, nonneg: bool = True):
        self.method = method
        self.nonneg = nonneg  # clip recalibrated samples at 0 (PM2.5 is non-negative)
        self._rng = np.random.default_rng(seed)
        self._r_grid: np.ndarray | None = None  # R evaluated on _GRID (isotonic)
        self._scale: float | None = None  # inflation factor (scaling)

    # ------------------------------------------------------------------ fit (val only)
    def fit(self, val_samples: np.ndarray, val_actuals: np.ndarray) -> RecalibrationModel:
        flat_s = _flatten(val_samples)  # (N, S)
        flat_y = np.asarray(val_actuals, dtype=float).reshape(-1)  # (N,)
        if self.method == "scaling":
            self._fit_scaling(flat_s, flat_y)
        else:
            self._fit_isotonic(flat_s, flat_y)
        return self

    def _fit_isotonic(self, flat_s: np.ndarray, flat_y: np.ndarray) -> None:
        # PIT values: predicted CDF at the truth = fraction of samples <= y.
        pit = np.mean(flat_s <= flat_y[:, None], axis=1)  # (N,)
        # Empirical CDF of the PIT values, evaluated at each PIT point.
        ecdf = stats.rankdata(pit, method="max") / pit.size
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso.fit(pit, ecdf)
        self._r_grid = np.clip(iso.predict(_GRID), 0.0, 1.0)

    def _fit_scaling(self, flat_s: np.ndarray, flat_y: np.ndarray) -> None:
        # Inflate deviations about the per-row median so val 90% coverage hits 0.90.
        med = np.median(flat_s, axis=1, keepdims=True)

        def cov_at(c: float) -> float:
            infl = med + c * (flat_s - med)
            lo = np.quantile(infl, 0.05, axis=1)
            hi = np.quantile(infl, 0.95, axis=1)
            return float(np.mean((flat_y >= lo) & (flat_y <= hi)))

        cands = np.linspace(0.5, 5.0, 91)
        errs = [abs(cov_at(c) - 0.90) for c in cands]
        self._scale = float(cands[int(np.argmin(errs))])

    # ------------------------------------------------------------------ transform (no labels)
    def transform(
        self, test_samples: np.ndarray, num_samples: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Recalibrated samples ``(W, num_samples, H)``. Takes no labels — leakage-safe."""
        rng = rng or self._rng
        samples = np.asarray(test_samples, dtype=float)
        n_w, _, n_h = samples.shape
        out = np.empty((n_w, num_samples, n_h), dtype=float)
        for w in range(n_w):
            for h in range(n_h):
                col = samples[w, :, h]
                if self.method == "scaling":
                    med = np.median(col)
                    infl = med + self._scale * (col - med)
                    out[w, :, h] = rng.choice(infl, size=num_samples, replace=True)
                else:
                    u = rng.uniform(size=num_samples)
                    lvl = self._r_inv(u)  # F^{-1}(R^{-1}(u))
                    out[w, :, h] = np.quantile(col, lvl)
        return np.clip(out, 0.0, None) if self.nonneg else out

    def _r_inv(self, u: np.ndarray) -> np.ndarray:
        """Inverse calibration map: given target coverage u, return the model quantile level."""
        # R maps _GRID -> _r_grid (ascending). Invert by interpolating u against _r_grid.
        return np.interp(u, self._r_grid, _GRID)


__all__ = ["RecalibrationModel"]
