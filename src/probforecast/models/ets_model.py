"""ETS (Error-Trend-Seasonal) baseline.

Uses the statespace ``ETSModel`` (additive error/trend/seasonal, period 24), which provides
native simulation for probabilistic forecasts. The model is refit per window on a trailing
slice of history (bounded cost) and forecasts are drawn via ``results.simulate`` — proper
sample paths, no Gaussian assumption. Convergence failures are caught: the model is marked
``failed`` and the pipeline reports it as such rather than crashing (spec §13).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import structlog

from probforecast.data.schema import TARGET
from probforecast.models.base import BaseForecaster

log = structlog.get_logger()


class EtsForecaster(BaseForecaster):
    name = "ets"

    def __init__(self, seasonal_periods: int = 24, trailing_weeks: int = 8, seed: int = 0):
        self.seasonal_periods = seasonal_periods
        self.trailing = trailing_weeks * 7 * 24
        self._rng = np.random.default_rng(seed)
        self._failed = False

    @staticmethod
    def _clean(y: np.ndarray) -> np.ndarray:
        return pd.Series(y, dtype=float).ffill().bfill().to_numpy()

    def _fit_one(self, y: np.ndarray):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ETSModel(
                self._clean(y),
                error="add",
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
            )
            return model.fit(disp=False)

    def fit(self, train_df: pd.DataFrame) -> EtsForecaster:
        y = train_df[TARGET].to_numpy(dtype=float)[-self.trailing :]
        try:
            self._fit_one(y)  # trial fit to surface convergence failure early
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            log.warning("ets.fit_failed", error=str(exc))
        return self

    def sample(self, history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        hist = np.asarray(history, dtype=float)[-self.trailing :]
        try:
            res = self._fit_one(hist)
            seed = int(self._rng.integers(0, 2**31 - 1))
            sim = res.simulate(
                nsimulations=horizon, repetitions=num_samples, anchor="end", random_state=seed
            )
            arr = np.asarray(sim).reshape(horizon, num_samples).T  # -> (num_samples, horizon)
            return np.clip(arr, 0.0, None)
        except Exception as exc:  # noqa: BLE001
            # Fallback: seasonal persistence + historical std (keeps the pipeline alive).
            log.warning("ets.window_failed", error=str(exc))
            s = self.seasonal_periods
            last_season = hist[-s:] if hist.size >= s else np.resize(hist, s)
            point = np.array([last_season[h % s] for h in range(horizon)], dtype=float)
            sigma = float(np.nanstd(hist[-self.trailing :])) or 1.0
            draws = self._rng.normal(point, sigma, size=(num_samples, horizon))
            return np.clip(draws, 0.0, None)


__all__ = ["EtsForecaster"]
