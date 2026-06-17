"""Seasonal Naive baseline.

Point forecast repeats the value from exactly one season (``season_length`` hours) ago.
Predictive samples come from bootstrapping the empirical seasonal-difference residuals
``e_t = y_t - y_{t-season}`` measured on the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from probforecast.data.schema import TARGET
from probforecast.models.base import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    name = "seasonal_naive"

    def __init__(self, season_length: int = 24, seed: int = 0):
        self.season_length = season_length
        self._rng = np.random.default_rng(seed)
        self._residuals: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> SeasonalNaiveForecaster:
        y = train_df[TARGET].to_numpy(dtype=float)
        s = self.season_length
        diff = y[s:] - y[:-s]
        self._residuals = diff[np.isfinite(diff)]
        if self._residuals.size == 0:
            self._residuals = np.zeros(1)
        return self

    def sample(self, history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        s = self.season_length
        hist = pd.Series(np.asarray(history, dtype=float)).ffill().bfill().to_numpy()
        last_season = hist[-s:] if hist.size >= s else np.resize(hist, s)
        # Repeat the last season across the horizon.
        point = np.array([last_season[h % s] for h in range(horizon)], dtype=float)
        resid = self._rng.choice(self._residuals, size=(num_samples, horizon), replace=True)
        draws = point[None, :] + resid
        return np.clip(draws, 0.0, None)


__all__ = ["SeasonalNaiveForecaster"]
