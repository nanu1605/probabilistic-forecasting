"""ARIMA baseline.

Order is selected with ``pmdarima.auto_arima`` (seasonal, ``m=24``) on a recent slice of the
training data; a ``statsmodels`` SARIMAX with that order is then fit and **walked forward**
across the test set via ``results.append`` (cheap state extension, no re-estimation). Each
window's Gaussian forecast (mean + standard error) is turned into sample paths.

Seasonal SARIMA with period 24 is expensive, so the initial fit uses the last
``arima.max_train_weeks`` of training data (documented trade-off, spec §13).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import structlog

from probforecast.data.schema import TARGET
from probforecast.models.base import BaseForecaster, normal_samples

log = structlog.get_logger()


class ArimaForecaster(BaseForecaster):
    name = "arima"

    def __init__(
        self,
        m: int = 24,
        max_train_weeks: int = 4,
        fallback_order: tuple[int, int, int] = (2, 1, 2),
        fallback_seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
        seed: int = 0,
    ):
        self.m = m
        self.max_train_weeks = max_train_weeks
        self.fallback_order = fallback_order
        self.fallback_seasonal_order = fallback_seasonal_order
        self._rng = np.random.default_rng(seed)
        self._res = None
        self._n_seen = 0

    @staticmethod
    def _clean(y: np.ndarray) -> np.ndarray:
        s = pd.Series(y, dtype=float).ffill().bfill()
        return s.to_numpy()

    def _select_order(self, y: np.ndarray):
        import pmdarima as pm

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = pm.auto_arima(
                    y,
                    seasonal=True,
                    m=self.m,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=3,
                    max_q=3,
                    max_P=2,
                    max_Q=2,  # fmt: skip
                )
            log.info("arima.order", order=model.order, seasonal_order=model.seasonal_order)
            return model.order, model.seasonal_order
        except Exception as exc:  # noqa: BLE001
            log.warning("arima.auto_failed", error=str(exc))
            return self.fallback_order, self.fallback_seasonal_order

    def fit(self, train_df: pd.DataFrame) -> ArimaForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_full = train_df[TARGET].to_numpy(dtype=float)
        n_init = min(len(y_full), self.max_train_weeks * 7 * 24)
        y = self._clean(y_full[-n_init:])

        order, seasonal_order = self._select_order(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._res = model.fit(disp=False)
        self._n_seen = len(y_full)
        self._order = order
        self._seasonal_order = seasonal_order
        return self

    def sample(self, history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        # Append any newly revealed actuals (between the last origin and this one).
        new = np.asarray(history, dtype=float)[self._n_seen :]
        if new.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._res = self._res.append(self._clean(new), refit=False)
            self._n_seen += new.size

        fc = self._res.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        se = np.asarray(fc.se_mean, dtype=float)
        return normal_samples(mean, se, num_samples, self._rng)


__all__ = ["ArimaForecaster"]
