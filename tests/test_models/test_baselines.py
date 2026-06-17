"""Baseline model tests: shapes, ordering, seasonal correctness, failure handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probforecast.data.schema import TARGET
from probforecast.models.ets_model import EtsForecaster
from probforecast.models.seasonal_naive import SeasonalNaiveForecaster


def _periodic_df(n: int = 480, period: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    t = np.arange(n)
    y = 50 + 20 * np.sin(2 * np.pi * t / period)
    return pd.DataFrame({"timestamp": idx, TARGET: y})


def test_seasonal_naive_reproduces_prior_period():
    df = _periodic_df()
    model = SeasonalNaiveForecaster(season_length=24).fit(df)
    history = df[TARGET].to_numpy()
    s = model.sample(history, horizon=24, num_samples=200)
    assert s.shape == (200, 24)
    # Median forecast should match the previous period (clean signal -> tiny residuals).
    median = np.median(s, axis=0)
    np.testing.assert_allclose(median, history[-24:], atol=2.0)


def test_seasonal_naive_quantiles_ordered():
    df = _periodic_df()
    model = SeasonalNaiveForecaster().fit(df)
    s = model.sample(df[TARGET].to_numpy(), horizon=24, num_samples=300)
    q05, q50, q95 = np.quantile(s, [0.05, 0.5, 0.95], axis=0)
    assert np.all(q05 <= q50) and np.all(q50 <= q95)


def test_seasonal_naive_handles_nan_history():
    df = _periodic_df()
    hist = df[TARGET].to_numpy().copy()
    hist[-3:] = np.nan  # gap right before origin
    model = SeasonalNaiveForecaster().fit(df)
    s = model.sample(hist, horizon=24, num_samples=100)
    assert np.all(np.isfinite(s))


def test_ets_sample_shape():
    df = _periodic_df(n=400)
    model = EtsForecaster(seasonal_periods=24, trailing_weeks=2).fit(df)
    s = model.sample(df[TARGET].to_numpy(), horizon=24, num_samples=100)
    assert s.shape == (100, 24)
    assert np.all(np.isfinite(s))


def test_ets_failure_is_marked_not_raised():
    # seasonal_periods larger than the data -> fit cannot succeed; must be caught.
    df = _periodic_df(n=40)
    model = EtsForecaster(seasonal_periods=24, trailing_weeks=1).fit(df.head(30))
    assert model.failed is True


@pytest.mark.slow
def test_arima_fit_predict_shape():
    from probforecast.models.arima_model import ArimaForecaster

    df = _periodic_df(n=400)
    model = ArimaForecaster(m=24, max_train_weeks=2).fit(df)
    s = model.sample(df[TARGET].to_numpy(), horizon=24, num_samples=100)
    assert s.shape == (100, 24)
    assert np.all(np.isfinite(s))
