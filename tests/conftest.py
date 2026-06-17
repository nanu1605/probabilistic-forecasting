"""Shared pytest fixtures: small synthetic data + loaded config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probforecast.config import load_config


@pytest.fixture(scope="session")
def config():
    """Project config loaded from configs/."""
    return load_config()


@pytest.fixture
def rng():
    """Deterministic numpy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def small_series(rng) -> pd.DataFrame:
    """A tiny hourly series (~10 days) with daily seasonality, for fast unit tests.

    Columns: timestamp (hourly, monotonic), value (>= 0), one covariate.
    """
    n = 240  # 10 days hourly
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    hours = np.arange(n)
    daily = 10.0 * np.sin(2 * np.pi * hours / 24.0)
    trend = 0.01 * hours
    noise = rng.normal(0, 1.0, size=n)
    value = np.clip(50.0 + daily + trend + noise, 0, None)
    covariate = daily + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"timestamp": idx, "value": value, "covariate": covariate})
