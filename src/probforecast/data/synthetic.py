"""Synthetic air-quality generator (fallback when the UCI dataset is unreachable).

Emits a DataFrame in the canonical raw layout (:data:`probforecast.data.schema.RAW_COLUMNS`)
so the rest of the pipeline is identical whether data is real or synthetic. The series has
daily + weekly seasonality, a slow trend, **heteroscedastic** noise (time-varying variance —
important for the calibration experiments), and covariates (two correlated with the target,
one pure noise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from probforecast.config import Config
from probforecast.data.schema import RAW_COLUMNS

_WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]  # fmt: skip

# Fixed seed → reproducible synthetic data (no wall-clock randomness).
_SYNTH_SEED = 12345


def generate_synthetic(cfg: Config) -> pd.DataFrame:
    """Generate a synthetic hourly series spanning the configured split date range."""
    rng = np.random.default_rng(_SYNTH_SEED)

    start = pd.Timestamp(cfg.data.split.train_start)
    end = pd.Timestamp(cfg.data.split.test_end) + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    t = np.arange(n, dtype=float)

    # Seasonal structure.
    daily = 18.0 * np.sin(2 * np.pi * t / 24.0)
    weekly = 8.0 * np.sin(2 * np.pi * t / 168.0)
    yearly = 25.0 * np.sin(2 * np.pi * t / (24.0 * 365.25))
    trend = 0.0008 * t

    # Two covariates correlated with the target signal, plus one noise covariate.
    temp = 12.0 + 14.0 * np.sin(2 * np.pi * (t / (24.0 * 365.25)) - 0.4) + rng.normal(0, 2.0, n)
    dewp = temp - 8.0 + rng.normal(0, 1.5, n)  # correlated with temp/target
    rain = rng.gamma(0.05, 2.0, n)  # noise-like covariate

    # Heteroscedastic noise: variance higher in winter (anti-correlated with yearly term).
    sigma = 8.0 + 6.0 * (1.0 + np.sin(2 * np.pi * t / (24.0 * 365.25) + np.pi)) / 2.0
    noise = rng.normal(0.0, 1.0, n) * sigma

    base = 70.0 + yearly + daily + weekly + trend
    # Couple covariates into the target so they carry signal.
    target = base - 0.6 * (temp - temp.mean()) + 0.3 * (dewp - dewp.mean()) + noise
    target = np.clip(target, 0.0, None)

    df = pd.DataFrame(
        {
            "No": np.arange(1, n + 1),
            "year": idx.year,
            "month": idx.month,
            "day": idx.day,
            "hour": idx.hour,
            "PM2.5": target,
            "PM10": np.clip(target * 1.4 + rng.normal(0, 10, n), 0, None),
            "SO2": np.clip(15.0 + rng.normal(0, 8, n), 0, None),
            "NO2": np.clip(40.0 + 0.2 * target + rng.normal(0, 10, n), 0, None),
            "CO": np.clip(900.0 + 3.0 * target + rng.normal(0, 200, n), 0, None),
            "O3": np.clip(55.0 - 0.1 * target + rng.normal(0, 15, n), 0, None),
            "TEMP": temp,
            "PRES": 1012.0 + rng.normal(0, 8, n),
            "DEWP": dewp,
            "RAIN": rain,
            "wd": rng.choice(_WIND_DIRS, size=n),
            "WSPM": np.clip(1.8 + rng.gamma(2.0, 0.6, n), 0, None),
            "station": cfg.data.station,
        }
    )

    # Inject a few realistic gaps in the target: a short one (<=6h, imputable) and a long one.
    if n > 1000:
        df.loc[300:303, "PM2.5"] = np.nan  # 4-hour gap -> forward-fillable
        df.loc[500:530, "PM2.5"] = np.nan  # 31-hour gap -> excluded

    return df[RAW_COLUMNS]


__all__ = ["generate_synthetic"]
