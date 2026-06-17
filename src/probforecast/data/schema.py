"""Canonical column constants + pandera schemas for the air-quality pipeline.

Both the real-UCI path and the synthetic fallback emit the *same* raw column set
(``RAW_COLUMNS``) so all downstream code is source-agnostic. After preprocessing,
each split is validated with :func:`validate_split`.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# --- Canonical raw layout (mirrors UCI Beijing Multi-Site per-station CSVs) ---
TARGET = "PM2.5"

# Numeric covariates passed through preprocessing (meteorology + co-pollutants).
COVARIATES = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
]

# Raw CSV columns as shipped by UCI (and reproduced by the synthetic generator).
RAW_COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",  # wind direction, 16-point compass (categorical)
    "WSPM",
    "station",
]

# Engineered feature columns produced by preprocess.py (cyclical encodings, lags, rolling).
CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "wd_sin",
    "wd_cos",
]
LAG_FEATURES = ["lag_1", "lag_24", "lag_168"]
ROLLING_FEATURES = ["roll_mean_24", "roll_std_24"]


def split_schema(*, require_target_non_null: bool = True) -> DataFrameSchema:
    """Pandera schema for a processed split.

    Args:
        require_target_non_null: when True, the target column must have no nulls
            (used for the eval-ready rows). Set False to validate frames that may
            still contain long-gap NaN targets.
    """
    columns: dict[str, Column] = {
        "timestamp": Column(
            "datetime64[ns]",
            nullable=False,
            # Strictly increasing within a split (no duplicate/backward timestamps).
            checks=Check(
                lambda s: s.is_monotonic_increasing and s.is_unique,
                error="timestamp must be strictly monotonic increasing",
            ),
        ),
        TARGET: Column(
            float,
            nullable=not require_target_non_null,
            checks=Check.ge(0.0, error="PM2.5 must be >= 0"),
        ),
        "TEMP": Column(float, nullable=True, checks=Check.in_range(-40.0, 50.0)),
        "PRES": Column(float, nullable=True, checks=Check.in_range(800.0, 1100.0)),
    }
    return DataFrameSchema(columns, strict=False, coerce=True)


def validate_split(df: pd.DataFrame, *, require_target_non_null: bool = True) -> pd.DataFrame:
    """Validate a processed split; raises ``pandera.errors.SchemaError`` on failure."""
    schema = split_schema(require_target_non_null=require_target_non_null)
    return schema.validate(df)


__all__ = [
    "TARGET",
    "COVARIATES",
    "RAW_COLUMNS",
    "CYCLICAL_FEATURES",
    "LAG_FEATURES",
    "ROLLING_FEATURES",
    "split_schema",
    "validate_split",
    "pa",
]
