"""Schema validation: valid splits pass; corrupt data is rejected."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from probforecast.data.schema import TARGET, validate_split


def _valid_frame(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": idx,
            TARGET: rng.uniform(0, 100, n),
            "TEMP": rng.uniform(-10, 30, n),
            "PRES": rng.uniform(990, 1030, n),
        }
    )


def test_valid_frame_passes():
    df = _valid_frame()
    validate_split(df)  # should not raise


def test_negative_target_rejected():
    df = _valid_frame()
    df.loc[5, TARGET] = -1.0
    with pytest.raises(SchemaError):
        validate_split(df)


def test_null_target_rejected_when_required():
    df = _valid_frame()
    df.loc[5, TARGET] = np.nan
    with pytest.raises(SchemaError):
        validate_split(df, require_target_non_null=True)


def test_null_target_allowed_when_not_required():
    df = _valid_frame()
    df.loc[5, TARGET] = np.nan
    validate_split(df, require_target_non_null=False)  # should not raise


def test_non_monotonic_timestamp_rejected():
    df = _valid_frame()
    ts = df["timestamp"].to_numpy()
    ts[10], ts[11] = ts[11], ts[10]  # swap -> not monotonic increasing
    df["timestamp"] = ts
    with pytest.raises(SchemaError):
        validate_split(df)


def test_temperature_out_of_range_rejected():
    df = _valid_frame()
    df.loc[3, "TEMP"] = 999.0
    with pytest.raises(SchemaError):
        validate_split(df)
