"""Known-answer tests for preprocessing: gap handling, lags, rolling, cyclical encodings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from probforecast.data.preprocess import (
    _add_cyclical,
    _add_lags_rolling,
    _impute,
    _to_continuous_hourly,
)
from probforecast.data.schema import TARGET


def _continuous_frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({TARGET: np.arange(n, dtype=float)}, index=idx)


def test_short_gap_filled_long_gap_nan():
    df = _continuous_frame(60)
    df.loc[df.index[10:14], TARGET] = np.nan  # 4h gap (<= 6) -> filled
    df.loc[df.index[20:35], TARGET] = np.nan  # 15h gap (> 6) -> entirely NaN

    out = _impute(df, max_gap=6)

    # Short gap forward-filled with the last valid value (index 9 -> 9.0).
    assert out[TARGET].iloc[10:14].notna().all()
    assert (out[TARGET].iloc[10:14] == 9.0).all()
    # Long gap left entirely NaN (no partial fill).
    assert out[TARGET].iloc[20:35].isna().all()


def test_reindex_fills_missing_hours():
    idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 04:00"])
    df = pd.DataFrame({"timestamp": idx, TARGET: [1.0, 2.0, 5.0]})
    out = _to_continuous_hourly(df)
    # 02:00 and 03:00 inserted -> 5 contiguous hourly rows.
    assert len(out) == 5
    assert out.index.freq == "h" or (out.index[1] - out.index[0]) == pd.Timedelta(hours=1)
    assert out[TARGET].isna().sum() == 2


def test_lags_are_past_values():
    df = _continuous_frame(200)
    out = _add_lags_rolling(df)
    # lag_k at row i equals target at row i-k.
    assert out["lag_1"].iloc[50] == df[TARGET].iloc[49]
    assert out["lag_24"].iloc[50] == df[TARGET].iloc[26]
    assert out["lag_168"].iloc[180] == df[TARGET].iloc[12]


def test_rolling_is_strictly_historical():
    df = _continuous_frame(200)
    out = _add_lags_rolling(df)
    # roll_mean_24 at row i is the mean of the 24 values ending at i-1 (excludes current).
    expected = df[TARGET].iloc[26:50].mean()
    assert np.isclose(out["roll_mean_24"].iloc[50], expected)
    # Current target must not leak into its own rolling feature.
    assert not np.isclose(out["roll_mean_24"].iloc[50], df[TARGET].iloc[27:51].mean())


def test_cyclical_encoding_known_values():
    df = _continuous_frame(48)  # starts at midnight
    out = _add_cyclical(df)
    # Hour 0 -> sin 0, cos 1.
    assert np.isclose(out["hour_sin"].iloc[0], 0.0, atol=1e-9)
    assert np.isclose(out["hour_cos"].iloc[0], 1.0, atol=1e-9)
    # Encodings bounded in [-1, 1].
    for col in ["hour_sin", "hour_cos", "dow_sin", "month_cos"]:
        assert out[col].abs().max() <= 1.0 + 1e-9
