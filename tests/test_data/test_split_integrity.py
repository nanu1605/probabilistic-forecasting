"""Temporal split integrity: no leakage, no overlap, rows reconcile, columns preserved."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probforecast.data.schema import TARGET
from probforecast.data.split import assert_no_leakage, split


def _processed_frame(config) -> pd.DataFrame:
    """A continuous hourly frame spanning all three split date ranges."""
    s = config.data.split
    idx = pd.date_range(s.train_start, pd.Timestamp(s.test_end) + pd.Timedelta(hours=23), freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": idx,
            TARGET: rng.uniform(0, 100, len(idx)),
            "feat_a": rng.normal(size=len(idx)),
        }
    )


def test_split_strict_ordering(config):
    df = _processed_frame(config)
    parts = split(config, df=df, write=False)
    assert parts["train"]["timestamp"].max() < parts["val"]["timestamp"].min()
    assert parts["val"]["timestamp"].max() < parts["test"]["timestamp"].min()


def test_split_no_row_overlap(config):
    df = _processed_frame(config)
    parts = split(config, df=df, write=False)
    ts = {k: set(v["timestamp"]) for k, v in parts.items()}
    assert ts["train"].isdisjoint(ts["val"])
    assert ts["val"].isdisjoint(ts["test"])
    assert ts["train"].isdisjoint(ts["test"])


def test_split_columns_preserved(config):
    df = _processed_frame(config)
    parts = split(config, df=df, write=False)
    for part in parts.values():
        assert list(part.columns) == list(df.columns)


def test_assert_no_leakage_raises_on_overlap(config):
    s = config.data.split
    # Build val that starts before train ends -> must raise.
    train = pd.DataFrame({"timestamp": pd.date_range(s.train_start, s.train_end, freq="h")})
    bad_val = pd.DataFrame(
        {"timestamp": pd.date_range(s.train_start, s.val_end, freq="h")}  # overlaps train
    )
    test = pd.DataFrame({"timestamp": pd.date_range(s.test_start, s.test_end, freq="h")})
    with pytest.raises((AssertionError, ValueError)):
        assert_no_leakage({"train": train, "val": bad_val, "test": test})


def test_assert_no_leakage_raises_on_empty(config):
    s = config.data.split
    train = pd.DataFrame({"timestamp": pd.date_range(s.train_start, s.train_end, freq="h")})
    val = pd.DataFrame({"timestamp": pd.date_range(s.val_start, s.val_end, freq="h")})
    empty = pd.DataFrame({"timestamp": pd.to_datetime([])})
    with pytest.raises(ValueError, match="empty"):
        assert_no_leakage({"train": train, "val": val, "test": empty})
