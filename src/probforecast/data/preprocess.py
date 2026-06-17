"""Clean, impute, and feature-engineer the air-quality series.

Produces one continuous hourly frame (regular ``freq="h"`` index, no rows dropped) so it is
GluonTS-ready. Target gaps ≤ ``ffill_max_gap_hours`` are forward-filled; longer gaps stay NaN
and are excluded from evaluation downstream. All engineered features are **strictly backward
-looking** (lags + rolling windows shifted by one step), so computing them on the full series
before the temporal split introduces no future leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from probforecast.config import Config, load_config
from probforecast.data.download import load_raw
from probforecast.data.schema import COVARIATES, TARGET

log = structlog.get_logger()

# 16-point compass -> degrees, for cyclical wind-direction encoding.
_COMPASS_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}  # fmt: skip


def _build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df


def _to_continuous_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicate timestamps, reindex onto a gap-free hourly index."""
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full)
    df.index.name = "timestamp"
    return df


def _impute(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Forward-fill target gaps whose full length <= max_gap; longer gaps stay entirely NaN.

    Covariates are exogenous and time-interpolated so engineered features stay non-null.
    """
    df = df.copy()
    na = df[TARGET].isna()
    # Label consecutive runs; the size of each NaN run is its gap length in hours.
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    filled = df[TARGET].ffill()
    long_gap = na & (run_len > max_gap)
    df[TARGET] = filled.where(~long_gap, np.nan)

    present = [c for c in COVARIATES if c in df.columns]
    df[present] = df[present].interpolate("time", limit_direction="both")
    return df


def _add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    for name, period, vals in [
        ("hour", 24, idx.hour),
        ("dow", 7, idx.dayofweek),
        ("month", 12, idx.month - 1),
    ]:
        ang = 2 * np.pi * np.asarray(vals) / period
        df[f"{name}_sin"] = np.sin(ang)
        df[f"{name}_cos"] = np.cos(ang)
    if "wd" in df.columns:
        deg = df["wd"].map(_COMPASS_DEG)
        rad = np.deg2rad(deg.astype(float))
        df["wd_sin"] = np.sin(rad)
        df["wd_cos"] = np.cos(rad)
    return df


def _add_lags_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Lags (past values) + 24h rolling stats shifted one step (strictly historical)."""
    df = df.copy()
    df["lag_1"] = df[TARGET].shift(1)
    df["lag_24"] = df[TARGET].shift(24)
    df["lag_168"] = df[TARGET].shift(168)
    roll = df[TARGET].rolling(window=24, min_periods=1)
    df["roll_mean_24"] = roll.mean().shift(1)
    df["roll_std_24"] = roll.std().shift(1)
    return df


def preprocess(cfg: Config | None = None, *, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the full preprocessing pipeline; write processed parquet; return the frame."""
    cfg = cfg or load_config()
    df = raw if raw is not None else load_raw(cfg)

    df = _build_timestamp(df)
    df = _to_continuous_hourly(df)

    raw_target_missing = float(df[TARGET].isna().mean())

    df = _impute(df, cfg.data.ffill_max_gap_hours)
    df = _add_cyclical(df)
    df = _add_lags_rolling(df)

    post_target_missing = float(df[TARGET].isna().mean())
    log.info(
        "preprocess.missingness",
        target_missing_raw=round(raw_target_missing, 4),
        target_missing_post_impute=round(post_target_missing, 4),
        rows=len(df),
    )

    df = df.reset_index()  # timestamp back to a column
    out = Path(cfg.data.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.data.station}.parquet"
    df.to_parquet(path, index=False)
    log.info("preprocess.written", path=str(path), cols=len(df.columns))
    return df


if __name__ == "__main__":
    preprocess()
