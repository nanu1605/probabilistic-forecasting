"""Strict temporal train/val/test split (no future leakage).

Cuts the processed frame by the configured date boundaries, asserts strict ordering and
no overlap, and writes ``data/splits/{train,val,test}.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from probforecast.config import Config, load_config

log = structlog.get_logger()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows with timestamp in [start 00:00, end 23:59:59]."""
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    mask = (df["timestamp"] >= lo) & (df["timestamp"] <= hi)
    return df.loc[mask].reset_index(drop=True)


def split(
    cfg: Config | None = None,
    *,
    df: pd.DataFrame | None = None,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Produce train/val/test splits, assert integrity, optionally write parquet, return dict."""
    cfg = cfg or load_config()
    if df is None:
        path = Path(cfg.data.processed_dir) / f"{cfg.data.station}.parquet"
        df = pd.read_parquet(path)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    s = cfg.data.split
    splits = {
        "train": _slice(df, s.train_start, s.train_end),
        "val": _slice(df, s.val_start, s.val_end),
        "test": _slice(df, s.test_start, s.test_end),
    }

    assert_no_leakage(splits)

    if write:
        out = Path(cfg.data.splits_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_parquet(out / f"{name}.parquet", index=False)
            log.info("split.written", name=name, rows=len(part))
    return splits


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Assert strict temporal ordering between splits and non-empty parts."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    for name, part in splits.items():
        if part.empty:
            raise ValueError(f"split '{name}' is empty — check split dates vs data range")
    train_max = tr["timestamp"].max()
    val_min, val_max = va["timestamp"].min(), va["timestamp"].max()
    test_min = te["timestamp"].min()
    if not (train_max < val_min < test_min):
        raise AssertionError(
            f"temporal leakage: max(train)={train_max}, min(val)={val_min}, min(test)={test_min}"
        )
    if not (val_max < test_min):
        raise AssertionError(f"temporal leakage: max(val)={val_max} >= min(test)={test_min}")


if __name__ == "__main__":
    split()
