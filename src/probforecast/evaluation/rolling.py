"""Shared rolling-origin evaluation harness.

Non-overlapping ``horizon``-step windows tile the test set. For window ``k`` the forecast
origin is ``len(train) + k*horizon`` in the concatenated train+test target series, and the
model only sees values strictly before that origin. Windows whose actuals contain any NaN
(long data gaps) are excluded. The same harness is reused for DeepAR (Phase 3) so every
model is evaluated under an identical protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from probforecast.data.schema import TARGET
from probforecast.models.base import BaseForecaster


@dataclass
class RollingResult:
    samples: np.ndarray  # (num_windows, num_samples, horizon)
    actuals: np.ndarray  # (num_windows, horizon)
    n_windows: int
    n_dropped: int


def rolling_forecast(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    num_samples: int,
    *,
    fit: bool = True,
) -> RollingResult:
    """Fit (optionally) then produce sample forecasts over non-overlapping test windows."""
    if fit:
        model.fit(train_df)

    train_y = train_df[TARGET].to_numpy(dtype=float)
    test_y = test_df[TARGET].to_numpy(dtype=float)
    full = np.concatenate([train_y, test_y])
    train_len = len(train_y)
    n_test = len(test_y)

    samples_list: list[np.ndarray] = []
    actuals_list: list[np.ndarray] = []
    n_dropped = 0

    n_windows = n_test // horizon
    for k in range(n_windows):
        origin = train_len + k * horizon
        actual = full[origin : origin + horizon]
        if actual.shape[0] < horizon or not np.all(np.isfinite(actual)):
            n_dropped += 1
            continue
        history = full[:origin]
        s = model.sample(history, horizon, num_samples)
        samples_list.append(np.asarray(s, dtype=float))
        actuals_list.append(actual)

    if not samples_list:
        raise RuntimeError(f"no valid windows for model '{model.name}' (all dropped)")

    return RollingResult(
        samples=np.stack(samples_list),
        actuals=np.stack(actuals_list),
        n_windows=len(samples_list),
        n_dropped=n_dropped,
    )


__all__ = ["RollingResult", "rolling_forecast"]
