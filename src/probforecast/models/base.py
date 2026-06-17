"""Common forecaster interface + helpers shared by all baseline models.

Every model exposes the same two-method contract so the rolling-origin evaluation harness
(:mod:`probforecast.evaluation.rolling`) is model-agnostic and the comparison against DeepAR
(Phase 3) is apples-to-apples:

- ``fit(train_df)``: train on the training split.
- ``sample(history, horizon, num_samples)``: given all target values strictly before the
  forecast origin, return ``(num_samples, horizon)`` sample paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract sample-emitting forecaster."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> BaseForecaster:
        """Fit on the training split (DataFrame with a 'timestamp' column + target)."""

    @abstractmethod
    def sample(self, history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        """Return forecast samples of shape ``(num_samples, horizon)``."""

    @property
    def failed(self) -> bool:
        """True if the model failed to fit (e.g. ETS non-convergence)."""
        return getattr(self, "_failed", False)


def normal_samples(
    mean: np.ndarray,
    sigma: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    *,
    nonneg: bool = True,
) -> np.ndarray:
    """Draw ``(num_samples, horizon)`` Gaussian samples from per-step mean/sigma.

    PM2.5 is non-negative, so samples are clipped at 0 by default.
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    draws = rng.normal(loc=mean, scale=sigma, size=(num_samples, mean.shape[0]))
    if nonneg:
        draws = np.clip(draws, 0.0, None)
    return draws


__all__ = ["BaseForecaster", "normal_samples"]
