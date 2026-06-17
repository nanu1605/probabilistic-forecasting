"""DeepAR tests: seeding reproducibility (fast) + train/predict/determinism (slow, CPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probforecast.config import load_config
from probforecast.data.schema import TARGET
from probforecast.utils.reproducibility import set_all_seeds


def _tiny_df(n: int = 400) -> pd.DataFrame:
    t = np.arange(n)
    y = 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(0).normal(0, 2, n)
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="h"), TARGET: y})


def _tiny_model(epochs: int = 2):
    from probforecast.models.deepar_model import DeepARForecaster

    cfg = load_config()
    params = dict(cfg.model_params["deepar"])
    return DeepARForecaster(params, accelerator="cpu", epochs=epochs, deterministic=True)


def test_set_all_seeds_reproducible():
    import torch

    set_all_seeds(7)
    a = torch.randn(5)
    set_all_seeds(7)
    b = torch.randn(5)
    assert torch.equal(a, b)


@pytest.mark.slow
def test_deepar_train_predict_shape():
    df = _tiny_df()
    set_all_seeds(42)
    model = _tiny_model().fit(df)
    s = model.sample(df[TARGET].to_numpy(), horizon=24, num_samples=100)
    assert s.shape == (100, 24)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)  # nonnegative_pred_samples


@pytest.mark.slow
def test_deepar_quantiles_ordered_and_crps_finite():
    from probforecast.evaluation.metrics import crps_ensemble

    df = _tiny_df()
    set_all_seeds(42)
    model = _tiny_model().fit(df)
    s = model.sample(df[TARGET].to_numpy(), horizon=24, num_samples=100)
    q05, q50, q95 = np.quantile(s, [0.05, 0.5, 0.95], axis=0)
    assert np.all(q05 <= q50) and np.all(q50 <= q95)
    # CRPS against an arbitrary horizon of truths.
    crps = crps_ensemble(df[TARGET].to_numpy()[-24:], np.moveaxis(s, 0, -1))
    assert np.isfinite(crps) and crps >= 0.0


@pytest.mark.slow
def test_deepar_same_seed_deterministic_cpu():
    df = _tiny_df()
    hist = df[TARGET].to_numpy()

    def run(seed: int) -> np.ndarray:
        set_all_seeds(seed)
        m = _tiny_model().fit(df)
        set_all_seeds(seed)  # align RNG state before sampling
        return m.sample(hist, horizon=24, num_samples=50)

    np.testing.assert_allclose(run(42), run(42))
    # Different seed -> different samples.
    assert not np.allclose(run(42), run(123))
