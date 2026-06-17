"""Phase 0 smoke tests: package imports, config loads and validates."""

from __future__ import annotations

import probforecast
from probforecast.config import Config


def test_package_version():
    assert probforecast.__version__ == "0.1.0"


def test_config_loads(config: Config):
    assert isinstance(config, Config)
    # Core horizons present and sane.
    assert config.forecast.context_length == 168
    assert config.forecast.prediction_length == 24
    assert config.forecast.num_samples == 100


def test_split_dates_ordered(config: Config):
    s = config.data.split
    assert s.train_end < s.val_start
    assert s.val_end < s.test_start


def test_five_seeds(config: Config):
    assert len(config.reproducibility.seeds) == 5


def test_model_params_present(config: Config):
    assert config.model_params["deepar"]["context_length"] == 168
    assert config.model_params["deepar"]["distr_output"] == "StudentT"


def test_small_series_fixture(small_series):
    assert (small_series["value"] >= 0).all()
    assert small_series["timestamp"].is_monotonic_increasing
    assert len(small_series) == 240
