"""Configuration loading via pydantic-settings.

Single source of truth for paths, horizons, thresholds. No hardcoded values elsewhere.
Load with ``load_config()`` (uses ``configs/config.yaml`` + ``configs/model_params.yaml``
by default). Override the config directory with the ``PROBFORECAST_CONFIG_DIR`` env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    # src/probforecast/config.py -> project root is two parents up from src/.
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    override = os.environ.get("PROBFORECAST_CONFIG_DIR")
    if override:
        return Path(override)
    return _project_root() / "configs"


class SplitConfig(BaseModel):
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


class DataConfig(BaseModel):
    station: str
    target: str
    raw_dir: str
    processed_dir: str
    splits_dir: str
    uci_url: str
    use_synthetic_fallback: bool
    ffill_max_gap_hours: int
    split: SplitConfig


class ForecastConfig(BaseModel):
    context_length: int
    prediction_length: int
    num_samples: int
    quantile_levels: list[float]


class EvaluationConfig(BaseModel):
    coverage_levels: list[float]
    horizon_checkpoints: list[int]


class ReproducibilityConfig(BaseModel):
    seeds: list[int]


class PathsConfig(BaseModel):
    metrics_dir: str
    images_dir: str
    mlflow_tracking_uri: str
    mlflow_experiment: str


class Config(BaseSettings):
    """Top-level project config. Populated from YAML, env vars override (prefix PROBFORECAST_)."""

    model_config = SettingsConfigDict(env_prefix="PROBFORECAST_", env_nested_delimiter="__")

    data: DataConfig
    forecast: ForecastConfig
    evaluation: EvaluationConfig
    reproducibility: ReproducibilityConfig
    paths: PathsConfig
    model_params: dict = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_dir: Path | str | None = None) -> Config:
    """Load and validate project configuration from the configs directory."""
    cfg_dir = Path(config_dir) if config_dir is not None else _config_dir()
    base = _read_yaml(cfg_dir / "config.yaml")
    model_params = _read_yaml(cfg_dir / "model_params.yaml")
    base["model_params"] = model_params
    return Config(**base)


__all__ = ["Config", "load_config"]
