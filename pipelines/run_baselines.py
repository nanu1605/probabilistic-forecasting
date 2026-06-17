"""Train + evaluate the classical baselines (Seasonal Naive, ARIMA, ETS).

Rolling-origin evaluation over the test set, uniform sample-based metrics, MLflow logging,
results to ``metrics/baselines_results.json`` and forecasts to ``metrics/baselines_forecasts.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import structlog

from probforecast.config import Config, load_config
from probforecast.evaluation.metrics import summarize_samples
from probforecast.evaluation.rolling import rolling_forecast
from probforecast.models.arima_model import ArimaForecaster
from probforecast.models.ets_model import EtsForecaster
from probforecast.models.seasonal_naive import SeasonalNaiveForecaster
from probforecast.plotting.forecasts import plot_forecast_windows

log = structlog.get_logger()


def _load_splits(cfg: Config) -> dict[str, pd.DataFrame]:
    d = Path(cfg.data.splits_dir)
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in ["train", "val", "test"]}


def _build_models(cfg: Config) -> list:
    mp = cfg.model_params
    sn = mp["seasonal_naive"]["season_length"]
    a = mp["arima"]
    return [
        SeasonalNaiveForecaster(season_length=sn),
        ArimaForecaster(
            m=a["m"],
            max_train_weeks=a["max_train_weeks"],
            fallback_order=tuple(a["fallback_order"]),
            fallback_seasonal_order=tuple(a["fallback_seasonal_order"]),
        ),
        EtsForecaster(seasonal_periods=mp["ets"]["seasonal_periods"]),
    ]


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    # MLflow metric names disallow '@' -> map coverage@90 to coverage_90.
    return {
        k.replace("@", "_"): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    splits = _load_splits(cfg)
    horizon = cfg.forecast.prediction_length
    num_samples = cfg.forecast.num_samples
    source = (
        (Path(cfg.data.raw_dir) / "SOURCE.txt").read_text().strip()
        if (Path(cfg.data.raw_dir) / "SOURCE.txt").exists()
        else "unknown"
    )

    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.paths.mlflow_experiment)

    results: dict[str, dict] = {}
    forecasts: dict[str, np.ndarray] = {}

    for model in _build_models(cfg):
        log.info("baseline.start", model=model.name)
        model.fit(splits["train"])
        if model.failed:
            results[model.name] = {"status": "failed"}
            log.warning("baseline.failed", model=model.name)
            continue

        roll = rolling_forecast(
            model, splits["train"], splits["test"], horizon, num_samples, fit=False
        )
        metrics = summarize_samples(roll.samples, roll.actuals, cfg.evaluation.coverage_levels)
        metrics["status"] = "ok"
        metrics["n_windows"] = roll.n_windows
        metrics["n_dropped"] = roll.n_dropped
        results[model.name] = metrics
        forecasts[f"{model.name}_samples"] = roll.samples
        forecasts[f"{model.name}_actuals"] = roll.actuals

        with mlflow.start_run(run_name=model.name):
            mlflow.set_tags({"model": model.name, "kind": "baseline", "data_source": source})
            mlflow.log_params(
                {
                    "context_length": cfg.forecast.context_length,
                    "horizon": horizon,
                    "num_samples": num_samples,
                }
            )
            mlflow.log_metrics(_scalar_metrics(metrics))
        log.info("baseline.done", model=model.name, crps=round(metrics["crps"], 3))

    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "baselines_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    np.savez(metrics_dir / "baselines_forecasts.npz", **forecasts)
    log.info("baselines.written", path=str(out_json))

    # Forecast visualization for the first non-failed model with forecasts.
    for name in ["arima", "seasonal_naive", "ets"]:
        if f"{name}_samples" in forecasts:
            plot_forecast_windows(
                forecasts[f"{name}_samples"],
                forecasts[f"{name}_actuals"],
                Path(cfg.paths.images_dir) / "baseline_forecasts.png",
                title=f"{name} — test forecasts (median + 80/95% intervals)",
            )
            break
    return results


if __name__ == "__main__":
    main()
