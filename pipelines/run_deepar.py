"""Train + evaluate DeepAR over 5 seeds; aggregate mean ± std; log to MLflow.

Uses the same rolling-origin harness and metrics as the baselines (apples-to-apples).
Writes ``metrics/deepar_results.json`` (per-seed + aggregate) and the best-seed forecast
plot/arrays; logs each seed as an MLflow run with full provenance metadata.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import structlog

from probforecast.config import Config, load_config
from probforecast.evaluation.metrics import summarize_samples
from probforecast.evaluation.rolling import collect_windows
from probforecast.models.deepar_model import DeepARForecaster
from probforecast.plotting.forecasts import plot_forecast_windows
from probforecast.utils.reproducibility import log_experiment_metadata, set_all_seeds

log = structlog.get_logger()


def _load_splits(cfg: Config) -> dict[str, pd.DataFrame]:
    d = Path(cfg.data.splits_dir)
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in ["train", "val", "test"]}


def _accelerator() -> str:
    import torch

    return "gpu" if torch.cuda.is_available() else "cpu"


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    return {
        k.replace("@", "_"): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _aggregate(per_seed: dict[int, dict]) -> dict[str, dict[str, float]]:
    keys = [
        k
        for k, v in next(iter(per_seed.values())).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    agg: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in per_seed], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return agg


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    splits = _load_splits(cfg)
    horizon = cfg.forecast.prediction_length
    num_samples = cfg.forecast.num_samples
    params = cfg.model_params["deepar"]
    accelerator = _accelerator()
    source = (
        (Path(cfg.data.raw_dir) / "SOURCE.txt").read_text().strip()
        if (Path(cfg.data.raw_dir) / "SOURCE.txt").exists()
        else "unknown"
    )
    log.info(
        "deepar.config",
        accelerator=accelerator,
        epochs=params["epochs"],
        seeds=cfg.reproducibility.seeds,
    )

    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.paths.mlflow_experiment)

    # Window contexts are identical across seeds (same protocol as baselines) — build once.
    histories, actuals, n_dropped = collect_windows(splits["train"], splits["test"], horizon)
    log.info("deepar.windows", n_windows=len(histories), n_dropped=n_dropped)

    per_seed: dict[int, dict] = {}
    best = {"seed": None, "crps": float("inf"), "samples": None, "actuals": None, "predictor": None}

    for seed in cfg.reproducibility.seeds:
        set_all_seeds(seed)
        model = DeepARForecaster(params, accelerator=accelerator, epochs=int(params["epochs"]))
        model.fit(splits["train"])
        samples = model.sample_windows(histories, num_samples)  # (W, num_samples, horizon)
        metrics = summarize_samples(samples, actuals, cfg.evaluation.coverage_levels)
        metrics["n_windows"] = len(histories)
        per_seed[seed] = metrics
        log.info("deepar.seed.done", seed=seed, crps=round(metrics["crps"], 3))

        meta = log_experiment_metadata({"seed": seed, "accelerator": accelerator})
        with mlflow.start_run(run_name=f"deepar_seed{seed}"):
            mlflow.set_tags({"model": "deepar", "kind": "deep", "data_source": source, **meta})
            mlflow.log_params(
                {
                    "seed": seed,
                    "context_length": cfg.forecast.context_length,
                    "horizon": horizon,
                    "epochs": params["epochs"],
                    "hidden_size": params["hidden_size"],
                    "distr_output": params["distr_output"],
                }
            )
            mlflow.log_metrics(_scalar_metrics(metrics))

        if metrics["crps"] < best["crps"]:
            best.update(
                seed=seed,
                crps=metrics["crps"],
                samples=samples,
                actuals=actuals,
                predictor=model._predictor,
            )

    aggregate = _aggregate(per_seed)
    results = {
        "per_seed": {str(s): per_seed[s] for s in per_seed},
        "aggregate": aggregate,
        "best_seed": best["seed"],
        "data_source": source,
    }

    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "deepar_results.json").write_text(json.dumps(results, indent=2))
    np.savez(
        metrics_dir / "deepar_forecasts.npz",
        deepar_samples=best["samples"],
        deepar_actuals=best["actuals"],
    )
    plot_forecast_windows(
        best["samples"],
        best["actuals"],
        Path(cfg.paths.images_dir) / "deepar_forecasts.png",
        title=f"DeepAR (best seed {best['seed']}) — test forecasts (median + 80/95% intervals)",
    )

    # Persist the best predictor as an MLflow artifact.
    try:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mlflow.start_run(run_name="deepar_best_artifact"),
        ):
            best["predictor"].serialize(Path(tmp))
            mlflow.log_artifacts(tmp, artifact_path="deepar_best_predictor")
    except Exception as exc:  # noqa: BLE001
        log.warning("deepar.artifact_failed", error=str(exc))

    crps_mean = aggregate["crps"]["mean"]
    log.info(
        "deepar.done",
        crps_mean=round(crps_mean, 3),
        crps_std=round(aggregate["crps"]["std"], 3),
        best_seed=best["seed"],
    )
    return results


if __name__ == "__main__":
    main()
