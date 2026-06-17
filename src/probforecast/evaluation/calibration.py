"""Calibration analysis: curve data, per-horizon stratification, ECE, pipeline entrypoint.

Thin layer over :mod:`probforecast.evaluation.metrics` (single source for the calibration-curve
and ECE computations). Diagnoses DeepAR's (mis)calibration from the forecasts saved in Phase 3 —
no retraining. Produces the "before" calibration plot + the per-horizon grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

from probforecast.config import Config, load_config
from probforecast.evaluation.metrics import compute_calibration_curve, compute_ece
from probforecast.plotting.calibration import (
    plot_calibration_curve,
    plot_calibration_per_horizon,
)

log = structlog.get_logger()

CalibrationData = list[tuple[float, float]]


def compute_calibration_data(
    samples: np.ndarray, actuals: np.ndarray, levels: list[float]
) -> CalibrationData:
    """Return sorted (predicted_coverage, observed_coverage) pairs over windows×horizon.

    ``samples`` (num_windows, num_samples, horizon); ``actuals`` (num_windows, horizon).
    """
    whs = np.moveaxis(np.asarray(samples, dtype=float), 1, -1)  # (W, H, S)
    n_w, n_h, n_s = whs.shape
    flat_samples = whs.reshape(n_w * n_h, n_s)
    flat_actuals = np.asarray(actuals, dtype=float).reshape(n_w * n_h)
    curve = compute_calibration_curve(flat_actuals, flat_samples, levels)
    return sorted(curve.items())


def compute_per_horizon_calibration(
    samples: np.ndarray,
    actuals: np.ndarray,
    levels: list[float],
    horizon_steps: list[int],
) -> dict[int, CalibrationData]:
    """Calibration curve restricted to each forecast step (hour). ``horizon_steps`` are 1-based."""
    samples = np.asarray(samples, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    out: dict[int, CalibrationData] = {}
    for h in horizon_steps:
        i = h - 1  # 1-based hour -> 0-based index
        s_h = samples[:, :, i]  # (W, S)
        a_h = actuals[:, i]  # (W,)
        curve = compute_calibration_curve(a_h, s_h, levels)
        out[h] = sorted(curve.items())
    return out


def _load_deepar_forecasts(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(Path(cfg.paths.metrics_dir) / "deepar_forecasts.npz")
    return npz["deepar_samples"], npz["deepar_actuals"]


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    samples, actuals = _load_deepar_forecasts(cfg)
    levels = cfg.evaluation.coverage_levels

    curve = compute_calibration_data(samples, actuals, levels)
    ece = compute_ece(dict(curve))
    per_horizon = compute_per_horizon_calibration(
        samples, actuals, levels, cfg.evaluation.horizon_checkpoints
    )

    images = Path(cfg.paths.images_dir)
    plot_calibration_curve(
        curve, "DeepAR (raw) — calibration", images / "calibration_before.png", ece=ece
    )
    plot_calibration_per_horizon(per_horizon, images / "calibration_per_horizon.png")

    results = {
        "ece": ece,
        "calibration_curve": {str(p): o for p, o in curve},
        "per_horizon_ece": {str(h): compute_ece(dict(c)) for h, c in per_horizon.items()},
    }
    out = Path(cfg.paths.metrics_dir) / "calibration_results.json"
    out.write_text(json.dumps(results, indent=2))

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.paths.mlflow_experiment)
        with mlflow.start_run(run_name="deepar_calibration"):
            mlflow.log_metric("ece_raw", ece)
            mlflow.log_metrics({f"ece_h{h}": compute_ece(dict(c)) for h, c in per_horizon.items()})
    except Exception as exc:  # noqa: BLE001
        log.warning("calibration.mlflow_failed", error=str(exc))

    log.info("calibration.done", ece=round(ece, 4), path=str(out))
    return results


if __name__ == "__main__":
    main()
