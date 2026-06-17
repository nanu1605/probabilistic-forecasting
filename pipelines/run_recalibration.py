"""Post-hoc recalibration of DeepAR: fit on validation, evaluate on test.

Self-contained: retrains DeepAR (best seed 1337), produces val + test forecasts, fits the
recalibration map on val ONLY, transforms test, and reports before/after metrics + the
centerpiece comparison plot + the full results table. Test labels are never seen during fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import structlog

from probforecast.config import Config, load_config
from probforecast.evaluation.calibration import compute_calibration_data
from probforecast.evaluation.metrics import compute_ece, summarize_samples
from probforecast.evaluation.recalibration import RecalibrationModel
from probforecast.evaluation.rolling import collect_windows
from probforecast.models.deepar_model import DeepARForecaster
from probforecast.plotting.calibration import (
    plot_calibration_comparison,
    plot_calibration_curve,
)
from probforecast.plotting.results_table import write_results_table
from probforecast.utils.reproducibility import log_experiment_metadata, set_all_seeds

log = structlog.get_logger()

_RECALIB_SEED = 1337  # Phase-3 best seed


def _load_splits(cfg: Config) -> dict[str, pd.DataFrame]:
    d = Path(cfg.data.splits_dir)
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in ["train", "val", "test"]}


def _accelerator() -> str:
    import torch

    return "gpu" if torch.cuda.is_available() else "cpu"


def _assemble_table(cfg: Config, recal_after: dict) -> list[dict]:
    md = Path(cfg.paths.metrics_dir)
    rows: list[dict] = []

    def cov(m, lvl):
        return m.get(f"coverage@{lvl}")

    baselines = json.loads((md / "baselines_results.json").read_text())
    for name, label in [("seasonal_naive", "Seasonal Naive"), ("arima", "ARIMA"), ("ets", "ETS")]:
        m = baselines.get(name, {})
        if m.get("status") != "ok":
            rows.append({"Model": label, "MAE": "failed"})
            continue
        rows.append(
            {
                "Model": label,
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "CRPS": m["crps"],
                "Cov@50": cov(m, 50),
                "Cov@80": cov(m, 80),
                "Cov@90": cov(m, 90),
                "ECE": m["ece"],
            }  # fmt: skip
        )

    dp = json.loads((md / "deepar_results.json").read_text())["aggregate"]

    def ms(metric):
        return f"{dp[metric]['mean']:.2f}±{dp[metric]['std']:.2f}"

    rows.append(
        {
            "Model": "DeepAR (raw)",
            "MAE": ms("mae"),
            "RMSE": ms("rmse"),
            "CRPS": ms("crps"),
            "Cov@50": ms("coverage@50"),
            "Cov@80": ms("coverage@80"),
            "Cov@90": ms("coverage@90"),
            "ECE": ms("ece"),
        }  # fmt: skip
    )
    rows.append(
        {
            "Model": "DeepAR (recalibrated)",
            "MAE": recal_after["mae"],
            "RMSE": recal_after["rmse"],
            "CRPS": recal_after["crps"],
            "Cov@50": cov(recal_after, 50),
            "Cov@80": cov(recal_after, 80),
            "Cov@90": cov(recal_after, 90),
            "ECE": recal_after["ece"],
        }  # fmt: skip
    )
    return rows


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    splits = _load_splits(cfg)
    horizon = cfg.forecast.prediction_length
    num_samples = cfg.forecast.num_samples
    levels = cfg.evaluation.coverage_levels
    params = cfg.model_params["deepar"]
    method = cfg.model_params["recalibration"]["method"]

    # Retrain DeepAR (best seed) and produce val + test forecasts under the shared protocol.
    set_all_seeds(_RECALIB_SEED)
    model = DeepARForecaster(params, accelerator=_accelerator(), epochs=int(params["epochs"]))
    model.fit(splits["train"])

    val_hist, val_actuals, _ = collect_windows(splits["train"], splits["val"], horizon)
    test_hist, test_actuals, _ = collect_windows(splits["train"], splits["test"], horizon)
    log.info("recal.windows", val=len(val_hist), test=len(test_hist))
    samples_val = model.sample_windows(val_hist, num_samples)
    samples_test = model.sample_windows(test_hist, num_samples)

    # Fit on VAL only; transform test (no labels).
    recal = RecalibrationModel(method=method, nonneg=True).fit(samples_val, val_actuals)
    recal_test = recal.transform(
        samples_test, num_samples, rng=np.random.default_rng(_RECALIB_SEED)
    )

    before = summarize_samples(samples_test, test_actuals, levels)
    after = summarize_samples(recal_test, test_actuals, levels)
    curve_before = compute_calibration_data(samples_test, test_actuals, levels)
    curve_after = compute_calibration_data(recal_test, test_actuals, levels)
    ece_before, ece_after = compute_ece(dict(curve_before)), compute_ece(dict(curve_after))

    # In-sample check: does the map calibrate the val set it was fit on? (isolates shift vs bug)
    recal_val = recal.transform(samples_val, num_samples, rng=np.random.default_rng(7))
    val_ece_before = compute_ece(dict(compute_calibration_data(samples_val, val_actuals, levels)))
    val_ece_after = compute_ece(dict(compute_calibration_data(recal_val, val_actuals, levels)))
    log.info(
        "recal.val_check",
        val_ece_before=round(val_ece_before, 4),
        val_ece_after=round(val_ece_after, 4),
    )

    # Diagnostic upper bound (LEAKAGE — fit on test): proves the recalibration METHOD is correct.
    # The gap between this and the honest val-fit result isolates val→test distribution shift as
    # the reason the no-leakage recalibration cannot fix test miscalibration. NOT a reported result.
    oracle = RecalibrationModel(method=method, nonneg=True).fit(samples_test, test_actuals)
    oracle_test = oracle.transform(samples_test, num_samples, rng=np.random.default_rng(0))
    oracle_ece = compute_ece(dict(compute_calibration_data(oracle_test, test_actuals, levels)))
    log.info("recal.oracle_check", oracle_test_ece_leakage=round(oracle_ece, 4))

    # Persist forecasts for Phase 6 reuse (avoids retraining).
    np.savez(
        Path(cfg.paths.metrics_dir) / "recalibration_forecasts.npz",
        samples_test=samples_test,
        recal_test=recal_test,
        test_actuals=test_actuals,
    )

    images = Path(cfg.paths.images_dir)
    plot_calibration_curve(
        curve_after, "DeepAR (recalibrated) — calibration", images / "calibration_after.png",
        ece=ece_after,
    )  # fmt: skip
    plot_calibration_comparison(
        curve_before, curve_after, images / "calibration_comparison.png",
        ece_before=ece_before, ece_after=ece_after,
    )  # fmt: skip

    results = {
        "method": method,
        "val_windows": len(val_hist),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "val_ece_before": val_ece_before,
        "val_ece_after": val_ece_after,
        "oracle_test_ece_leakage": oracle_ece,
        "coverage90_before": before["coverage@90"],
        "coverage90_after": after["coverage@90"],
        "crps_before": before["crps"],
        "crps_after": after["crps"],
        "width90_before": before["width@90"],
        "width90_after": after["width@90"],
        "metrics_before": {k: v for k, v in before.items() if isinstance(v, (int, float))},
        "metrics_after": {k: v for k, v in after.items() if isinstance(v, (int, float))},
    }
    md = Path(cfg.paths.metrics_dir)
    (md / "recalibration_results.json").write_text(json.dumps(results, indent=2))

    rows = _assemble_table(cfg, after)
    (md / "full_results_table.json").write_text(json.dumps(rows, indent=2))
    write_results_table(rows, images / "results_table.md")

    try:
        mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.paths.mlflow_experiment)
        meta = log_experiment_metadata({"method": method, "seed": _RECALIB_SEED})
        with mlflow.start_run(run_name="deepar_recalibrated"):
            mlflow.set_tags({"model": "deepar_recal", "kind": "recalibration", **meta})
            mlflow.log_params({"method": method, "val_windows": len(val_hist)})
            mlflow.log_metrics(
                {
                    "ece_before": ece_before,
                    "ece_after": ece_after,
                    "crps_before": before["crps"],
                    "crps_after": after["crps"],
                    "coverage_90_before": before["coverage@90"],
                    "coverage_90_after": after["coverage@90"],
                    "width_90_before": before["width@90"],
                    "width_90_after": after["width@90"],
                }  # fmt: skip
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("recal.mlflow_failed", error=str(exc))

    log.info(
        "recal.done",
        ece_before=round(ece_before, 4),
        ece_after=round(ece_after, 4),
        cov90_before=round(before["coverage@90"], 3),
        cov90_after=round(after["coverage@90"], 3),
    )
    return results


if __name__ == "__main__":
    main()
