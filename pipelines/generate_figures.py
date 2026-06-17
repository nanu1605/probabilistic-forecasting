"""Regenerate every plot from saved predictions/metrics (no model retraining).

Reads the npz forecast dumps + metrics JSON written by the pipelines and rebuilds all figures in
``docs/images/`` via the plotting helpers. Degrades gracefully: a missing artifact is logged and
skipped rather than fatal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from probforecast.config import load_config
from probforecast.evaluation.calibration import (
    compute_calibration_data,
    compute_per_horizon_calibration,
)
from probforecast.evaluation.metrics import compute_ece
from probforecast.plotting.architecture import plot_architecture
from probforecast.plotting.calibration import (
    plot_calibration_comparison,
    plot_calibration_curve,
    plot_calibration_per_horizon,
)
from probforecast.plotting.forecasts import plot_forecast_windows
from probforecast.plotting.horizon import plot_performance_vs_horizon

log = structlog.get_logger()


def main(cfg=None) -> None:
    cfg = cfg or load_config()
    md = Path(cfg.paths.metrics_dir)
    images = Path(cfg.paths.images_dir)
    levels = cfg.evaluation.coverage_levels

    # Architecture diagram (no data dependency).
    plot_architecture(images / "architecture.png")
    log.info("figures.architecture_done")

    # Baseline forecast windows.
    bpath = md / "baselines_forecasts.npz"
    if bpath.exists():
        b = np.load(bpath)
        for name in ["arima", "seasonal_naive", "ets"]:
            if f"{name}_samples" in b:
                plot_forecast_windows(
                    b[f"{name}_samples"], b[f"{name}_actuals"],
                    images / "baseline_forecasts.png",
                    title=f"{name} — test forecasts (median + 80/95% intervals)",
                )  # fmt: skip
                break
        log.info("figures.baselines_done")
    else:
        log.warning("figures.skip", missing=str(bpath))

    # DeepAR forecasts + calibration (before) + per-horizon + performance-vs-horizon.
    dpath = md / "deepar_forecasts.npz"
    if dpath.exists():
        d = np.load(dpath)
        s, a = d["deepar_samples"], d["deepar_actuals"]
        plot_forecast_windows(
            s, a, images / "deepar_forecasts.png",
            title="DeepAR — test forecasts (median + 80/95% intervals)",
        )  # fmt: skip
        curve = compute_calibration_data(s, a, levels)
        plot_calibration_curve(
            curve, "DeepAR (raw) — calibration", images / "calibration_before.png",
            ece=compute_ece(dict(curve)),
        )  # fmt: skip
        ph = compute_per_horizon_calibration(s, a, levels, cfg.evaluation.horizon_checkpoints)
        plot_calibration_per_horizon(ph, images / "calibration_per_horizon.png")
        plot_performance_vs_horizon(
            s, a, images / "performance_vs_horizon.png", title="DeepAR — error vs horizon"
        )
        log.info("figures.deepar_done")
    else:
        log.warning("figures.skip", missing=str(dpath))

    # Recalibration comparison (money shot) + after curve.
    rpath = md / "recalibration_forecasts.npz"
    if rpath.exists():
        r = np.load(rpath)
        st, rt, ta = r["samples_test"], r["recal_test"], r["test_actuals"]
        cb = compute_calibration_data(st, ta, levels)
        ca = compute_calibration_data(rt, ta, levels)
        eb, ea = compute_ece(dict(cb)), compute_ece(dict(ca))
        plot_calibration_curve(
            ca, "DeepAR (recalibrated) — calibration", images / "calibration_after.png", ece=ea
        )
        plot_calibration_comparison(
            cb, ca, images / "calibration_comparison.png", ece_before=eb, ece_after=ea
        )
        log.info("figures.recal_done")
    else:
        log.warning("figures.skip", missing=str(rpath))

    log.info("figures.done")


if __name__ == "__main__":
    main()
