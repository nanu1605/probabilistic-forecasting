"""Calibration plots: single curve, before/after comparison (the money shot), per-horizon grid."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CalibrationData = list[tuple[float, float]]


def _xy(data: CalibrationData):
    pred = [p for p, _ in data]
    obs = [o for _, o in data]
    return pred, obs


def plot_calibration_curve(
    data: CalibrationData, title: str, save_path: str | Path, *, ece: float | None = None
) -> Path:
    """Observed vs predicted coverage with the perfect-calibration diagonal."""
    pred, obs = _xy(data)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    label = "DeepAR (raw)" + (f"  (ECE={ece:.3f})" if ece is not None else "")
    ax.plot(pred, obs, marker="o", color="tab:blue", label=label)
    ax.set(
        title=title,
        xlabel="predicted coverage",
        ylabel="observed coverage",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_calibration_comparison(
    before: CalibrationData,
    after: CalibrationData,
    save_path: str | Path,
    *,
    ece_before: float | None = None,
    ece_after: float | None = None,
    title: str = "DeepAR calibration: raw vs recalibrated",
) -> Path:
    """The money shot: both curves + diagonal, ECE annotated for each."""
    pb, ob = _xy(before)
    pa, oa = _xy(after)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    lb = "DeepAR (raw)" + (f"  ECE={ece_before:.3f}" if ece_before is not None else "")
    la = "DeepAR (recalibrated)" + (f"  ECE={ece_after:.3f}" if ece_after is not None else "")
    ax.plot(pb, ob, marker="o", color="tab:blue", label=lb)
    ax.plot(pa, oa, marker="s", color="tab:green", label=la)
    ax.set(
        title=title,
        xlabel="predicted coverage",
        ylabel="observed coverage",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_calibration_per_horizon(
    per_horizon: dict[int, CalibrationData], save_path: str | Path
) -> Path:
    """One calibration curve per forecast step (hour), showing horizon degradation."""
    steps = sorted(per_horizon)
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, h in zip(axes[0], steps, strict=False):
        pred, obs = _xy(per_horizon[h])
        ax.plot([0, 1], [0, 1], ls="--", color="gray")
        ax.plot(pred, obs, marker="o", color="tab:blue")
        ax.set(title=f"horizon h+{h}", xlabel="predicted", xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("observed coverage")
    fig.suptitle("DeepAR calibration by forecast horizon")
    fig.tight_layout()
    return _save(fig, save_path)


def _save(fig, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


__all__ = [
    "plot_calibration_curve",
    "plot_calibration_comparison",
    "plot_calibration_per_horizon",
]
