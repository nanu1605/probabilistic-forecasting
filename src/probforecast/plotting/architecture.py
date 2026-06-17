"""Render the pipeline architecture diagram (reproducible — no manual figure)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_BOX = {"boxstyle": "round,pad=0.5", "linewidth": 1.4}


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch((x - w / 2, y - h / 2), w, h, **_BOX, facecolor=color, edgecolor="#333")
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=9, zorder=5)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="#555", linewidth=1.2
        )
    )


def plot_architecture(save_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    blue, green, orange, gray, purple = "#cfe3f7", "#cdeccd", "#fde2c4", "#e8e8e8", "#e6d6f2"

    _box(ax, 5, 13, 5.2, 1.0, "Raw Data (DVC)\nUCI Beijing air-quality · synthetic fallback", gray)
    _box(ax, 5, 11.3, 5.2, 1.0,
         "Preprocessing\nclean → impute → features → temporal split\n(pandera schema)", blue)  # fmt: skip
    _box(ax, 2.6, 9.3, 3.6, 1.0, "Classical Baselines\nSeasonal Naive · ARIMA · ETS", green)
    _box(ax, 7.4, 9.3, 3.6, 1.0, "DeepAR\nGluonTS · 5 seeds · MLflow", green)
    _box(ax, 5, 7.3, 5.6, 1.0,
         "Evaluation Engine\nMAE RMSE CRPS Coverage ECE Winkler\n(uniform 100-sample inputs)", orange)  # fmt: skip
    _box(ax, 5, 5.3, 5.6, 1.0,
         "Post-hoc Recalibration\nisotonic (Kuleshov) on val → re-eval on test", purple)  # fmt: skip
    _box(ax, 5, 3.3, 5.6, 1.0,
         "Artifacts & Report\ncalibration_comparison.png · results_table · report.md", gray)  # fmt: skip

    _arrow(ax, 5, 12.5, 5, 11.8)
    _arrow(ax, 5, 10.8, 2.6, 9.8)
    _arrow(ax, 5, 10.8, 7.4, 9.8)
    _arrow(ax, 2.6, 8.8, 5, 7.8)
    _arrow(ax, 7.4, 8.8, 5, 7.8)
    _arrow(ax, 5, 6.8, 5, 5.8)
    _arrow(ax, 5, 4.8, 5, 3.8)

    ax.set_title("probforecast — pipeline architecture", fontsize=12, pad=10)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


__all__ = ["plot_architecture"]
