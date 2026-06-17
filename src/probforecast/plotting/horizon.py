"""Performance-vs-horizon plot (bonus): how point + distributional error grow with lead time."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import properscoring as ps  # noqa: E402


def plot_performance_vs_horizon(
    samples: np.ndarray, actuals: np.ndarray, save_path: str | Path, *, title: str
) -> Path:
    """Per-step MAE and CRPS vs forecast horizon. samples (W,S,H), actuals (W,H)."""
    samples = np.asarray(samples, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    horizon = actuals.shape[1]
    steps = np.arange(1, horizon + 1)

    median = np.median(samples, axis=1)  # (W, H)
    mae = np.mean(np.abs(median - actuals), axis=0)  # (H,)
    # CRPS per step: ensemble on last axis.
    crps = np.array(
        [np.mean(ps.crps_ensemble(actuals[:, h], samples[:, :, h])) for h in range(horizon)]
    )

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, mae, marker="o", color="tab:blue", label="MAE")
    ax1.set(xlabel="forecast horizon (hours ahead)", ylabel="MAE")
    ax2 = ax1.twinx()
    ax2.plot(steps, crps, marker="s", color="tab:red", label="CRPS")
    ax2.set_ylabel("CRPS")
    ax1.set_title(title)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


__all__ = ["plot_performance_vs_horizon"]
