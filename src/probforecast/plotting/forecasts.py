"""Plot forecast windows: actuals vs median with shaded prediction intervals."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probforecast.evaluation.metrics import central_interval  # noqa: E402


def plot_forecast_windows(
    samples: np.ndarray,
    actuals: np.ndarray,
    save_path: str | Path,
    *,
    title: str,
    window_indices: list[int] | None = None,
) -> Path:
    """Plot representative windows. samples (W,S,H), actuals (W,H)."""
    n_w = samples.shape[0]
    if window_indices is None:
        window_indices = list(np.linspace(0, n_w - 1, num=min(4, n_w), dtype=int))

    n = len(window_indices)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    for ax, w in zip(axes[0], window_indices, strict=False):
        s = samples[w]  # (S, H)
        median = np.median(s, axis=0)
        steps = np.arange(s.shape[1])
        for cov, alpha_shade in [(0.95, 0.15), (0.80, 0.25)]:
            lower, upper = central_interval(s, cov, sample_axis=0)
            ax.fill_between(steps, lower, upper, color="tab:blue", alpha=alpha_shade)
        ax.plot(steps, median, color="tab:blue", lw=1.5, label="median")
        ax.plot(steps, actuals[w], color="black", lw=1.2, ls="--", label="actual")
        ax.set(title=f"window {w}", xlabel="horizon (h)")
    axes[0][0].set_ylabel("PM2.5")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


__all__ = ["plot_forecast_windows"]
