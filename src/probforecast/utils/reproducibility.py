"""Reproducibility helpers: seeding + experiment metadata capture."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (+ cuDNN determinism) and Lightning."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import lightning.pytorch as pl

        pl.seed_everything(seed, workers=True)
    except Exception:  # noqa: BLE001 — lightning optional / alt import path
        try:
            import pytorch_lightning as pl

            pl.seed_everything(seed, workers=True)
        except Exception:  # noqa: BLE001
            pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def log_experiment_metadata(extra: dict | None = None) -> dict[str, str]:
    """Capture run provenance: git SHA, dvc.lock hash, library + Python versions."""
    import gluonts
    import torch

    meta = {
        "git_sha": _git_sha(),
        "dvc_lock_hash": _file_hash("dvc.lock"),
        "config_hash": _file_hash("configs/config.yaml"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gluonts_version": gluonts.__version__,
    }
    if extra:
        meta.update({k: str(v) for k, v in extra.items()})
    return meta


__all__ = ["set_all_seeds", "log_experiment_metadata"]
