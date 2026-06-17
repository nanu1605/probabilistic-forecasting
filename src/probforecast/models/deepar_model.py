"""DeepAR forecaster (GluonTS, PyTorch backend).

Wraps ``gluonts.torch.model.deepar.DeepAREstimator`` behind the project's
:class:`~probforecast.models.base.BaseForecaster` interface so it plugs into the same
rolling-origin harness and metrics as the classical baselines. Trains with fixed epochs
(no early stopping → the validation split stays untouched for Phase 5 recalibration).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from probforecast.data.schema import TARGET
from probforecast.models.base import BaseForecaster

# GluonTS triggers a torch 2.9 indexing deprecation internally; harmless, silence the flood.
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")

_DATASET_START = "2010-01-01"  # arbitrary anchor; only the regular hourly freq matters


def _distr_output(name: str):
    from gluonts.torch.distributions import NormalOutput, StudentTOutput

    return {"StudentT": StudentTOutput, "Normal": NormalOutput}.get(name, StudentTOutput)()


class DeepARForecaster(BaseForecaster):
    name = "deepar"

    def __init__(
        self,
        params: dict,
        *,
        freq: str = "h",
        accelerator: str = "auto",
        devices: int = 1,
        epochs: int | None = None,
        deterministic: bool = False,
    ):
        self.p = params
        self.freq = freq
        self.accelerator = accelerator
        self.devices = devices
        self.epochs = epochs if epochs is not None else int(params["epochs"])
        self.deterministic = deterministic
        self.prediction_length = int(params["prediction_length"])
        self.context_length = int(params["context_length"])
        self.num_parallel_samples = int(params.get("num_eval_samples", 100))
        self._predictor = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _clean(values: np.ndarray) -> np.ndarray:
        return pd.Series(np.asarray(values, dtype=float)).ffill().bfill().to_numpy()

    def _series_df(self, values: np.ndarray) -> pd.DataFrame:
        vals = self._clean(values)
        idx = pd.date_range(_DATASET_START, periods=len(vals), freq=self.freq)
        return pd.DataFrame({"target": vals}, index=idx)

    def _dataset(self, values: np.ndarray):
        from gluonts.dataset.pandas import PandasDataset

        return PandasDataset(self._series_df(values), target="target", freq=self.freq)

    def _multi_dataset(self, histories: list[np.ndarray]):
        from gluonts.dataset.pandas import PandasDataset

        frames = {str(i): self._series_df(h) for i, h in enumerate(histories)}
        return PandasDataset(frames, target="target", freq=self.freq)

    def _build_estimator(self):
        from gluonts.torch.model.deepar import DeepAREstimator

        trainer_kwargs = {
            "max_epochs": self.epochs,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        }
        if self.deterministic:
            trainer_kwargs["deterministic"] = True
        return DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_layers=int(self.p["num_layers"]),
            hidden_size=int(self.p["hidden_size"]),
            dropout_rate=float(self.p["dropout_rate"]),
            lr=float(self.p["lr"]),
            distr_output=_distr_output(str(self.p["distr_output"])),
            num_parallel_samples=self.num_parallel_samples,
            batch_size=int(self.p["batch_size"]),
            num_batches_per_epoch=int(self.p["num_batches_per_epoch"]),
            nonnegative_pred_samples=True,
            trainer_kwargs=trainer_kwargs,
        )

    # ------------------------------------------------------------------ interface
    def fit(self, train_df: pd.DataFrame) -> DeepARForecaster:
        ds = self._dataset(train_df[TARGET].to_numpy(dtype=float))
        self._predictor = self._build_estimator().train(training_data=ds)
        return self

    def sample(self, history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        if horizon != self.prediction_length:
            raise ValueError(
                f"DeepAR fixed prediction_length={self.prediction_length}, got horizon={horizon}"
            )
        ds = self._dataset(history)
        forecast = next(iter(self._predictor.predict(ds)))
        return self._fit_samples(np.asarray(forecast.samples, dtype=float), num_samples)

    def sample_windows(self, histories: list[np.ndarray], num_samples: int) -> np.ndarray:
        """Batch-predict all windows in one pass → ``(num_windows, num_samples, horizon)``.

        One ``predictor.predict`` call over a multi-series dataset (vs one call per window):
        avoids per-window Lightning overhead, the dominant cost at evaluation time.
        """
        ds = self._multi_dataset(histories)
        forecasts = list(self._predictor.predict(ds))  # preserves input order
        out = [
            self._fit_samples(np.asarray(f.samples, dtype=float), num_samples) for f in forecasts
        ]
        return np.stack(out)

    def _fit_samples(self, samples: np.ndarray, num_samples: int) -> np.ndarray:
        if samples.shape[0] >= num_samples:
            return samples[:num_samples]
        idx = np.random.default_rng(0).integers(0, samples.shape[0], size=num_samples)
        return samples[idx]


__all__ = ["DeepARForecaster"]
