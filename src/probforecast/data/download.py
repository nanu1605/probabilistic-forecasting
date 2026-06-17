"""Fetch the UCI Beijing air-quality dataset, or fall back to synthetic generation.

``make data`` runs this first. It attempts to download + extract the UCI zip; on any
network/extraction failure (and when ``use_synthetic_fallback`` is set) it generates a
synthetic series with the identical raw layout. The chosen path is recorded in
``data/raw/SOURCE.txt`` and logged.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import structlog

from probforecast.config import Config, load_config
from probforecast.data.schema import RAW_COLUMNS
from probforecast.data.synthetic import generate_synthetic

log = structlog.get_logger()

_DOWNLOAD_TIMEOUT_S = 60


def _raw_csv_path(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / f"PRSA_Data_{cfg.data.station}.csv"


def _source_marker(cfg: Config) -> Path:
    return Path(cfg.data.raw_dir) / "SOURCE.txt"


def _try_download_uci(cfg: Config) -> pd.DataFrame | None:
    """Download + extract the UCI zip and return the chosen station's raw frame, or None."""
    try:
        log.info("uci.download.start", url=cfg.data.uci_url)
        req = urllib.request.Request(cfg.data.uci_url, headers={"User-Agent": "probforecast/0.1"})
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:  # noqa: S310
            blob = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        # UCI ships an outer zip containing a nested "PRSA*.zip" (the real per-station CSVs)
        # plus decoy files (data.csv/test.csv). Always descend into the inner PRSA zip first.
        inner = [m for m in zf.namelist() if m.lower().endswith(".zip") and "prsa" in m.lower()]
        if inner:
            zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        match = [m for m in members if cfg.data.station.lower() in m.lower()]
        if not match:
            log.warning("uci.download.station_not_found", station=cfg.data.station)
            return None
        df = pd.read_csv(io.BytesIO(zf.read(match[0])))
        log.info("uci.download.ok", member=match[0], rows=len(df))
        return df
    except Exception as exc:  # noqa: BLE001 — any failure routes to fallback
        log.warning("uci.download.failed", error=str(exc))
        return None


def download_raw(cfg: Config | None = None, *, force: bool = False) -> Path:
    """Ensure the raw station CSV exists; return its path. Real UCI first, synthetic fallback."""
    cfg = cfg or load_config()
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = _raw_csv_path(cfg)

    if out.exists() and not force:
        log.info("raw.exists", path=str(out))
        return out

    df = _try_download_uci(cfg)
    source = "uci"
    if df is None:
        if not cfg.data.use_synthetic_fallback:
            raise RuntimeError("UCI download failed and synthetic fallback is disabled.")
        log.info("synthetic.generate.start")
        df = generate_synthetic(cfg)
        source = "synthetic"

    # Normalise to canonical raw columns (UCI CSVs already match; be defensive).
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"raw frame missing columns: {missing}")
    df = df[RAW_COLUMNS]
    df.to_csv(out, index=False)
    _source_marker(cfg).write_text(f"{source}\n")
    log.info("raw.written", path=str(out), source=source, rows=len(df))
    return out


def load_raw(cfg: Config | None = None) -> pd.DataFrame:
    """Load the raw station CSV (downloading/generating it if absent)."""
    cfg = cfg or load_config()
    path = _raw_csv_path(cfg)
    if not path.exists():
        download_raw(cfg)
    return pd.read_csv(path)


if __name__ == "__main__":
    download_raw()
