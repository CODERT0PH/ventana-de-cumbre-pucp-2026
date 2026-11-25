"""Carga de la data oficial, ventanas temporales y cortes de validación sin fuga de datos.

Convenciones de la competencia:
  - GFS (17x13): fila 0 = norte.  WRF y DEM (70x35): fila 0 = sur.  Columna 0 = oeste.
  - times: 'YYYYMMDDHH' en UTC (hora de Perú = UTC - 5).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import EVAL_HOURS


def find(name: str) -> str:
    """Busca un archivo de la competencia en Kaggle, en data/ del repo o en la carpeta actual."""
    patterns = [f"/kaggle/input/**/{name}", f"../data/**/{name}", f"data/**/{name}", f"**/{name}"]
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No encuentro {name}. Baja la data de Kaggle a data/ (ver README).")


@dataclass
class CompetitionData:
    Xtr: np.ndarray      # (N, 17, 13, 4) GFS
    Ytr: np.ndarray      # (N, 70, 35, 4) WRF
    ttr: np.ndarray      # (N,) str
    Xte: np.ndarray      # (M, 17, 13, 4)
    tte: np.ndarray      # (M,) str
    static: dict         # dem, gfs_lats, gfs_lons, wrf_lats, wrf_lons


def load_competition() -> CompetitionData:
    tr = np.load(find("train.npz"))
    te = np.load(find("test.npz"))
    st = dict(np.load(find("static.npz")))
    return CompetitionData(tr["X"], tr["Y"], tr["times"].astype(str),
                           te["X"], te["times"].astype(str), st)


def hours_of(t) -> np.ndarray:
    """Hora UTC (0-23) de cada timestamp 'YYYYMMDDHH'."""
    return np.array([int(str(k)[-2:]) for k in t])


def make_windows(X: np.ndarray, t, lags=(2, 1, 0)) -> np.ndarray:
    """Ventanas [t-2, t-1, t] por tiempo real (no por índice), igual que el starter:
    si falta una hora previa por un hueco en la data, se usa la hora t.
    Devuelve (N, len(lags), 17, 13, 4)."""
    ts = pd.to_datetime(np.asarray(t).astype(str), format="%Y%m%d%H")
    pos = {k: i for i, k in enumerate(ts)}
    out = np.empty((len(X), len(lags)) + X.shape[1:], np.float32)
    for i, k in enumerate(ts):
        for j, lag in enumerate(lags):
            idx = pos.get(k - pd.Timedelta(hours=lag))
            out[i, j] = X[idx] if idx is not None else X[i]
    return out


def eval_mask(t) -> np.ndarray:
    """True en las horas que evalúa Kaggle (cada 3 horas)."""
    return np.isin(hours_of(t), EVAL_HOURS)


def temporal_holdout(t, val_start="2024061600", purge_hours=48):
    """Holdout principal: desde val_start hasta el final. Se purgan `purge_hours` antes
    del corte para que ninguna ventana de train toque la validación."""
    ts = pd.to_datetime(np.asarray(t).astype(str), format="%Y%m%d%H")
    cut = pd.Timestamp(pd.to_datetime(val_start, format="%Y%m%d%H"))
    val = ts >= cut
    trn = ts < cut - pd.Timedelta(hours=purge_hours)
    return np.asarray(trn), np.asarray(val)


def blocked_folds(t, n_folds=5, purge_hours=48):
    """CV por bloques temporales contiguos con purga a ambos lados del bloque de validación.
    Nunca split aleatorio: horas consecutivas son casi idénticas."""
    ts = pd.to_datetime(np.asarray(t).astype(str), format="%Y%m%d%H")
    edges = pd.date_range(ts.min(), ts.max() + pd.Timedelta(hours=1), periods=n_folds + 1)
    purge = pd.Timedelta(hours=purge_hours)
    folds = []
    for a, b in zip(edges[:-1], edges[1:]):
        val = (ts >= a) & (ts < b)
        trn = (ts < a - purge) | (ts >= b + purge)
        folds.append((np.asarray(trn), np.asarray(val)))
    return folds
