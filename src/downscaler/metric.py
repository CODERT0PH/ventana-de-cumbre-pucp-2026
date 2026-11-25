"""Métrica oficial del leaderboard: skill score frente a la climatología por hora.

    score = 1 - mean_v( RMSE_v(modelo) / RMSE_v(climatología) ),  v en {RH, T, U, V}

0 = climatología, 1 = perfecto. La climatología se calcula SOLO con el train del fold.
"""
from __future__ import annotations

import numpy as np

from . import VARS
from .data import hours_of


def climatology(Y: np.ndarray, t) -> np.ndarray:
    """Media por hora UTC y por píxel: (24, 70, 35, 4). Horas sin datos -> media global."""
    h = hours_of(t)
    glob_mean = Y.mean(0)
    return np.stack([Y[h == k].mean(0) if (h == k).any() else glob_mean for k in range(24)])


def rmse_per_var(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred.astype(np.float64) - true) ** 2).mean(axis=(0, 1, 2)))


def skill_score(pred: np.ndarray, true: np.ndarray, t, clim: np.ndarray):
    """Devuelve (score, dict con el skill por variable)."""
    ref = clim[hours_of(t)]
    r_model = rmse_per_var(pred, true)
    r_clim = rmse_per_var(ref, true)
    ratio = r_model / r_clim
    return float(1 - ratio.mean()), {v: float(1 - r) for v, r in zip(VARS, ratio)}


def loss_weights(clim_rmse: np.ndarray, model_rmse: np.ndarray | None = None) -> np.ndarray:
    """Pesos por variable para una pérdida MSE alineada con la métrica.
    Sin model_rmse: 1/RMSE_clim^2. Con model_rmse: 1/(RMSE_modelo * RMSE_clim),
    que es el gradiente exacto de la métrica (fase de ajuste fino)."""
    w = 1.0 / (clim_rmse ** 2) if model_rmse is None else 1.0 / (model_rmse * clim_rmse)
    return (w / w.mean()).astype(np.float32)
