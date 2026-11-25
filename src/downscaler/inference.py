"""Preprocesamiento compartido + carga de pesos para inferencia.

`Preprocessor` guarda todo lo que el modelo necesita fuera de los pesos (normalización,
climatología, canales estáticos). Se ajusta con el train del fold y se guarda en
models/artefactos.npz, para que la app use EXACTAMENTE lo mismo que el submission.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import hours_of
from .features import lapse_rate_correction, terrain_features
from .metric import climatology
from .model import DownscalerUNet, sampling_grid

STATIC_KEYS = ["dem", "dem_anom", "dz_dx", "dz_dy", "slope", "aspect_sin", "aspect_cos", "tpi_local", "lapse_T"]


@dataclass
class Preprocessor:
    mu_x: np.ndarray          # (4,)
    sd_x: np.ndarray          # (4,)
    clim: np.ndarray          # (24, 70, 35, 4)
    anom_sd: np.ndarray       # (4,)
    static_ch: np.ndarray     # (S, 70, 35) normalizado
    gfs_lats: np.ndarray
    gfs_lons: np.ndarray
    wrf_lats: np.ndarray
    wrf_lons: np.ndarray

    @classmethod
    def fit(cls, Xtr, Ytr, ttr, static) -> "Preprocessor":
        clim = climatology(Ytr, ttr)
        anom = Ytr - clim[hours_of(ttr)]
        tf = terrain_features(static["dem"])
        tf["lapse_T"] = lapse_rate_correction(tf["dem_anom"])
        ch = np.stack([tf[k] for k in STATIC_KEYS]).astype(np.float32)
        ch = (ch - ch.mean((1, 2), keepdims=True)) / (ch.std((1, 2), keepdims=True) + 1e-6)
        return cls(Xtr.mean((0, 1, 2)).astype(np.float32), Xtr.std((0, 1, 2)).astype(np.float32),
                   clim.astype(np.float32), anom.std((0, 1, 2)).astype(np.float32), ch,
                   np.asarray(static["gfs_lats"]), np.asarray(static["gfs_lons"]),
                   np.asarray(static["wrf_lats"]), np.asarray(static["wrf_lons"]))

    def save(self, path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path) -> "Preprocessor":
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def norm_x(self, W: np.ndarray) -> np.ndarray:
        return ((W - self.mu_x) / self.sd_x).astype(np.float32)

    def target(self, Y: np.ndarray, t) -> np.ndarray:
        return ((Y - self.clim[hours_of(t)]) / self.anom_sd).astype(np.float32)

    def to_physical(self, anom_norm: np.ndarray, t) -> np.ndarray:
        return anom_norm * self.anom_sd + self.clim[hours_of(t)]

    def build_model(self, k: int = 3, **kw) -> DownscalerUNet:
        grid = sampling_grid(self.gfs_lats, self.gfs_lons, self.wrf_lats, self.wrf_lons)
        static = torch.tensor(self.static_ch[None], dtype=torch.float32)
        return DownscalerUNet(grid, static, k=k, **kw)


class Downscaler:
    """Lo que usa el producto: GFS (N, 3, 17, 13, 4) en unidades físicas -> mapas (N, 70, 35, 4)."""

    def __init__(self, weights_path, artifacts_path, device: str | None = None, **model_kw):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pre = Preprocessor.load(artifacts_path)
        self.model = self.pre.build_model(**model_kw).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, Xw: np.ndarray, times, batch_size: int = 64) -> np.ndarray:
        h_all = hours_of(times)
        out = []
        for s in range(0, len(Xw), batch_size):
            x = torch.tensor(self.pre.norm_x(Xw[s:s + batch_size]), device=self.device)
            h = torch.tensor(h_all[s:s + batch_size], device=self.device, dtype=torch.float32)
            out.append(self.model(x, h).cpu().numpy())
        return self.pre.to_physical(np.concatenate(out), times).astype(np.float32)


def default_paths(root: Path | str = ".") -> tuple[Path, Path]:
    root = Path(root)
    return root / "models" / "modelo_final.pt", root / "models" / "artefactos.npz"
