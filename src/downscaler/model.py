"""Modelo de downscaling: encoder en la grilla GFS -> interpolación por coordenadas reales
-> U-Net en la grilla de 3 km con el relieve y sus derivadas como canales estáticos.

Predice la ANOMALÍA normalizada respecto de la climatología por hora (como el starter);
inference.py la vuelve a unidades físicas.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sampling_grid(gfs_lats, gfs_lons, wrf_lats, wrf_lons) -> torch.Tensor:
    """Grilla para F.grid_sample: posición de cada celda WRF dentro de la grilla GFS. (1, 70, 35, 2)."""
    LAT, LON = np.meshgrid(np.asarray(wrf_lats), np.asarray(wrf_lons), indexing="ij")
    gy = (LAT - gfs_lats[0]) / (gfs_lats[-1] - gfs_lats[0]) * 2 - 1   # GFS fila 0 = norte
    gx = (LON - gfs_lons[0]) / (gfs_lons[-1] - gfs_lons[0]) * 2 - 1
    return torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)


def _block(cin, cout, dropout=0.0):
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
              nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU()]
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class DownscalerUNet(nn.Module):
    """
    x      : (B, k, 17, 13, 4) ventanas GFS normalizadas
    hours  : (B,) hora UTC (float)
    salida : (B, 70, 35, 4) anomalía normalizada
    """

    def __init__(self, grid: torch.Tensor, static: torch.Tensor, k: int = 3,
                 width: int = 48, depth: int = 3, dropout: float = 0.0, n_learned_static: int = 8):
        super().__init__()
        self.register_buffer("grid", grid)                  # (1, 70, 35, 2)
        self.register_buffer("static", static)              # (1, S, 70, 35)
        self.H, self.W = grid.shape[1], grid.shape[2]
        cin_gfs = 4 * k + 2
        self.gfs_enc = _block(cin_gfs, width)
        self.learned_static = nn.Parameter(torch.zeros(1, n_learned_static, self.H, self.W))
        c0 = width + cin_gfs + static.shape[1] + n_learned_static

        widths = [width * 2 ** i for i in range(depth)]
        self.downs = nn.ModuleList()
        cin = c0
        for w in widths:
            self.downs.append(_block(cin, w, dropout))
            cin = w
        self.mid = _block(widths[-1], widths[-1] * 2, dropout)
        self.ups, self.up_convs = nn.ModuleList(), nn.ModuleList()
        cin = widths[-1] * 2
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.up_convs.append(_block(2 * w, w, dropout))
            cin = w
        self.head = nn.Conv2d(cin, 4, 1)
        self.mult = 2 ** depth

    def forward(self, x: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        B, k = x.shape[:2]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 4 * k, x.shape[2], x.shape[3])
        ang = 2 * np.pi * hours.float() / 24
        hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None]
        x = torch.cat([x, hh.expand(B, 2, x.shape[2], x.shape[3])], 1)

        g = self.grid.expand(B, -1, -1, -1)
        feats = F.grid_sample(self.gfs_enc(x), g, align_corners=True)
        raw = F.grid_sample(x, g, align_corners=True)                 # GFS interpolado, como pista directa
        z = torch.cat([feats, raw, self.static.expand(B, -1, -1, -1),
                       self.learned_static.expand(B, -1, -1, -1)], 1)

        ph, pw = (-self.H) % self.mult, (-self.W) % self.mult
        z = F.pad(z, (0, pw, 0, ph), mode="replicate")
        skips = []
        for i, down in enumerate(self.downs):
            z = down(z)
            skips.append(z)
            z = F.max_pool2d(z, 2)
        z = self.mid(z)
        for up, conv, s in zip(self.ups, self.up_convs, reversed(skips)):
            z = conv(torch.cat([up(z), s], 1))
        out = self.head(z)[:, :, :self.H, :self.W]
        return out.permute(0, 2, 3, 1)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """MSE por variable con pesos alineados a la métrica (ver metric.loss_weights)."""
    return (((pred - target) ** 2).mean(dim=(0, 1, 2)) * w).mean()
