"""Feature engineering SOLO a partir de la data oficial (X, times, static.npz).

1. Relieve x temperatura: diferencia entre el DEM real y el relieve "visto" a 27 km.
2. Viento x pendiente: pendiente, orientación, posición topográfica (valle/cresta).
3. Humedad x temperatura: punto de rocío y humedad específica (Magnus).
"""
from __future__ import annotations

import numpy as np

LAPSE_RATE = 0.0065  # K/m, gradiente térmico estándar


def box_smooth(a: np.ndarray, k: int) -> np.ndarray:
    p = k // 2
    ap = np.pad(a.astype(np.float32), p, mode="edge")
    out = np.zeros(a.shape, np.float32)
    for di in range(k):
        for dj in range(k):
            out += ap[di:di + a.shape[0], dj:dj + a.shape[1]]
    return out / (k * k)


def terrain_features(dem: np.ndarray, cell_m: float = 3000.0) -> dict[str, np.ndarray]:
    """Mapas estáticos (70, 35) derivados del DEM (fila 0 = sur)."""
    dem = dem.astype(np.float32)
    dz_dy, dz_dx = np.gradient(dem, cell_m)          # y crece hacia el norte
    slope = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    coarse = box_smooth(dem, 9)                        # ~27 km: lo que "ve" el GFS
    return {
        "dem": dem,
        "dem_anom": dem - coarse,                      # >0 cresta/nevado, <0 valle
        "dz_dx": dz_dx,
        "dz_dy": dz_dy,
        "slope": slope,
        "aspect_sin": np.where(slope > 0, -dz_dx / (slope + 1e-9), 0).astype(np.float32),
        "aspect_cos": np.where(slope > 0, -dz_dy / (slope + 1e-9), 0).astype(np.float32),
        "tpi_local": dem - box_smooth(dem, 3),         # índice de posición topográfica
    }


def lapse_rate_correction(dem_anom: np.ndarray) -> np.ndarray:
    """Corrección de temperatura (K) por la diferencia de altura frente a la celda de 27 km."""
    return (-LAPSE_RATE * dem_anom).astype(np.float32)


def upslope_wind(U: np.ndarray, V: np.ndarray, dz_dx: np.ndarray, dz_dy: np.ndarray) -> np.ndarray:
    """Viento pendiente arriba (m/s * m/m). Positivo = el aire sube la ladera."""
    return U * dz_dx + V * dz_dy


def dewpoint_k(T_k: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """Punto de rocío (K) por Magnus. Se conserva mejor que la RH al cambiar de altura."""
    T_c = T_k - 273.15
    a, b = 17.625, 243.04
    g = np.log(np.clip(RH, 1e-3, 100) / 100.0) + a * T_c / (b + T_c)
    return b * g / (a - g) + 273.15


def specific_humidity(T_k: np.ndarray, RH: np.ndarray, p_hpa: float | np.ndarray = 700.0) -> np.ndarray:
    """Humedad específica aproximada (kg/kg) con una presión fija o estimada por altura."""
    T_c = T_k - 273.15
    es = 6.1094 * np.exp(17.625 * T_c / (T_c + 243.04))
    e = np.clip(RH, 0, 100) / 100.0 * es
    return 0.622 * e / (p_hpa - 0.378 * e)


def hour_encoding(hours: np.ndarray) -> np.ndarray:
    ang = 2 * np.pi * np.asarray(hours) / 24.0
    return np.stack([np.sin(ang), np.cos(ang)], -1).astype(np.float32)
