"""Lógica de negocio de Ventana de Cumbre: de los mapas de 3 km del modelo al plan de salida.

Entrada: mapas del modelo (N, 70, 35, 4) en unidades físicas, sus tiempos UTC y una ruta
(lista de puntos con lat, lon, altura). Salida: condiciones por punto y hora, semáforo y
mejor hora de salida. Todo depende de la salida del modelo: sin el modelo no hay plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

LAPSE_RATE = 0.0065  # K/m

# Umbrales por perfil del grupo: (viento máx m/s, sensación térmica mín °C, RH máx %)
PERFILES = {
    "principiante": (8.0, -10.0, 95.0),
    "intermedio": (11.0, -15.0, 97.0),
    "experto": (14.0, -20.0, 99.0),
}


def utc_to_lima(t: str) -> datetime:
    return datetime.strptime(str(t), "%Y%m%d%H") - timedelta(hours=5)


def bilinear_weights(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray):
    """Índices y pesos para leer un punto de la grilla WRF (fila 0 = sur, lats crecientes)."""
    fi = float(np.interp(lat, lats, np.arange(len(lats))))
    fj = float(np.interp(lon, lons, np.arange(len(lons))))
    i0 = int(min(max(np.floor(fi), 0), len(lats) - 2))
    j0 = int(min(max(np.floor(fj), 0), len(lons) - 2))
    di, dj = fi - i0, fj - j0
    return i0, j0, di, dj


def sample_point(maps: np.ndarray, lat, lon, lats, lons) -> np.ndarray:
    """(N, 70, 35, C) -> (N, C) en el punto (lat, lon)."""
    i0, j0, di, dj = bilinear_weights(lat, lon, lats, lons)
    a, b = maps[:, i0, j0], maps[:, i0, j0 + 1]
    c, d = maps[:, i0 + 1, j0], maps[:, i0 + 1, j0 + 1]
    return (a * (1 - dj) + b * dj) * (1 - di) + (c * (1 - dj) + d * dj) * di


def wind_chill_c(T_c: np.ndarray, wind_ms: np.ndarray) -> np.ndarray:
    """Sensación térmica (fórmula canadiense/NWS); válida con T <= 10 °C y viento > 1.3 m/s."""
    v = np.maximum(wind_ms * 3.6, 4.8)
    wc = 13.12 + 0.6215 * T_c - 11.37 * v ** 0.16 + 0.3965 * T_c * v ** 0.16
    return np.where((T_c <= 10) & (wind_ms > 1.3), wc, T_c)


@dataclass
class PointConditions:
    nombre: str
    altura_m: float
    T_c: np.ndarray
    viento_ms: np.ndarray
    sensacion_c: np.ndarray
    RH: np.ndarray


def route_conditions(maps, times, ruta: dict, static: dict) -> list[PointConditions]:
    lats, lons, dem = static["wrf_lats"], static["wrf_lons"], static["dem"]
    out = []
    for p in ruta["puntos"]:
        v = sample_point(maps, p["lat"], p["lon"], lats, lons)
        dem_cell = float(sample_point(dem[None, ..., None], p["lat"], p["lon"], lats, lons)[0, 0])
        T_c = v[:, 1] - 273.15 - LAPSE_RATE * (p["altura_m"] - dem_cell)   # ajuste fino a la altura del punto
        wind = np.hypot(v[:, 2], v[:, 3])
        out.append(PointConditions(p["nombre"], p["altura_m"], T_c, wind, wind_chill_c(T_c, wind), v[:, 0]))
    return out


def traffic_light(pc: PointConditions, perfil: str) -> np.ndarray:
    """0 = verde, 1 = amarillo, 2 = rojo, por hora."""
    wmax, scmin, rhmax = PERFILES[perfil]
    rojo = (pc.viento_ms > wmax) | (pc.sensacion_c < scmin)
    amarillo = (pc.viento_ms > 0.75 * wmax) | (pc.sensacion_c < scmin + 5) | (pc.RH > rhmax)
    return np.where(rojo, 2, np.where(amarillo, 1, 0))


def best_start(conds: list[PointConditions], times, perfil: str, horas_por_tramo: list[int]):
    """Prueba cada hora de salida: el grupo llega al punto i tras sum(horas_por_tramo[:i]) horas.
    Devuelve (hora_salida_utc, peor_semáforo, detalle) de la mejor opción."""
    lights = [traffic_light(c, perfil) for c in conds]
    offsets = np.concatenate([[0], np.cumsum(horas_por_tramo)]).astype(int)
    n = len(times)
    best = None
    for s in range(n - offsets[-1]):
        worst = max(int(lights[i][s + offsets[i]]) for i in range(len(conds)))
        margin = -sum(float(conds[i].sensacion_c[s + offsets[i]]) for i in range(len(conds)))
        key = (worst, margin)
        if best is None or key < best[0]:
            best = (key, s)
    if best is None:
        return None
    s = best[1]
    detalle = [{"punto": c.nombre, "llegada_lima": utc_to_lima(times[s + offsets[i]]).strftime("%a %d/%m %H:00"),
                "T_c": round(float(c.T_c[s + offsets[i]]), 1),
                "viento_ms": round(float(c.viento_ms[s + offsets[i]]), 1),
                "sensacion_c": round(float(c.sensacion_c[s + offsets[i]]), 1),
                "semaforo": int(lights[i][s + offsets[i]])} for i, c in enumerate(conds)]
    return times[s], best[0][0], detalle
