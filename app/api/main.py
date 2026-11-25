"""API de Ventana de Cumbre (FastAPI).

GFS actual (Open-Meteo, con caché) -> nuestro modelo de Kaggle (mismos pesos) -> plan por ruta.

Correr:  uvicorn app.api.main:app --reload   (desde la raíz del repo)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))           # gfs_forecast.py y route_plan.py

from downscaler.data import make_windows  # noqa: E402
from downscaler.inference import Downscaler  # noqa: E402
from route_plan import PERFILES, best_start, route_conditions, utc_to_lima  # noqa: E402

WEIGHTS = Path(os.getenv("WEIGHTS_PATH", ROOT / "models" / "modelo_final.pt"))
ARTIFACTS = Path(os.getenv("ARTIFACTS_PATH", ROOT / "models" / "artefactos.npz"))
STATIC = Path(os.getenv("STATIC_PATH", ROOT / "data" / "static.npz"))
GFS_CACHE = Path(os.getenv("GFS_CACHE", APP_DIR / "gfs_cache.npz"))
CACHE_HOURS = float(os.getenv("GFS_CACHE_HOURS", "3"))
RUTAS = json.loads((APP_DIR / "rutas" / "rutas.json").read_text(encoding="utf-8"))["rutas"]

app = FastAPI(title="Ventana de Cumbre", version="0.1.0")
_state: dict = {}


def _model() -> Downscaler:
    if "model" not in _state:
        faltan = [str(p) for p in (WEIGHTS, ARTIFACTS, STATIC) if not p.exists()]
        if faltan:
            raise HTTPException(503, f"Faltan archivos del modelo: {faltan}. Ver README.")
        _state["model"] = Downscaler(WEIGHTS, ARTIFACTS)
        _state["static"] = dict(np.load(STATIC))
    return _state["model"]


def _forecast(horas: int):
    """Pronóstico downscaled, recalculado como máximo cada CACHE_HOURS horas."""
    key = ("fc", horas)
    hit = _state.get(key)
    if hit and time.time() - hit[0] < CACHE_HOURS * 3600:
        return hit[1], hit[2]
    model = _model()            # primero el modelo: sin él no tiene sentido bajar el GFS
    from gfs_forecast import get_gfs_forecast
    try:
        X, t = get_gfs_forecast(forecast_hours=horas, cache_path=str(GFS_CACHE), cache_max_age_h=CACHE_HOURS)
    except Exception as e:  # Open-Meteo caído: usamos la última descarga guardada
        if not GFS_CACHE.exists():
            raise HTTPException(503, f"No se pudo descargar el GFS y no hay caché: {e}")
        d = np.load(GFS_CACHE)
        X, t = d["X"], d["times"]
    Xw, t = make_windows(X, t)
    maps = model.predict(Xw, t)
    _state[key] = (time.time(), maps, t)
    return maps, t


@app.get("/health")
def health():
    return {"ok": True, "modelo_listo": WEIGHTS.exists() and ARTIFACTS.exists(), "static": STATIC.exists()}


@app.get("/rutas")
def rutas():
    return [{"id": r["id"], "nombre": r["nombre"], "puntos": r["puntos"]} for r in RUTAS]


@app.get("/plan")
def plan(ruta_id: str, perfil: str = Query("intermedio", enum=list(PERFILES)), horas: int = Query(72, ge=12, le=384)):
    ruta = next((r for r in RUTAS if r["id"] == ruta_id), None)
    if ruta is None:
        raise HTTPException(404, f"Ruta desconocida: {ruta_id}")
    maps, t = _forecast(horas)
    conds = route_conditions(maps, t, ruta, _state["static"])
    res = best_start(conds, t, perfil, ruta["horas_por_tramo"])
    if res is None:
        raise HTTPException(422, "El pronóstico es más corto que la ruta.")
    salida, semaforo, detalle = res
    return {
        "ruta": ruta["nombre"], "perfil": perfil,
        "mejor_salida_lima": utc_to_lima(salida).strftime("%a %d/%m %H:00"),
        "semaforo": ["verde", "amarillo", "rojo"][semaforo],
        "detalle": detalle,
        "horas_pronostico": [utc_to_lima(x).strftime("%d/%m %H:00") for x in t],
    }
