import numpy as np
import pandas as pd

from route_plan import best_start, route_conditions, wind_chill_c


def test_wind_chill_colder_than_air():
    assert wind_chill_c(np.array([-5.0]), np.array([10.0]))[0] < -5.0


def test_best_start_prefers_calm_hours(static):
    t = pd.date_range("2024-07-01", periods=24, freq="h").strftime("%Y%m%d%H").to_numpy()
    maps = np.zeros((24, 70, 35, 4), np.float32)
    maps[..., 0], maps[..., 1] = 60, 275
    maps[12:, ..., 2] = 20                   # mucho viento desde la hora 12
    ruta = {"puntos": [{"nombre": "A", "lat": -9.5, "lon": -77.6, "altura_m": 3500},
                       {"nombre": "B", "lat": -9.4, "lon": -77.5, "altura_m": 4500}],
            "horas_por_tramo": [3]}
    conds = route_conditions(maps, t, ruta, static)
    salida, semaforo, detalle = best_start(conds, t, "intermedio", ruta["horas_por_tramo"])
    assert semaforo < 2 and int(str(salida)[-2:]) + 3 < 12
    assert detalle[1]["T_c"] < detalle[0]["T_c"]          # más alto = más frío
