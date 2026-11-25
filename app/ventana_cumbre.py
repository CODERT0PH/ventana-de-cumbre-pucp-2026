"""
Ventana de Cumbre · Hackathon IA PUCP 2026

¿Cuándo conviene atacar cumbre en la Cordillera Blanca en las próximas horas?

    GFS actual (Open-Meteo) -> modelo de downscaling (3 km) -> condiciones en cada cumbre
                            -> horas buenas / regulares / malas -> "ventanas de cumbre"

Para cada cumbre de app/cumbres.json:
  1. Toma el píxel de 3 km más cercano en la salida del modelo (RH, T, U, V).
  2. Corrige la temperatura desde la altura del píxel hasta la altura real de la cumbre
     (gradiente de 6.5 °C/km), porque un píxel de 3 km "promedia" el relieve.
  3. Calcula viento, sensación térmica y humedad, y clasifica cada hora.
  4. Busca bloques de horas buenas seguidas dentro del horario típico de ataque
     (de madrugada al mediodía, hora de Lima): esas son las ventanas de cumbre.

Salidas (en app/salidas/):
  ventana_cumbre.png   tablero cumbre x hora con el semáforo
  ventana_cumbre.json  todo el detalle, para una web, un bot o un LLM que lo explique

El modelo viene de `predecir()` en ejemplo_producto.py. Hoy es el modelo de RELLENO de la
plantilla: cuando el equipo tenga su modelo de Kaggle, se cambia ahí y este producto lo usa solo.

Uso:
    python app/ventana_cumbre.py                       # próximas 48 h
    python app/ventana_cumbre.py --horas 72
    python app/ventana_cumbre.py --desde 2026-07-01 --hasta 2026-07-03   # fechas pasadas
    python app/ventana_cumbre.py --gfs-npz gfs_cache.npz                 # sin internet, con una descarga guardada
"""
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ejemplo_producto import predecir
from gfs_forecast import get_gfs_forecast, make_windows

APP = Path(__file__).resolve().parent
RAIZ = APP.parent

GRADIENTE_K_POR_M = 0.0065   # la temperatura baja ~6.5 °C por cada km de altura

# Umbrales del semáforo (ajustables con guías de montaña). Valores en la cumbre.
UMBRALES = {
    "viento_bueno_ms": 10.0,      # < 36 km/h: se puede avanzar en arista
    "viento_regular_ms": 15.0,    # < 54 km/h: incómodo, riesgo en tramos expuestos
    "sensacion_buena_c": -20.0,   # > -20 °C: frío manejable con equipo de altura
    "sensacion_regular_c": -28.0, # > -28 °C: riesgo de congelamiento en exposición larga
    "rh_buena": 80.0,             # < 80 %: menos probabilidad de nube / nevada
    "rh_regular": 92.0,
}
HORA_ATAQUE_LIMA = (0, 13)        # el ataque suele salir de madrugada y bajar antes del mediodía
NIVELES = ["mala", "regular", "buena"]


# ---------------------------------------------------------------------------
# Física simple
# ---------------------------------------------------------------------------
def sensacion_termica(t_c, viento_ms):
    """Wind chill (fórmula NOAA/Environment Canada). Válida para T <= 10 °C y viento > 1.3 m/s."""
    v = np.maximum(viento_ms, 0.0) * 3.6                    # km/h
    wc = 13.12 + 0.6215 * t_c - 11.37 * v ** 0.16 + 0.3965 * t_c * v ** 0.16
    return np.where((t_c <= 10) & (v > 4.8), wc, t_c)


def clasificar(viento, sensacion, rh, u=UMBRALES):
    """0 = mala, 1 = regular, 2 = buena. Manda la peor de las tres condiciones."""
    def nivel(bueno, regular):
        return np.where(bueno, 2, np.where(regular, 1, 0))
    nv = nivel(viento < u["viento_bueno_ms"], viento < u["viento_regular_ms"])
    ns = nivel(sensacion > u["sensacion_buena_c"], sensacion > u["sensacion_regular_c"])
    nh = nivel(rh < u["rh_buena"], rh < u["rh_regular"])
    return np.minimum(np.minimum(nv, ns), nh)


# ---------------------------------------------------------------------------
# De la salida del modelo a cada cumbre
# ---------------------------------------------------------------------------
def pixel_mas_cercano(lat, lon, wrf_lats, wrf_lons):
    """Índices (fila, columna) del píxel WRF más cercano. Fila 0 = sur."""
    if not (wrf_lats.min() <= lat <= wrf_lats.max() and wrf_lons.min() <= lon <= wrf_lons.max()):
        return None
    return int(np.abs(wrf_lats - lat).argmin()), int(np.abs(wrf_lons - lon).argmin())


def condiciones_cumbre(Y, cumbre, static):
    """Serie horaria en la cumbre: T (°C), viento (m/s), sensación (°C), RH (%) y nivel."""
    ij = pixel_mas_cercano(cumbre["lat"], cumbre["lon"], static["wrf_lats"], static["wrf_lons"])
    if ij is None:
        return None
    i, j = ij
    z_pixel = float(static["dem"][i, j])
    rh = np.clip(Y[:, i, j, 0], 0, 100)
    t_c = Y[:, i, j, 1] - 273.15 - GRADIENTE_K_POR_M * (cumbre["altitud_m"] - z_pixel)
    viento = np.hypot(Y[:, i, j, 2], Y[:, i, j, 3])
    sens = sensacion_termica(t_c, viento)
    return {
        "pixel": [i, j], "altitud_pixel_m": round(z_pixel),
        "T_c": t_c, "viento_ms": viento, "sensacion_c": sens, "RH": rh,
        "nivel": clasificar(viento, sens, rh),
    }


def a_lima(t_utc):
    return datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)


def buscar_ventanas(nivel, times, min_horas=4, horario=HORA_ATAQUE_LIMA):
    """Bloques de >= min_horas horas 'buenas' seguidas dentro del horario de ataque (hora de Lima)."""
    horas = [a_lima(t) for t in times]
    ok = [n == 2 and horario[0] <= h.hour <= horario[1] for n, h in zip(nivel, horas)]
    ventanas, ini = [], None
    for k in range(len(ok) + 1):
        seguida = k < len(ok) and ok[k] and (ini is None or horas[k] - horas[k - 1] == timedelta(hours=1))
        if seguida and ini is None:
            ini = k
        elif not seguida and ini is not None:
            if k - ini >= min_horas:
                ventanas.append((ini, k - 1))
            ini = k if (k < len(ok) and ok[k]) else None
    return ventanas


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
def guardar_tablero(resultados, times, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    nombres = [r["nombre"] for r in resultados]
    M = np.array([r["cond"]["nivel"] for r in resultados], dtype=float)
    horas = [a_lima(t) for t in times]
    colores = ["#d64541", "#f2b134", "#3aa655"]

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(times) + 3), 0.6 * len(nombres) + 2), constrained_layout=True)
    ax.imshow(M, aspect="auto", cmap=ListedColormap(colores), vmin=0, vmax=2, interpolation="nearest")
    for fila, r in enumerate(resultados):
        for a, b in r["ventanas_idx"]:
            ax.add_patch(plt.Rectangle((a - 0.5, fila - 0.5), b - a + 1, 1, fill=False, lw=2.2, ec="black"))
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels([f"{r['nombre']} ({r['altitud_m']} m)" for r in resultados])
    marcas = [k for k, h in enumerate(horas) if h.hour % 6 == 0]
    ax.set_xticks(marcas)
    ax.set_xticklabels([horas[k].strftime("%a %d\n%H:00") for k in marcas], fontsize=8)
    ax.set_xlabel("Hora de Lima")
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colores, NIVELES)]
              + [Patch(fill=False, ec="black", lw=2, label="ventana de cumbre")],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title(titulo)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def resumen_json(resultados, times, modelo):
    def serie(r, k):
        return [round(float(x), 1) for x in r["cond"][k]]
    return {
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modelo": modelo,
        "horas_lima": [a_lima(t).strftime("%Y-%m-%d %H:00") for t in times],
        "umbrales": UMBRALES,
        "horario_ataque_lima": list(HORA_ATAQUE_LIMA),
        "cumbres": [{
            "nombre": r["nombre"], "altitud_m": r["altitud_m"],
            "pixel_wrf": r["cond"]["pixel"], "altitud_pixel_m": r["cond"]["altitud_pixel_m"],
            "ventanas": [{"inicio": a_lima(times[a]).strftime("%Y-%m-%d %H:00"),
                          "fin": a_lima(times[b]).strftime("%Y-%m-%d %H:00"),
                          "horas": b - a + 1} for a, b in r["ventanas_idx"]],
            "T_c": serie(r, "T_c"), "viento_ms": serie(r, "viento_ms"),
            "sensacion_c": serie(r, "sensacion_c"), "RH": serie(r, "RH"),
            "nivel": [NIVELES[int(n)] for n in r["cond"]["nivel"]],
        } for r in resultados],
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Ventana de Cumbre: mejores horas para atacar cumbre")
    ap.add_argument("--static", default=str(RAIZ / "data" / "static.npz"), help="ruta a static.npz (de Kaggle)")
    ap.add_argument("--cumbres", default=str(APP / "cumbres.json"))
    ap.add_argument("--horas", type=int, default=48, help="horas de pronóstico hacia adelante")
    ap.add_argument("--desde", help="fecha pasada YYYY-MM-DD (opcional)")
    ap.add_argument("--hasta", help="fecha pasada YYYY-MM-DD (opcional)")
    ap.add_argument("--gfs-npz", help="usar un GFS ya descargado (npz con X y times) en vez de Open-Meteo")
    ap.add_argument("--min-horas", type=int, default=4, help="horas buenas seguidas para contar como ventana")
    ap.add_argument("--salidas", default=str(APP / "salidas"))
    a = ap.parse_args()

    if not os.path.exists(a.static):
        raise SystemExit(f"No encuentro {a.static}. Baja la data de Kaggle a la carpeta data/ (ver README).")
    static = dict(np.load(a.static))
    cumbres = json.load(open(a.cumbres, encoding="utf-8"))["cumbres"]

    if a.gfs_npz:
        d = np.load(a.gfs_npz)
        X, times = d["X"], d["times"].astype(str)
    elif a.desde:
        X, times = get_gfs_forecast(start_date=a.desde, end_date=a.hasta or a.desde)
    else:
        X, times = get_gfs_forecast(forecast_hours=a.horas, cache_path=str(APP / "gfs_cache.npz"))
    Xw, times = make_windows(X, times)
    Y = predecir(Xw, static)
    assert Y.shape[1:] == (70, 35, 4), f"predecir() debe devolver (N, 70, 35, 4), devolvió {Y.shape}"

    resultados = []
    for c in cumbres:
        cond = condiciones_cumbre(Y, c, static)
        if cond is None:
            print(f"  (aviso) {c['nombre']} está fuera del dominio de 3 km; se omite.")
            continue
        resultados.append({**c, "cond": cond, "ventanas_idx": buscar_ventanas(cond["nivel"], times, a.min_horas)})
    if not resultados:
        raise SystemExit("Ninguna cumbre de cumbres.json cae dentro del dominio del modelo.")

    os.makedirs(a.salidas, exist_ok=True)
    periodo = f"{a_lima(times[0]):%a %d/%m %H:00} a {a_lima(times[-1]):%a %d/%m %H:00} (hora de Lima)"
    guardar_tablero(resultados, times, os.path.join(a.salidas, "ventana_cumbre.png"), f"Ventana de Cumbre · {periodo}")
    modelo = "relleno de la plantilla (interpolación + altura)"   # cámbialo al poner tu modelo en predecir()
    with open(os.path.join(a.salidas, "ventana_cumbre.json"), "w", encoding="utf-8") as f:
        json.dump(resumen_json(resultados, times, modelo), f, ensure_ascii=False, indent=1)

    print(f"Periodo: {periodo}\n")
    for r in resultados:
        cd = r["cond"]
        print(f"{r['nombre']} ({r['altitud_m']} m) · T {cd['T_c'].min():.0f} a {cd['T_c'].max():.0f} °C · "
              f"viento máx {cd['viento_ms'].max() * 3.6:.0f} km/h · sensación mín {cd['sensacion_c'].min():.0f} °C")
        if r["ventanas_idx"]:
            for x, y in r["ventanas_idx"]:
                print(f"   ventana: {a_lima(times[x]):%a %d/%m %H:00} a {a_lima(times[y]):%H:00} ({y - x + 1} h)")
        else:
            print("   sin ventana de cumbre en este periodo")
    print(f"\nTablero y detalle en {a.salidas}/ventana_cumbre.png y .json")


if __name__ == "__main__":
    main()
