"""Ventana de Cumbre · núcleo de downscaling GFS (27 km) -> WRF (3 km).

El mismo código lo usan el notebook de submission (train/) y el producto (app/),
para que el preprocesamiento y el modelo sean idénticos en Kaggle y en la app.
"""

VARS = ["RH", "T", "U", "V"]
UNITS = ["%", "K", "m/s", "m/s"]
GFS_SHAPE = (17, 13)
WRF_SHAPE = (70, 35)
EVAL_HOURS = tuple(range(0, 24, 3))  # Kaggle evalúa 00, 03, ..., 21 UTC
