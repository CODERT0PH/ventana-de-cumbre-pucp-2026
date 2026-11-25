import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def static():
    """static.npz sintético con la misma forma y orientación que el oficial."""
    rng = np.random.default_rng(0)
    wrf_lats = np.linspace(-10.5, -8.6, 70)          # fila 0 = sur
    wrf_lons = np.linspace(-78.2, -77.2, 35)
    LAT, LON = np.meshgrid(wrf_lats, wrf_lons, indexing="ij")
    dem = 3000 + 1500 * np.sin(LAT * 6) * np.cos(LON * 5) + rng.normal(0, 50, LAT.shape)
    return {
        "dem": dem.astype(np.float32),
        "gfs_lats": np.round(np.arange(-7.25, -11.25 - 1e-9, -0.25), 2),
        "gfs_lons": np.round(np.arange(-79.0, -76.0 + 1e-9, 0.25), 2),
        "wrf_lats": wrf_lats, "wrf_lons": wrf_lons,
    }


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(1)
    t = pd.date_range("2024-06-01", periods=96, freq="h").strftime("%Y%m%d%H").to_numpy()
    X = rng.normal([70, 280, 0, 0], [10, 5, 3, 3], (96, 17, 13, 4)).astype(np.float32)
    Y = rng.normal([70, 278, 0, 0], [12, 7, 4, 4], (96, 70, 35, 4)).astype(np.float32)
    return X, Y, t
