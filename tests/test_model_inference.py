import numpy as np
import torch

from downscaler.data import make_windows
from downscaler.features import dewpoint_k, terrain_features
from downscaler.inference import Downscaler, Preprocessor


def test_terrain_and_dewpoint(static):
    tf = terrain_features(static["dem"])
    assert all(v.shape == (70, 35) for v in tf.values())
    td = dewpoint_k(np.array([283.15]), np.array([100.0]))
    assert abs(td[0] - 283.15) < 0.05


def test_end_to_end_save_load_predict(tmp_path, static, toy_data):
    X, Y, t = toy_data
    pre = Preprocessor.fit(X, Y, t, static)
    model = pre.build_model(width=16, depth=2)
    W = make_windows(X, t)
    out = model(torch.tensor(pre.norm_x(W[:4])), torch.zeros(4))
    assert out.shape == (4, 70, 35, 4)

    torch.save(model.state_dict(), tmp_path / "m.pt")
    pre.save(tmp_path / "a.npz")
    ds = Downscaler(tmp_path / "m.pt", tmp_path / "a.npz", device="cpu", width=16, depth=2)
    P = ds.predict(W[:5], t[:5])
    assert P.shape == (5, 70, 35, 4) and np.isfinite(P).all()
    assert 200 < P[..., 1].mean() < 350    # vuelve a Kelvin
