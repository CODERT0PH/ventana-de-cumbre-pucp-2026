import numpy as np
import pandas as pd

from downscaler.data import blocked_folds, eval_mask, make_windows, temporal_holdout
from downscaler.metric import climatology, skill_score


def test_windows_follow_real_time_with_gaps():
    t = np.array(["2024060100", "2024060101", "2024060103"])     # falta la hora 02
    X = np.arange(3, dtype=np.float32)[:, None, None, None] * np.ones((3, 17, 13, 4), np.float32)
    W = make_windows(X, t)
    assert W.shape == (3, 3, 17, 13, 4)
    assert W[1, :, 0, 0, 0].tolist() == [1, 0, 1]     # t-2 no existe -> usa t (como el starter)
    assert W[2, :, 0, 0, 0].tolist() == [1, 2, 2]     # t-1 (hora 02) falta -> usa t


def test_metric_zero_for_climatology_and_one_for_perfect(toy_data):
    _, Y, t = toy_data
    clim = climatology(Y, t)
    s_clim, _ = skill_score(clim[[int(k[-2:]) for k in t]], Y, t, clim)
    s_perf, per_var = skill_score(Y, Y, t, clim)
    assert abs(s_clim) < 1e-9 and abs(s_perf - 1) < 1e-9 and set(per_var) == {"RH", "T", "U", "V"}


def test_eval_mask_every_three_hours(toy_data):
    _, _, t = toy_data
    assert eval_mask(t).sum() == 96 // 3


def test_holdout_and_folds_have_purge():
    t = pd.date_range("2024-02-18", "2024-06-30 23:00", freq="h").strftime("%Y%m%d%H").to_numpy()
    trn, val = temporal_holdout(t, purge_hours=48)
    assert not (trn & val).any()
    gap = pd.to_datetime(t[val][0], format="%Y%m%d%H") - pd.to_datetime(t[trn][-1], format="%Y%m%d%H")
    assert gap >= pd.Timedelta(hours=48)
    for trn, val in blocked_folds(t, 5, 48):
        assert not (trn & val).any() and val.sum() > 0
