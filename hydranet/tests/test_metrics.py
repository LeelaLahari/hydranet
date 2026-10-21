import numpy as np
from src.evaluation.metrics import nse, kge, pbias, iou, f1, contingency

def test_nse_perfect():
    q = np.random.rand(100) + 1
    assert nse(q, q) == 1.0

def test_nse_worse_than_mean():
    q = np.ones(100)
    assert nse(q, np.zeros(100)) < 0

def test_kge_perfect():
    q = np.random.rand(100) + 2
    assert abs(kge(q, q) - 1) < 1e-9

def test_extent_scores():
    a = np.zeros((10, 10)); b = np.zeros((10, 10))
    a[:5, :5] = 1; b[:5, :5] = 1
    assert iou(a, b) == 1.0 and f1(a, b) == 1.0
    c = contingency(a, b)
    assert c["POD"] == 1.0 and c["FAR"] == 0.0 and c["CSI"] == 1.0
    a[0, 9] = 1  # one false alarm
    c = contingency(a, b)
    assert 0 < c["FAR"] < 1

def test_pbias_sign():
    q = np.ones(50)
    assert pbias(q, q * 2) > 0  # overestimation -> positive
