import numpy as np

from pitch_detectors import algorithms


def test_ensemble(record):
    alg = algorithms.Ensemble(record.a, record.fs, algorithms=algorithms.ALGORITHMS)
    assert alg.f0.shape == alg.t.shape
    alg_from_cache = algorithms.Ensemble(record.a, record.fs, algorithms_cache=alg._algorithms)
    assert np.array_equal(alg.t, alg_from_cache.t)
    assert np.array_equal(alg.f0, alg_from_cache.f0, equal_nan=True)
