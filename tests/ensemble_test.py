import os
from unittest import mock

import numpy as np

from pitch_detectors import algorithms
from pitch_detectors.schemas import F0


def test_ensemble(record, environ):
    if os.environ.get('PITCH_DETECTORS_GPU') == 'false':
        env = environ
    else:
        env = environ | {'PITCH_DETECTORS_GPU': 'true'}
    with mock.patch.dict(os.environ, env):
        alg = algorithms.Ensemble(record.a, record.fs, algorithms=algorithms.ALGORITHMS)
        assert alg.f0.shape == alg.t.shape
        algorithms_cache = {k: F0(alg.t, alg.f0) for k, alg in alg._algorithms.items()}
        alg_from_cache = algorithms.Ensemble(record.a, record.fs, algorithms_cache=algorithms_cache)
        assert np.array_equal(alg.t, alg_from_cache.t)
        assert np.array_equal(alg.f0, alg_from_cache.f0, equal_nan=True)
