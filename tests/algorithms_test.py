import pytest

from pitch_detectors.algorithms import ALGORITHMS


@pytest.mark.order(3)
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_detection(algorithm, record):
    alg = algorithm(record.a, record.fs)
    assert alg.f0.shape == alg.t.shape
