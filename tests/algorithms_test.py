import pytest

from pitch_detectors.algorithms import ALGORITHMS


@pytest.mark.order(3)
@pytest.mark.filterwarnings('ignore:pkg_resources is deprecated as an API')
@pytest.mark.filterwarnings('ignore:Deprecated call to `pkg_resources.declare_namespace')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated')
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_detection(algorithm, record):
    algorithm(record.a, record.fs)
