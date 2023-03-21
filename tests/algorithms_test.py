import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pitch_detectors import util
from pitch_detectors.algorithms import ALGORITHMS


@dataclasses.dataclass
class Record:
    a: np.ndarray
    fs: int


@pytest.fixture
def record():
    fs, a = util.load_wav(Path(__file__).parent.parent / 'data' / 'b1a5da49d564a7341e7e1327aa3f229a.wav')
    return Record(a, fs)


@pytest.mark.order(3)
@pytest.mark.filterwarnings('ignore:pkg_resources is deprecated as an API')
@pytest.mark.filterwarnings('ignore:Deprecated call to `pkg_resources.declare_namespace')
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_detection(algorithm, record):
    algorithm(record.a, record.fs)
