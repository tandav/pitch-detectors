import dataclasses
from pathlib import Path

import numpy as np
import pytest
from dsplib.scale import minmax_scaler
from scipy.io import wavfile

from pitch_detectors.algorithms import ALGORITHMS


@dataclasses.dataclass
class Record:
    a: np.ndarray
    fs: int


@pytest.fixture
def record(rescale: float = 100000):
    """audio and fs"""
    fs, a = wavfile.read(Path(__file__).parent / 'data' / 'b1a5da49d564a7341e7e1327aa3f229a.wav')
    a = minmax_scaler(a, a.min(), a.max(), -rescale, rescale).astype(np.float32)
    assert a.dtype == np.float32
    return Record(a, fs)


@pytest.mark.filterwarnings('ignore:pkg_resources is deprecated as an API')
@pytest.mark.filterwarnings('ignore:Deprecated call to `pkg_resources.declare_namespace')
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_detection(algorithm, record):
    algorithm(record.a, record.fs)
