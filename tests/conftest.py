from pathlib import Path

import pytest

from pitch_detectors import util
from pitch_detectors.schemas import Record


@pytest.fixture
def record():
    fs, a = util.load_wav(Path(__file__).parent.parent / 'data' / 'b1a5da49d564a7341e7e1327aa3f229a.wav')
    return Record(fs, a)


@pytest.fixture
def environ():
    return {
        'PITCH_DETECTORS_GPU_MEMORY_LIMIT': 'true',
        'PITCH_DETECTORS_PENN_CHECKPOINT_PATH': '/home/tandav/docs/bhairava/libmv/data/fcnf0++.pt',
        'PITCH_DETECTORS_SPICE_MODEL_PATH': '/home/tandav/docs/bhairava/libmv/data/spice_model',
        'LD_LIBRARY_PATH': util.ld_library_path(),
    }
