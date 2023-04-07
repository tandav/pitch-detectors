import os
from pathlib import Path

import pytest

from pitch_detectors import util
from pitch_detectors.schemas import Record


@pytest.fixture
def record():
    fs, a = util.load_wav(Path(__file__).parent.parent / 'data' / 'b1a5da49d564a7341e7e1327aa3f229a.wav')
    return Record(fs=fs, a=a)


@pytest.fixture
def environ():
    env = {
        'PITCH_DETECTORS_GPU_MEMORY_LIMIT': 'true',
        'LD_LIBRARY_PATH': util.ld_library_path(),
    }
    if 'PITCH_DETECTORS_PENN_CHECKPOINT_PATH' in os.environ:
        env['PITCH_DETECTORS_PENN_CHECKPOINT_PATH'] = os.environ['PITCH_DETECTORS_PENN_CHECKPOINT_PATH']
    if 'PITCH_DETECTORS_SPICE_MODEL_PATH' in os.environ:
        env['PITCH_DETECTORS_SPICE_MODEL_PATH'] = os.environ['PITCH_DETECTORS_SPICE_MODEL_PATH']
    return env


@pytest.fixture
def subprocess_warning():
    return '''\
    Running in subprocess to avoid pytorch cuda import caching.
    It's difficult to disable gpu after it has been initialized.
    It is also difficult to set LD_LIBRARY_PATH without a subprocess.
    '''
