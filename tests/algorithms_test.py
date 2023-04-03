import subprocess
import sys

import pytest

from pitch_detectors.algorithms import ALGORITHMS


@pytest.mark.order(3)
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('gpu', ['true', 'false'])
def test_detection(algorithm, environ, gpu):
    env = environ | {
        'PITCH_DETECTORS_ALGORITHM': algorithm.name(),
        'PITCH_DETECTORS_GPU': gpu,
    }
    # run in subprocess to avoid pytorch cuda import caching
    # it's difficult to disable gpu after it has been initialized
    subprocess.check_call([sys.executable, 'scripts/run_algorithm.py'], env=env)
