import os
import subprocess
import sys

import pytest

from pitch_detectors.algorithms import ALGORITHMS


@pytest.mark.order(3)
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('gpu', ['false'] if os.environ.get('PITCH_DETECTORS_GPU') == 'false' else ['true', 'false'])
def test_detection(algorithm, environ, gpu, subprocess_warning):
    env = environ | {
        'PITCH_DETECTORS_ALGORITHM': algorithm.name(),
        'PITCH_DETECTORS_GPU': gpu,
    }
    print(subprocess_warning)
    subprocess.check_call([sys.executable, 'scripts/run_algorithm.py'], env=env)
