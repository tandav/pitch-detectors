import os
import subprocess
import sys

import pytest

from pitch_detectors import algorithms


@pytest.mark.order(3)
@pytest.mark.parametrize('algorithm', algorithms.ALGORITHMS)
@pytest.mark.parametrize('gpu', ['false'] if os.environ.get('PITCH_DETECTORS_GPU') == 'false' else ['true', 'false'])
def test_detection(algorithm, environ, gpu, subprocess_warning):
    env = environ | {
        'PITCH_DETECTORS_ALGORITHM': algorithm.name(),
        'PITCH_DETECTORS_GPU': gpu,
    }
    print(subprocess_warning)
    subprocess.check_call([sys.executable, 'scripts/run_algorithm.py'], env=env)


def test_uses_gpu_framework():
    assert algorithms.Crepe.uses_gpu_framework is True
    assert algorithms.Ensemble .uses_gpu_framework is True
    assert algorithms.Penn.uses_gpu_framework is True
    assert algorithms.PipTrack.uses_gpu_framework is False
    assert algorithms.PraatAC.uses_gpu_framework is False
    assert algorithms.PraatCC.uses_gpu_framework is False
    assert algorithms.PraatSHS.uses_gpu_framework is False
    assert algorithms.Pyin.uses_gpu_framework is False
    assert algorithms.Rapt.uses_gpu_framework is False
    assert algorithms.Reaper.uses_gpu_framework is False
    assert algorithms.Spice.uses_gpu_framework is True
    assert algorithms.Swipe.uses_gpu_framework is False
    assert algorithms.TorchCrepe.uses_gpu_framework is True
    assert algorithms.TorchYin.uses_gpu_framework is True
    assert algorithms.World.uses_gpu_framework is False
    assert algorithms.Yaapt.uses_gpu_framework is False
    assert algorithms.Yin.uses_gpu_framework is False
