import os
import subprocess
import sys

import pytest

from pitch_detectors import algorithms


@pytest.mark.order(3)
@pytest.mark.parametrize('gpu', ['true', 'false'], ids=['gpu', 'cpu'])
@pytest.mark.parametrize('algorithm', (*algorithms.algorithms, 'Ensemble'))
def test_detection(algorithm, gpu):
    env = {
        'PITCH_DETECTORS_GPU_MEMORY_LIMIT': 'true',
        'PITCH_DETECTORS_AUDIO_PATH': 'data/b1a5da49d564a7341e7e1327aa3f229a.wav',
        'PATH': '',  # for some reason this line prevents SIGSEGV for Spice algorithm
        # 'PATH': '/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',  # this is from docker history of base cuda image https://hub.docker.com/layers/nvidia/cuda/12.4.1-cudnn-devel-ubuntu22.04/images/sha256-0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4?context=explore
        'PITCH_DETECTORS_ALGORITHM': algorithm,
        'PITCH_DETECTORS_GPU': gpu,
    }
    if 'PITCH_DETECTORS_PENN_CHECKPOINT_PATH' in os.environ:
        env['PITCH_DETECTORS_PENN_CHECKPOINT_PATH'] = os.environ['PITCH_DETECTORS_PENN_CHECKPOINT_PATH']
    if 'PITCH_DETECTORS_SPICE_MODEL_PATH' in os.environ:
        env['PITCH_DETECTORS_SPICE_MODEL_PATH'] = os.environ['PITCH_DETECTORS_SPICE_MODEL_PATH']
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
