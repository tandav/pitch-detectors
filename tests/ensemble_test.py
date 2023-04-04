import os
import subprocess
import sys


def test_ensemble(environ, subprocess_warning):
    print(subprocess_warning)
    environ['PITCH_DETECTORS_ALGORITHM'] = 'ensemble'
    environ['PITCH_DETECTORS_GPU'] = os.environ.get('PITCH_DETECTORS_GPU', 'true')
    subprocess.check_call([sys.executable, 'scripts/run_algorithm.py'], env=environ)
