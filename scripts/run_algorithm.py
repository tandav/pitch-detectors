import argparse
import os

import numpy as np

from pitch_detectors import algorithms
from pitch_detectors import util
from pitch_detectors.algorithms.ensemble import Ensemble
from pitch_detectors.algorithms.ensemble import vote_and_median
from pitch_detectors.schemas import F0


def main(
    audio_path: str,
    algorithm: str,
) -> None:
    fs, a = util.load_wav(audio_path)
    if algorithm == 'ensemble':
        alg = Ensemble(a, fs, algorithms=algorithms.ALGORITHMS)
        algorithms_cache = {k.name(): F0(t=alg.t, f0=alg.f0) for k, alg in alg._algorithms.items()}
        vm = vote_and_median(algorithms_cache, alg.seconds)
        assert np.array_equal(alg.t, vm.t)
        assert np.array_equal(alg.f0, vm.f0, equal_nan=True)
    else:
        alg = getattr(algorithms, os.environ['PITCH_DETECTORS_ALGORITHM'])(a, fs)

    assert alg.f0.shape == alg.t.shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-path', type=str, default=os.environ.get('PITCH_DETECTORS_AUDIO_PATH', 'data/b1a5da49d564a7341e7e1327aa3f229a.wav'))
    parser.add_argument('--algorithm', type=str, default=os.environ.get('PITCH_DETECTORS_ALGORITHM'))
    args = parser.parse_args()
    main(**vars(args))
