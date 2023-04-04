import argparse
import os

import numpy as np

from pitch_detectors import algorithms
from pitch_detectors import util
from pitch_detectors.schemas import F0


def main(
    audio_path: str,
    algorithm: str,
) -> None:
    fs, a = util.load_wav(audio_path)
    if algorithm == 'ensemble':
        alg = algorithms.Ensemble(a, fs, algorithms=algorithms.ALGORITHMS)
    else:
        alg = getattr(algorithms, os.environ['PITCH_DETECTORS_ALGORITHM'])(a, fs)

    assert alg.f0.shape == alg.t.shape

    if algorithm == 'ensemble':
        algorithms_cache = {k: F0(alg.t, alg.f0) for k, alg in alg._algorithms.items()}
        alg_from_cache = algorithms.Ensemble(a, fs, algorithms_cache=algorithms_cache)
        assert np.array_equal(alg.t, alg_from_cache.t)
        assert np.array_equal(alg.f0, alg_from_cache.f0, equal_nan=True)
        # data = alg.dict()
        # data['algorithms_cache'] = {k: _alg.dict() for k, _alg in alg._algorithms.items()}
        # print(json.dumps(alg.dict(), allow_nan=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-path', type=str, default=os.environ.get('PITCH_DETECTORS_AUDIO_PATH', 'data/b1a5da49d564a7341e7e1327aa3f229a.wav'))
    parser.add_argument('--algorithm', type=str, default=os.environ.get('PITCH_DETECTORS_ALGORITHM'))
    args = parser.parse_args()
    main(**vars(args))
