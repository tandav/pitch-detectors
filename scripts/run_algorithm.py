import os

from pitch_detectors import algorithms
from pitch_detectors import util


def main() -> None:
    fs, a = util.load_wav('data/b1a5da49d564a7341e7e1327aa3f229a.wav')
    alg = getattr(algorithms, os.environ['PITCH_DETECTORS_ALGORITHM'])(a, fs)
    if alg.f0.shape != alg.t.shape:
        raise AssertionError(f'f0.shape != t.shape: {alg.f0.shape} != {alg.t.shape}')


if __name__ == '__main__':
    main()
