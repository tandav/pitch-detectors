import argparse
import os

from pitch_detectors import algorithms
from pitch_detectors import util


def main(audio_path: str, algorithm: str) -> None:
    fs, a = util.load_wav(audio_path)
    alg = getattr(algorithms, algorithm)(a, fs)
    assert alg.f0.shape == alg.t.shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-path', type=str, default=os.environ.get('PITCH_DETECTORS_AUDIO_PATH'))
    parser.add_argument('--algorithm', type=str, default=os.environ.get('PITCH_DETECTORS_ALGORITHM'))
    args = parser.parse_args()
    main(**vars(args))
