import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class World(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int):
        import pyworld
        super().__init__(a, fs)
        f0, sp, ap = pyworld.wav2world(a.astype(float), fs)
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
