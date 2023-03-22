import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class Swipe(PitchDetector):
    """https://pysptk.readthedocs.io/en/stable/generated/pysptk.sptk.swipe.html"""

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import pysptk
        super().__init__(a, fs, hz_min, hz_max)
        f0 = pysptk.sptk.swipe(self.a, fs=self.fs, min=self.hz_min, max=self.hz_max, hopsize=250)
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
