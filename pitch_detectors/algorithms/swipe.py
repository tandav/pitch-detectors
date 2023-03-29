import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector


class Swipe(PitchDetector):
    """https://pysptk.readthedocs.io/en/stable/generated/pysptk.sptk.swipe.html"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
    ):
        import pysptk
        super().__init__(a, fs)
        f0 = pysptk.sptk.swipe(self.a, fs=self.fs, min=hz_min, max=hz_max, hopsize=250)
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
