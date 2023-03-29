import numpy as np

from pitch_detectors import config
from pitch_detectors.algorithms.base import PitchDetector


class Yin(PitchDetector):
    """https://librosa.org/doc/latest/generated/librosa.yin.html#librosa.yin"""

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        hz_min: float = config.HZ_MIN,
        hz_max: float = config.HZ_MAX,
        trough_threshold: float = 0.1,
    ):
        import librosa
        super().__init__(a, fs)
        f0 = librosa.yin(
            self.a, sr=self.fs, fmin=hz_min, fmax=hz_max,
            frame_length=2048,
            trough_threshold=trough_threshold,
        )
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
