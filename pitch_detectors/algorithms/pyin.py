import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class Pyin(PitchDetector):
    """https://librosa.org/doc/latest/generated/librosa.pyin.html"""

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import librosa
        super().__init__(a, fs, hz_min, hz_max)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.a, sr=self.fs, fmin=self.hz_min, fmax=self.hz_max,
            resolution=0.1,  # Resolution of the pitch bins. 0.01 corresponds to cents.
            frame_length=2048,
        )
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])
