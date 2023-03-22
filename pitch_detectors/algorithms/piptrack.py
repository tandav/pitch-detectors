import numpy as np

from pitch_detectors.algorithms.base import PitchDetector


class PipTrack(PitchDetector):
    """https://librosa.org/doc/latest/generated/librosa.piptrack.html"""

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600, threshold: float = 0.1):
        import librosa
        super().__init__(a, fs, hz_min, hz_max)
        pitches, magnitudes = librosa.piptrack(
            y=a,
            sr=fs,
            fmin=hz_min,
            fmax=hz_max,
            threshold=threshold,
        )
        max_indexes = np.argmax(magnitudes, axis=0)
        f0 = pitches[max_indexes, range(magnitudes.shape[1])]
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, self.f0.shape[0])
