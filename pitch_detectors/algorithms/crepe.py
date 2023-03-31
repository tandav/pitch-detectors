import numpy as np

from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TensorflowGPU


class Crepe(TensorflowGPU, PitchDetector):
    """https://github.com/marl/crepe"""

    def __init__(self, a: np.ndarray, fs: int, confidence_threshold: float = 0.8):
        TensorflowGPU.__init__(self)
        PitchDetector.__init__(self, a, fs)
        import crepe

        self.t, self.f0, self.confidence, self.activation = crepe.predict(self.a, sr=self.fs, viterbi=True)
        self.f0[self.confidence < confidence_threshold] = np.nan
