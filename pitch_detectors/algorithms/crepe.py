import numpy as np

from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TensorflowGPU


class Crepe(TensorflowGPU, PitchDetector):
    """https://github.com/marl/crepe"""

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600, confidence_threshold: float = 0.8):
        import crepe
        import tensorflow as tf
        super().__init__(a, fs, hz_min, hz_max)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.t, self.f0, self.confidence, self.activation = crepe.predict(self.a, sr=self.fs, viterbi=True)
        self.f0[self.confidence < confidence_threshold] = np.nan
