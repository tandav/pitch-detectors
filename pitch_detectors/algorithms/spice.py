import numpy as np

from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.algorithms.base import TensorflowGPU


class Spice(TensorflowGPU, PitchDetector):
    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        confidence_threshold: float = 0.8,
        expected_sample_rate: int = 16000,
        spice_model_path: str = '/spice_model',
    ):
        import resampy
        import tensorflow as tf
        import tensorflow_hub as hub
        a = resampy.resample(a, fs, expected_sample_rate)
        super().__init__(a, fs)
        model = hub.load(spice_model_path)
        model_output = model.signatures['serving_default'](tf.constant(a, tf.float32))
        confidence = 1.0 - model_output['uncertainty']
        self.f0 = self.output2hz(model_output['pitch'].numpy())
        self.f0[confidence < confidence_threshold] = np.nan
        self.t = np.linspace(0, self.seconds, self.f0.shape[0])

    def output2hz(
        self,
        pitch_output: np.ndarray,
        pt_offset: float = 25.58,
        pt_slope: float = 63.07,
        fmin: float = 10.0,
        bins_per_octave: float = 12.0,
    ) -> np.ndarray:
        """convert pitch from the model output [0.0, 1.0] range to absolute values in Hz."""
        cqt_bin = pitch_output * pt_slope + pt_offset
        return fmin * 2.0 ** (1.0 * cqt_bin / bins_per_octave)