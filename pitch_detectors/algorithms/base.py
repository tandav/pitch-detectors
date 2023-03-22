import os

import numpy as np

from pitch_detectors import util


class PitchDetector:
    use_gpu = False

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        self.a = a
        self.fs = fs
        self.hz_min = hz_min
        self.hz_max = hz_max
        self.seconds = len(a) / fs
        self.f0: np.ndarray
        self.t: np.ndarray
        if (
            os.environ.get('PITCH_DETECTORS_GPU') == 'true' and
            self.use_gpu and
            not self.gpu_available()
        ):
            raise ConnectionError(f'gpu must be available for {self.name()} algorithm')

    def dict(self) -> dict[str, list[float | None]]:
        return {'f0': util.nan_to_none(self.f0.tolist()), 't': self.t.tolist()}

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def gpu_available(self) -> bool:
        return False


class TensorflowGPU:
    use_gpu = True

    def gpu_available(self) -> bool:
        import tensorflow as tf
        return bool(tf.config.experimental.list_physical_devices('GPU'))


class TorchGPU:
    use_gpu = True

    def gpu_available(self) -> bool:
        import torch
        return torch.cuda.is_available()  # type: ignore
