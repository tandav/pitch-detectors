import os

import numpy as np

from pitch_detectors import util


class PitchDetector:
    use_gpu = False

    def __init__(self, a: np.ndarray, fs: int):
        self.a = a
        self.fs = fs
        self.seconds = len(a) / fs
        self.f0: np.ndarray
        self.t: np.ndarray

    def dict(self) -> dict[str, list[float | None]]:
        return {'f0': util.nan_to_none(self.f0.tolist()), 't': self.t.tolist()}

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def gpu_available(self) -> bool:
        return False


class TensorflowGPU:
    use_gpu = True
    memory_limit_initialized = False

    def __init__(self) -> None:
        import tensorflow as tf
        self.tf = tf

        if (
            os.environ.get('PITCH_DETECTORS_GPU') == 'true' and
            not self.gpu_available()
        ):
            raise ConnectionError('gpu must be available')

        self.set_memory_limit()

    def set_memory_limit(self) -> None:
        if TensorflowGPU.memory_limit_initialized:
            return
        gpus = self.gpus
        for gpu in gpus:
            self.tf.config.experimental.set_memory_growth(gpu, True)
        TensorflowGPU.memory_limit_initialized = True

    @property
    def gpus(self) -> list[str]:
        return self.tf.config.experimental.list_physical_devices('GPU')  # type: ignore

    def gpu_available(self) -> bool:
        return bool(self.gpus)


class TorchGPU:
    use_gpu = True

    def gpu_available(self) -> bool:
        import torch
        return torch.cuda.is_available()  # type: ignore
