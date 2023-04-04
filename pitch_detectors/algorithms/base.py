import os

import numpy as np

from pitch_detectors import util


class PitchDetector:
    gpu_capable = False

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


class UsesGPU:
    gpu_capable = True
    memory_limit_initialized = False

    def __init__(self, gpu: bool | None = None) -> None:
        self.gpu = gpu or os.environ.get('PITCH_DETECTORS_GPU') == 'true'
        if self.gpu and not self.gpu_available():
            raise ConnectionError('gpu must be available')
        if not self.gpu:
            self.disable_gpu()
            if self.gpu_available():
                raise ConnectionError('gpu must not be available')

    def gpu_available(self) -> bool:
        return False

    def disable_gpu(self) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TensorflowGPU(UsesGPU):

    def __init__(self, gpu: bool | None = None) -> None:

        import tensorflow as tf
        self.tf = tf
        super().__init__(gpu)
        if self.gpu_available() and os.environ.get('PITCH_DETECTORS_GPU_MEMORY_LIMIT') == 'true':
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


class TorchGPU(UsesGPU):

    def __init__(self, gpu: bool | None = None) -> None:
        import torch
        self.torch = torch
        super().__init__(gpu)
        if self.gpu_available() and os.environ.get('PITCH_DETECTORS_GPU_MEMORY_LIMIT') == 'true':
            self.set_memory_limit()

    def set_memory_limit(self) -> None:
        if TorchGPU.memory_limit_initialized:
            return
        self.torch.cuda.set_per_process_memory_fraction(fraction=1 / 8, device=0)
        TorchGPU.memory_limit_initialized = True

    def gpu_available(self) -> bool:
        return self.torch.cuda.is_available()  # type: ignore
