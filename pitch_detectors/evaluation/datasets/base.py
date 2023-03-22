import abc
from collections.abc import Iterator
from pathlib import Path

from pitch_detectors.schemas import F0
from pitch_detectors.schemas import Wav


class Dataset(abc.ABC):

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    @abc.abstractmethod
    def dataset_dir(cls) -> Path:
        ...

    @classmethod
    @abc.abstractmethod
    def wav_dir(cls) -> Path:
        ...

    @classmethod
    def iter_wav_files(cls) -> Iterator[Path]:
        yield from cls.wav_dir().glob('*.wav')

    @classmethod
    @abc.abstractmethod
    def load_wav(cls, wav_path: Path) -> Wav:
        ...

    @classmethod
    @abc.abstractmethod
    def load_true(cls, wav_path: Path, seconds: float) -> F0:
        ...
