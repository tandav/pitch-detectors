import csv
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pitch_detectors.evaluation.datasets.base import Dataset
from pitch_detectors.schemas import F0
from pitch_detectors.schemas import Wav


class MDBStemSynth(Dataset):

    @classmethod
    def dataset_dir(cls) -> Path:
        return Path(os.environ.get('DATASET_DIR_MDB_STEM_SYNTH', 'f0-datasets/mdb-stem-synth/MDB-stem-synth'))

    @classmethod
    def wav_dir(cls) -> Path:
        return cls.dataset_dir() / 'audio_stems'

    @classmethod
    def iter_wav_files(cls) -> Iterator[Path]:
        it = cls.wav_dir().glob('*.wav')
        it = (f for f in it if not f.name.startswith('.'))
        yield from it

    @classmethod
    def load_wav(cls, wav_path: Path) -> Wav:
        fs, a = wavfile.read(wav_path)
        a = a.astype(np.float32)
        return Wav(fs=fs, a=a)

    @classmethod
    def load_true(cls, wav_path: Path, seconds: float) -> F0:
        file = (cls.dataset_dir() / 'annotation_stems' / wav_path.name).with_suffix('.csv')
        t, f0 = [], []
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for _t, _f0 in reader:
                t.append(_t)
                f0.append(_f0)
        t = np.array(t)
        f0 = np.array(f0)
        return F0(t=t, f0=f0)
