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
        self.f0: np.ndarray | None = None
        self.t: np.ndarray | None = None

    def dict(self) -> dict[str, list[float | None]]:
        if self.f0 is None:
            raise ValueError('f0 must be not None')
        if self.t is None:
            raise ValueError('t must be not None')
        return {'f0': util.nan_to_none(self.f0.tolist()), 't': self.t.tolist()}

    @classmethod
    def name(cls) -> str:
        return cls.__class__.__name__


class PraatAC(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import parselmouth
        super().__init__(a, fs, hz_min, hz_max)
        self.signal = parselmouth.Sound(self.a, sampling_frequency=self.fs)
        self.pitch_obj = self.signal.to_pitch_ac(pitch_floor=self.hz_min, pitch_ceiling=self.hz_max, very_accurate=True)
        self.f0 = self.pitch_obj.selected_array['frequency']
        self.f0[self.f0 == 0] = np.nan
        self.t = self.pitch_obj.xs()


class PraatCC(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import parselmouth
        super().__init__(a, fs, hz_min, hz_max)
        self.signal = parselmouth.Sound(self.a, sampling_frequency=self.fs)
        self.pitch_obj = self.signal.to_pitch_cc(pitch_floor=self.hz_min, pitch_ceiling=self.hz_max, very_accurate=True)
        self.f0 = self.pitch_obj.selected_array['frequency']
        self.f0[self.f0 == 0] = np.nan
        self.t = self.pitch_obj.xs()


class PraatSHS(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import parselmouth
        super().__init__(a, fs, hz_min, hz_max)
        self.signal = parselmouth.Sound(self.a, sampling_frequency=self.fs)
        self.pitch_obj = self.signal.to_pitch_shs(minimum_pitch=self.hz_min, maximum_frequency_component=self.hz_max)
        self.f0 = self.pitch_obj.selected_array['frequency']
        self.f0[self.f0 == 0] = np.nan
        self.t = self.pitch_obj.xs()


class Pyin(PitchDetector):
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


class Crepe(PitchDetector):
    use_gpu = True

    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600, confidence_threshold: float = 0.8):
        import crepe
        import tensorflow as tf
        super().__init__(a, fs, hz_min, hz_max)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('Crepe requires a GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.t, self.f0, self.confidence, self.activation = crepe.predict(self.a, sr=self.fs, viterbi=True)
        self.f0[self.confidence < confidence_threshold] = np.nan


class TorchCrepe(PitchDetector):
    use_gpu = True

    def __init__(
        self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600, confidence_threshold: float = 0.8,
        batch_size: int = 2048,
        device: str | None = None,
    ):
        import torch
        import torchcrepe
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            torch.device(device)

        super().__init__(a, fs, hz_min, hz_max)
        if not torch.cuda.is_available():
            raise RuntimeError('TorchCrepe requires a GPU')

        f0, confidence = torchcrepe.predict(
            torch.from_numpy(a[np.newaxis, ...]),
            fs,
            hop_length=int(fs / 100),  # 10 ms
            fmin=self.hz_min,
            fmax=self.hz_max,
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
        )
        win_length = 3
        f0 = torchcrepe.filter.mean(f0, win_length)
        confidence = torchcrepe.filter.median(confidence, win_length)

        f0 = f0.ravel().numpy()
        confidence = confidence.ravel().numpy()
        f0[confidence < confidence_threshold] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])


class Yaapt(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import amfm_decompy.basic_tools as basic
        from amfm_decompy import pYAAPT
        super().__init__(a, fs, hz_min, hz_max)
        self.signal = basic.SignalObj(data=self.a, fs=self.fs)
        f0 = pYAAPT.yaapt(self.signal, f0_min=self.hz_min, f0_max=self.hz_max, frame_length=15)
        f0 = f0.samp_values
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])


class Rapt(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import pysptk
        super().__init__(a, fs, hz_min, hz_max)
        f0 = pysptk.sptk.rapt(self.a, fs=self.fs, min=self.hz_min, max=self.hz_max, hopsize=250)
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])


class Swipe(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import pysptk
        super().__init__(a, fs, hz_min, hz_max)
        f0 = pysptk.sptk.swipe(self.a, fs=self.fs, min=self.hz_min, max=self.hz_max, hopsize=250)
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])


class Reaper(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import dsplib.scale
        import pyreaper
        int16_info = np.iinfo(np.int16)
        a = dsplib.scale.minmax_scaler(a, np.min(a), np.max(a), int16_info.min, int16_info.max).round().astype(np.int16)
        super().__init__(a, fs, hz_min, hz_max)
        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(self.a, fs=self.fs, minf0=self.hz_min, maxf0=self.hz_max, frame_period=0.01)
        f0[f0 == -1] = np.nan
        self.f0 = f0
        self.t = f0_times


class Spice(PitchDetector):
    """https://ai.googleblog.com/2019/11/spice-self-supervised-pitch-estimation.html"""
    use_gpu = True

    def __init__(
        self,
        a: np.ndarray,
        fs: int,
        confidence_threshold: float = 0.8,
        expected_sample_rate: int = 16000,
        spice_model_path: str = 'data/spice_model/',
    ):
        import resampy
        import tensorflow as tf
        import tensorflow_hub as hub
        a = resampy.resample(a, fs, expected_sample_rate)
        super().__init__(a, fs)
        model = hub.load(spice_model_path)
        model_output = model.signatures['serving_default'](tf.constant(a, tf.float32))
        confidence = 1.0 - model_output['uncertainty']
        f0 = self.output2hz(model_output['pitch'].numpy())
        f0[confidence < confidence_threshold] = np.nan

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


class World(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int):
        import pyworld
        super().__init__(a, fs)
        f0, sp, ap = pyworld.wav2world(a.astype(float), fs)
        f0[f0 == 0] = np.nan
        self.f0 = f0
        self.t = np.linspace(0, self.seconds, f0.shape[0])


class TorchYin(PitchDetector):
    def __init__(self, a: np.ndarray, fs: int, hz_min: float = 75, hz_max: float = 600):
        import torch
        import torchyin
        a = torch.from_numpy(a)
        super().__init__(a, fs, hz_min, hz_max)
        f0 = torchyin.estimate(self.a, sample_rate=self.fs, pitch_min=self.hz_min, pitch_max=self.hz_max)
        f0[f0 == 0] = np.nan
        self.f0 = f0[:-1]
        self.t = np.linspace(0, self.seconds, f0.shape[0])[1:]


ALGORITHMS = (
    PraatAC,
    PraatCC,
    PraatSHS,
    Pyin,
    Reaper,
    Crepe,
    TorchCrepe,
    Yaapt,
    Swipe,
    Rapt,
    World,
    TorchYin,
)

cpu_algorithms = (
    'PraatAC',
    'PraatCC',
    'PraatSHS',
    'Pyin',
    'Reaper',
    'Yaapt',
    'Rapt',
    'World',
    'TorchYin',
)

gpu_algorithms = (
    'Crepe',
    'TorchCrepe',
    'Swipe',
)

algorithms = cpu_algorithms + gpu_algorithms
