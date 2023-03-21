from pathlib import Path

import mir_eval
import numpy as np
from dsplib.scale import minmax_scaler
from scipy.io import wavfile

from pitch_detectors import algorithms

MIR_1K_DIR = Path('MIR-1K')
WAV_DIR = MIR_1K_DIR / 'Wavfile'


def midi_to_freq(midi_n, ref_frequency=440.0):
    if (midi_n == 0):
        return 0
    else:
        return ref_frequency * 2**((midi_n - 69) / 12)


def load_f0_true(wav_path: Path, seconds: float):
    pitch_label_dir = wav_path.parent.parent / 'PitchLabel'
    f0_path = (pitch_label_dir / wav_path.stem).with_suffix('.pv')
    f0 = []
    with open(f0_path) as f:
        for line in f:
            line = line.strip()
            if line == '0':
                f0.append(float('nan'))
            else:
                f0.append(midi_to_freq(float(line)))  # todoo fix
    f0 = np.array(f0)
    # t = np.arange(0.02, seconds - 0.02, 0.02)
    # assert t.shape == f0.shape
    t = np.linspace(0.02, seconds, len(f0))
    return t, f0


def resample_f0(
    pitch: algorithms.PitchDetector,
    t_resampled: np.ndarray,
):
    f0_resampled = np.full_like(t_resampled, fill_value=np.nan)
    notna_slices = np.ma.clump_unmasked(np.ma.masked_invalid(pitch.f0))
    for sl in notna_slices:
        t_slice = pitch.t[sl]
        f0_slice = pitch.f0[sl]
        t_start, t_stop = t_slice[0], t_slice[-1]
        mask = (t_start < t_resampled) & (t_resampled < t_stop)
        t_interp = t_resampled[mask]
        f0_interp = np.interp(t_interp, t_slice, f0_slice)
        f0_resampled[mask] = f0_interp
    return f0_resampled


def raw_pitch_accuracy(
    ref_f0,
    est_f0,
    cent_tolerance=50,
):
    ref_voicing = np.isfinite(ref_f0)
    est_voicing = np.isfinite(est_f0)
    ref_cent = mir_eval.melody.hz2cents(ref_f0)
    est_cent = mir_eval.melody.hz2cents(est_f0)
    return mir_eval.melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance)


def main():
    with open(MIR_1K_DIR / 'evaluation.csv', 'w') as f:
        for wav_path in sorted(WAV_DIR.glob('*.wav')):
            fs, a = wavfile.read(wav_path)
            seconds = len(a) / fs
            a = a[:, 1].astype(np.float32)
            rescale = 100000
            a = minmax_scaler(a, a.min(), a.max(), -rescale, rescale).astype(np.float32)
            t_true, f0_true = load_f0_true(wav_path, seconds)
            for algorithm in algorithms.ALGORITHMS:
                pitch = algorithm(a, fs)
                f0 = resample_f0(pitch, t_resampled=t_true)
                score = raw_pitch_accuracy(f0_true, f0)
                print(wav_path.stem, algorithm.name(), score, sep=',')
                print(wav_path.stem, algorithm.name(), score, sep=',', file=f)


if __name__ == '__main__':
    main()
