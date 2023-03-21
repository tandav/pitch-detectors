import argparse
import os
from pathlib import Path

import mir_eval
import numpy as np
import tqdm
from dsplib.scale import minmax_scaler
from musiclib.pitch import Pitch
from redis import Redis
from scipy.io import wavfile

from pitch_detectors import algorithms

MIR_1K_DIR = Path('MIR-1K')
WAV_DIR = MIR_1K_DIR / 'Wavfile'


def load_f0_true(wav_path: Path, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    p = Pitch()
    pitch_label_dir = wav_path.parent.parent / 'PitchLabel'
    f0_path = (pitch_label_dir / wav_path.stem).with_suffix('.pv')
    f0 = []
    with open(f0_path) as f:
        for _line in f:
            line = _line.strip()
            if line == '0':
                f0.append(float('nan'))
            else:
                f0.append(p.note_i_to_hz(float(line)))
    f0 = np.array(f0)
    # t = np.arange(0.02, seconds - 0.02, 0.02)
    # assert t.shape == f0.shape
    t = np.linspace(0.02, seconds, len(f0))
    return t, f0


def resample_f0(
    pitch: algorithms.PitchDetector,
    t_resampled: np.ndarray,
) -> np.ndarray:
    f0_resampled = np.full_like(t_resampled, fill_value=np.nan)
    notna_slices = np.ma.clump_unmasked(np.ma.masked_invalid(pitch.f0))
    for slice_ in notna_slices:
        t_slice = pitch.t[slice_]
        f0_slice = pitch.f0[slice_]
        t_start, t_stop = t_slice[0], t_slice[-1]
        mask = (t_start < t_resampled) & (t_resampled < t_stop)
        t_interp = t_resampled[mask]
        f0_interp = np.interp(t_interp, t_slice, f0_slice)
        f0_resampled[mask] = f0_interp
    return f0_resampled


def raw_pitch_accuracy(
    ref_f0: np.ndarray,
    est_f0: np.ndarray,
    cent_tolerance: float = 50,
) -> float:
    ref_voicing = np.isfinite(ref_f0)
    est_voicing = np.isfinite(est_f0)
    ref_cent = mir_eval.melody.hz2cents(ref_f0)
    est_cent = mir_eval.melody.hz2cents(est_f0)
    score: float = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance)
    return score


def evaluate_one(
    redis: Redis[str],
    algorithm: type[algorithms.PitchDetector],
    wav_path: Path,
) -> str:
    key = f'pitch_detectors:evaluation:{algorithm.name()}:{wav_path.stem}'
    if redis.exists(key):
        return key
    fs, a = wavfile.read(wav_path)
    seconds = len(a) / fs
    a = a[:, 1].astype(np.float32)
    rescale = 100000
    a = minmax_scaler(a, a.min(), a.max(), -rescale, rescale).astype(np.float32)
    t_true, f0_true = load_f0_true(wav_path, seconds)
    pitch = algorithm(a, fs)
    f0 = resample_f0(pitch, t_resampled=t_true)
    score = raw_pitch_accuracy(f0_true, f0)
    redis.set(key, score)
    return key


def evaluate_all(redis: Redis[str]) -> None:
    t = tqdm.tqdm(sorted(WAV_DIR.glob('*.wav')))
    for wav_path in t:
        for algorithm in tqdm.tqdm(algorithms.ALGORITHMS, leave=False):
            key = evaluate_one(redis, algorithm, wav_path)
            t.set_description(key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    if (args.algorithm is None) ^ (args.file is None):
        raise ValueError('you must specify both algorithm and file or neither')
    redis = Redis.from_url(os.environ['REDIS_URL'], decode_responses=True)
    if args.algorithm is not None and args.file is not None:
        evaluate_one(redis, algorithm=getattr(algorithms, args.algorithm), wav_path=WAV_DIR / args.file)
        raise SystemExit(0)
    evaluate_all(redis)
