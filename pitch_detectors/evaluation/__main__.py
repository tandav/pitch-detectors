import argparse
import os
from pathlib import Path

import mir_eval
import numpy as np
import tqdm
from dsplib.scale import minmax_scaler
from redis import Redis

from pitch_detectors import algorithms
from pitch_detectors import util
from pitch_detectors.algorithms.base import PitchDetector
from pitch_detectors.evaluation import datasets
from pitch_detectors.evaluation.datasets.base import Dataset


def resample_f0(
    pitch: PitchDetector,
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
    redis: Redis,  # type: ignore
    algorithm: type[PitchDetector],
    wav_path: Path,
    source_hashes: dict[str, str],
    dataset: type[Dataset],
) -> str:
    source_hash = source_hashes[algorithm.name().lower()]
    key = f'pitch_detectors:evaluation:{dataset.name()}:{wav_path.stem}:{algorithm.name()}:{source_hash}'
    if redis.exists(key):
        return key
    wav = dataset.load_wav(wav_path)
    seconds = len(wav.a) / wav.fs
    rescale = 100000
    a = minmax_scaler(wav.a, wav.a.min(), wav.a.max(), -rescale, rescale).astype(np.float32)
    true = dataset.load_true(wav_path, seconds)
    pitch = algorithm(a, wav.fs)
    f0 = resample_f0(pitch, t_resampled=true.t)
    score = raw_pitch_accuracy(true.f0, f0)
    redis.set(key, score)
    return key


def evaluate_all(redis: Redis, source_hashes, dataset: type[Dataset]) -> None:  # type: ignore
    t = tqdm.tqdm(sorted(dataset.iter_wav_files()))
    for wav_path in t:
        for algorithm in tqdm.tqdm(algorithms.ALGORITHMS, leave=False):
            key = evaluate_one(redis, algorithm, wav_path, source_hashes, dataset)
            t.set_description(key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    if (args.algorithm is None) ^ (args.file is None):
        raise ValueError('you must specify both algorithm and file or neither')

    if args.dataset is None:
        _datasets = datasets.all_datasets
    else:
        _datasets = (getattr(datasets, args.dataset),)  # type: ignore

    redis = Redis.from_url(os.environ['REDIS_URL'], decode_responses=True)
    source_hashes = util.source_hashes()
    redis.hset('pitch_detectors:source_hashes', mapping=source_hashes)  # type: ignore

    for _dataset in _datasets:
        if args.algorithm is not None and args.file is not None:
            evaluate_one(
                redis,
                algorithm=getattr(algorithms, args.algorithm),
                wav_path=_dataset.wav_dir() / args.file,
                source_hashes=source_hashes,
                dataset=_dataset,  # type: ignore
            )
            raise SystemExit(0)
        evaluate_all(redis, source_hashes, _dataset)  # type: ignore
