import os
import re
from pathlib import Path

import numpy as np
import pipe21 as P
from redis import Redis
from tabulate import tabulate

from pitch_detectors import algorithms
from pitch_detectors import util


def get_key_fields(s: str) -> dict[str, str]:
    _, _, dataset, wav_path, algorithm, algorithm_sha256 = s.split(':')
    return {
        'dataset': dataset,
        'wav_path': wav_path,
        'algorithm': algorithm,
        'algorithm_sha256': algorithm_sha256,
    }


def delete_stale_metrics(redis: Redis) -> int:  # type: ignore
    keys = redis.keys('pitch_detectors:evaluation:*')
    source_hashes = util.source_hashes()
    pipeline = redis.pipeline()
    for key in keys:
        key_dict = get_key_fields(key)
        if source_hashes[key_dict['algorithm'].lower()] != key_dict['algorithm_sha256']:
            pipeline.delete(key)
    return sum(pipeline.execute())


def update_readme(
    repl: str,
    file: str = 'README.md',
    start: str = '<!-- table-start -->',
    stop: str = '<!-- table-stop -->',
) -> None:
    _file = Path(file)
    text = _file.read_text()
    new = re.sub(fr'{start}(.*){stop}', f'{start}\n{repl}\n{stop}', text, flags=re.DOTALL)
    _file.write_text(new)


def main() -> None:
    redis = Redis.from_url(os.environ['REDIS_URL'], decode_responses=True)
    delete_stale_metrics(redis)

    keys = redis.keys('pitch_detectors:evaluation:*')
    pipeline = redis.pipeline()
    for key in keys:
        pipeline.get(key)
    scores = pipeline.execute()
    key_score = list(zip(keys, scores, strict=True))

    algorithm_scores = (
        key_score
        | P.MapKeys(get_key_fields)
        | P.MapValues(lambda x: {'raw_pitch_accuracy': float(x)})
        | P.Map(lambda kv: kv[0] | kv[1])
        | P.MapApply(lambda x: x.pop('algorithm_sha256'))
        | P.Sorted(key=lambda x: x['algorithm'])
        | P.GroupBy(lambda x: x['algorithm'])
        | P.MapValues(lambda it: it | P.Map(lambda x: x['raw_pitch_accuracy']) | P.Pipe(list))
        | P.Pipe(list)
    )

    def add_cls(kv: dict[str, str]) -> dict[str, str]:
        cls = getattr(algorithms, kv['algorithm'])
        kv['algorithm'] = f'[{cls.name()}]({cls.__doc__})'
        kv['cpu'] = '✓'
        kv['gpu'] = '✓' if cls.use_gpu else ''
        kv['[MIR-1K](https://www.kaggle.com/datasets/datongmuyuyi/mir1k) accuracy'] = f"{kv.pop('mean'):<1.3f} ± {kv.pop('std'):<1.3f}"
        return kv

    table = (
        algorithm_scores
        | P.MapValues(lambda x: {'mean': np.mean(x).round(3), 'std': np.std(x).round(3)})
        | P.Map(lambda kv: {'algorithm': kv[0]} | kv[1])
        | P.Sorted(key=lambda kv: kv['algorithm'])
        | P.Map(add_cls)
        | P.Pipe(list)
    )
    table = tabulate(table, headers='keys', tablefmt='github')
    print(table)
    update_readme(table)


if __name__ == '__main__':
    print(os.environ['REDIS_URL'])
    main()
