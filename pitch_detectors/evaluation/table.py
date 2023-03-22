import operator
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import TypeAlias

import numpy as np
import pipe21 as P
from redis import Redis
from tabulate import tabulate

from pitch_detectors import algorithms
from pitch_detectors import util
from pitch_detectors.evaluation import datasets

DictStr: TypeAlias = dict[str, str]


def get_key_fields(s: str) -> DictStr:
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

    def group_key(x: DictStr) -> tuple[str, str]:
        return x['algorithm'], x['dataset']

    algorithm_scores = (
        key_score
        | P.MapKeys(get_key_fields)
        | P.MapValues(lambda x: {'raw_pitch_accuracy': float(x)})
        | P.Map(lambda kv: kv[0] | kv[1])
        | P.MapApply(lambda x: x.pop('algorithm_sha256'))
        | P.Sorted(key=group_key)
        | P.GroupBy(group_key)
        | P.MapValues(lambda it: it | P.Map(lambda x: x['raw_pitch_accuracy']) | P.Pipe(list))
        | P.Pipe(list)
    )

    def datasets_stats(it: Iterable[dict[str, Any]]) -> DictStr:
        out = {}
        for x in it:
            cls = getattr(datasets, x['dataset'])
            dataset = x['dataset']
            out[f'[{dataset}]({cls.__doc__}) accuracy'] = f"{x.pop('mean'):<1.3f} ± {x.pop('std'):<1.3f}"
        return out

    def add_cls(kv: DictStr) -> DictStr:
        cls = getattr(algorithms, kv['algorithm'])
        kv['algorithm'] = f'[{cls.name()}]({cls.__doc__})'
        kv['cpu'] = '✓'
        kv['gpu'] = '✓' if cls.use_gpu else ''
        return kv

    def sort_keys(kv: DictStr) -> DictStr:
        keys = kv.keys()
        to_sort = ['algorithm', 'cpu', 'gpu']
        rest_keys = sorted(keys - set(to_sort))
        return {k: kv[k] for k in to_sort + rest_keys}

    table = (
        algorithm_scores
        | P.MapValues(lambda x: {'mean': np.mean(x).round(3), 'std': np.std(x).round(3)})
        | P.Map(lambda kv: {'algorithm': kv[0][0], 'dataset': kv[0][1]} | kv[1])
        | P.Sorted(key=operator.itemgetter('algorithm', 'dataset'))
        | P.GroupBy(operator.itemgetter('algorithm'))
        | P.MapValues(datasets_stats)
        | P.Map(lambda kv: {'algorithm': kv[0]} | kv[1])
        | P.Map(add_cls)
        | P.Map(sort_keys)
        | P.Pipe(list)
    )
    table = tabulate(table, headers='keys', tablefmt='github')
    print(table)
    update_readme(table)


if __name__ == '__main__':
    print(os.environ['REDIS_URL'])
    main()
