"""Fixed original-heldout cohort, with equal observations for all 100 tasks.

This selects identities only. It never estimates success from loss, moves an
episode across splits, or re-admits quarantined evaluation sources.
"""
import hashlib

import numpy as np

from .sampling import digest_json


class FixedEvalPlan:
    def __init__(self, dataset, dataset_to_model, *, per_task=16, seed=31, batch_size=32):
        mapping = {int(key): int(value) for key, value in dataset_to_model.items()}
        if dataset.split != 'eval' or set(mapping) != set(range(100)) or set(mapping.values()) != set(range(100)):
            raise ValueError('Original protected eval and complete 100-task mapping required')
        if per_task < 1 or (50 * per_task) % batch_size or seed < 0:
            raise ValueError('Equal per-task evaluation must have complete fixed-size group batches')
        intervals = [[] for _ in range(100)]
        for serial, ep in enumerate(dataset.episodes):
            task = int(ep['row']['task_index'])
            if ep['split'] != 'eval' or task not in mapping:
                raise ValueError('Non-eval episode in heldout cohort')
            intervals[task].append(tuple(map(int, dataset.offsets[serial:serial + 2])))
        selected = {'old50': [], 'new50': []}
        for task, spans in enumerate(intervals):
            if not spans:
                raise ValueError(f'Missing original heldout task {task}')
            starts, stops = np.asarray(spans, dtype=np.int64).T
            offsets = np.r_[0, np.cumsum(stops - starts)]
            if offsets[-1] < per_task:
                raise ValueError('Insufficient distinct heldout anchors')
            local = np.random.default_rng(np.random.SeedSequence([seed, task])).choice(
                int(offsets[-1]), size=per_task, replace=False)
            serial = np.searchsorted(offsets, local, side='right') - 1
            selected['old50' if mapping[task] < 50 else 'new50'].extend((starts[serial] + local - offsets[serial]).tolist())
        self.indices = {}
        self.batch_size = batch_size
        self.identities = []
        for number, (group, indices) in enumerate(selected.items()):
            values = np.asarray(indices, dtype=np.int64)
            np.random.default_rng(np.random.SeedSequence([seed, 100 + number])).shuffle(values)
            values.flags.writeable = False
            self.indices[group] = values
            for index in values:
                ep, frame = dataset.locate(index); row = ep['row']
                self.identities.append(dict(group=group, index=int(index), episode=int(row['episode_index']),
                    task=int(row['task_index']), instance=int(row['task_instance_id']),
                    raw_episode=int(row['raw_episode_id']), frame=int(frame), phase=int(ep['phase']), split='eval'))
        if len({row['index'] for row in self.identities}) != 100 * per_task:
            raise ValueError('Repeated heldout anchor')
        self.contract = dict(schema='champion-fixed-original-eval-v1', per_task=per_task,
            seed=seed, batch_size=batch_size, observations=100 * per_task,
            original_eval_episodes=len(dataset.episodes), preserved_quarantines=dataset.excluded_episode_counts,
            dataset_to_model=mapping, identities_sha256=digest_json(self.identities),
            index_sha256={group: hashlib.sha256(values.astype('<i8').tobytes()).hexdigest()
                          for group, values in self.indices.items()},
            noise_protocol='Fixed independent evaluation key seed31 folded by group and batch; never training step',
            scope='Original heldout loss cohort only; not all eval frames and not physical Q/SR')
        self.fingerprint = digest_json(self.contract)

    def batches(self, group):
        values = self.indices[group]
        for start in range(0, len(values), self.batch_size):
            yield values[start:start + self.batch_size].copy()
