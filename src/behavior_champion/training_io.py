"""Shared native-data transforms and ordered, identity-checked loader batches.

No model-state mutation, data reshuffling, or GPU placement occurs here. The
optimizer cursor owns ordering; a prefetched batch is not a consumed update.
"""
import hashlib

import numpy as np

from .contracts import delta_actions, pad32, state23_from_proprio61, training_stage
from .sampling import digest_json


def combine_chunks(chunks, expected_indices, *, split):
    if not chunks or split not in ('train', 'eval'):
        raise ValueError('Explicit nonempty protected split required')
    identities = [item for chunk in chunks for item in chunk['identity']]
    indices = np.array([item['index'] for item in identities], dtype=np.int64)
    expected = np.asarray(expected_indices, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError('Prefetch order does not match the optimizer/evaluation cursor')
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('Duplicate anchors inside one batch')
    if any(item['split'] != split or item['frame'] % 16 != item['phase'] for item in identities):
        raise ValueError('Protected split or original per-video stride phase changed')
    source = {key: np.concatenate([chunk[key] for chunk in chunks])
              for key in chunks[0] if key != 'identity'}
    for name, field in (('task_id', 'task'), ('frame', 'frame'), ('length', 'length')):
        source[name] = np.array([row[field] for row in identities], dtype=np.int32)
    counts = np.bincount(source['task_id'], minlength=100)
    if len(counts) != 100:
        raise ValueError('Unknown task identity')
    return source, dict(sampled_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                        identities_sha256=digest_json(identities), task_counts=counts.tolist(),
                        samples=len(indices), split=split)


class NativeBatchTransform:
    def __init__(self, layout, normalizer, tokenizer):
        self.layout, self.normalizer, self.tokenizer = layout, normalizer, tokenizer

    def __call__(self, source):
        import jax
        examples = []
        for i in range(len(source['task_id'])):
            task = self.layout['dataset_to_model'][str(int(source['task_id'][i]))]
            state = state23_from_proprio61(source['state61'][i])
            row = self.tokenizer(self.normalizer(dict(
                state=state, actions=delta_actions(source['actions23'][i], state))))
            row['state'], row['actions'] = pad32(row['state']), pad32(row['actions'])
            row['image'] = {key: source[key][i] for key in
                            ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')}
            if any(image.dtype != np.uint8 for image in row['image'].values()):
                raise ValueError('Native uint8 pixels required; normalize on the assigned GPU')
            row['image_mask'] = {key: np.bool_(True) for key in row['image']}
            row['tokenized_prompt'] = np.array([task, training_stage(
                int(source['frame'][i]), int(source['length'][i]), self.layout['num_stages'][task])], dtype=np.int32)
            row['tokenized_prompt_mask'] = np.ones(2, dtype=bool)
            examples.append(row)
        return jax.tree.map(lambda *xs: np.stack(xs), *examples)


def unpack_device_batch(batch):
    from b1k.models.observation import Observation
    from .frontend import resize_uint8_rgb
    fields = {key: value for key, value in batch.items() if key != 'actions'}
    fields['image'] = {key: resize_uint8_rgb(value) for key, value in fields['image'].items()}
    return Observation.from_dict(fields), batch['actions']


class EvalAccumulator:
    """Exact per-example accounting; never silently broadcast missing metrics."""
    def __init__(self, mapping, *, per_task=16):
        self.mapping = {int(key): int(value) for key, value in mapping.items()}
        self.per_task = per_task
        self.rows = []
        self.seen = set()

    def add(self, indices, task_ids, metrics):
        indices, task_ids = list(map(int, indices)), list(map(int, task_ids))
        if not metrics or any(np.asarray(value).shape != (len(indices),) for value in metrics.values()):
            raise ValueError('Every heldout metric must have one scalar per example')
        if len(task_ids) != len(indices) or any(not np.isfinite(value).all() for value in metrics.values()):
            raise ValueError('Nonfinite or misaligned heldout metrics')
        if len(set(indices)) != len(indices) or self.seen.intersection(indices):
            raise ValueError('Heldout observation evaluated twice')
        for offset, (index, task) in enumerate(zip(indices, task_ids, strict=True)):
            if task not in self.mapping:
                raise ValueError('Unknown heldout task')
            self.rows.append(dict(index=index, task=task, metrics={
                key: float(value[offset]) for key, value in metrics.items()}))
        self.seen.update(indices)

    def finish(self):
        counts = np.bincount([row['task'] for row in self.rows], minlength=100)
        if counts.shape != (100,) or not np.all(counts == self.per_task):
            raise ValueError('Incomplete original-heldout cohort; do not report a partial mean as complete')
        keys = set(self.rows[0]['metrics'])
        if any(set(row['metrics']) != keys for row in self.rows):
            raise ValueError('Metric coverage changed inside evaluation')
        groups = dict(all=self.rows,
            old50=[row for row in self.rows if self.mapping[row['task']] < 50],
            new50=[row for row in self.rows if self.mapping[row['task']] >= 50])
        groups.update({f'task_{task:03d}': [row for row in self.rows if row['task'] == task]
                       for task in range(100)})
        return {name: dict(observations=len(rows), **{
            key: float(np.mean([row['metrics'][key] for row in rows])) for key in sorted(keys)})
            for name, rows in groups.items()}
