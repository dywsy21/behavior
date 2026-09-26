"""Worker-local lazy RGB/Parquet dataset for the opt-in composite action SFT.

No legacy dataset/trainer is modified. A prepared-but-not-released dataset is
only usable with an explicit allow_prepared flag for CPU/model loading checks.
Returned items are encoded tensors, never the private source row or futures.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import os
from pathlib import Path

from audit_expert_action_capacity import snapshot
from audit_trajectory_codec import CORPUS_SHA
import native_trajectory_codec as codec
from prepare_full_annotation import packed, strict_json
from trajectory_text import ids_digest
from trajectory_video import CurrentVideoReader, validate_reference
import trajectory_modeling as modeling


class TrajectoryDataset:
    def __init__(self, root, manifest_sha256, processor, *, split='train', allow_prepared=False):
        if split not in ('train', 'validation', 'test'): raise ValueError('Explicit valid split required')
        self.root = Path(root).resolve(); self.processor = processor; self.split = split
        self.manifest = strict_json(snapshot(self.root / 'manifest.json', {}, manifest_sha256))
        m = self.manifest
        if ((self.root / 'failure.json').exists() or m['schema'] != 'h85-composite-action-sft-v1'
                or m['status'] != 'ACTION_SFT_FORMAT_COMPLETE_CAPACITY_PENDING' or m['protocol'] != codec.VERSION
                or m['corpus_manifest_sha256'] != CORPUS_SHA
                or m['training_eligible'] is not False or allow_prepared is not True):
            raise ValueError('Incomplete, wrong-protocol, or not explicitly permitted prepared dataset')
        corpus = strict_json(snapshot(Path(m['corpus_root']) / 'manifest.json', {}, CORPUS_SHA))
        original = {s['episode']: s for s in corpus['shards']}
        self.entries = []; self.ends = []; count = 0
        for source in m['sources']:
            s = original[source['episode']]
            if (any(source[k] != s[k] for k in ('task', 'instance', 'episode', 'split'))
                    or source['source_sha256'] != s['shard_sha256'] or source['source_shard'] != s['shard']):
                raise ValueError('Prepared/source shard identity mismatch')
            if source['split'] != split: continue
            for file in source['files']:
                if file['kind'] != 'shards': continue
                relative = Path(file['path'])
                if (relative.parts != ('shards', split, f'episode_{s["episode"]:06d}.parquet')
                        or file['samples'] != source['samples'] or file['samples'] <= 0):
                    raise ValueError('Prepared split/path/count mismatch')
                count += file['samples']; self.ends.append(count); self.entries.append((file, s))
        if count != m['samples_by_split'][split] or not count:
            raise ValueError('Incomplete or empty requested split')
        self._pid = os.getpid(); self._cache = OrderedDict(); self._reader = None

    def __len__(self): return self.ends[-1]

    def close(self):
        if self._reader is not None: self._reader.close()
        self._reader = None; self._cache.clear()

    def _row(self, index):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if type(index) is not int or not 0 <= index < len(self): raise IndexError(index)
        if self._pid != os.getpid():
            self.close(); self._pid = os.getpid()
        slot = bisect_right(self.ends, index); file, source = self.entries[slot]
        if slot not in self._cache:
            path = (self.root / file['path']).resolve()
            if not path.is_relative_to(self.root): raise ValueError('Prepared shard escaped dataset root')
            data = snapshot(path, {}, file['sha256'])
            if len(data) != file['bytes']: raise ValueError('Prepared shard size changed')
            rows = pq.read_table(pa.BufferReader(data), use_threads=False).to_pylist()
            if len(rows) != file['samples']: raise ValueError('Prepared shard count changed')
            self._cache[slot] = rows
        self._cache.move_to_end(slot)
        while len(self._cache) > 2: self._cache.popitem(last=False)
        row = self._cache[slot][index - (self.ends[slot - 1] if slot else 0)]
        for view in ('head', 'left_wrist', 'right_wrist'): validate_reference(row, source, view)
        if row['quarantine_reason']: raise ValueError('Quarantined target entered normal dataset')
        return row, source

    def __getitem__(self, index):
        row, source = self._row(index)
        actor = strict_json(row['actor_json']); codec.validate_actor(actor)
        actual = codec.actor_from_state(row['task'], row['active_instruction'], row['observation_state'])
        if actor != actual or hashlib.sha256(row['target_json'].encode()).hexdigest() != row['target_sha256']:
            raise ValueError('Prepared current actor/target binding changed')
        if self._reader is None: self._reader = CurrentVideoReader()
        images, _ = self._reader.read(row, source)
        try:
            result = modeling.encode(self.processor, actor, images, target=row['target_json'])
        finally:
            for image in images.values(): image.close()
        import torch
        if any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu' for value in result.values()):
            raise ValueError('Dataset processor must return only CPU tensors')
        n = row['prefix_tokens']
        if (result['input_ids'].shape[1] != row['total_tokens']
                or ids_digest(result['input_ids'][0, :n].tolist()) != row['prefix_ids_sha256']
                or ids_digest(result['input_ids'][0, n:].tolist()) != row['target_ids_sha256']
                or int((result['labels'] != -100).sum()) != row['target_tokens']):
            raise ValueError('Actual training tensors disagree with fully audited text targets')
        return result
