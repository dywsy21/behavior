"""Independently recount all SFT shards and load 50 real RGB training items.

CPU processor only. This does not certify forward/backward, throughput or SR.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from audit_expert_action_capacity import snapshot
from audit_trajectory_codec import MODEL_SPEC, CORPUS_SHA, percentiles
from audit_trajectory_encoding import PROCESSOR_FILES
from prepare_full_annotation import packed, strict_json, forbid_source_output
from prepare_trajectory_sft import ENCODING_SHA, MAX_ROWS
from trajectory_dataset import TrajectoryDataset
from trajectory_video import video_identity
import trajectory_modeling as modeling

REPO = Path(__file__).resolve().parents[2]
MAX_SECONDS = 600
MAX_BYTES = 4 * 1024**2


def recount(root, manifest, wanted_ids, deadline, *, pins=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pins = {} if pins is None else pins
    seen = set(); counts = Counter(); task_counts = Counter(); tokens = Counter()
    indices = {}; accepted_train = 0; checked_bytes = 0; files = []
    target_lengths = []
    for source in manifest['sources']:
        for file in source['files']:
            if time.monotonic() > deadline: raise TimeoutError('Dataset recount deadline')
            relative = Path(file['path']); path = Path(root) / relative
            if (relative.parts != (file['kind'], source['split'], f'episode_{source["episode"]:06d}.parquet')
                    or file['kind'] not in ('shards', 'quarantine') or not path.resolve().is_relative_to(Path(root).resolve())):
                raise ValueError('SFT audit path/split mismatch')
            data = snapshot(path, pins, file['sha256'])
            if len(data) != file['bytes']: raise ValueError('SFT file bytes changed')
            columns = ['id', 'source_split', 'task_index', 'task_instance_id', 'episode_index', 'frame_index',
                       'prefix_tokens', 'target_tokens', 'total_tokens', 'quarantine_reason']
            rows = pq.read_table(pa.BufferReader(data), columns=columns, use_threads=False).to_pylist()
            if len(rows) != file['samples']: raise ValueError('SFT file count changed')
            accepted = file['kind'] == 'shards'
            for offset, row in enumerate(rows):
                frame = row['frame_index']
                reason = ('response_over_budget' if row['target_tokens'] > modeling.MAX_RESPONSE else
                          'context_over_budget' if row['total_tokens'] > modeling.MAX_CONTEXT else '')
                if (row['id'] in seen or row['id'] != f't{source["task"]}_i{source["instance"]}_e{source["episode"]}_f{frame:06d}'
                        or any(row[k] != source[s] for k, s in (('source_split', 'split'), ('task_index', 'task'),
                                        ('task_instance_id', 'instance'), ('episode_index', 'episode')))
                        or row['total_tokens'] != row['prefix_tokens'] + row['target_tokens']
                        or row['prefix_tokens'] <= 0 or row['target_tokens'] < 2
                        or accepted == bool(row['quarantine_reason'])
                        or row['quarantine_reason'] != reason
                        or (accepted and (row['target_tokens'] > modeling.MAX_RESPONSE or row['total_tokens'] > modeling.MAX_CONTEXT))):
                    raise ValueError('SFT identity, quarantine, or token accounting mismatch')
                seen.add(row['id']); counts[(file['kind'], source['split'])] += 1
                if accepted:
                    task_counts[(source['task'], source['split'])] += 1
                    tokens[(source['split'], 'total')] += row['total_tokens']
                    tokens[(source['split'], 'target')] += row['target_tokens']
                    target_lengths.append(row['target_tokens'])
                    if row['id'] in wanted_ids:
                        if source['split'] != 'train': raise ValueError('Review item found outside TRAIN')
                        indices[row['id']] = accepted_train + offset
            if accepted and source['split'] == 'train': accepted_train += len(rows)
            checked_bytes += len(data); files.append(file)
    expected_paths = {'launch.json', 'loader_preflight.json', 'manifest.json'} | {f['path'] for f in files}
    actual_paths = {p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file()}
    if actual_paths != expected_paths: raise ValueError('SFT dataset file set changed')
    if (len(seen) != MAX_ROWS or len(files) != len({f['path'] for f in files})
            or set(indices) != set(wanted_ids)
            or any(counts[('shards', split)] != value for split, value in manifest['samples_by_split'].items())
            or any(task_counts[(t, s)] != manifest['samples_by_task_split'][str(t)][s]
                   for t in range(5) for s in ('train', 'validation', 'test'))
            or sum(n for (kind, _), n in counts.items() if kind == 'quarantine') != manifest['quarantined']
            or checked_bytes != manifest['shard_bytes']):
        raise ValueError('SFT full-dataset totals changed')
    summary = {'total_original_rows': len(seen), 'unique_ids': len(seen), 'files_checked': len(files),
               'bytes_checked': checked_bytes, 'accepted_by_split': {s: counts[('shards', s)] for s in ('train', 'validation', 'test')},
               'quarantine_by_split': {s: counts[('quarantine', s)] for s in ('train', 'validation', 'test')},
               'accepted_by_task_split': {str(t): {s: task_counts[(t, s)] for s in ('train', 'validation', 'test')} for t in range(5)},
               'tokens_by_split': {s: {k: tokens[(s, k)] for k in ('total', 'target')} for s in ('train', 'validation', 'test')},
               'accepted_target_tokens': percentiles(target_lengths)}
    return summary, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'encoding-root', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(); started = time.monotonic(); pins = {}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Explicit offline CPU-only process required')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean frozen source required')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = strict_json(snapshot(args.dataset / 'manifest.json', pins, args.manifest_sha256))
    corpus = strict_json(snapshot(Path(manifest['corpus_root']) / 'manifest.json', pins, CORPUS_SHA))
    previous = strict_json(snapshot(args.encoding_root / 'result.json', pins, ENCODING_SHA))
    previous_rows = [strict_json(line) for line in snapshot(args.encoding_root / 'rows.jsonl', pins, previous['rows_sha256']).splitlines()]
    if len(previous_rows) != 50 or len({r['id'] for r in previous_rows}) != 50:
        raise ValueError('Exact 50-case prior encoding evidence required')
    spec = strict_json(snapshot(MODEL_SPEC, pins)); model = Path(spec['model'])
    for name in PROCESSOR_FILES: snapshot(model / name, pins, spec['model_files'][name])
    for name in ('audit_trajectory_dataset.py', 'trajectory_dataset.py', 'trajectory_modeling.py', 'trajectory_video.py', 'native_trajectory_codec.py'):
        snapshot(REPO / 'scripts/vlm_sft' / name, pins)
    forbidden = [args.dataset, args.encoding_root, Path(manifest['corpus_root']), model, REPO]
    forbidden.extend(Path(v['path']).resolve().parent for s in corpus['shards'] for v in s['videos'].values())
    forbid_source_output(args.output, *forbidden)
    if args.output.exists(): raise FileExistsError('Preserve previous dataset audit')
    os.sched_setaffinity(0, set(range(48, 52)))
    import torch
    import transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized(): raise ValueError('Unexpected CUDA initialization')
    args.output.mkdir()
    launch = {'schema': 'h85-actual-sft-dataset-check-v1', 'code_commit': commit, 'pid': os.getpid(),
              'utc': datetime.now(timezone.utc).isoformat(), 'dataset_manifest_sha256': args.manifest_sha256,
              'max_seconds': MAX_SECONDS, 'max_bytes': MAX_BYTES, 'model_weight_loads': 0,
              'training_updates': 0, 'controls': 0, 'training_eligible': False,
              'torch': torch.__version__, 'transformers': transformers.__version__}
    (args.output / 'launch.json').write_bytes(packed(launch) + b'\n')
    dataset = None
    try:
        summary, indices = recount(args.dataset, manifest, {r['id'] for r in previous_rows}, started + MAX_SECONDS - 30, pins=pins)
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        dataset = TrajectoryDataset(args.dataset, args.manifest_sha256, processor, allow_prepared=True)
        records = []; shortest = longest = None
        for old in previous_rows:
            if time.monotonic() - started > MAX_SECONDS - 20: raise TimeoutError('Actual dataset deadline')
            before = time.monotonic(); value = dataset[indices[old['id']]]
            digest = hashlib.sha256(value['pixel_values'].numpy().tobytes()).hexdigest()
            if digest != old['pixels_tensor_sha256']: raise ValueError('Actual dataset RGB payload differs from original encoding')
            n = value['input_ids'].shape[1]
            if n != old['total_tokens'] or int((value['labels'] != -100).sum()) != old['supervised_tokens']:
                raise ValueError('Actual item length/supervision differs from original encoding')
            if shortest is None or n < shortest['input_ids'].shape[1]: shortest = value
            if longest is None or n > longest['input_ids'].shape[1]: longest = value
            records.append({'id': old['id'], 'index': indices[old['id']], 'total_tokens': n,
                            'supervised_tokens': old['supervised_tokens'], 'pixels_tensor_sha256': digest,
                            'cpu_load_encode_seconds': time.monotonic() - before})
        batch = modeling.collate([shortest, longest], processor.tokenizer.pad_token_id)
        if not torch.all(batch['labels'][batch['attention_mask'] == 0] == -100): raise ValueError('Padding supervised')
        for path, digest in pins.items(): snapshot(path, {}, digest)
        for source in corpus['shards']:
            for video in source['videos'].values(): video_identity(video)
        if torch.cuda.is_initialized(): raise ValueError('Dataset audit initialized CUDA')
        result = {**launch, 'status': 'CPU_DATASET_LOADING_VERIFIED_CAPACITY_PENDING', 'full_recount': summary,
                  'real_train_items_checked': len(records), 'real_train_images_checked': 3 * len(records),
                  'all_pixels_equal_prior_checked_encoder': True, 'real_left_padding_checked': True,
                  'wall_seconds': time.monotonic() - started, 'cuda_initialized': False,
                  'cpu_load_encode_seconds': percentiles([r['cpu_load_encode_seconds'] for r in records]),
                  'records': records, 'input_sha256': pins, 'gpu_forward_backward': 'NOT_RUN',
                  'training_throughput_and_three_hour_capacity': 'NOT_MEASURED'}
        content = packed(result) + b'\n'
        if len(content) > MAX_BYTES or result['wall_seconds'] > MAX_SECONDS: raise ValueError('Dataset audit budget')
        (args.output / 'result.json').write_bytes(content)
    except BaseException as exc:
        (args.output / 'failure.json').write_bytes(packed({'error': repr(exc), 'seconds': time.monotonic() - started}) + b'\n')
        raise
    finally:
        if dataset is not None: dataset.close()


if __name__ == '__main__': main()
