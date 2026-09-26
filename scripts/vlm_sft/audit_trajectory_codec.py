"""CPU-only real-corpus/tokenizer feasibility audit; never a training release."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from audit_expert_action_capacity import snapshot
import native_trajectory_codec as codec
from prepare_full_annotation import packed, strict_json, forbid_source_output

REPO = Path(__file__).resolve().parents[2]
CORPUS_SHA = '851b3cd9709f5dc9db2b378078ece93e9cc1f625278f0afa05994db5e82787e4'
MODEL_SPEC = REPO / 'configs/semantic_robot/h45_shared_small_vlm_probe.json'
MAX_SECONDS = 1200
MAX_ROWS = 12000
MAX_BYTES = 64 * 1024**2


def select_sources(manifest):
    selected = []
    for task in range(5):
        sources = [s for s in manifest['shards'] if s['split'] == 'train' and s['task'] == task and s['samples'] > 0]
        sources.sort(key=lambda s: hashlib.sha256(f'h85-codec:41:{task}:{s["instance"]}'.encode()).hexdigest())
        if len(sources) < 4: raise ValueError('Four TRAIN sources per task required')
        selected.extend(sources[:4])
    if len({(s['task'], s['instance'], s['episode']) for s in selected}) != 20:
        raise ValueError('Repeated source identity')
    return selected


def sampled_rows(rows):
    if not rows: raise ValueError('Empty source')
    indices = np.linspace(0, len(rows) - 1, min(600, len(rows)), dtype=int)
    if len(np.unique(indices)) != len(indices): raise ValueError('Duplicate time sampling')
    return [rows[int(i)] for i in indices]


def check_row(row, shard):
    f = row['frame_index']
    if (type(f) is not int or f < 0 or row['id'] != f't{shard["task"]}_i{shard["instance"]}_e{shard["episode"]}_f{f:06d}'
            or row['source_split'] != 'train' or row['task_index'] != shard['task']
            or row['task_instance_id'] != shard['instance'] or row['episode_index'] != shard['episode']
            or row['timestamp_s'] != f / 30):
        raise ValueError('Wrong source row identity/time/split')


def dense_target(actions, state):
    a = codec.finite(actions, (16, 23)); q = codec.current_anchor(state)
    obj = {'v': codec.VERSION}
    for name, sl in codec.JOINT_ACTION.items():
        values = codec._quantized(a[:, sl] - q[codec.JOINT_ANCHOR[name]], codec.JOINT_UNIT)
        obj[name] = [[i, *v.tolist()] for i, v in enumerate(values)]
    for name, values in [('b', a[:, :3]), ('g', a[:, [14, 22]])]:
        quantized = codec._quantized(values, codec.COMMAND_UNIT)
        obj[name] = [[i, *v.tolist()] for i, v in enumerate(quantized)]
    return json.dumps(obj, separators=(',', ':'))


def percentiles(values):
    if not values: raise ValueError('No observations for statistics')
    return dict(zip(('min', 'p50', 'p90', 'p95', 'p99', 'max'),
                    map(float, np.percentile(values, [0, 50, 90, 95, 99, 100]))))


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic(); pins = {}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Explicit CPU-only process required')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean immutable source required')
    code = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    for name in ('audit_trajectory_codec.py', 'native_trajectory_codec.py', 'audit_expert_action_capacity.py', 'prepare_full_annotation.py'):
        snapshot(REPO / 'scripts/vlm_sft' / name, pins)
    manifest = strict_json(snapshot(args.corpus / 'manifest.json', pins, CORPUS_SHA))
    if (manifest['status'] != 'INTERMEDIATE_NATIVE_ACTION_CORPUS_NOT_SFT_RELEASE'
            or manifest['training_eligible'] is not False or (args.corpus / 'failure.json').exists()):
        raise ValueError('Wrong intermediate corpus')
    spec = strict_json(snapshot(MODEL_SPEC, pins)); model = Path(spec['model'])
    tokenizer_bytes = snapshot(model / 'tokenizer.json', pins, spec['model_files']['tokenizer.json'])
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    eos = tokenizer.token_to_id('<|im_end|>')
    if type(eos) is not int: raise ValueError('Native end-of-message token absent')
    selected = select_sources(manifest)
    protected = [args.corpus, REPO, model]
    for shard in selected:
        protected.append(Path(shard['source_parquet']).resolve().parents[2])
        protected.extend(Path(v['path']).resolve().parent for v in shard['videos'].values())
    forbid_source_output(args.output, *protected)
    if args.output.exists(): raise FileExistsError('Preserve prior audit')
    if shutil.disk_usage(args.output.parent).free < 80 * 1024**3: raise ValueError('Disk reserve')
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    os.sched_setaffinity(0, set(range(48, 52)))
    args.output.mkdir()
    launch = {'schema': 'h85-trajectory-codec-audit-v1', 'code_commit': code, 'pid': os.getpid(),
              'utc': datetime.now(timezone.utc).isoformat(), 'protocol': codec.VERSION,
              'max_seconds': MAX_SECONDS, 'max_rows': MAX_ROWS, 'max_bytes': MAX_BYTES,
              'model_calls': 0, 'training_updates': 0, 'controls': 0, 'training_eligible': False,
              'corpus_manifest_sha256': CORPUS_SHA, 'tokenizer_sha256': spec['model_files']['tokenizer.json']}
    (args.output / 'launch.json').write_bytes(packed(launch) + b'\n')
    records = []; seen = set(); examples = []
    try:
        for shard in selected:
            path = args.corpus / shard['shard']
            data = snapshot(path, pins, shard['shard_sha256'])
            if len(data) != shard['shard_bytes'] or path.parent.name != 'train': raise ValueError('Shard layout changed')
            rows = pq.read_table(pa.BufferReader(data), use_threads=False).to_pylist()
            if len(rows) != shard['samples']: raise ValueError('Shard count changed')
            for row in sampled_rows(rows):
                if time.monotonic() - started > MAX_SECONDS - 20: raise TimeoutError('CPU audit wall budget')
                check_row(row, shard)
                if row['id'] in seen or len(records) >= MAX_ROWS: raise ValueError('Duplicate row or audit row limit')
                seen.add(row['id'])
                actor = codec.actor_from_state(row['task'], row['active_instruction'], row['observation_state'])
                target = codec.encode(row['expert_action'], row['observation_state'])
                decoded = codec.decode(target, actor['proprio']['q_rad'])
                err = codec.errors(row['expert_action'], decoded)
                native_err = codec.errors(row['expert_action'], decoded.astype(np.float32))
                cast_err = codec.errors(decoded, decoded.astype(np.float32))
                if any(native_err[k] > codec.ERROR_LIMITS[k] + cast_err[k] + 1e-10 for k in err):
                    raise ValueError('Native float32 exceeds interpolation plus actual cast error')
                ids = tokenizer.encode(target, add_special_tokens=False).ids
                if tokenizer.decode(ids, skip_special_tokens=False) != target: raise ValueError('Tokenizer changes target text')
                dense = dense_target(row['expert_action'], row['observation_state'])
                dense_ids = tokenizer.encode(dense, add_special_tokens=False).ids
                prompt_text = codec.SYSTEM + '\n' + codec.prompt(actor)
                record = {'id': row['id'], 'task': shard['task'], 'instance': shard['instance'], 'episode': shard['episode'],
                          'target_tokens_including_eos': len(ids) + 1, 'dense_tokens_including_eos': len(dense_ids) + 1,
                          'text_only_prefix_tokens_NOT_IMAGE_CHAT_PREFIX': len(tokenizer.encode(prompt_text, add_special_tokens=False).ids),
                          'roundtrip_errors': err, 'native_float32_errors': native_err, 'native_cast_errors': cast_err,
                          'group_knots': {k: len(codec.parse(target)[k]) for k in codec.GROUP_WIDTHS},
                          'target_sha256': hashlib.sha256(target.encode()).hexdigest()}
                records.append(record)
                if len(examples) < 10: examples.append({'id': row['id'], 'actor': actor, 'target': target})
            print(json.dumps({'sources_done': len({r['episode'] for r in records}), 'samples': len(records),
                              'seconds': time.monotonic() - started}), flush=True)
        for path, pin in pins.items(): snapshot(path, {}, pin)
        if len({r['episode'] for r in records}) != 20: raise ValueError('Incomplete audit sources')
        results = b''.join(packed(r) + b'\n' for r in records)
        if len(results) > MAX_BYTES - 1024**2: raise ValueError('Audit output byte limit')
        (args.output / 'rows.jsonl').write_bytes(results)
        elapsed = time.monotonic() - started
        if elapsed > MAX_SECONDS: raise TimeoutError('Final audit wall budget')
        result = {**launch, 'status': 'OFFLINE_CODEC_FEASIBILITY_ONLY', 'wall_seconds': elapsed,
                  'samples': len(records), 'sources': [{k: s[k] for k in ('task', 'instance', 'episode', 'samples')} for s in selected],
                  'samples_per_task': dict(Counter(r['task'] for r in records)),
                  'target_tokens': percentiles([r['target_tokens_including_eos'] for r in records]),
                  'dense_tokens': percentiles([r['dense_tokens_including_eos'] for r in records]),
                  'text_only_prefix_tokens': percentiles([r['text_only_prefix_tokens_NOT_IMAGE_CHAT_PREFIX'] for r in records]),
                  'token_ratio': sum(r['target_tokens_including_eos'] for r in records) / sum(r['dense_tokens_including_eos'] for r in records),
                  'error_limits': codec.ERROR_LIMITS,
                  'worst_errors': {k: max(r['roundtrip_errors'][k] for r in records) for k in codec.ERROR_LIMITS},
                  'worst_native_float32_errors': {k: max(r['native_float32_errors'][k] for r in records) for k in codec.ERROR_LIMITS},
                  'worst_native_cast_errors': {k: max(r['native_cast_errors'][k] for r in records) for k in codec.ERROR_LIMITS},
                  'target_over_1024_tokens': sum(r['target_tokens_including_eos'] > 1024 for r in records),
                  'rows_sha256': hashlib.sha256(results).hexdigest(), 'rows_bytes': len(results), 'examples': examples,
                  'input_sha256': pins, 'rgb_decoded': False, 'full_chat_tensor_and_mask_gate': 'PENDING',
                  'three_hour_capacity': 'NOT_MEASURED', 'execution_safety_or_success_certified': False}
        (args.output / 'result.json').write_bytes(packed(result) + b'\n')
    except BaseException as exc:
        (args.output / 'failure.json').write_bytes(packed({'error': repr(exc), 'samples': len(records),
                                                         'wall_seconds': time.monotonic() - started}) + b'\n')
        raise


if __name__ == '__main__': main()
