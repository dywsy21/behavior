"""Export the complete native corpus as opt-in composite VLM action supervision.

Keeps every original row/float32 control; overlength targets are quarantined,
never truncated. Does not start training or modify an online control interface.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from audit_expert_action_capacity import snapshot
from audit_trajectory_codec import CORPUS_SHA, MODEL_SPEC, percentiles
from audit_trajectory_encoding import REVIEW_SHA, HUMAN_SHA, PROCESSOR_FILES, checked_cases
import native_trajectory_codec as codec
from prepare_full_annotation import packed, strict_json, forbid_source_output
from trajectory_text import prefix_ids, ids_digest
from trajectory_video import CurrentVideoReader, validate_reference, video_identity
from trajectory_modeling import MAX_CONTEXT, MAX_RESPONSE

REPO = Path(__file__).resolve().parents[2]
ENCODING_SHA = '8ef15cbfec342c69eddf886c2d836528c3f9d6d4ec3591b567e1b9b591bed428'
MAX_SECONDS = 1800
MAX_BYTES = 4 * 1024**3
MAX_ROWS = 256214
MAX_SHARD_BYTES = 64 * 1024**2
SCHEMA = 'h85-composite-action-sft-v1'
EXTRA_FIELDS = ('actor_json', 'target_json', 'prefix_tokens', 'target_tokens', 'total_tokens',
                'prefix_ids_sha256', 'target_ids_sha256', 'target_sha256', 'quarantine_reason')


def prepare_row(row, tokenizer, template):
    actor = codec.actor_from_state(row['task'], row['active_instruction'], row['observation_state'])
    target = codec.encode(row['expert_action'], row['observation_state'])
    decoded = codec.decode(target, actor['proprio']['q_rad'])
    errors = codec.errors(row['expert_action'], decoded)
    native_errors = codec.errors(row['expert_action'], decoded.astype(np.float32))
    cast_errors = codec.errors(decoded, decoded.astype(np.float32))
    if any(native_errors[k] > codec.ERROR_LIMITS[k] + cast_errors[k] + 1e-10 for k in errors):
        raise ValueError('Native float32 reconstruction violates error contract')
    ids = prefix_ids(tokenizer, template, actor)
    response = tokenizer.encode(target, add_special_tokens=False)
    if tokenizer.decode(response, skip_special_tokens=False) != target:
        raise ValueError('Action text tokenizer roundtrip failed')
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos != tokenizer.eos_token_id or tokenizer.convert_ids_to_tokens(eos) != '<|im_end|>':
        raise ValueError('Registered native EOS required')
    response.append(eos)
    reason = ('response_over_budget' if len(response) > MAX_RESPONSE else
              'context_over_budget' if len(ids) + len(response) > MAX_CONTEXT else '')
    output = {**row, 'actor_json': packed(actor).decode(), 'target_json': target,
              'prefix_tokens': len(ids), 'target_tokens': len(response), 'total_tokens': len(ids) + len(response),
              'prefix_ids_sha256': ids_digest(ids), 'target_ids_sha256': ids_digest(response),
              'target_sha256': hashlib.sha256(target.encode()).hexdigest(), 'quarantine_reason': reason}
    return output, native_errors


def preflight(corpus, review_root, encoding_root, model_path, pins, deadline):
    """Reuse the real-processor evidence, and independently decode raw videos."""
    from transformers import AutoTokenizer
    review = strict_json(snapshot(review_root / 'manifest.json', pins, REVIEW_SHA))
    human = strict_json(snapshot(REPO / 'configs/vlm_sft/h85_parent_action_review_v1.json', pins, HUMAN_SHA))
    cases = checked_cases(review, human)
    evidence = strict_json(snapshot(encoding_root / 'result.json', pins, ENCODING_SHA))
    if (evidence['status'] != 'CPU_MULTIMODAL_ENCODING_CHECKED_NOT_TRAIN_RELEASE'
            or evidence['review_manifest_sha256'] != REVIEW_SHA or evidence['cases'] != 50
            or (encoding_root / 'failure.json').exists()): raise ValueError('Real processor evidence required')
    rows = [strict_json(line) for line in snapshot(encoding_root / 'rows.jsonl', pins, evidence['rows_sha256']).splitlines()]
    encoded = {r['id']: r for r in rows}
    if len(rows) != 50 or set(encoded) != {c['id'] for c in cases}: raise ValueError('Encoding cohort mismatch')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    template = snapshot(model_path / 'chat_template.jinja', pins).decode()
    shards = {s['episode']: s for s in corpus['shards']}
    checked = []
    with CurrentVideoReader() as reader:
        for case in cases:
            if time.monotonic() > deadline: raise TimeoutError('Preflight budget')
            row = case['row']; source = shards[case['episode']]
            actor = codec.actor_from_state(row['task'], row['active_instruction'], row['observation_state'])
            ids = prefix_ids(tokenizer, template, actor)
            if ids_digest(ids) != encoded[row['id']]['prefix_ids_sha256']:
                raise ValueError('Pixel-independent prefix differs from actual three-image processor')
            images, receipts = reader.read(row, source)
            original = {r['view']: r for r in case['images'] if r['offset'] == 0}
            pixels = {}
            try:
                for view, image in images.items():
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest != original[view]['pixels_sha256']:
                        raise ValueError('Actual video loader differs from reviewed current original PNG')
                    pixels[view] = digest
            finally:
                for image in images.values(): image.close()
            checked.append({'id': row['id'], 'prefix_ids_sha256': ids_digest(ids), 'pixels_sha256': pixels,
                            'decoded': receipts})
    return {'cases': len(checked), 'current_images': len(checked) * 3, 'all_prefix_ids_equal_actual_processor': True,
            'all_video_pixels_equal_current_png': True, 'rows': checked}


def init_worker(model, asset_pins, deadline, stopped):
    import pyarrow as pa
    from transformers import AutoTokenizer
    global TOKENIZER, TEMPLATE, DEADLINE, STOPPED
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    for path, digest in asset_pins.items(): snapshot(path, {}, digest)
    TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    TEMPLATE = Path(model, 'chat_template.jinja').read_text()
    DEADLINE = deadline
    STOPPED = stopped


def bounded():
    if time.monotonic() > DEADLINE or STOPPED.is_set():
        raise TimeoutError('SFT export stopped or deadline reached')


def build_shard(source, corpus_root, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    bounded()
    relative = Path(source['shard'])
    if relative.parts != ('shards', source['split'], f'episode_{source["episode"]:06d}.parquet'):
        raise ValueError('Unsafe source shard layout')
    data = snapshot(Path(corpus_root) / relative, {}, source['shard_sha256'])
    if len(data) != source['shard_bytes']: raise ValueError('Source shard size mismatch')
    table = pq.read_table(pa.BufferReader(data), use_threads=False)
    rows = table.to_pylist()
    if len(rows) != source['samples'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Source row count/uniqueness mismatch')
    for v in source['videos'].values(): video_identity(v)
    accepted = []; quarantine = []; lengths = []; errors = dict.fromkeys(codec.ERROR_LIMITS, 0.)
    for row in rows:
        bounded()
        for view in ('head', 'left_wrist', 'right_wrist'): validate_reference(row, source, view)
        record, err = prepare_row(row, TOKENIZER, TEMPLATE)
        for k in errors: errors[k] = max(errors[k], err[k])
        (quarantine if record['quarantine_reason'] else accepted).append(record)
        lengths.append([record[k] for k in ('prefix_tokens', 'target_tokens', 'total_tokens')])
    schema = table.schema
    for name in EXTRA_FIELDS:
        schema = schema.append(pa.field(name, pa.int32() if name in ('prefix_tokens', 'target_tokens', 'total_tokens') else pa.string()))
    files = []; written = 0
    for kind, values in (('shards', accepted), ('quarantine', quarantine)):
        if not values: continue
        rel = Path(kind) / source['split'] / relative.name
        dest = Path(output) / rel
        if dest.exists(): raise FileExistsError('Never overwrite exported shard')
        result = pa.Table.from_pylist(values, schema=schema)
        buffer = pa.BufferOutputStream(); pq.write_table(result, buffer, compression='zstd')
        saved = buffer.getvalue().to_pybytes()
        written += len(saved)
        if written > MAX_SHARD_BYTES: raise ValueError('Per-source SFT byte limit')
        bounded()
        with dest.open('xb') as stream: stream.write(saved)
        restored = pq.read_table(pa.BufferReader(saved), use_threads=False).to_pylist()
        if restored != values: raise ValueError('SFT parquet altered original state/control/target')
        files.append({'kind': kind, 'path': rel.as_posix(), 'sha256': hashlib.sha256(saved).hexdigest(),
                      'bytes': len(saved), 'samples': len(values)})
    for v in source['videos'].values(): video_identity(v)
    return {'task': source['task'], 'instance': source['instance'], 'episode': source['episode'],
            'split': source['split'], 'source_shard': source['shard'], 'source_sha256': source['shard_sha256'],
            'samples': len(accepted), 'quarantined': len(quarantine), 'files': files, 'lengths': lengths,
            'quarantine_reasons': dict(Counter(r['quarantine_reason'] for r in quarantine)),
            'worst_native_float32_errors': errors}


def verify_export_files(output, results, sources):
    """Seal only the current exact files, not stale post-worker receipts."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(output).resolve()
    expected = {'launch.json', 'loader_preflight.json'}
    total = 0
    for result in results:
        for file in result['files']:
            relative = Path(file['path'])
            if (relative.parts != (file['kind'], result['split'], f'episode_{result["episode"]:06d}.parquet')
                    or file['kind'] not in ('shards', 'quarantine') or file['path'] in expected):
                raise ValueError('Output file identity repeated or invalid')
            expected.add(file['path']); path = output / relative
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                raise ValueError('Output file escaped export root')
            content = snapshot(path, {}, file['sha256'])
            if len(content) != file['bytes']: raise ValueError('Post-worker output size changed')
            table = pq.read_table(pa.BufferReader(content), use_threads=False)
            if table.num_rows != file['samples']: raise ValueError('Post-worker output row count changed')
            total += len(content)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != expected: raise ValueError('Missing or unexpected export files')
    for source in sources:
        for video in source['videos'].values(): video_identity(video)
    if total + sum((output / f).stat().st_size for f in ('launch.json', 'loader_preflight.json')) > MAX_BYTES - 1024**2:
        raise ValueError('Final output byte limit')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'review-root', 'encoding-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); pins = {}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('CPU-only offline export required')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean frozen source required')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    for name in ('prepare_trajectory_sft.py', 'trajectory_text.py', 'trajectory_video.py', 'trajectory_modeling.py',
                 'native_trajectory_codec.py', 'prepare_visual_review.py', 'audit_trajectory_encoding.py'):
        snapshot(REPO / 'scripts/vlm_sft' / name, pins)
    manifest = strict_json(snapshot(args.corpus / 'manifest.json', pins, CORPUS_SHA))
    if (manifest['status'] != 'INTERMEDIATE_NATIVE_ACTION_CORPUS_NOT_SFT_RELEASE'
            or manifest['samples'] != MAX_ROWS or len(manifest['shards']) != 466
            or (args.corpus / 'failure.json').exists()): raise ValueError('Exact complete source corpus required')
    spec = strict_json(snapshot(MODEL_SPEC, pins)); model = Path(spec['model'])
    assets = {}
    for name in PROCESSOR_FILES:
        snapshot(model / name, pins, spec['model_files'][name]); assets[str(model / name)] = spec['model_files'][name]
    protected = [REPO, model, args.corpus, args.review_root, args.encoding_root]
    for source in manifest['shards']:
        protected.append(Path(source['source_parquet']).resolve().parents[2])
        protected.extend(Path(v['path']).resolve().parent for v in source['videos'].values())
    forbid_source_output(args.output, *protected)
    if args.output.exists(): raise FileExistsError('Preserve previous SFT export')
    if shutil.disk_usage(args.output.parent).free < MAX_BYTES + 80 * 1024**3: raise ValueError('Disk reserve')
    os.sched_setaffinity(0, set(range(48, 56)))
    args.output.mkdir()
    launch = {'schema': SCHEMA, 'code_commit': commit, 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
              'corpus_root': str(args.corpus), 'corpus_manifest_sha256': CORPUS_SHA, 'protocol': codec.VERSION,
              'model': str(model), 'model_revision': spec['revision'], 'max_seconds': MAX_SECONDS,
              'max_bytes': MAX_BYTES, 'max_rows': MAX_ROWS, 'workers': 4, 'training_eligible': False,
              'training_updates': 0, 'model_weight_loads': 0, 'controls': 0}
    (args.output / 'launch.json').write_bytes(packed(launch) + b'\n')
    results = []; total_bytes = 0
    try:
        before = time.monotonic()
        checked = preflight(manifest, args.review_root, args.encoding_root, model, pins, started + 600)
        checked['wall_seconds'] = time.monotonic() - before
        (args.output / 'loader_preflight.json').write_bytes(packed(checked) + b'\n')
        print(json.dumps({'loader_preflight_cases': checked['cases'], 'seconds': checked['wall_seconds']}), flush=True)
        for name in ('shards', 'quarantine'):
            for split in ('train', 'validation', 'test'): (args.output / name / split).mkdir(parents=True)
        # No new source selection: export exactly the immutable corpus cohort.
        context = mp.get_context('spawn')
        stopped = context.Event()
        pool = ProcessPoolExecutor(max_workers=4, mp_context=context, initializer=init_worker,
                                   initargs=(str(model), assets, started + MAX_SECONDS - 30, stopped))
        futures = []
        try:
            futures = [pool.submit(build_shard, s, args.corpus, args.output) for s in manifest['shards']]
            for future in as_completed(futures):
                result = future.result(); results.append(result)
                total_bytes += sum(f['bytes'] for f in result['files'])
                if total_bytes > MAX_BYTES - 5 * MAX_SHARD_BYTES or shutil.disk_usage(args.output).free < 80 * 1024**3:
                    raise ValueError('SFT export disk budget exceeded')
                if len(results) % 20 == 0:
                    print(json.dumps({'shards': len(results), 'accepted': sum(r['samples'] for r in results),
                                      'quarantined': sum(r['quarantined'] for r in results),
                                      'seconds': time.monotonic() - started}), flush=True)
        except BaseException:
            stopped.set()
            for future in futures: future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        results.sort(key=lambda r: r['episode'])
        if sum(r['samples'] + r['quarantined'] for r in results) != MAX_ROWS or len(results) != 466:
            raise ValueError('Incomplete full-corpus accounting')
        for path, digest in pins.items(): snapshot(path, {}, digest)
        for source in manifest['shards']:
            snapshot(args.corpus / source['shard'], {}, source['shard_sha256'])
        if verify_export_files(args.output, results, manifest['shards']) != total_bytes:
            raise ValueError('Final output accounting changed')
        lengths = [v for r in results for v in r.pop('lengths')]
        elapsed = time.monotonic() - started
        if elapsed > MAX_SECONDS: raise TimeoutError('Final SFT deadline')
        counts = {split: sum(r['samples'] for r in results if r['split'] == split) for split in ('train', 'validation', 'test')}
        status = {**launch, 'status': 'ACTION_SFT_FORMAT_COMPLETE_CAPACITY_PENDING', 'wall_seconds': elapsed,
                  'samples': sum(counts.values()), 'samples_by_split': counts,
                  'samples_by_task_split': {str(t): {s: sum(r['samples'] for r in results if r['split'] == s and r['task'] == t)
                                             for s in counts} for t in range(5)},
                  'quarantined': sum(r['quarantined'] for r in results), 'shard_bytes': total_bytes,
                  'original_rows_preserved': MAX_ROWS, 'sources': results, 'input_sha256': pins,
                  'prefix_tokens_all': percentiles([v[0] for v in lengths]),
                  'target_tokens_all': percentiles([v[1] for v in lengths]),
                  'total_tokens_all': percentiles([v[2] for v in lengths]),
                  'worst_native_float32_errors': {k: max(r['worst_native_float32_errors'][k] for r in results) for k in codec.ERROR_LIMITS},
                  'loader_preflight_sha256': hashlib.sha256((args.output / 'loader_preflight.json').read_bytes()).hexdigest(),
                  'max_context_tokens': MAX_CONTEXT, 'max_response_tokens': MAX_RESPONSE,
                  'original_float32_roundtrip': True, 'current_video_loader_checked_cases': 50,
                  'full_dataset_pixels_individually_reviewed': False, 'gpu_forward_backward': 'PENDING',
                  'three_hour_training_capacity': 'PENDING_ACTUAL_THROUGHPUT',
                  'online_control_interface_changed': False}
        (args.output / 'manifest.json').write_bytes(packed(status) + b'\n')
    except BaseException as exc:
        (args.output / 'failure.json').write_bytes(packed({'error': repr(exc), 'completed_shards': len(results),
                                                         'seconds': time.monotonic() - started}) + b'\n')
        raise


if __name__ == '__main__': main()
