"""CPU-only H84 annotation inventory, deterministic shards and fail-closed resume.

No model import/load, label approval, training export or process termination.
Resume skips *recorded proposals*, not 'correct labels'. Errors remain retryable.
Source/gold metadata never enters the image-only actor prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/vlm_sft/h84_full_annotation_plan_v1.json'
VIEWS = ('head', 'left_wrist', 'right_wrist')
SPLITS = ('train', 'validation', 'test')
SCHEMA = 'h84-cpu-annotation-queue-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return digest(Path(path).read_bytes())


def checked_bytes(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Original regular file required: {path}')
    value = path.read_bytes()
    if digest(value) != expected:
        raise ValueError(f'SHA changed: {path}')
    return value


def json_lines(data):
    return [strict_json(line) for line in data.splitlines() if line.strip()]


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def non_finite(value):
        raise ValueError('Non-finite JSON number: ' + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=non_finite)


def forbid_source_output(output, *sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or source in output.parents:
            raise ValueError('Output must not modify a frozen source directory')


def load_inputs(metadata):
    """One in-memory snapshot; validate every input before constructing output."""
    plan_bytes = PLAN.read_bytes()
    plan = strict_json(plan_bytes)
    files = {name: checked_bytes(Path(metadata) / name, pin)
             for name, pin in plan['metadata_sha256'].items()}
    checked_bytes(REPO / plan['actor_protocol_file'], plan['actor_protocol_sha256'])
    reviews = [strict_json(checked_bytes(REPO / name, pin))
               for name, pin in plan['review_files'].items()]
    original = strict_json(files['source_plan.json'])
    seal = strict_json(files['manifest.json'])
    complete = strict_json(files['complete.json'])
    if (complete['manifest_sha256'] != plan['metadata_sha256']['manifest.json'] or
            complete['training_eligible'] is not False or seal['training_eligible'] is not False or
            seal['images_manifest_sha256'] != plan['metadata_sha256']['images.jsonl'] or
            seal['source_plan_sha256'] != plan['metadata_sha256']['source_plan.json'] or
            original['source_identity']['tasks_sha256'] != plan['metadata_sha256']['tasks.jsonl']):
        raise ValueError('Broken source seals')
    tasks = json_lines(files['tasks.jsonl'])
    if sorted(row['task_index'] for row in tasks) != list(range(5)):
        raise ValueError('Task scope changed')
    return plan, digest(plan_bytes), json_lines(files['images.jsonl']), original, reviews


def validate_states(states, original, expected):
    sources = {}
    expected_states = set()
    forbidden = set(map(tuple, original['source_identity']['protected_groups']))
    for source in original['sources']:
        group = (source['task'], source['instance'])
        if group in sources or group in forbidden or source['split'] not in SPLITS:
            raise ValueError('Duplicate/protected source group or split')
        sources[group] = source
        for frame in source['selected_frames']:
            item = (*group, source['episode'], frame)
            if item in expected_states:
                raise ValueError('Duplicate planned state')
            expected_states.add(item)
    seen = set()
    for row in states:
        key = (row['task'], row['instance'], row['episode'], row['frame'])
        if key in seen or key not in expected_states:
            raise ValueError('Duplicate/unplanned raw state')
        seen.add(key)
        source = sources[key[:2]]
        name = f't{key[0]}_i{key[1]}_e{key[2]}_f{key[3]:06d}'
        if (row['id'] != name or row['split'] != source['split'] or
                row['training_eligible'] is not False or
                set(row['images']) != set(VIEWS) or set(row['image_receipts']) != set(VIEWS)):
            raise ValueError('State identity, split or raw-only status changed')
        for view in VIEWS:
            receipt = row['image_receipts'][view]
            size = 720 if view == 'head' else 480
            if (row['images'][view] != f'images/{name}_{view}.png' or
                    receipt['resolution'] != [size, size] or receipt['raw_resolution_preserved'] is not True or
                    any(not re.fullmatch('[0-9a-f]{64}', receipt[k])
                        for k in ('png_sha256', 'raw_pixels_sha256'))):
                raise ValueError('Raw view/path/receipt changed')
    if seen != expected_states or len(states) != expected['states'] or len(sources) != expected['groups']:
        raise ValueError('Raw source coverage is incomplete')


def calibration_rows(states, reviews, plan):
    by_id = {r['id']: r for r in states}
    protected = set()
    for review in reviews:
        if review['training_eligible'] is not False or review['teacher_outputs_seen_before_labeling'] is not False:
            raise ValueError('Calibration must be pre-teacher, not student training')
        groups = set(map(tuple, review['protect_groups_in_future_student_releases']))
        actual = set()
        for row in review['rows']:
            raw = by_id[row['id']]
            if raw['split'] != 'train':
                raise ValueError('Heldout image in calibration')
            actual.add((raw['task'], raw['instance']))
        if groups != actual:
            raise ValueError('Calibration protection is not source-group-complete')
        protected |= groups
    if len(protected) != plan['expected']['protected_train_groups']:
        raise ValueError('Calibration exclusion count changed')
    # The new gold remains presence-only; it does not invent missing boxes.
    review = reviews[1]
    if (review['schema'] != 'h84-task24-parent-presence-v1' or review['boxes_reviewed'] is not False or
            review['heldout_used'] is not False or review['views'] != list(VIEWS)):
        raise ValueError('Unexpected expanded presence gold')
    seen = set()
    counts = Counter()
    rows = []
    for item in review['rows']:
        raw = by_id[item['id']]
        for query, labels in item['labels'].items():
            if query not in plan['queries_by_task'][str(raw['task'])] or not re.fullmatch('[PNU]{3}', labels):
                raise ValueError('Invalid reviewed query/visibility')
            for view, label in zip(VIEWS, labels):
                key = (item['id'], view, query)
                if key in seen:
                    raise ValueError('Duplicate calibration query')
                seen.add(key)
                counts[label] += 1
                rows.append({'image_id': item['id'] + '_' + view, 'image': raw['images'][view],
                             'query': query, 'label': label, 'note': item['note'],
                             'png_sha256': raw['image_receipts'][view]['png_sha256'],
                             'raw_pixels_sha256': raw['image_receipts'][view]['raw_pixels_sha256'],
                             'resolution': raw['image_receipts'][view]['resolution'],
                             'source_split': 'train', 'role': 'annotation_calibration',
                             'training_eligible': False, 'boxes_reviewed': False})
    if counts != review['counts']:
        raise ValueError('Human presence counts changed')
    return protected, rows


def make_inventory(plan, plan_sha, states, protected):
    images, jobs = [], []
    ordered = sorted(states, key=lambda r: (SPLITS.index(r['split']), r['task'], r['instance'], r['frame']))
    for row in ordered:
        role = ('annotation_calibration' if (row['task'], row['instance']) in protected else
                'student_candidate' if row['split'] == 'train' else row['split'])
        for view in VIEWS:
            receipt = row['image_receipts'][view]
            image_id = row['id'] + '_' + view
            image = {'image_id': image_id, 'state_id': row['id'], 'image': row['images'][view],
                     'task': row['task'], 'instance': row['instance'], 'episode': row['episode'],
                     'frame': row['frame'], 'source_split': row['split'], 'role': role, 'view': view,
                     'png_sha256': receipt['png_sha256'], 'raw_pixels_sha256': receipt['raw_pixels_sha256'],
                     'resolution': receipt['resolution'], 'training_eligible': False}
            images.append(image)
            for query in plan['queries_by_task'][str(row['task'])]:
                job_id = digest((plan_sha + '\0' + image_id + '\0' + query).encode())
                jobs.append({'job_id': job_id, 'image_id': image_id, 'image': image['image'],
                             'png_sha256': receipt['png_sha256'], 'query': query,
                             'source_split': row['split'], 'role': role, 'task': row['task'],
                             'status': 'PENDING', 'training_eligible': False})
    if (len(images) != plan['expected']['images'] or len(jobs) != plan['expected']['jobs'] or
            len({r['image_id'] for r in images}) != len(images) or
            len({r['job_id'] for r in jobs}) != len(jobs)):
        raise ValueError('Image/job coverage or uniqueness failed')
    return images, jobs


def inventory(metadata):
    plan, plan_sha, states, original, reviews = load_inputs(metadata)
    validate_states(states, original, plan['expected'])
    protected, calibration = calibration_rows(states, reviews, plan)
    images, jobs = make_inventory(plan, plan_sha, states, protected)
    return plan, plan_sha, images, jobs, calibration, sorted(protected)


def write_jsonl(path, rows):
    with path.open('xb') as stream:
        for row in rows:
            stream.write(packed(row) + b'\n')
    return {'sha256': sha(path), 'rows': len(rows), 'bytes': path.stat().st_size}


def create_queue(metadata, output):
    start = time.monotonic()
    plan, pin, images, jobs, calibration, protected = inventory(metadata)
    output = Path(output)
    # New directory only; never write into a frozen RAW or previous run.
    forbid_source_output(output, plan['raw_root'], metadata)
    output.mkdir(parents=True, exist_ok=False)
    files = {'image_index.jsonl': write_jsonl(output / 'image_index.jsonl', images),
             'calibration_presence.jsonl': write_jsonl(output / 'calibration_presence.jsonl', calibration)}
    (output / 'shards').mkdir()
    for split in SPLITS:
        for task in range(5):
            selected = [r for r in jobs if r['source_split'] == split and r['task'] == task]
            for index, offset in enumerate(range(0, len(selected), plan['shard_size'])):
                name = f'shards/{split}_task{task}_{index:04d}.jsonl'
                files[name] = write_jsonl(output / name, selected[offset:offset + plan['shard_size']])
    # Revalidate identities after writing; an interrupted or drifting build has no seal.
    current = load_inputs(metadata)
    if current[1] != pin:
        raise ValueError('Plan changed during queue construction')
    seal = {'schema': SCHEMA, 'status': 'CPU_READY_LABELS_PENDING', 'plan_sha256': pin,
            'source_metadata_sha256': plan['metadata_sha256'], 'raw_root': plan['raw_root'],
            'builder_sha256': sha(Path(__file__)), 'files': files,
            'images': len(images), 'jobs': len(jobs), 'shards': len(files) - 2,
            'images_by_role': dict(Counter(r['role'] for r in images)),
            'jobs_by_role': dict(Counter(r['role'] for r in jobs)),
            'protected_train_groups': protected, 'calibration_presence_pairs': len(calibration),
            'new_model_calls': 0, 'training_updates': 0, 'controls': 0,
            'accepted_labels': 0, 'training_eligible': False, 'wall_seconds': time.monotonic() - start}
    (output / 'queue.json').write_bytes(packed(seal) + b'\n')
    return seal


def load_queue(queue, metadata):
    queue = Path(queue)
    plan, pin, images, jobs, calibration, protected = inventory(metadata)
    seal_bytes = (queue / 'queue.json').read_bytes()
    seal = strict_json(seal_bytes)
    required = {'schema', 'status', 'plan_sha256', 'source_metadata_sha256', 'raw_root',
                'builder_sha256', 'files', 'images', 'jobs', 'shards', 'images_by_role',
                'jobs_by_role', 'protected_train_groups', 'calibration_presence_pairs',
                'new_model_calls', 'training_updates', 'controls', 'accepted_labels',
                'training_eligible', 'wall_seconds'}
    if set(seal) != required:
        raise ValueError('Exact queue seal schema required')
    if (seal.get('schema') != SCHEMA or seal.get('status') != 'CPU_READY_LABELS_PENDING' or
            seal.get('plan_sha256') != pin or seal.get('source_metadata_sha256') != plan['metadata_sha256'] or
            seal.get('raw_root') != plan['raw_root'] or seal.get('protected_train_groups') != [list(x) for x in protected] or
            seal.get('training_eligible') is not False or seal.get('accepted_labels') != 0):
        raise ValueError('Incomplete/stale queue seal')
    if (seal['builder_sha256'] != sha(Path(__file__)) or
            any(type(seal[k]) is not int or seal[k] != 0
                for k in ('new_model_calls', 'training_updates', 'controls', 'accepted_labels')) or
            seal['images_by_role'] != dict(Counter(r['role'] for r in images)) or
            seal['jobs_by_role'] != dict(Counter(r['role'] for r in jobs)) or
            seal['calibration_presence_pairs'] != len(calibration) or
            type(seal['wall_seconds']) not in (int, float) or
            not math.isfinite(seal['wall_seconds']) or not 0 < seal['wall_seconds'] <= 900):
        raise ValueError('Queue builder, budget or summary changed')
    expected = {'image_index.jsonl': images, 'calibration_presence.jsonl': calibration}
    for split in SPLITS:
        for task in range(5):
            selected = [r for r in jobs if r['source_split'] == split and r['task'] == task]
            for index, offset in enumerate(range(0, len(selected), plan['shard_size'])):
                expected[f'shards/{split}_task{task}_{index:04d}.jsonl'] = selected[offset:offset + plan['shard_size']]
    if set(seal['files']) != set(expected):
        raise ValueError('Missing/extra queue shard')
    for name, records in expected.items():
        content = checked_bytes(queue / name, seal['files'][name]['sha256'])
        if (seal['files'][name] != {'sha256': digest(content), 'rows': len(records), 'bytes': len(content)} or
                content != b''.join(packed(r) + b'\n' for r in records)):
            raise ValueError('Queue differs from sealed original sources')
    if any(seal[k] != value for k, value in (('images', len(images)), ('jobs', len(jobs)), ('shards', len(expected) - 2))):
        raise ValueError('Queue totals changed')
    return {r['job_id']: r for r in jobs}, {**seal, 'verified_queue_sha256': digest(seal_bytes)}


def resume_status(jobs, attempts, queue_sha):
    """Strict append-order attempts; a valid proposal never means data release.

    Model workers must close/flush an attempts file before snapshotting it.
    Truncated, duplicate, foreign-source or conflicting histories fail closed.
    """
    from visual_grounding import parse_answer
    seen, latest, finished = set(), {}, set()
    common = {'schema', 'attempt_id', 'job_id', 'queue_sha256', 'png_sha256', 'query',
              'status', 'model_identity_sha256', 'protocol_sha256', 'response', 'error'}
    for item in attempts:
        if set(item) != common or item['schema'] != 'h84-annotation-attempt-v1':
            raise ValueError('Exact attempt schema required; no release flags')
        job_id = item['job_id']
        if job_id not in jobs or item['queue_sha256'] != queue_sha:
            raise ValueError('Foreign job/queue attempt')
        if not isinstance(item['attempt_id'], str) or not item['attempt_id'] or item['attempt_id'] in seen or job_id in finished:
            raise ValueError('Duplicate attempt or overwrite of completed proposal')
        seen.add(item['attempt_id'])
        job = jobs[job_id]
        if item['png_sha256'] != job['png_sha256'] or item['query'] != job['query']:
            raise ValueError('Attempt source/query changed')
        for field in ('model_identity_sha256', 'protocol_sha256'):
            if not isinstance(item[field], str) or not re.fullmatch('[0-9a-f]{64}', item[field]):
                raise ValueError('Model/protocol evidence hash required')
        if item['status'] == 'PROPOSED':
            if item['error'] is not None or not isinstance(item['response'], str):
                raise ValueError('Ambiguous proposal')
            target = parse_answer(item['response'])
            latest[job_id] = ('REVIEW_REQUIRED' if target['visibility'] == 'uncertain' else 'QUALITY_PENDING')
            finished.add(job_id)
        elif item['status'] == 'ERROR':
            if item['response'] is not None or not isinstance(item['error'], str) or not item['error']:
                raise ValueError('Explicit error evidence required')
            latest[job_id] = 'RETRY_PENDING'
        else:
            raise ValueError('Attempts cannot approve labels or train data')
    rows = [{**job, 'status': latest.get(key, 'PENDING')} for key, job in jobs.items()]
    return rows, {'counts': dict(Counter(r['status'] for r in rows)), 'attempts': len(seen),
                  'accepted_labels': 0, 'training_eligible': False}


def write_resume(queue, metadata, attempt_paths, output):
    jobs, seal = load_queue(queue, metadata)
    queue_pin = seal['verified_queue_sha256']
    attempts, pins = [], []
    for path in attempt_paths:
        data = Path(path).read_bytes()
        # A final newline distinguishes a closed append from a partial JSON record.
        if data and not data.endswith(b'\n'):
            raise ValueError('Unsealed/truncated attempts file')
        attempts.extend(json_lines(data))
        pins.append({'path': str(Path(path).resolve()), 'sha256': digest(data)})
    rows, summary = resume_status(jobs, attempts, queue_pin)
    output = Path(output)
    forbid_source_output(output, queue, metadata, seal['raw_root'])
    output.mkdir(parents=True, exist_ok=False)
    files = {'status.jsonl': write_jsonl(output / 'status.jsonl', rows)}
    retry = [r for r in rows if r['status'] in ('PENDING', 'RETRY_PENDING')]
    files['pending.jsonl'] = write_jsonl(output / 'pending.jsonl', retry)
    summary.update(schema='h84-annotation-resume-v1', queue_sha256=queue_pin,
                   attempt_files=pins, files=files, pending=len(retry))
    if (sha(Path(queue) / 'queue.json') != queue_pin or
            any(sha(item['path']) != item['sha256'] for item in pins)):
        raise ValueError('Input evidence changed during resume snapshot')
    (output / 'resume.json').write_bytes(packed(summary) + b'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'resume'))
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--attempts', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.operation == 'prepare':
        if args.queue or args.attempts:
            parser.error('prepare does not accept inference attempts')
        result = create_queue(args.metadata, args.output)
    else:
        if not args.queue:
            parser.error('resume requires --queue')
        result = write_resume(args.queue, args.metadata, args.attempts, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
