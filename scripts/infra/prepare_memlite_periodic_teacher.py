"""Select TRAIN-only periodic-history candidates for human review, not fitting.

No new outcome truth, no low-action targets, no decoding or GPU work. Original
accepted labels/splits stay immutable; a candidate is not a training admission.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rl/memlite_online/code'))
from recovery_periodic_teacher import periodic_annotation_trace, command_kind, unknown_feedback
from behavior_champion.release import AcceptedRelease, sha256
from behavior_champion.sampling import digest_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location('periodic_original_labels', REPO / 'src/g05/data/memlite_stage1_labels.py')
    labels = importlib.util.module_from_spec(spec); spec.loader.exec_module(labels)
    accepted = AcceptedRelease(args.release)
    import numpy as np
    for name in ('train_candidates.npy', 'train_tasks.npy'):
        accepted.verify_file(name)
    original_candidates = np.load(args.release / 'train_candidates.npy', mmap_mode='r')
    original_tasks = np.load(args.release / 'train_tasks.npy', mmap_mode='r')
    limits = dict(initial=2, transition=8, continuation=8)
    chosen, counts, tasks = defaultdict(list), Counter(), defaultdict(Counter)
    started = time.monotonic(); original_cursor = 0
    for serial, episode in accepted.episodes('train'):
        anchors = labels.anchor_records(episode['segments'], episode['phase'])
        records, audit = periodic_annotation_trace(anchors, episode['segments'])
        task = int(episode['row']['task_index'])
        counts['episodes'] += 1; counts['calls'] += len(records)
        counts['episodes_ended_at_annotation_gap'] += audit['first_unannotated_scheduled_call'] is not None
        counts['known_calls_omitted_after_gap'] += audit['omitted_known_calls_after_gap']
        for number, record in enumerate(records):
            segment = episode['segments'][record['segment']]
            kind = command_kind(record, segment); counts[kind] += 1; tasks[task][kind] += 1
            identity = dict(episode_serial=serial, original_candidate_index=original_cursor + record['original_anchor'],
                episode=int(episode['row']['episode_index']), task=task,
                instance=int(episode['row']['task_instance_id']), raw_episode=int(episode['row']['raw_episode_id']),
                frame=record['frame'], phase=int(episode['phase']), split='train', kind=kind)
            index = identity['original_candidate_index']
            if (not np.array_equal(original_candidates[index], [serial, record['original_anchor']])
                    or original_tasks[index] != task):
                raise ValueError('Periodic candidate identity differs from original TRAIN sampler')
            score = int.from_bytes(hashlib.sha256(('periodic128-v1:' + digest_json(identity)).encode()).digest(), 'big')
            heap = chosen[(task, kind)]
            if len(heap) >= limits[kind] and score >= -heap[0][0]: continue
            target = labels.projection(segment, branch='high', task_name=episode['task_name'],
                previous_intent=record['previous_intent'], previous_parent=record['previous_parent'], history=record['history'])
            target['execution_feedback'] = unknown_feedback(number, records, episode['segments'])
            target['low_action_supervision_mask'] = False
            row = dict(identity=identity, projection=target, annotation_trace=audit,
                original_stride16_previous_intent=anchors[record['original_anchor']][2],
                human_admission=False, teacher_kind='annotation_periodic_command_trace_not_actor_rollout')
            entry = (-score, identity['original_candidate_index'], row)
            if len(heap) < limits[kind]: heapq.heappush(heap, entry)
            else: heapq.heapreplace(heap, entry)
        original_cursor += len(anchors)
    rows = [entry[2] for key in sorted(chosen) for entry in sorted(chosen[key], key=lambda row: -row[0])]
    if set(tasks) != set(range(100)) or original_cursor != 12299471:
        raise ValueError('Original protected TRAIN universe changed')
    if any(row['projection']['outcome_supervision_mask'] or row['projection']['low_action_supervision_mask'] for row in rows):
        raise ValueError('New outcome/BC supervision is forbidden in this candidate preparation')
    args.output.mkdir(parents=True)
    with (args.output / 'candidates.jsonl').open('x') as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True) + '\n')
    result = dict(status='metadata_candidates_pending_human_review_not_training_admitted',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        release_manifest_sha256=sha256(args.release / 'manifest.json'),
        candidate_sha256=sha256(args.output / 'candidates.jsonl'), original_train_anchors=original_cursor,
        counts=dict(counts), selected=len(rows), selected_counts={f'{task}:{kind}':len(heap) for (task,kind),heap in sorted(chosen.items())},
        per_task={str(task):dict(count) for task,count in sorted(tasks.items())},
        preserved_quarantines=accepted.excluded_splits, original_eval_excluded=1000,
        new_outcome_labels=0, low_action_labels=0, decoded_images=0, optimizer_updates=0,
        human_review_complete=False, seconds=time.monotonic()-started,
        limitation='Only annotation-derived call histories; no physical recovery/result truth or on-policy provenance. Unknown scheduled calls truncate usable prefixes; never assume an unobserved gap command.')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('per_task','selected_counts')}), flush=True)


if __name__ == '__main__': main()
