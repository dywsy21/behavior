"""Read-only full metadata/annotation count for the user's fixed stride grid.

Not a train/validation split or label release; no RGB/depth/model loaded.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time


def phase(task, raw_episode, instance, seed=17):
    identity = json.dumps([seed, task, raw_episode, instance], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big') % 16


def count_anchors(length, offset):
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError('Length must be a nonnegative integer')
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < 16:
        raise ValueError('One fixed phase in 0..15 is required')
    return max(0, (length - 1 - offset) // 16 + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    started = time.monotonic()
    rows = []
    tasks = defaultdict(lambda: Counter(episodes=0, raw_frames=0, observation_candidates=0,
                                         complete_32_action_candidates=0))
    differences = Counter()
    missing_annotations = []
    seen = set()
    for path in sorted((args.root / 'meta/episodes').glob('chunk-*/file-*.parquet')):
        for row in pq.read_table(path).to_pylist():
            task, episode, length = int(row['task_index']), int(row['episode_index']), int(row['length'])
            raw, instance = int(row['raw_episode_id']), int(row['task_instance_id'])
            if (task, raw, instance) in seen:
                raise ValueError('Duplicate logical episode identity')
            seen.add((task, raw, instance))
            offset = phase(task, raw, instance)
            candidates = count_anchors(length, offset)
            full = count_anchors(max(0, length-31), offset)
            tasks[task].update(episodes=1, raw_frames=length, observation_candidates=candidates,
                               complete_32_action_candidates=full)
            annotation_path = args.root / row['annotation_path']
            if annotation_path.is_file():
                annotation = json.loads(annotation_path.read_text())
                duration = annotation.get('meta_data', {}).get('task_duration')
                if isinstance(duration, int):
                    differences[length-duration] += 1
                else:
                    differences['unknown_duration'] += 1
            else:
                missing_annotations.append(str(row['annotation_path']))
            rows.append(dict(task=task, episode=episode, raw_episode_id=raw, instance=instance,
                             length=length, offset=offset, observation_candidates=candidates,
                             complete_32_action_candidates=full))
    if set(tasks) != set(range(100)) or len(rows) != 20000:
        raise ValueError('Expected all 100 tasks and 20000 episodes')
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    (args.output / 'episode_phases.json').write_bytes(raw)
    total = sum(x['observation_candidates'] for x in rows)
    result = dict(status='complete', seed=17, stride=16, phases_fixed_across_epochs=True,
                  same_phase_for_three_cameras=True, tasks=dict(sorted(tasks.items())),
                  episodes=len(rows), raw_frames=sum(x['length'] for x in rows),
                  observation_candidates_per_pass=total, observation_candidates_two_passes=2*total,
                  complete_32_action_candidates=sum(x['complete_32_action_candidates'] for x in rows),
                  annotation_duration_differences=dict(differences), missing_annotations=missing_annotations,
                  before_split_and_label_eligibility=True, formal_training_index_released=False,
                  phase_manifest_sha256=hashlib.sha256(raw).hexdigest(), seconds=time.monotonic()-started)
    with (args.output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'}), flush=True)


if __name__ == '__main__':
    main()
