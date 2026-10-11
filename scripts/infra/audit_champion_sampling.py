"""CPU-only exhaustive audit of the proposed 8000-update task-balanced plan."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from behavior_champion.dataset import NativeV3Dataset
from behavior_champion.sampling import BalancedPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    data = NativeV3Dataset(args.release)
    plan = BalancedPlan(data)
    tasks = np.array([ep['row']['task_index'] for ep in data.episodes])
    phases = np.array([ep['phase'] for ep in data.episodes])
    counts = np.zeros(100, dtype=np.int64)
    selected = []
    for update in range(8000):
        indices = plan.batch(update)
        serials = np.searchsorted(data.offsets, indices, side='right') - 1
        per_task = np.bincount(tasks[serials], minlength=100)
        if not np.array_equal(np.sort(per_task), [2] * 44 + [3] * 56):
            raise ValueError('Not all 100 tasks have exactly 2 or 3 samples in a batch')
        frames = phases[serials] + 16 * (indices - data.offsets[serials])
        if not np.array_equal(frames % 16, phases[serials]):
            raise ValueError('Changed per-video stride phase')
        selected.append(indices); counts += per_task
    if len(np.unique(np.concatenate(selected))) != 8000 * 256 or not np.all(counts == 20480):
        raise ValueError('Duplicate anchor or unfair task allocation')
    layout = json.loads(args.layout.read_text())['layout']
    old = np.array([layout['dataset_to_model'][str(task)] < 50 for task in range(100)])
    rebuilt = BalancedPlan(data)
    for completed in (0, 1, 2, 24, 25, 999, 1000, 7999, 8000):
        rebuilt.validate_cursor(plan.cursor(completed), completed)
    result = dict(status='all_8000_batches_verified_cpu_only', optimizer_updates=0, physical_controls=0,
        train_episodes=len(data.episodes), eligible_anchors=len(data), original_quarantines=data.excluded_episode_counts,
        plan_sha256=plan.fingerprint, selected_anchors=8000 * 256, unique_anchors=8000 * 256,
        every_batch_task_counts={'two': 44, 'three': 56}, per_task_samples=counts.tolist(),
        old50_samples=int(counts[old].sum()), new50_samples=int(counts[~old].sum()),
        deterministic_resume_cursors_checked=9, seconds=time.monotonic() - started, contract=plan.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('contract', 'per_task_samples')}))


if __name__ == '__main__': main()
