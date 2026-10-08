"""Incremental TRAIN progress only. Never label this a fixed-policy evaluation."""
from collections import defaultdict
import json
from pathlib import Path
import time

from checkpoint_io import atomic_json


class TrainingMonitor:
    def __init__(self, job):
        self.job = Path(job)
        self.offsets = {}
        self.episodes = {}
        self.completed = {}
        self.total_controls = 0
        self.created = time.time()
        self.last_effect_hour = 0
        self.last_successes = 0

    def refresh(self):
        for path in sorted(self.job.glob('collectors/gpu_*/cycle_*/reward_steps.jsonl')):
            offset = self.offsets.get(str(path), 0)
            with path.open('rb') as stream:
                stream.seek(offset)
                for _ in range(20000):
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        stream.seek(start)
                        break
                    row = json.loads(line)
                    episode_id = row['episode']['episode_id']
                    entry = self.episodes.setdefault(episode_id, dict(task=row['task'], controls=0,
                        initial_q=row['official_q'], max_q=row['official_q'], final_q=None,
                        max_skill=0., success=False, terminal=False))
                    entry.update(controls=row['episode_control_step'], q=row['official_q'],
                        max_q=max(entry['max_q'], row['official_q']),
                        max_skill=max(entry['max_skill'], row.get('skill_potential', 0.)),
                        success=entry['success'] or bool(row['official_success']),
                        terminal=bool(row['task_terminal']))
                    if row['task_terminal']:
                        entry['final_q'] = row['official_q']
                    self.total_controls += 1
                self.offsets[str(path)] = stream.tell()
        for path in sorted(self.job.glob('collectors/gpu_*/cycle_*/eval_*/json/*.json')):
            if str(path) in self.completed or not path.parent.parent.parent.joinpath(
                    path.parent.parent.name.removeprefix('eval_') + '.completed').exists():
                continue
            row = json.loads(path.read_text())
            self.completed[str(path)] = dict(task=path.parent.parent.name.removeprefix('eval_'),
                final_q=row['q_score']['final'], success=bool(row['success']), path=str(path))
        by_task = defaultdict(list)
        for row in self.completed.values():
            by_task[row['task']].append(row)
        per_task = {task: dict(completed=len(rows), successes=sum(r['success'] for r in rows),
            mean_q=sum(r['final_q'] for r in rows) / len(rows),
            sr=sum(r['success'] for r in rows) / len(rows)) for task, rows in by_task.items()}
        updates = []
        for path in sorted(self.job.glob('policies/gpu_*/latest_update.json')):
            updates.append(json.loads(path.read_text()))
        versions = [u['update'] for u in updates]
        result = dict(updated=time.time(), metric_scope='on_policy_TRAIN_not_independent_evaluation',
            controls=self.total_controls, episodes_seen=len(self.episodes),
            task_coverage=len({e['task'] for e in self.episodes.values()}), expected_tasks=100,
            completed_episodes=len(self.completed), completed_task_coverage=len(per_task),
            successes=sum(r['success'] for r in self.completed.values()),
            completed_only_sr=(sum(r['success'] for r in self.completed.values()) / len(self.completed)
                               if self.completed else None),
            macro_q_covered=(sum(t['mean_q'] for t in per_task.values()) / len(per_task) if per_task else None),
            macro_sr_covered=(sum(t['sr'] for t in per_task.values()) / len(per_task) if per_task else None),
            episodes_with_q_increase=sum(e['max_q'] > e['initial_q'] + 1e-6 for e in self.episodes.values()),
            versions=versions, synchronized_last_commits=len(versions) == 8 and len(set(versions)) == 1,
            per_task=per_task, latest_rank0_update=updates[0] if updates else None,
            controls_per_wall_second=self.total_controls / max(1., time.time() - self.created))
        atomic_json(self.job / 'training_summary.json', result)
        elapsed_hours = int((time.time() - self.created) / 3600)
        if result['successes'] > self.last_successes or elapsed_hours // 2 > self.last_effect_hour // 2:
            event = dict(time=time.time(), elapsed_hours=elapsed_hours,
                kind='new_train_success' if result['successes'] > self.last_successes else 'effect_checkpoint',
                successes=result['successes'], completed=result['completed_episodes'],
                task_coverage=result['task_coverage'], macro_q_covered=result['macro_q_covered'],
                episodes_with_q_increase=result['episodes_with_q_increase'],
                interpretation='Training-only evidence; changing task mix and policy, not a causal improvement claim')
            with (self.job/'effect_events.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            self.last_effect_hour = elapsed_hours
            self.last_successes = result['successes']
        return result
