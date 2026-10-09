"""Read existing TRAIN logs without simulator/model imports or remote writes.

Bound each file to its size at the start of its scan; incomplete final lines are
ignored. This is diagnostic evidence, never a fixed-policy evaluation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def lines(path):
    limit = path.stat().st_size
    with path.open('rb') as stream:
        while stream.tell() < limit:
            line = stream.readline(limit - stream.tell())
            if not line.endswith(b'\n'):
                break
            yield json.loads(line)


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), min=values[0], median=statistics.median(values),
                p90=values[int(.9 * (len(values) - 1))], max=values[-1],
                mean=statistics.mean(values))


def audit(job):
    started = time.time()
    manifest = json.loads((job / 'manifest.json').read_text())
    episodes, chunks = {}, {}
    totals = Counter()
    for path in sorted(job.glob('collectors/gpu_*/cycle_*/reward_steps.jsonl')):
        rank = int(path.parents[1].name.removeprefix('gpu_'))
        for row in lines(path):
            eid = row['episode']['episode_id']
            e = episodes.setdefault(eid, dict(task=row['task'], rank=rank,
                instance=row['episode']['instance_id'], controls=0,
                initial_q=row['official_q'], max_q=row['official_q'], final_q=None,
                q_increase_steps=0, skill_min=1., skill_max=0.,
                shape_sum=0., shape_abs_sum=0., terminal_q_sum=0.,
                terminal=False, last_q=row['official_q']))
            q = row['official_q']
            e['q_increase_steps'] += int(q > e['last_q'] + 1e-6)
            e.update(controls=row['episode_control_step'], last_q=q,
                     max_q=max(e['max_q'], q), terminal=row['task_terminal'])
            e['skill_min'] = min(e['skill_min'], row['skill_potential'])
            e['skill_max'] = max(e['skill_max'], row['skill_potential'])
            e['shape_sum'] += row['shaping']
            e['shape_abs_sum'] += abs(row['shaping'])
            e['terminal_q_sum'] += row['terminal_q_reward']
            if row['task_terminal']:
                e['final_q'] = q
            c = chunks.setdefault((rank, row['experience_id']), dict(
                n=0, discounted_reward=0., terminal_q=0., q_change=False))
            c['discounted_reward'] += .99999 ** c['n'] * row['reward']
            c['n'] += 1
            c['terminal_q'] += row['terminal_q_reward']
            c['q_change'] |= q > e['initial_q'] + 1e-6
            totals['controls'] += 1
            totals['positive_terminal_q_steps'] += row['terminal_q_reward'] > 0
            totals['official_success_steps'] += bool(row['official_success'])

    plans = defaultdict(list)
    decisions, outcomes, verbs = Counter(), Counter(), Counter()
    for path in sorted(job.glob('policies/gpu_*/planner_events.jsonl')):
        for row in lines(path):
            event = row['event']
            skills = json.loads(event['active_skills_semantic_json'])
            key = json.dumps(skills, sort_keys=True)
            plans[row['episode']['episode_id']].append((row['control_step'], key))
            decisions[event['decision']] += 1
            outcomes[event.get('previous_outcome')] += 1
            verbs.update(s['verb'] for s in skills)
    for eid, e in episodes.items():
        rows = sorted(plans[eid])
        spans = []
        previous, start = None, 0
        for step, key in rows:
            if key != previous:
                if previous is not None:
                    spans.append(step - start)
                start, previous = step, key
        if previous is not None:
            spans.append(e['controls'] - start)
        e.update(plans=len(rows), skill_segments=len(spans),
                 longest_unchanged_intent_controls=max(spans, default=0))

    update_rows = list(lines(job / 'policies/gpu_0/checkpoints/updates.jsonl'))
    completed = []
    cycles_per_task = Counter()
    for path in sorted(job.glob('collectors/gpu_*/status.json')):
        for cycle in json.loads(path.read_text())['cycles']:
            cycles_per_task[cycle['task']] += 1
            for row in cycle['official_episodes']:
                completed.append(dict(task=cycle['task'], instance=row['instance_id'],
                    controls=row['steps'], q=row['q_score']['final'],
                    success=row['success'], normalized_time=row['time']['normalized_time']))
    per_task = defaultdict(list)
    for row in completed:
        per_task[row['task']].append(row['q'])
    recent = update_rows[-40:]
    return dict(started=started, finished=time.time(), source=manifest['source_commit'],
        job=str(job), scope='on_policy_TRAIN_read_only_non_atomic_snapshot',
        manifest_sha256=hashlib.sha256((job/'manifest.json').read_bytes()).hexdigest(),
        totals=dict(totals), episodes=list(episodes.values()), completed=completed,
        summary=dict(episodes_seen=len(episodes), completed=len(completed),
            successes=sum(row['success'] for row in completed),
            terminal_q_positive_episodes=sum(row['q'] > 0 for row in completed),
            completed_task_macro_q=statistics.mean(statistics.mean(v) for v in per_task.values()),
            episodes_q_increase=sum(e['max_q'] > e['initial_q']+1e-6 for e in episodes.values()),
            completed_episodes_lost_all_q=sum(e['max_q'] > 0 and e['final_q'] == 0 for e in episodes.values()),
            cycles_per_task=dict(cycles_per_task),
            normalized_time=stats([row['normalized_time'] for row in completed]),
            q_positive_results=[row for row in completed if row['q'] > 0],
            chunks_seen=len(chunks), terminal_q_positive_chunks=sum(c['terminal_q'] > 0 for c in chunks.values()),
            discounted_chunk_rewards=stats([c['discounted_reward'] for c in chunks.values()]),
            abs_chunk_reward_below_1e_3=sum(abs(c['discounted_reward']) < 1e-3 for c in chunks.values()),
            planner_decisions=dict(decisions), planner_outcomes=dict(outcomes), planner_verbs=dict(verbs),
            completed_intent_static_fraction=stats([e['longest_unchanged_intent_controls']/e['controls']
                for e in episodes.values() if e['terminal']]),
            completed_single_skill_episodes=sum(e['terminal'] and e['skill_segments'] == 1 for e in episodes.values()),
            unchanged_kl_max=max(r['before']['mean_approx_kl'] for r in update_rows),
            actor_updates=len(update_rows), actor_lr_counts=dict(Counter(str(r['accepted_actor_lr']) for r in update_rows)),
            recent_post_kl=stats([r['post_update']['mean_approx_kl'] for r in recent]),
            recent_clip=stats([r['post_update']['clip_fraction'] for r in recent]),
            recent_value_loss=stats([r['before']['value_loss'] for r in recent]),
            gae_control_efold=-16/math.log(.95 * .99999**16),
            normal_dual_env_window_controls=32*16))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.job), allow_nan=False))
