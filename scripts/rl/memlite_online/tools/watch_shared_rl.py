"""Read-only SSH observer and private W&B metric mirror; never edits the trainer.

The key is read from stdin or a no-echo prompt, not argv or a credential file.
Run in a separate, pinned local SDK environment when the GPU host has bad egress.
"""
import argparse
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


REMOTE_READER = r'''
import json,pathlib,sys
job=pathlib.Path(sys.argv[1]); offsets=json.loads(sys.argv[2])
def read(name): return json.loads((job/name).read_text())
def lines(name):
    p=job/name; offset=offsets.get(name,0); rows=[]
    if not p.exists(): return rows
    if p.stat().st_size<offset: raise ValueError('Append-only history shrank')
    with p.open('rb') as f:
        f.seek(offset)
        for _ in range(10000):
            start=f.tell(); line=f.readline()
            if not line or not line.endswith(b'\n'):
                f.seek(start); break
            row=json.loads(line)
            if name.endswith('updates.jsonl'):
                row={k:row[k] for k in ['update','actor_updates','actor_updated',
                  'global_experiences','accepted_actor_lr','actor_grad_norm',
                  'critic_grad_norm','actor_max_delta','before','post_update']}
            rows.append(row)
        offsets[name]=f.tell()
    return rows
s=read('training_summary.json'); s.pop('latest_rank0_update',None)
manifest=read('manifest.json'); state=read('status.json')
print(json.dumps(dict(summary=s,state={k:state.get(k) for k in
 ['status','started','deadline','source','wandb_url','updated']},
 training=manifest['training'],
 history=lines('monitor_history.jsonl'),
 updates=lines('policies/gpu_0/checkpoints/updates.jsonl'),
 effect_events=lines('effect_events.jsonl'),offsets=offsets,
 recovery_candidates=len(list((job/'recovery').glob('gpu_*/*.zip'))),
 aborted=(job/'ABORT').exists())))
'''


def numeric_fields(row, names, prefix):
    return {prefix + key: row[key] for key in names
            if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])}


def safe_api_errors(body, key):
    # Only bounded server error messages, never headers/request/settings/raw body.
    messages = [str(e.get('message', 'unspecified')) for e in body.get('errors', [])]
    return [re.sub(r'wandb_v1_[A-Za-z0-9_-]+', '[REDACTED]', message.replace(key, '[REDACTED]'))[:240]
            for message in messages[:3]]


def train_metrics(row):
    result = numeric_fields(row, ('controls', 'completed_episodes', 'task_coverage',
        'successes', 'macro_q_covered', 'macro_sr_covered', 'episodes_with_q_increase',
        'controls_per_wall_second', 'completed_task_coverage'), 'train/')
    result['train/control_steps'] = row['controls']
    if row.get('versions'):
        result['train/min_policy_version'] = min(row['versions'])
        result['train/max_policy_version'] = max(row['versions'])
    for task, values in row.get('per_task', {}).items():
        result.update(numeric_fields(values, ('completed', 'successes', 'mean_q', 'sr'),
                                     'task/' + task + '/'))
    return result


def ppo_metrics(row):
    result = numeric_fields(row, ('update', 'actor_updates', 'actor_updated',
        'global_experiences', 'accepted_actor_lr', 'actor_grad_norm', 'critic_grad_norm',
        'actor_max_delta'), 'ppo/')
    for phase in ('before', 'post_update'):
        result.update(numeric_fields(row[phase], ('mean_approx_kl', 'clip_fraction',
            'ratio_mean', 'policy_loss', 'value_loss'), 'ppo/' + phase + '/'))
    return result


def findings(snapshot, now):
    s, state = snapshot['summary'], snapshot['state']
    events = []
    if snapshot['aborted'] or state['status'] == 'failed':
        events.append(('training_failed', 'Training stopped with failure; inspect original run.'))
    if state['status'] not in ('completed', 'failed') and now - s['updated'] > 300:
        events.append(('monitor_stale', 'Training monitor has not refreshed for over five minutes.'))
    if state['status'] == 'completed':
        events.append(('training_finished', 'Training supervisor finished; checkpoint and outcome audit required.'))
    if s['completed_episodes']:
        events.append(('first_complete_train_episode',
            'Complete TRAIN episodes are available; changing policy/task mix is not independent evaluation.'))
    if s['episodes_with_q_increase']:
        events.append(('first_train_q_increase',
            'Official Q increased in at least one TRAIN episode; not proof of improvement over SFT.'))
    if s['successes']:
        events.append(('first_train_success', 'Official success observed in TRAIN, not held-out SR.'))
    if now - state['started'] >= 4 * 3600 and not s['episodes_with_q_increase']:
        events.append(('four_hours_no_q_increase',
            'Four hours since launch with no observed official Q increase; inspect behavior and reward signal.'))
    return events


def fetch(job, offsets):
    command = 'python3 -c ' + shlex.quote(REMOTE_READER) + ' ' + shlex.quote(job) + ' ' + shlex.quote(json.dumps(offsets))
    result = subprocess.run(['ssh', '-p', '10383', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=12', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        'root@42.192.34.154', command], capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError('Read-only SSH snapshot failed')
    return json.loads(result.stdout)


def atomic_json(path, value):
    pending = path.with_suffix('.tmp')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--expected-user', required=True)
    parser.add_argument('--project', default='behavior-memlite-rl')
    parser.add_argument('--inspect-account', action='store_true')
    args = parser.parse_args()
    # No credentials in argv, source, tracebacks, shared .netrc or training env.
    key = getpass.getpass('W&B key (not saved): ') if sys.stdin.isatty() else sys.stdin.readline().strip()
    if not key:
        raise ValueError('An explicit account key is required')
    import requests
    session = requests.Session()
    session.auth = ('api', key)

    def gql(query, variables=None):
        response = session.post('https://api.wandb.ai/graphql',
            json={'query': query, 'variables': variables or {}}, timeout=25)
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400 or body.get('errors'):
            print(json.dumps(dict(api_failure=True, http_status=response.status_code,
                operation='mutation' if query.startswith('mutation') else 'query',
                messages=safe_api_errors(body, key))), flush=True)
        response.raise_for_status()
        if body.get('errors'):
            raise RuntimeError('W&B GraphQL request rejected; no error body logged')
        return body['data']

    viewer = gql('query { viewer { username entity } }')['viewer']
    if viewer['username'] != args.expected_user or viewer['entity'] != args.entity:
        raise ValueError('Verified account identity differs from requested destination')
    variables = {'entity': args.entity, 'project': args.project}
    project_query = 'query($entity:String!,$project:String!){project(name:$project,entityName:$entity){name access}}'
    project = gql(project_query, variables)['project']
    if args.inspect_account:
        schema = gql('query { __type(name:"UpsertModelInput") {inputFields {name type {kind name ofType {name kind}}}}}')
        print(json.dumps(dict(viewer=viewer, project=project, input_schema=schema)))
        return
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (args.output/'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output/'identity.json').exists():
        raise ValueError('Existing observer run: do not duplicate history; use a fresh explicit attempt')
    if project is None:
        gql('mutation($entity:String!,$project:String!){upsertModel(input:{name:$project,entityName:$entity,access:"PRIVATE"}){model{name access}}}', variables)
        project = gql(project_query, variables)['project']
    if project is None or project['access'] != 'PRIVATE':
        raise ValueError('Destination must be private; not changing an existing public project')
    first = fetch(args.job, {})
    import wandb
    run = wandb.init(entity=args.entity, project=args.project,
        name=Path(args.job).name + '-owner-mirror', dir=str(args.output),
        job_type='read-only-monitor', tags=['TRAIN-only', 'metric-mirror'],
        config=dict(original_job=args.job, source_commit=first['state']['source'],
            original_run=first['state']['wandb_url'], training=first['training'],
            metric_scope='on_policy_TRAIN_not_independent_evaluation',
            policy_or_environment_changed=False),
        settings=wandb.Settings(api_key=key, base_url='https://api.wandb.ai',
            mode='online', console='off', disable_git=True, save_code=False,
            x_disable_stats=True, x_disable_meta=True, init_timeout=45, silent=True))
    run.define_metric('train/control_steps')
    run.define_metric('train/*', step_metric='train/control_steps')
    run.define_metric('task/*', step_metric='train/control_steps')
    run.define_metric('ppo/update')
    run.define_metric('ppo/*', step_metric='ppo/update')
    identity = dict(pid=os.getpid(), url=run.url, run_id=run.id, entity=args.entity,
        project=args.project, started=time.time(), remote_job=args.job,
        auth_persisted=False, training_process_restarted=False)
    atomic_json(args.output/'identity.json', identity)
    print(json.dumps(identity), flush=True)
    offsets, seen = {}, set()
    try:
        snapshot = first
        while True:
            for row in snapshot['history']:
                run.log(train_metrics(row))
            for row in snapshot['updates']:
                run.log(ppo_metrics(row))
            run.log(train_metrics(snapshot['summary']))
            offsets = snapshot['offsets']
            now = time.time()
            for name, message in findings(snapshot, now):
                if name in seen:
                    continue
                event = dict(time=now, kind=name, message=message,
                    controls=snapshot['summary']['controls'],
                    completed=snapshot['summary']['completed_episodes'],
                    task_coverage=snapshot['summary']['task_coverage'])
                with (args.output/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                run.summary['watch/' + name] = event
                seen.add(name)
                print(json.dumps(event), flush=True)
            status = dict(updated=now, remote_state=snapshot['state'],
                summary=snapshot['summary'], recovery_candidates=snapshot['recovery_candidates'],
                offsets=offsets, findings=sorted(seen), url=run.url)
            atomic_json(args.output/'status.json', status)
            with (args.output/'snapshots.jsonl').open('a') as stream:
                stream.write(json.dumps(status) + '\n')
            if snapshot['state']['status'] in ('completed', 'failed') or now > snapshot['state']['deadline'] + 600:
                break
            while True:
                time.sleep(60)
                try:
                    snapshot = fetch(args.job, offsets)
                    break
                except Exception as error:
                    # No raw request/SDK exception, argv or environment in logs.
                    atomic_json(args.output/'observer_error.json',
                                dict(time=time.time(), error_type=type(error).__name__))
                    if time.time() > first['state']['deadline'] + 600:
                        raise RuntimeError('Observer deadline reached while SSH was unavailable') from None
    finally:
        run.finish()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'observer_failed': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
