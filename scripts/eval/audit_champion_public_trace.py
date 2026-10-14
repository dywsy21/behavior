"""Audit recorded champion protocol state without importing a model or simulator.

This proves only invariants observable in the journal: connection/reset/task
identity, displayed stage recurrence and inpainting lengths, and delivery clocks.
It does not prove raw image/action/RNG identity, physical execution, or success.
An active trace is audited only through a stable, newline-complete byte prefix.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def audit_lines(lines, *, allowed_tasks=None, require_closed=False):
    connections, episodes = {}, []
    digest = hashlib.sha256(); size = count = deliveries = 0
    allowed = None if allowed_tasks is None else set(allowed_tasks)

    def finish(connection, previous):
        session = connection['session']
        if session is None:
            require(previous is None, 'Foreign previous session on reset/close')
            return
        require(isinstance(previous, dict), 'Missing prior session finish')
        expected = dict(episode=session['episode'], task=session['task'],
            clock_kind='policy_delivery', delivered_controls=session['deliveries'] * 20,
            physically_confirmed_controls=None, predictions=session['deliveries'],
            unused_offered_controls=0, optimizer_updates=0)
        require(all(previous.get(k) == v for k, v in expected.items()), 'Previous delivery session differs')
        session['closed'] = True; connection['session'] = None

    for raw in lines:
        require(isinstance(raw, bytes) and raw.endswith(b'\n'), 'Need complete original journal lines')
        digest.update(raw); size += len(raw); count += 1
        row = json.loads(raw); kind = row['kind']; cid = row['connection']
        require(isinstance(cid, str) and bool(cid), 'Missing connection identity')
        if kind == 'connected':
            require(cid not in connections, 'Reused connection identity')
            connections[cid] = dict(epoch=0, session=None, closed=False)
            continue
        require(cid in connections and not connections[cid]['closed'], 'Foreign/closed connection event')
        connection = connections[cid]
        if kind in ('reset', 'closed'):
            finish(connection, row.get('previous'))
            connection['epoch'] += 1
            connection['closed'] = kind == 'closed'
            continue
        require(kind == 'delivered', 'Connection error or unknown event requires diagnosis')
        d = row.get('generated_display')
        require(isinstance(d, dict), 'Chunk20 must have a fresh prediction display')
        task = row['task']
        require(integer(task) and 0 <= task < 100 and (allowed is None or task in allowed), 'Foreign task')
        require(connection['epoch'] > 0, 'Registered client must reset before first inference')
        episode = f"{cid}/reset-{connection['epoch']}/task-{task}"
        require(row.get('episode') == d.get('episode') == episode and d.get('task') == task,
                'Crossed task/episode identity')
        if connection['session'] is None:
            connection['session'] = dict(episode=episode, connection=cid, epoch=connection['epoch'],
                task=task, deliveries=0, stage=0, votes=[], stages=None, legacy=None, closed=False)
            episodes.append(connection['session'])
        s = connection['session']
        require(s['episode'] == episode and s['task'] == task, 'Task changed without reset')
        require(row.get('clock_kind') == d.get('clock_kind') == 'policy_delivery'
                and row.get('physically_confirmed_controls') is None
                and d.get('physically_confirmed_controls') is None, 'Fabricated physical clock')
        require(all(integer(v) for v in (row.get('sent_controls'), row.get('prediction'),
                row.get('delivered_controls'), d.get('delivered_controls')))
                and row['sent_controls'] == 20 and row['prediction'] == s['deliveries'] + 1
                and row['delivered_controls'] == (s['deliveries'] + 1) * 20
                and d['delivered_controls'] == s['deliveries'] * 20, 'Wrong per-session delivery clock')
        require(isinstance(row.get('action_sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', row['action_sha256']) is not None, 'Malformed action digest')
        probabilities = d.get('stage_probabilities')
        require(isinstance(probabilities, list) and 5 <= len(probabilities) <= 15
                and all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                and abs(sum(probabilities) - 1) < 1e-6, 'Invalid legal-stage probabilities')
        require(type(d.get('legacy_corrections')) is bool and type(d.get('compressed')) is bool,
                'Ambiguous correction/compression mode')
        if s['stages'] is None:
            s['stages'], s['legacy'] = len(probabilities), d['legacy_corrections']
        require(s['stages'] == len(probabilities) and s['legacy'] == d['legacy_corrections'],
                'Stage layout or legacy mode changed within a session')
        for k in ('input_stage', 'corrected_stage', 'next_stage', 'predicted_stage'):
            require(integer(d.get(k)) and 0 <= d[k] < s['stages'], 'Invalid displayed stage: ' + k)
        predicted = probabilities.index(max(probabilities))
        require(d['input_stage'] == s['stage'] and d['predicted_stage'] == predicted,
                'Stage history or prediction is not local to this session')
        require(type(d.get('stage_probability')) in (int, float)
                and abs(d['stage_probability'] - probabilities[predicted]) < 1e-6,
                'Reported predicted probability differs')
        require(d.get('inpainting_input_steps') == (0 if s['deliveries'] == 0 else 4)
                and d.get('inpainting_next_start') == (26 if d['compressed'] else 20),
                'Wrong initial or continuing inpainting boundary')
        if d['corrected_stage'] != s['stage']:
            require(s['legacy'], 'Stage correction without registered legacy mode')
            s['stage'] = d['corrected_stage']; s['votes'] = []
        s['votes'] = (s['votes'] + [predicted])[-3:]
        if len(s['votes']) == 3 and s['stage'] + 1 <= s['stages'] - 1:
            if s['votes'].count(s['stage'] + 1) >= 2 or s['votes'].count(s['stage'] + 2) == 3:
                s['stage'] += 1; s['votes'] = []
            elif s['stage'] > 0 and s['votes'].count(s['stage'] - 1) == 3:
                s['stage'] -= 1; s['votes'] = []
        require(d.get('next_stage') == s['stage'] and d.get('stage_votes') == s['votes'],
                'Displayed vote recurrence differs from per-session history')
        s['deliveries'] += 1; deliveries += 1
    require(count > 0 and bool(connections), 'Empty journal is not an isolation audit')
    if require_closed:
        require(all(c['closed'] for c in connections.values()), 'Live connection; no terminal audit')
    return dict(schema='champion_protocol_prefix_audit_v1', prefix_bytes=size,
        prefix_sha256=digest.hexdigest(), lines=count, connections=len(connections),
        episodes=len(episodes), closed_episodes=sum(s['closed'] for s in episodes),
        active_connections=sum(not c['closed'] for c in connections.values()), deliveries=deliveries,
        delivered_controls=deliveries * 20, physically_confirmed_controls=None,
        task_ids=sorted({s['task'] for s in episodes}), require_closed=require_closed,
        session_summaries=episodes, original_outputs_modified=False,
        scope='Recorded identity, displayed stage/vote/inpainting recurrence and delivery clock only',
        does_not_prove=['raw observation/action/RNG values', 'physical execution', 'success',
                        'events beyond this exact prefix'])


def audit_file(path, *, allowed_tasks=None, require_closed=False):
    path = Path(path).resolve(); captured = path.stat().st_size
    def lines():
        with path.open('rb') as stream:
            remaining = captured
            while remaining:
                raw = stream.readline(remaining); remaining -= len(raw)
                if not raw or not raw.endswith(b'\n'):
                    break
                yield raw
    result = audit_lines(lines(), allowed_tasks=allowed_tasks, require_closed=require_closed)
    if require_closed:
        require(result['prefix_bytes'] == captured, 'Terminal trace has an incomplete final line')
    digest = hashlib.sha256(); remaining = result['prefix_bytes']
    with path.open('rb') as stream:
        while remaining:
            raw = stream.read(min(remaining, 8 << 20))
            require(bool(raw), 'Journal truncated during audit')
            digest.update(raw); remaining -= len(raw)
    require(digest.hexdigest() == result['prefix_sha256'], 'Journal prefix changed during audit')
    result.update(trace=str(path), captured_bytes=captured, ignored_incomplete_tail_bytes=captured-result['prefix_bytes'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allowed-tasks', type=int, nargs='+', required=True)
    parser.add_argument('--require-closed', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or args.output.resolve() == args.trace.resolve():
        raise ValueError('New audit output required; never overwrite original evidence')
    result = audit_file(args.trace, allowed_tasks=args.allowed_tasks, require_closed=args.require_closed)
    result['observed_at'] = time.time()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'session_summaries'}))
