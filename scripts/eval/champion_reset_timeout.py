"""Certify the observed fruit-reset watchdog interruption, not arbitrary timeouts.

The full original journal is pinned. Its four exact shutdown disconnects are
separately checked before auditing the protocol projection; no original is edited.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from audit_champion_public_trace import audit_lines, require

JOURNAL_SHA = '2468f84dbff45be493a32bc33fd7a6f8090f9dfb77a7488752d90f0eaaa932e9'
PREFIX_SHA = 'a8540b71a5326049145b4640bc824c4187f09cdbbdb348083220e11319bc6203'
OFFICIAL_SHA = '5885dceea636ce206accbb26d153b90c3004e3d5e959eced27529487c18ae870'


def terminal_protocol(raw, *, previous, current, stopped_at, sampled_at):
    lines = raw.splitlines(keepends=True)
    require(all(x.endswith(b'\n') for x in lines), 'Truncated terminal journal')
    rows = [json.loads(x) for x in lines]
    before = [line for line, row in zip(lines, rows) if row['time'] < current['started']]
    after = [row for row in rows if row['time'] >= current['started']]
    require(len(after) == 8, 'Unexpected events after next batch began')
    disconnected = []
    for error, closed in zip(after[::2], after[1::2]):
        require(error.get('kind') == 'connection_error'
                and error.get('error') == 'ConnectionClosedError(None, None, None)'
                and closed.get('kind') == 'closed' and error['connection'] == closed['connection']
                and sampled_at < error['time'] <= closed['time'] <= stopped_at
                and stopped_at - error['time'] < 120, 'Not an exact terminal disconnect pair')
        disconnected.append(error['connection'])
    require(len(set(disconnected)) == 4, 'Wrong terminal connection set')
    prefix = audit_lines(before, allowed_tasks={39, 62})
    projection = audit_lines([line for line, row in zip(lines, rows) if row['kind'] != 'connection_error'],
                             allowed_tasks={39, 62}, require_closed=True)
    sessions = [s for s in projection['session_summaries'] if s['task'] == 39]
    require(len(sessions) == 4 and {s['connection'] for s in sessions} == set(disconnected)
            and all(s['epoch'] == 1 and s['deliveries'] == 626 for s in sessions)
            and previous['actual_applied_controls'] == [12519] * 4,
            'The next batch received a new session or actions')
    require(not any(r['kind'] not in ('connection_error', 'closed') for r in after),
            'Post-start policy/reset activity')
    return dict(original_sha256=hashlib.sha256(raw).hexdigest(), original_bytes=len(raw),
                before_next_batch_prefix=prefix, terminal_protocol_projection=projection,
                separately_verified_disconnects=after[::2], next_batch_policy_deliveries=0,
                original_journal_modified=False)


def certify(job, observation):
    import continue_champion_final as c
    job, observation = Path(job).resolve(), Path(observation).resolve()
    state = c.read(job / 'status.json'); obs = c.read(observation)
    require(state == obs['terminal']['status'] and state['status'] == 'needs_diagnosis'
            and set(state['failed_gpus']) == {'2'} and (job / 'STOP').exists(), 'Wrong stopped incident')
    _, original, _ = c.root_registration(job)
    wp = job / 'workers/gpu_2'; w = c.read(wp / 'status.json')
    require(w == state['failed_gpus']['2']
            and w['error'] == "TimeoutError('No model progress for 30 minutes')"
            and w['part'] == 'spraying_fruit_trees-n4-i0', 'Not the diagnosed reset watchdog')
    c.require_exited([state['pid'], *state['worker_pids'].values(), w['sim_pid'], w['model_pid']])
    part = job / 'parts' / w['part']; ps = c.read(part / 'status.json')
    previous, current = c.read(part / 'attempts/00.json'), c.read(part / 'attempts/04.json')
    require(previous['status'] == 'completed' and previous['instance_ids'] == [301, 302, 303, 304]
            and current['status'] == 'running' and current['instance_ids'] == [305, 306, 307, 308]
            and current['indices'] == [4, 5, 6, 7] and current['task'] == 'spraying_fruit_trees'
            and ps['pid'] == w['sim_pid'] and ps['updated'] < current['started'], 'Physical boundary changed')
    require(sorted(p.name for p in (part / 'attempts').glob('*.json')) == ['00.json', '04.json'],
            'Additional physical attempt')
    c.shared.require_no_uncommitted_metrics(part)
    require({r['instance_id'] for _, r in c.completed_metrics(part)} == set(range(301, 305)),
            'Unexpected scored next-batch case')
    sample = obs['last_pre_guard_sample']; sampled = datetime.fromisoformat(sample['observed_at']).timestamp()
    require(sample['pid'] == w['sim_pid'] and sample['nonblocking'] is True
            and sample['source_or_policy_modified'] is False
            and current['started'] < sampled < w['finished'] and w['finished'] - sampled < 120,
            'Missing recent unmodified reset stack')
    frames = sample['main_thread']['frames']
    require(sample['main_thread']['pid'] == w['sim_pid'] and sample['main_thread']['active'] is True
            and {'remove_particle_by_name', 'remove_all_group_particles', 'update_handles',
                 '_load_instance_state', 'load_batch', 'load_fn', 'evaluate_instances_batched'}
                <= {f['name'] for f in frames}, 'Not the sampled particle cleanup path')
    official = next(Path(f['filename']) for f in frames if f['name'] == 'evaluate_instances_batched')
    require(c.sha256(official) == OFFICIAL_SHA, 'Official load-before-step sequence changed')
    model = Path(w['model_path']); final = c.read(model / 'result.json')
    c.validate_weights(final, original, 2)
    trace = model / 'events.jsonl'; raw = trace.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == JOURNAL_SHA, 'Not the diagnosed immutable terminal trace')
    proof = terminal_protocol(raw, previous=previous, current=current, stopped_at=w['finished'], sampled_at=sampled)
    require(proof['before_next_batch_prefix']['prefix_sha256'] == PREFIX_SHA, 'Prior inspected prefix changed')
    log = wp / (w['part'] + '.log')
    # The completed previous physical receipt postdates these UI callbacks;
    # the stale coarse status does not. Do not claim later next-batch physics.
    ui = c.shared.nonfatal_property_ui(log.read_text(),
        dict(status='running', updated=previous['finished'], control_steps=previous['actual_applied_controls']), previous)
    require(ui is not None and ui['last_exception_time'] < previous['finished'],
            'Additional simulator failure needs diagnosis')
    evidence = [c.binding(p) for p in (observation, job/'manifest.json', job/'status.json', job/'STOP',
        wp/'status.json', part/'status.json', part/'attempts/00.json', part/'attempts/04.json',
        model/'result.json', model/'status.json', trace, log, official)]
    return dict(schema='champion_precontrol_reset_certificate_v1', job=str(job), gpu=2, part=w['part'],
        original_attempt=c.binding(part/'attempts/04.json'), observation=c.binding(observation),
        evidence=evidence, proof=proof, reviewed_previous_batch_ui=ui,
        reason='Coordinator model-only watchdog interrupted official particle cleanup before next-batch policy reset/delivery',
        rerun_allowed=True, best_of_n=False, policy_failure_claimed=False)


def verify_certificate(path):
    import continue_champion_final as c
    row = c.read(path); c.check_binding(row['observation'])
    require(row == certify(row['job'], row['observation']['path']), 'Reset certificate no longer matches originals')
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True); p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    require(not a.output.exists() and a.output.parent.is_dir(), 'Fresh certificate output required')
    result = certify(a.job, a.observation)
    with a.output.open('x') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(a.output), next_batch_policy_deliveries=0, cases=4)))
