"""One H72 diagnostic with the original H71 native/physical/resource gates."""
import argparse
import hashlib
import json
from pathlib import Path

import launch_h69 as gate
from probe_observation_clock import audit_digest, validate_installed

COMMAND, MANIFEST, IDENTITY = gate.command, gate.validate_manifest, gate.identity


def identity(base):
    validate_installed()
    return IDENTITY(base)


def command(base):
    result = COMMAND(base)
    result[1] = str(base.REPO/'scripts/semantic_robot/probe_observation_clock.py')
    return result


def validate_manifest(manifest, base, normal_digest_commit, normal_digest):
    MANIFEST(manifest, base, normal_digest_commit, audit_digest(normal_digest))


def validate_result(result, normal_digest):
    # Completed DIAGNOSIS, not a relaxed production gate. Preserve gate_ok,
    # failures and every original motion threshold in the raw worker result.
    exact = {'status': 'complete', 'task': 0, 'prefix_controls': 0,
             'diagnostic_replay_controls': 0, 'native_profile': 'a100_full_v1',
             'implementation_digest': audit_digest(normal_digest), 'model_calls': 0,
             'full_task_success_rate_claim': False}
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in exact.items()):
        raise ValueError('Exact H72 diagnostic identity failed')
    if not 12 <= len(result.get('decisions', [])) <= 24 or not 0 < result.get('controls', 0) <= 1536:
        raise ValueError('H72 did not reach the original base-motion diagnostic')
    action = result['decisions'][11]
    if (action.get('decision') != 11 or action.get('action') !=
            {'part':'base', 'move':'back', 'scale':'fine', 'frame':'base'} or
            action.get('accepted_before_motion') is not True or
            not isinstance(action.get('control_start'), int) or
            not isinstance(action.get('control_end'), int) or
            not action['control_start'] < action['control_end'] < result['controls'] or
            action.get('feedback', {}).get('control_ticks') != action['control_end']-action['control_start']):
        raise ValueError('H72 base motion was not actually executed')
    folder = gate.ROOT/'gate'
    audit = json.loads((folder/'clock_audit.json').read_text())
    if (audit.get('purpose') != 'observation_clock_diagnostic_not_policy_gate' or
            audit.get('extra_controls') != 0 or audit.get('extra_renders') != 0 or
            audit.get('instrumented_timing_not_identical_to_H71') is not True or
            audit.get('truth_sent_to_actor_or_odometer') is not False):
        raise ValueError('Private observation audit boundary failed')
    raw = (folder/'PRIVILEGED_clock_pose.jsonl').read_bytes()
    rows = [json.loads(x) for x in raw.splitlines()]
    if (not 1 <= len(rows) <= 8192 or audit.get('events') != len(rows) or
            audit.get('journal_sha256') != hashlib.sha256(raw).hexdigest() or
            rows[-1]['phase'] != 'final_after_safe_hold' or
            not any(r['phase'] == 'capture_after_render' for r in rows)):
        raise ValueError('Incomplete H72 observation journal')
    source = folder/'decision_011'
    chain = json.loads((source/'action_motion.json').read_text())
    if chain.get('control_start') != action['control_start'] or chain.get('control_end') != action['control_end']:
        raise ValueError('H72 base chain does not bind executed controls')
    paths = [source/'depth_receipt.json']
    cursor = action['control_start']
    for segment in chain['segments']:
        end = segment['control_end']
        if segment['control_start'] != cursor or not 0 < end-cursor <= 6:
            raise ValueError('Missing H72 base substep')
        paths.append(source/'motion_substeps'/f'control_{end:06d}'/'depth_receipt.json')
        cursor = end
    if len(paths) < 2 or cursor != action['control_end']: raise ValueError('Missing H72 base captures')
    for path in paths:
        camera = json.loads(path.read_text())['head']
        events = [r for r in rows if r['snapshot_id'] == camera['snapshot_id']]
        expected = [('capture_before', None), *[('capture_after_render', i) for i in range(1,5)], ('capture_return', None)]
        if [(r['phase'],r['render_index']) for r in events] != expected:
            raise ValueError('Incomplete H72 four-render capture sequence')
        if any(events[-1]['head'][key] != camera[key] for key in ('rgb_sha256','depth_sha256')):
            raise ValueError('H72 capture does not match runner RGB-D')


def configure():
    gate.ROOT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h72_observation_clock_v1')
    gate.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h72_observation_clock_v1')
    gate.WALL_SECONDS = 1200
    gate.command, gate.validate_manifest, gate.validate_result = command, validate_manifest, validate_result
    gate.identity = identity
    base = gate.configure()
    base.ENTRYPOINT = Path(__file__).resolve()
    return base


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch', action='store_true'); modes.add_argument('--supervise', action='store_true')
    args = parser.parse_args(); base = configure()
    (gate.launch if args.launch else gate.supervise)(base)
