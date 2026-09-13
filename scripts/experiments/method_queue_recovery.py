"""One explicit recovery of six zero-formal-update 2026-09-13 jobs.

This is not an automatic retry mechanism. Failed v1 receipts stay immutable;
each declared v2 retains its original recipe/parent and a bounded 5+500 budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path('/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913')
DECLARED = {
    'fm_beta_stratified_v2': ('fm_beta_stratified_v1', 'fm', '191a5393dfd57b8e6749bc93e1487328a6d5fed57fad26a76903f8203adde866'),
    'fm_exec_weight2_v2': ('fm_exec_weight2_v1', 'fm', 'a6b18a6c325b69b2609720ac95bd0b1ef6624eaf98ed4585faa2a80803cb4231'),
    'fm_action_control_v2': ('fm_action_control_v1', 'action', 'e16d8670fbf0f70193b670a7fb3d736552ab8d48e06f28b3ca16a6a87cd242eb'),
    'joint_a4_fulltrain_v2': ('joint_a4_fulltrain_v1', 'action', '2d8595a640def732d4891ed0497654fdcecd58e51be8d31866f80b9229bb6c03'),
    'ki_a4_fulltrain_v2': ('ki_a4_fulltrain_v1', 'action', 'b9a7eec4cabe36688ff7badc6620bde93b16f3c3ba69cb92fe9823f3c4a9ff06'),
    'ar_a4_marker_fulltrain_v2': ('ar_a4_marker_fulltrain_v1', 'action', 'c14c484b88c40e99ffb68bc570b656dfa1bcc5b6a50ff02decc75506a4212c21'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def assert_exited(pid):
    if type(pid) is not int or pid < 1:
        raise RuntimeError('Invalid recorded process identity')
    path = Path('/proc') / str(pid) / 'stat'
    try:
        state = path.read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return
    if state not in ('Z', 'X'):
        raise RuntimeError('Recorded old PID is live or reused; do not recover it')


def recovery_identity(previous, output, kind):
    previous, output = Path(previous), Path(output)
    allowed = DECLARED.get(output.name)
    if (output.parent != BASE or previous.parent != BASE or allowed is None
            or (previous.name, kind) != allowed[:2]):
        raise ValueError('Only the six predeclared v1-to-v2 recoveries are permitted')
    if previous.is_symlink() or output.is_symlink():
        raise RuntimeError('Recovery roots must not redirect to another run')
    method_path, status_path, launch_path = [previous / name for name in
        ('method_spec.json', 'status.json', 'launch.json')]
    if digest(method_path) != allowed[2]:
        raise RuntimeError('Failed original recipe identity changed')
    status, launch = read(status_path), read(launch_path)
    if status.get('state') != 'failed' or status.get('automatic_retry') is not False:
        raise RuntimeError('Recovery requires an already failed non-retrying v1')
    assert_exited(launch['supervisor_pid'])
    if (previous / 'formal').exists():
        raise RuntimeError('A formal run already exists; this recovery is zero-formal-update only')
    smoke = previous / 'smoke'
    if kind == 'fm' and previous.name == 'fm_beta_stratified_v1':
        receipt = read(smoke / 'coordination_run_receipt.json')
        assert_exited(receipt['pid'])
        if (receipt['state'] != 'failed' or (smoke / 'train.log').stat().st_size != 0
                or (previous / 'smoke.log').read_text().count(
                    'AssertionError: Fine-tuning assumes at least one GPU is available!') != 4):
            raise RuntimeError('Beta v1 is not the audited four-rank zero-update CUDA failure')
        allowed_smoke = {'train.log', 'coordination_run_receipt.json', '.hydra/config.yaml',
                         '.hydra/hydra.yaml', '.hydra/overrides.yaml'}
        if {str(p.relative_to(smoke)) for p in smoke.rglob('*') if p.is_file()} != allowed_smoke:
            raise RuntimeError('Unexpected smoke artifacts; zero updates no longer established')
    elif smoke.exists():
        raise RuntimeError('Downstream v1 must not have entered smoke training')
    return dict(previous_run=str(previous), previous_method_sha256=allowed[2],
                previous_status_sha256=digest(status_path), previous_launch_sha256=digest(launch_path),
                previous_supervisor_pid=launch['supervisor_pid'], kind=kind,
                zero_formal_updates_verified=True, zero_smoke_updates_verified=True,
                preserves_v1=True, automatic_retry=False)


def validate_recovery(spec, kind):
    receipt = spec.get('recovery_from')
    if receipt is None:
        if Path(spec['output']).name in DECLARED:
            raise RuntimeError('A declared v2 needs an explicit failed-v1 recovery receipt')
        return
    actual = recovery_identity(receipt['previous_run'], spec['output'], kind)
    if receipt != actual:
        raise RuntimeError('Recovery receipt identity changed')
    old = read(Path(receipt['previous_run']) / 'method_spec.json')
    fields = ('parent_sha256', 'parent_path', 'initialization') + (
        ('trial', 'settings', 'max_updates', 'smoke_updates', 'seed') if kind == 'fm' else
        ('route', 'recipe', 'conditioning'))
    if any(spec[key] != old[key] for key in fields):
        raise RuntimeError('Recovery must retain the original method, parent and finite budget')
    if kind == 'action' and (bool(spec.get('marker_rows')) != bool(old.get('marker_rows'))
                            or spec.get('marker_recipe') != old.get('marker_recipe')):
        raise RuntimeError('Recovery cannot silently change marker adaptation')
