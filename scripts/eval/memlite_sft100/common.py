"""Pinned paths and pure bookkeeping for the one-pass SFT100 evaluation."""
import hashlib
import csv
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
ROOT = Path('/run/ti/rl_memlite_stage1_20261006')
SIM_ROOT = Path('/run/ti/behavior_stage3_20260930')
G05 = ROOT / 'code/g05_sft_6af1ab9'
G05_COMMIT = '6af1ab983bb6c666f723c0367c3d6d4cc5ed4678'
OFFICIAL_COMMIT = 'a8247a8cc1633fe1ca0cc66aa07243d46c64f155'
OFFICIAL = ROOT / 'code/official_behavior_a8247a8/OmniGibson'
DATA = ROOT / 'assets_readiness/sim_data'
MODEL_PYTHON = Path('/home/tione/notebook/baselines/GalaxeaVLA/.venv/bin/python')
SIM_PYTHON = SIM_ROOT / 'envs/sim/bin/python'
CHECKPOINTS = {
    'high': ('step_00048045_save_0027.pt', '3683f719ebb033cacf77afdc0330b25f43a3c3e5b704b0bda9623608f092c3d1'),
    'low': ('step_00098414_save_0021.pt', 'd4d76099780281fb21ece6ab2dd6cd4deb643a9169c6c4fb2c130e1cce4cb470'),
}


def bootstrap():
    tools = REPO / 'scripts/rl/memlite_online/tools'
    sys.path.insert(0, str(tools))
    from bootstrap import bootstrap as pinned_bootstrap
    pinned_bootstrap()
    sys.path.insert(0, str(SOURCE))
    sys.path.append(str(SIM_ROOT / 'tools'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def expected_cases(tasks):
    if len(tasks) != 100 or len(set(tasks)) != 100:
        raise ValueError('Exactly 100 unique official tasks required')
    return {(t, instance, 0) for t in tasks for instance in range(301, 311)}


def load_official_task_names(path):
    """Protocol IDs map to canonical names, never the legacy prose instruction."""
    with Path(path).open() as stream:
        rows=list(csv.DictReader(stream))
    mapping={int(row['Task ID']):row['Task'] for row in rows}
    if len(rows)!=100 or set(mapping)!=set(range(100)) or len(set(mapping.values()))!=100:
        raise ValueError('Official100 task ID/name mapping is incomplete or ambiguous')
    return mapping


def aggregate(tasks, records):
    """No silent missing-case removal, duplicate selection or peak-Q reporting."""
    import math
    expected = expected_cases(tasks)
    seen = {}
    for record in records:
        key = (record['task'], record['instance_id'], record['rollout_id'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or repeated evaluation case: ' + str(key))
        q = record['q_score']['final']
        if isinstance(q, bool) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('Invalid official final Q')
        if type(record['success']) is not bool:
            raise ValueError('Official success must be Boolean')
        seen[key] = record
    per_task = {}
    for task in tasks:
        rows = [r for k, r in seen.items() if k[0] == task]
        q = sum(r['q_score']['final'] for r in rows)
        sr = sum(r['success'] for r in rows)
        per_task[task] = dict(completed=len(rows), expected=10, q_score_sum=q,
                              success_count=sr, q_score_missing_as_zero=q / 10,
                              sr_missing_as_zero=sr / 10,
                              completed_only_q=q / len(rows) if rows else None)
    n = len(seen)
    return dict(status='complete' if n == 1000 else 'incomplete', expected=1000, completed=n,
                missing=1000 - n, official_q_score=sum(r['q_score']['final'] for r in seen.values()) / 1000,
                official_sr=sum(r['success'] for r in seen.values()) / 1000,
                missing_case_convention='zero; do not call partial values full-model performance',
                completed_only_q=(sum(r['q_score']['final'] for r in seen.values()) / n if n else None),
                completed_only_sr=(sum(r['success'] for r in seen.values()) / n if n else None),
                tasks=per_task)
