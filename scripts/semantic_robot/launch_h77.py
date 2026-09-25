"""Second pose gate: task3, exactly H75's runtime implementation and thresholds.

This experiment-only launcher is pinned by its source commit, not added to the
actor implementation digest: it changes no worker code, flags or native helper.
H75's gate remains valid only while that complete implementation digest matches.
"""
import argparse
import hashlib
import json
from pathlib import Path

import launch_h73 as synchronous

gate = synchronous.gate
COMMAND = gate.command
IDENTITY = synchronous.identity
DIGEST = '7bad2e9021b6a48dd298e571b213c19e763453b8c593ba871afbc404f0c765f8'
FIRST = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h75_render_batch_v1/gate/result.json')
FIRST_SHA = 'd1cc900f093ed37d0078f5776652a0344b010c05e6404660a2010cf8d7b24bab'


def identity(base):
    commit = IDENTITY(base)
    from run_v2 import implementation_digest
    if implementation_digest() != DIGEST:
        raise ValueError('H77 must run the unchanged H75 implementation')
    if hashlib.sha256(FIRST.read_bytes()).hexdigest() != FIRST_SHA:
        raise ValueError('Reviewed first pose gate changed')
    first = json.loads(FIRST.read_text())
    if first['implementation_digest'] != DIGEST or first['task'] != 0 or first['gate_ok'] is not True:
        raise ValueError('Missing first pose gate prerequisite')
    return commit


def command(base):
    args = COMMAND(base)
    args[args.index('--task')+1] = '3'
    return args


def validate_manifest(manifest, base, commit, digest):
    exact = {'code_commit': commit, 'implementation_digest': digest, 'instance': 242,
             'task': 3, 'task_name': 'cleaning_up_plates_and_food', 'split': 'train', 'seed': 0,
             'training_updates': 0, 'model_identity': None, 'actor_scene_truth': False,
             'prefix_is_expert_not_agent': False, 'diagnostic_replay_requested': False,
             'native_profile': 'a100_full_v1', 'args': gate.expected_args(base)}
    if digest != DIGEST:
        raise ValueError('H77 implementation changed')
    for key, value in exact.items():
        if key not in manifest or json.dumps(manifest[key],sort_keys=True) != json.dumps(value,sort_keys=True):
            raise ValueError('Exact task3 manifest mismatch: '+key)


def validate_basic_result(result, digest):
    # Same scalar/24-action criteria as H69/H75, with actual task3 identity.
    # The reused synchronous validator additionally checks the entire journal
    # and every camera receipt; no task ID is rewritten in any result object.
    exact = {'status': 'complete', 'task': 3, 'prefix_controls': 0, 'diagnostic_replay_controls': 0,
             'gate_ok': True, 'gate_failures': [], 'native_profile': 'a100_full_v1',
             'implementation_digest': digest, 'finger_kinematics': True, 'press_cycle_v1': True,
             'press_finger_asset_sha256': gate.ASSET_SHA, 'model_calls': 0,
             'contact_geometry': False, 'full_task_success_rate_claim': False}
    exact.update({flag.replace('-','_'):True for flag in gate.FLAGS if flag!='structured-planning'})
    if digest != DIGEST or any(type(result.get(k)) is not type(v) or result[k]!=v for k,v in exact.items()):
        raise ValueError('Task3 gate did not pass its exact source/profile and original criteria')
    if len(result.get('decisions',[]))!=24 or not 0<result.get('controls',0)<=1536:
        raise ValueError('Incomplete or over-budget task3 gate')


def configure():
    synchronous.configure()
    gate.ROOT=Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h77_task3_gate_v1')
    gate.RUNTIME=Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h77_task3_gate_v1')
    gate.command,gate.identity,gate.validate_manifest=command,identity,validate_manifest
    synchronous.ORIGINAL_VALIDATE=validate_basic_result
    base=gate.configure();base.ENTRYPOINT=Path(__file__).resolve()
    return base


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch',action='store_true');modes.add_argument('--supervise',action='store_true')
    args=parser.parse_args();base=configure()
    (gate.launch if args.launch else gate.supervise)(base)
