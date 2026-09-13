"""Immutable three-instance A4+B-final autonomous radio evaluation contract."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from paired_a3_a4_actions import WORK, CHECKPOINTS, sha, read, require_hash

HERE = Path(__file__).resolve().parent
RUNTIME = WORK / 'native_a3_aligned_full_runtime_v2'
BASE = WORK / 'native_a3_aligned_five_task_development_v3.json'
BASE_SHA = '2722f9404b046ff69b443774d65fe1cb7b0bf4c25369d67b988cb7889924d730'
HIGH_SHA = 'd4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef'
INSTANCE_ROOT = Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/datasets/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances')
RECIPE_NAMES = ('radio_full_manifest.py', 'run_radio_full.py', 'serve_low_fm_full.py',
                'a4_radio_full_campaign.py', 'paired_a3_a4_actions.py')


def episodes():
    return [dict(task_index=0, task_name='turning_on_radio', max_steps=3224,
                 mode='public_test', instance_id=instance, seed=0, wall_budget_seconds=5024)
            for instance in (301, 302, 303)]


def validate_manifest(value, *, verify_files=True):
    required = dict(kind='A4_Bfinal_radio_three_public_instances_v1', immutable=True,
        teacher_prefix_actions=0, oracle_subgoals_used=False, physical_feedback_trained=False,
        policy_seed=17, low_num_obs_steps=6, low_trainability_profile='low_ae_lora_history',
        low_checkpoint_step=2500, high_phase_steps=1500, high_total_B_ancestry_steps=5000,
        physical_diagnostics_policy_input=False, training_admissible=False,
        reinject=False, teacher_eligible=False, low_checkpoint_sha256=CHECKPOINTS['A4'][1],
        high_checkpoint_sha256=HIGH_SHA, base_manifest_sha256=BASE_SHA,
        action_execution_start_index=0, padded_dimensions=[7, 8, 17, 18],
        max_policy_controls=9672, episodes=episodes())
    for key, expected in required.items():
        if value.get(key) != expected or type(value.get(key)) is not type(expected):
            raise ValueError('Frozen radio conditions differ: ' + key)
    if Path(value['low_checkpoint_path']) != CHECKPOINTS['A4'][0]:
        raise ValueError('Wrong A4 checkpoint path')
    if set(value['recipe_files']) != {str(HERE / name) for name in RECIPE_NAMES}:
        raise ValueError('Incomplete evaluation recipe identity')
    if [row['instance_id'] for row in value['instance_files']] != [301, 302, 303]:
        raise ValueError('Missing official initial states')
    if verify_files:
        require_hash(BASE, BASE_SHA)
        base = read(BASE)
        for key in ('variant', 'high_checkpoint_path', 'high_phase_lineage_path', 'robot_config_path',
                    'tasks_path', 'physical_oracle_path', 'runtime_files'):
            if value[key] != base[key]:
                raise ValueError('Previously validated controller/environment changed: ' + key)
        for label in ('high_checkpoint', 'low_checkpoint', 'robot_config', 'tasks',
                      'high_run_receipt', 'low_run_receipt', 'high_phase_lineage', 'physical_oracle'):
            require_hash(value[label + '_path'], value[label + '_sha256'])
        for item in value['runtime_files'] + value['instance_files']:
            require_hash(item['path'], item['sha256'])
        for path, digest in value['recipe_files'].items():
            require_hash(path, digest)
        require_hash(RUNTIME / 'inference_alignment_receipt.json', value['inference_alignment_receipt_sha256'])
    return value


def build_manifest(commit):
    require_hash(BASE, BASE_SHA)
    base = read(BASE)
    manifest = deepcopy(base)
    low_run = CHECKPOINTS['A4'][0].parents[1]
    receipt = read(low_run / 'coordination_run_receipt.json')
    if receipt['state'] != 'complete' or receipt['returncode'] != 0:
        raise ValueError('A4 has not completed successfully')
    manifest.update(kind='A4_Bfinal_radio_three_public_instances_v1', episodes=episodes(),
        low_checkpoint_path=str(CHECKPOINTS['A4'][0]), low_checkpoint_sha256=CHECKPOINTS['A4'][1],
        low_checkpoint_step=2500, low_run_receipt_path=str(low_run / 'coordination_run_receipt.json'),
        low_run_receipt_sha256=sha(low_run / 'coordination_run_receipt.json'),
        low_training_source=receipt['source_root'], base_manifest_sha256=BASE_SHA,
        oracle_subgoals_used=False, training_admissible=False, max_policy_controls=9672,
        action_execution_start_index=0, padded_dimensions=[7, 8, 17, 18], code_commit=commit,
        evaluation_scope='three predeclared public-test initial states; 301 reused development comparison; 302/303 additional instances, not a population or official leaderboard estimate',
        recipe_files={str(HERE / name): sha(HERE / name) for name in RECIPE_NAMES}, instance_files=[])
    # These were the old A3 launcher's diagnostics, not inputs for this cohort.
    for key in ('aligned_launcher_sha256', 'actual_wire_probe_sha256', 'prior_a3_comparison_manifest_sha256'):
        manifest.pop(key, None)
    for instance in (301, 302, 303):
        path = INSTANCE_ROOT / f'house_double_floor_lower_task_turning_on_radio_0_{instance}_template-tro_state.json'
        manifest['instance_files'].append(dict(instance_id=instance, path=str(path), sha256=sha(path)))
    return validate_manifest(manifest)
