"""One A4 fixed-GRASP diagnostic, reusing the audited A3 window and runtime.

No new teacher, high-level calls, policy deployment, or training. The private
service belongs to this supervisor and is closed after the bounded rollout.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from paired_a3_a4_actions import WORK, CHECKPOINTS, PYTHON, sha, read, write_new, require_hash

HERE = Path(__file__).resolve().parent
MAIN = Path('/mnt/sdc1/robodojo/GalaxeaVLA')
DEV = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908')
ADAPTER = DEV / 'sim_runtime/production_native_oracle_low_v1'
BEHPY = Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python')
RUNTIME = WORK / 'native_a3_aligned_prefix_runtime_v2'
SOURCE = WORK / 'a3_history_serving_candidate_v1'
COMPOSITION = WORK / 'a3_serving_composition_v2'
RUN = Path('/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal')
OUTPUT = Path('/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1')
OLD = WORK / 'a3_aligned_radio_e121_l1_v2'
CHECKPOINT_SHA = CHECKPOINTS['A4'][1]
PORT = 8782


def imports():
    sys.path[:0] = [str(RUNTIME), str(WORK / 'c1_v2'), str(ADAPTER), str(SOURCE / 'src'), str(SOURCE)]


def validate_window(manifest, window, context):
    if (manifest['source_split'] != 'train' or manifest['source_episode_index'] != 121
            or context.split != 'train' or window.official_mode != 'train'
            or context.task_id != 0 or context.instance_id != 138
            or len(window.frozen_window().prefix_actions) != 448
            or window.execute_steps != 16 or window.max_chunks != 80
            or manifest['policy_seed'] != 17):
        raise ValueError('Only the approved original radio L1 window/budget is allowed')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Never retry or overwrite the existing L1 run')
    if subprocess.check_output(['git', '-C', str(HERE.parents[1]), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Use a clean pinned Git worktree')
    prior = read(OLD / 'manifest.json')
    old_result = read(OLD / 'actual_rollout/collection_result.json')
    if old_result['status'] != 'complete' or old_result['expected_checkpoint_sha256'] != CHECKPOINTS['A3'][1]:
        raise ValueError('Require the actually completed aligned A3 comparison')
    for path, digest in prior['files'].items():
        require_hash(path, digest)
    prep = read(WORK / 'a3_aligned_runtimes_v2_preparation.json')
    runtime = prep['roots']['prefix']
    if (prep['status'] != 'complete' or runtime['path'] != str(RUNTIME)
            or runtime['files'] != {p.name: sha(p) for p in RUNTIME.glob('*.py')}):
        raise ValueError('The tested corrected runtime changed')
    completed = read(RUN / 'coordination_run_receipt.json')
    if completed['state'] != 'complete' or completed['returncode'] != 0:
        raise ValueError('A4 training has not completed normally')
    require_hash(CHECKPOINTS['A4'][0], CHECKPOINT_SHA)
    paired = read(OUTPUT.parent / 'a4_paired_actions_20260913_v1/result.json')
    if paired['status'] != 'complete' or paired['model_calls'] != 170:
        raise ValueError('Actual paired-action diagnostic must finish first')
    imports()
    from c1_feedback.official_factory_c1 import load_c1_window_and_context
    from native_a2_source_contract import validate_source_receipt
    window, context = load_c1_window_and_context(prior['window_path'], prior['context_path'])
    validate_window(prior, window, context)
    validate_source_receipt(COMPOSITION / 'source_receipt.json', training_source=completed['source_root'],
        serving_source=SOURCE, snapshot=COMPOSITION / 'native_snapshot.json', config=COMPOSITION / 'serving_config.yaml')
    copied = read(prior['context_path'])
    copied['run_id'] = 'actual-A4-aligned-radio-e121-prefix-history-L1-20260913-v1'
    OUTPUT.mkdir()
    write_new(OUTPUT / 'event_context.json', copied)
    files = dict(prior['files'])
    for path in [HERE / 'a4_prefix_pilot.py', HERE / 'serve_low_fm_prefix.py',
                 HERE / 'paired_a3_a4_actions.py', HERE / 'a4_prefix_wire_probe.py',
                 OUTPUT / 'event_context.json', RUN / 'coordination_run_receipt.json',
                 OLD / 'manifest.json', OLD / 'actual_rollout/collection_result.json',
                 RUNTIME / 'inference_alignment_receipt.json']:
        files[str(path)] = sha(path)
    manifest = dict(prior, kind='A4_aligned_original_radio_L1_v1', files=files,
        context_path=str(OUTPUT / 'event_context.json'), checkpoint_path=str(CHECKPOINTS['A4'][0]),
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_step=2500, policy_uri=f'ws://127.0.0.1:{PORT}',
        gpu_model=0, gpu_simulator=1, code_commit=subprocess.check_output([
            'git', '-C', str(HERE.parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
        comparison_manifest_sha256=sha(OLD / 'manifest.json'),
        only_model_weight_changed_from_aligned_A3=True, model_call_budget=87)
    write_new(OUTPUT / 'manifest.json', manifest)
    return manifest


def validate():
    manifest = read(OUTPUT / 'manifest.json')
    if (manifest['kind'] != 'A4_aligned_original_radio_L1_v1'
            or manifest['checkpoint_sha256'] != CHECKPOINT_SHA or manifest['checkpoint_step'] != 2500
            or manifest['policy_uri'] != f'ws://127.0.0.1:{PORT}'
            or manifest['training_admissible'] is not False):
        raise ValueError('A4 L1 run identity differs')
    for path, digest in manifest['files'].items():
        require_hash(path, digest)
    imports()
    from c1_feedback.official_factory_c1 import load_c1_window_and_context
    window, context = load_c1_window_and_context(manifest['window_path'], manifest['context_path'])
    validate_window(manifest, window, context)
    return manifest, window, context


def free_mib(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())


def supervisor():
    manifest, _, _ = validate()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false')
    env.pop('PYTHONPATH', None)
    env.pop('MEMLITE_CANDIDATE_ROOT', None)
    if free_mib(0) < 35*1024 or free_mib(1) < 25*1024:
        raise RuntimeError('GPU headroom insufficient; other jobs remain untouched')
    with socket.socket() as check:
        check.bind(('127.0.0.1', PORT))
    service_command = [str(PYTHON), str(HERE / 'serve_low_fm_prefix.py'), '--run', str(RUN),
        '--checkpoint', manifest['checkpoint_path'], '--checkpoint-step', '2500', '--checkpoint-sha', CHECKPOINT_SHA,
        '--runtime', str(RUNTIME), '--config', str(COMPOSITION / 'serving_config.yaml'), '--adapter', str(ADAPTER),
        '--output-dir', str(OUTPUT / 'service'), '--port', str(PORT), '--serving-source', str(SOURCE),
        '--snapshot', str(COMPOSITION / 'native_snapshot.json'), '--source-receipt', str(COMPOSITION / 'source_receipt.json'),
        '--initial-action-count', '448', '--prefix-window-sha', manifest['window_sha256'],
        '--inference-alignment-receipt', str(RUNTIME / 'inference_alignment_receipt.json')]
    with (OUTPUT / 'service.log').open('x') as log:
        service = subprocess.Popen(service_command, cwd=MAIN, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
    write_new(OUTPUT / 'service.launch.json', dict(pid=service.pid, command=service_command, started_unix=time.time()))
    try:
        ready = False
        for _ in range(180):
            if service.poll() is not None:
                raise RuntimeError('Private service exited before ready; inspect service.log')
            if 'native_skill_fm_history6 websocket ready' in (OUTPUT / 'service.log').read_text():
                ready = True
                break
            time.sleep(5)
        if not ready:
            raise TimeoutError('No service ready after 15 minutes; no automatic retry')
        load_receipt = read(OUTPUT / 'service/load_receipt.json')
        if (load_receipt['checkpoint_sha256'] != CHECKPOINT_SHA or load_receipt['checkpoint_step'] != 2500
                or not load_receipt['checkpoint_load']['full_base_and_adapter_bitwise_equal']
                or load_receipt['service_entry_sha256'] != sha(HERE / 'serve_low_fm_prefix.py')):
            raise ValueError('Private service loaded another checkpoint or recipe')
        with (OUTPUT / 'wire_probe.log').open('x') as log:
            subprocess.run([str(PYTHON), str(HERE / 'a4_prefix_wire_probe.py')], cwd=MAIN, env=env,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        wire = read(OUTPUT / 'wire_probe/result.json')
        if not wire['passed'] or wire['model_calls'] != 7 or not wire['all_alignment_admissions_correct']:
            raise ValueError('Actual seven-call wire check failed')
        sim_env = dict(env)
        sim_env.pop('CUDA_VISIBLE_DEVICES', None)
        sim_env.update(OMNI_KIT_ACCEPT_EULA='1', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_NO_OMNI_LOGS='1',
            OMNIGIBSON_GPU_ID='1', BEHAVIOR_ACTION_STEPS='1',
            OMNIGIBSON_APPDATA_PATH=str(WORK / 'kit_c1_gpu1_appdata_v1'),
            LD_LIBRARY_PATH='/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib')
        command = [str(BEHPY), str(HERE / 'a4_prefix_pilot.py'), 'simulate']
        with (OUTPUT / 'rollout.log').open('x') as log:
            simulation = subprocess.Popen(command, cwd=WORK, env=sim_env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT)
            write_new(OUTPUT / 'rollout.launch.json', dict(pid=simulation.pid, command=command,
                started_unix=time.time(), manifest_sha256=sha(OUTPUT / 'manifest.json')))
            if simulation.wait() != 0:
                raise RuntimeError('Bounded physics diagnostic failed; no automatic retry')
        result = read(OUTPUT / 'actual_rollout/collection_result.json')
        if (result['status'] != 'complete' or result['prefix_actions_executed'] != 448
                or result['native_policy_actions_consumed'] > 1280
                or result['low_service_identity']['checkpoint_sha256'] != CHECKPOINT_SHA
                or result['training_admissible'] or result['full_task_success_rate_claim']):
            raise ValueError('Physical result/identity/budget failed')
        write_new(OUTPUT / 'completion.json', dict(status='complete', completed_unix=time.time(),
            result_sha256=sha(OUTPUT / 'actual_rollout/collection_result.json'),
            physical_actions=result['prefix_actions_executed'] + result['native_policy_actions_consumed'],
            training_admissible=False, model_success_rate_claim=False, personal_review_pending=True))
    finally:
        # Only this supervisor's own private child; never stop another service.
        if service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()
        write_new(OUTPUT / 'private_service_exit.json', dict(pid=service.pid, returncode=service.returncode,
                                                           other_jobs_stopped=False))


def start():
    lock = (OUTPUT.parent / 'a4_radio_l1_gpu1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare()
    command = [str(PYTHON), str(HERE / 'a4_prefix_pilot.py'), 'supervise']
    with (OUTPUT / 'supervisor.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(lock.fileno(),))
    write_new(OUTPUT / 'launch.json', dict(pid=child.pid, command=command, started_unix=time.time(),
        manifest_sha256=sha(OUTPUT / 'manifest.json'), max_policy_controls=1280, original_prefix_controls=448,
        no_automatic_retry=True, other_jobs_stopped=False))
    print(json.dumps(dict(pid=child.pid, output=str(OUTPUT))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('start', 'supervise', 'simulate'))
    operation = parser.parse_args().operation
    if operation == 'start':
        start()
        return
    try:
        if operation == 'supervise':
            supervisor()
        else:
            manifest, window, context = validate()
            from a2_prefix_official_factory import run_official_a2_prefix_window
            result = run_official_a2_prefix_window(window=window, context=context, source_root=SOURCE,
                policy_uri=manifest['policy_uri'], expected_checkpoint_sha256=CHECKPOINT_SHA,
                prefix_window_sha256=manifest['window_sha256'], output_dir=OUTPUT / 'actual_rollout', gpu=1, policy_seed=17)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException:
        path = OUTPUT / f'{operation}_failure.json'
        if not path.exists():
            write_new(path, dict(status='failed', traceback=traceback.format_exc(), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
