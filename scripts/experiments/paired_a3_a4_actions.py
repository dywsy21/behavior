"""Frozen-input A3/A4 diagnostic, not training, deployment, or a success rate.

Reuses the audited original train radio episode 121/instance 138 cache. Each
weight gets the same 42 states, two seeds, and one repeat (170 total calls).
Workers are separate processes so the first model releases all GPU memory.
The source snapshots and old experiment artifacts are read-only dependencies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
SOURCE = WORK / 'a2_lora_history_candidate_v7_samplercoverage'
SOURCE_SHA = '356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281'
CACHE = WORK / 'a3_radio_e121_matched_original_inputs_v2'
RAW = WORK / 'demo_radio_e121_alignment_v2_persist'
CONFIG = WORK / 'a3_samplercoverage_real_processor_v2_microbatch2/diagnostic_processor_config.yaml'
OUTPUT = Path('/mnt/sdc1/robodojo/behavior_dev/a4_paired_actions_20260913_v1')
PYTHON = Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python3.10')
ANCHORS = tuple(range(448, 1120, 16))
SEEDS = (17, 29)
PAD_DIMS = [7, 8, 17, 18]
CHECKPOINTS = {
    'A3': (WORK / 'formal_a3_episodecoverage_5000_v1/checkpoints/step_5000.pt',
           '865193f1c8a257d72ea159bd7a46cebf945b0fd24905110b831f0e8a3438e940', 5000),
    'A4': (Path('/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt'),
           '6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269', 2500),
}
HELPERS = {
    'probe_a3_original_train_capacity_v1.py': 'e38beeec268671cfedbfcac0e10d79716292443293cc7c0f138f02d0fc7dfec7',
    'preflight_a3_parent_gpu_v2.py': '052dd0ec1b127647f18222426b1840187cc71f02871bd8d5b80d85823c3b8792',
    'demo_action_alignment_probe_v2.py': 'd9b7df505bc0df3770cfed253cc2f7cb7c13ac53bc4b019b279d7188f1a0c231',
}
STATS_SHA = '846bcbeac181df5555cb5d40d4183d61743a556df5cf8c8a8fb1bc17e2a40b19'
CACHE_SHA = '6319f9d7920c59d4be67945c5bc60a3a907d10d4066b29bdd45aab47f4912737'
CONFIG_SHA = 'bc6674b6d89d0eaca38ca9e215c1ff75c8e5d20be499c2e209d01a70d7c36c2f'
MANIFEST_SHA = 'c0138e093765e634374d762cd9b5de5fe17628baa14419121b9d8aed0a946c8d'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Frozen dependency changed: {path}')


def arm_metrics(relative, current_q, raw_absolute):
    """Compare against raw controls, NOT inverse-normalized clipped targets."""
    if (len(relative) != 16 or len(raw_absolute) != 16 or len(current_q) != 7
            or any(len(row) != 7 for row in [*relative, *raw_absolute])):
        raise ValueError('Expected first 16 future actions, seven right-arm joints')
    p = [float(x) for row in relative for x in row]
    t = [float(row[j]) - float(current_q[j]) for row in raw_absolute for j in range(7)]
    if not all(math.isfinite(x) for x in [*p, *t, *current_q]):
        raise ValueError('Nonfinite prediction, observed state, or target')
    rms = lambda values: math.sqrt(sum(x*x for x in values) / len(values))
    err, hold = rms([x-y for x, y in zip(p, t)]), rms(t)
    pe, te = sum(x*x for x in p), sum(x*x for x in t)
    dot = sum(x*y for x, y in zip(p, t))
    return dict(error_rms_rad=err, hold_error_rms_rad=hold, predicted_change_rms_rad=rms(p),
                beats_current_q_hold=err < hold,
                projection_gain=dot/te if te > 1e-16 else None,
                cosine=dot/math.sqrt(pe*te) if pe*te > 1e-24 else None)


def index_rows(rows):
    selected = {}
    for row in rows:
        if row['repeat']:
            continue
        key = row['frame'], row['seed']
        if key in selected:
            raise ValueError('Duplicate state/seed')
        selected[key] = row
    if set(selected) != {(f, s) for f in ANCHORS for s in SEEDS}:
        raise ValueError('Missing or extra state/seed')
    return selected


def paired_summary(a, b):
    first, second = index_rows(a), index_rows(b)
    result = {}
    for seed in SEEDS:
        keys = [k for k in sorted(first) if k[1] == seed]
        for name, subset in [('all', keys), ('active_ge_0.005rad', [
                k for k in keys if first[k]['right_arm']['hold_error_rms_rad'] >= .005])]:
            group = {'pairs': len(subset)}
            if subset:
                for version, table in [('A3', first), ('A4', second)]:
                    group[version] = {'mean_' + metric: statistics.mean(
                        table[k]['right_arm'][metric] for k in subset)
                        for metric in ('error_rms_rad', 'hold_error_rms_rad', 'predicted_change_rms_rad')}
                    group[version]['beats_current_q_hold'] = sum(
                        table[k]['right_arm']['beats_current_q_hold'] for k in subset)
                    group[version]['right_gripper_command_rmse'] = statistics.mean(
                        table[k]['right_gripper_command_rmse'] for k in subset)
                delta = [second[k]['right_arm']['error_rms_rad'] - first[k]['right_arm']['error_rms_rad']
                         for k in subset]
                group.update(mean_A4_minus_A3_error_rad=statistics.mean(delta),
                             median_A4_minus_A3_error_rad=statistics.median(delta),
                             A4_lower_error_count=sum(d < 0 for d in delta))
            result[f'seed{seed}/{name}'] = group
    return result


def check_launch():
    launch = read(OUTPUT / 'launch.json')
    require_hash(__file__, launch['recipe_sha256'])
    if launch['model_call_budget'] != 170 or launch['optimizer_steps'] != 0:
        raise ValueError('Unexpected diagnostic budget')


def predict(version):
    check_launch()
    destination = OUTPUT / version
    destination.mkdir(exist_ok=False)
    for name, digest in HELPERS.items():
        require_hash(WORK / name, digest)
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts'), str(SOURCE), str(WORK)]
    os.chdir(SOURCE)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.data.processor_utils import build_processors
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.utils.training.coordination_runtime import source_tree_sha256
    from preflight_memlite_skillfm_gpu import _load_and_configure_gpu_model, _stage_contract
    from preflight_a3_parent_gpu_v2 import verify_training_parent_tensors
    from probe_a3_original_train_capacity_v1 import target_free_sample, compare_prefix

    if source_tree_sha256(SOURCE) != SOURCE_SHA:
        raise ValueError('Training model source changed')
    require_hash(CACHE / 'actual_original_batches.pt', CACHE_SHA)
    require_hash(CONFIG, CONFIG_SHA)
    require_hash(RAW / 'manifest.json', MANIFEST_SHA)
    receipt = read(CACHE / 'input_receipt.json')
    if (receipt['status'] != 'complete' or not receipt['original_train_dataset_split']
            or receipt['anchors'] != list(ANCHORS) or len(receipt['records']) != 42
            or receipt['cache_sha256'] != CACHE_SHA
            or receipt['training_source_sha256'] != SOURCE_SHA
            or receipt['processor_config_sha256'] != CONFIG_SHA
            or receipt['original_manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('Original train-input receipt failed')
    manifest = read(RAW / 'manifest.json')
    if (manifest['episode_index'], manifest['instance_id'], manifest['split']) != (121, 138, 'train'):
        raise ValueError('Not the approved original train instance')
    for key in ('controls', 'original_proprio'):
        require_hash(manifest[key]['path'], manifest[key]['sha256'])
    raw_actions = np.load(manifest['controls']['path'], allow_pickle=False)
    raw_states = np.load(manifest['original_proprio']['path'], allow_pickle=False)
    checkpoint_path, digest, expected_step = CHECKPOINTS[version]
    require_hash(checkpoint_path, digest)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Only reserved GPU1 is allowed')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.40)
    register_default_resolvers()
    cfg = OmegaConf.load(CONFIG)
    require_hash(cfg.datastatics_path, STATS_SHA)
    cfg.model.pretrained_ckpt = str(checkpoint_path)
    device = torch.device('cuda:0')
    model, parent, _, _, _ = _load_and_configure_gpu_model(
        cfg, checkpoint_path, device, _stage_contract(cfg, 'b'))
    load_receipt = verify_training_parent_tensors(model, parent, expected_step=expected_step)
    if load_receipt['state_entries'] != 1138 or load_receipt['adapter_entries'] != 192:
        raise ValueError('Incomplete neural model restoration')
    del parent
    gc.collect()
    model.apply_fp32_params()
    model.eval()
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.datastatics_path))
    fields = processor.processors['galaxea_r1pro']._normalizer.normalizers['action']
    original_batches = torch.load(CACHE / 'actual_original_batches.pt', map_location='cpu', weights_only=False)
    if len(original_batches) != len(ANCHORS):
        raise ValueError('Cache length differs')
    old = read(CACHE / 'result.json')
    if old['status'] != 'complete' or old['checkpoint_sha256'] != CHECKPOINTS['A3'][1]:
        raise ValueError('A3 prior prediction identity differs')
    old_rows = index_rows(old['rows'])
    rows, prefix_records = [], []
    with torch.inference_mode():
        for frame, original, record in zip(ANCHORS, original_batches, receipt['records']):
            if (record['frame'] != frame or original['action'].shape != (1, 32, 27)
                    or original['action_is_pad'][0, :16].any()
                    or original['action_dim_is_pad'].shape != (1, 27)
                    or original['action_dim_is_pad'][0].nonzero().flatten().tolist() != PAD_DIMS
                    or len(original['pixel_values']) != 3
                    or any(v.shape[:2] != (1, 6) for v in original['pixel_values'].values())):
                raise ValueError('Input clock/history/valid-control contract differs')
            current_q = raw_states[frame, 28:35]
            if not np.array_equal(current_q, np.asarray(record['raw_right_q'])):
                raise ValueError('Original frame/current joint state mismatch')
            expected = fields['right_arm'].forward(torch.from_numpy(
                raw_actions[frame:frame+32, 15:22] - current_q).unsqueeze(0))
            valid = ~original['action_is_pad'][0]
            if not torch.allclose(expected[0, valid], original['action'][0, valid, 10:17], atol=1e-5, rtol=0):
                raise ValueError('Valid train targets differ from the raw source')
            expected_gripper = fields['right_gripper'].forward(
                torch.from_numpy(raw_actions[frame:frame+32, 22:23]).unsqueeze(0))
            if not torch.allclose(expected_gripper[0, valid], original['action'][0, valid, 19:20], atol=1e-5, rtol=0):
                raise ValueError('Gripper command mapping differs')
            batch = dict_apply(deepcopy(original), lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            samples = [target_free_sample(s) for s in batch['samples']]
            ti, _, tm, split = model.processor.encode_train(batch['samples'], device=device, training=False,
                max_chunk_token_length=model.max_chunk_token_length, max_pad_token_length=model.max_pad_token_length)
            ii, im = model.processor.encode_inference(samples, device=device, mode='ar', training=False)
            equality = compare_prefix(ti, tm, split, ii, im)[0]
            if not equality['token_ids_equal'] or not equality['modality_mask_equal']:
                raise ValueError('Inference prefix differs from training context')
            prefix_records.append(dict(frame=frame, **equality))
            first_prediction = None
            for seed, repeat in [(17, False), (29, False)] + ([(17, True)] if frame == ANCHORS[0] else []):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    state = model.prefill(samples, batch['pixel_values'])
                    generated = model.generate_action(state, samples,
                        action_dim_is_pad=batch['action_dim_is_pad'], action_gt=None)
                prediction = generated['action'].detach().float().cpu()
                if (generated['selected_action_source'] != 'fm' or prediction.shape != (1, 32, 27)
                        or not torch.isfinite(prediction).all() or torch.count_nonzero(prediction[:, :, PAD_DIMS])):
                    raise ValueError('Invalid continuous FM output or padding leakage')
                if repeat and not torch.equal(first_prediction, prediction):
                    raise ValueError('Same state and seed did not reproduce')
                if seed == 17 and not repeat:
                    first_prediction = prediction.clone()
                if version == 'A3' and not repeat:
                    reference = old_rows[(frame, seed)]
                    require_hash(reference['prediction_file'], reference['prediction_sha256'])
                    saved = torch.load(reference['prediction_file'], map_location='cpu', weights_only=True)
                    if not torch.equal(prediction, saved):
                        raise ValueError(f'A3 same-input baseline changed at frame={frame}, seed={seed}')
                path = destination / f'f{frame:08d}_seed{seed}{"_repeat" if repeat else ""}.pt'
                if path.exists():
                    raise FileExistsError(path)
                torch.save(prediction, path)
                relative = fields['right_arm'].backward(prediction[:, :, 10:17])[0, :16]
                gripper = fields['right_gripper'].backward(prediction[:, :, 19:20])[0, :16, 0].numpy()
                rows.append(dict(frame=frame, seed=seed, repeat=repeat,
                    prediction_file=str(path), prediction_sha256=sha(path),
                    right_arm=arm_metrics(relative.tolist(), current_q.tolist(), raw_actions[frame:frame+16, 15:22].tolist()),
                    right_gripper_command_rmse=float(np.sqrt(np.mean((gripper-raw_actions[frame:frame+16, 22])**2))),
                    right_gripper_predicted16=gripper.tolist(), right_gripper_raw_target16=raw_actions[frame:frame+16, 22].tolist()))
                del state, generated
            print(json.dumps(dict(version=version, frame=frame, model_calls=len(rows), last_arm=rows[-1]['right_arm'])), flush=True)
            del batch
    if len(rows) != 85:
        raise ValueError('Incorrect actual generation count')
    write_new(destination / 'result.json', dict(status='complete', version=version, rows=rows,
        model_calls=len(rows), optimizer_steps=0, physics_actions=0, checkpoint_sha256=digest,
        source_sha256=SOURCE_SHA, parent_load=load_receipt, same_seed_repeat_identical=True,
        prefix_identity=prefix_records, inference_received_targets=False,
        A3_prior_predictions_bitwise_equal=(True if version == 'A3' else None),
        original_input_receipt_sha256=sha(CACHE / 'input_receipt.json'), recipe_sha256=sha(__file__),
        first_executed_action_index=0, training_admissible=False, model_success_rate_claim=False))


def run():
    check_launch()
    for version in ('A3', 'A4'):
        with (OUTPUT / f'{version}.log').open('x') as log:
            subprocess.run([str(PYTHON), str(Path(__file__).resolve()), '--worker', version],
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    a, b = [read(OUTPUT / version / 'result.json') for version in ('A3', 'A4')]
    if any(r['status'] != 'complete' or r['model_calls'] != 85 for r in (a, b)):
        raise ValueError('Both actual predictions must complete')
    write_new(OUTPUT / 'result.json', dict(status='complete', completed_unix=time.time(),
        recipe_sha256=sha(__file__), model_calls=170, optimizer_steps=0, physics_actions=0,
        result_sha256={v: sha(OUTPUT / v / 'result.json') for v in ('A3', 'A4')},
        comparisons=paired_summary(a['rows'], b['rows']), training_admissible=False,
        model_success_rate_claim=False, limitations=[
            'One original train episode and one expert continuation, not closed-loop success.',
            'The 0.005rad motion threshold is descriptive, not a physical success label.',
            'No wrong-intent intervention in this diagnostic; condition obedience is not established.',
            'Raw absolute target error avoids the original normalizer clipping distortion.']))


def launch():
    if OUTPUT.exists():
        raise FileExistsError('No retry or overwrite of an existing diagnostic')
    if subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                'status', '--porcelain'], text=True).strip():
        raise ValueError('Recipe must run from a clean, pinned Git worktree')
    lock = (OUTPUT.parent / 'a4_paired_actions_gpu1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).strip())
    if free < 35*1024:
        raise RuntimeError('Insufficient GPU1 memory; no other jobs stopped')
    OUTPUT.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    env.pop('MEMLITE_CANDIDATE_ROOT', None)
    command = [str(PYTHON), str(Path(__file__).resolve()), '--run']
    # Finish the immutable contract before creating the child (no partial-JSON race).
    write_new(OUTPUT / 'launch.json', dict(prepared_unix=time.time(), command=command,
        recipe_commit=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                               'rev-parse', 'HEAD'], text=True).strip(),
        recipe_sha256=sha(__file__), gpu=1, gpu_free_mib_before=free, model_call_budget=170,
        optimizer_steps=0, physics_actions=0, other_jobs_stopped=False))
    with (OUTPUT / 'supervisor.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
    write_new(OUTPUT / 'process.json', dict(pid=child.pid, started_unix=time.time(), command=command))
    print(json.dumps(dict(pid=child.pid, output=str(OUTPUT))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--worker', choices=('A3', 'A4'))
    args = parser.parse_args()
    if args.start:
        launch()
        return
    try:
        for _ in range(50):
            if (OUTPUT / 'launch.json').exists():
                break
            time.sleep(.1)
        if args.worker:
            predict(args.worker)
        else:
            run()
    except BaseException:
        failure = OUTPUT / f'{args.worker or "supervisor"}_failure.json'
        if not failure.exists():
            write_new(failure, dict(status='failed', traceback=traceback.format_exc(),
                                   model_success_rate_claim=False, no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
