"""CPU-only codec gate on audited original training inputs; no VLM or rollout.

Compare original32, first16+last-value codec padding, and a direct16 encode.
Direct16 unsupported shapes are recorded, never silently resized or repaired.
All errors are in normalized action coordinates, not physical success metrics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
SOURCE = WORK / 'a2_lora_history_candidate_v7_samplercoverage'
SOURCE_SHA = '356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281'
INPUT = WORK / 'a3_samplercoverage_real_processor_v2_microbatch2'
INPUT_SHA = '237acf01b29bd0d6806ed1a11d9033a747640b3ea9e0bf7246b7a62b4e92be81'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError('output must be a new absolute directory')
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Run from a clean pinned Git worktree')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts'), str(SOURCE)]
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.training.coordination_runtime import source_tree_sha256
    from g05.tokenizer.interface.vq_base import VQActionTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    if torch.cuda.is_available():
        raise RuntimeError('This codec gate must not allocate CUDA')
    if source_tree_sha256(SOURCE) != SOURCE_SHA or sha(INPUT / 'actual_cpu_batches.pt') != INPUT_SHA:
        raise RuntimeError('Source or original train input identity changed')
    receipt = json.loads((INPUT / 'result.json').read_text())
    config_path = INPUT / 'diagnostic_processor_config.yaml'
    if (receipt['status'] != 'complete' or not receipt['train_only'] or not receipt['all_five_tasks']
            or receipt['actual_batch_file_sha256'] != INPUT_SHA
            or receipt['processor_config_sha256'] != sha(config_path)):
        raise RuntimeError('Audited original-train receipt is invalid')
    register_default_resolvers()
    cfg = OmegaConf.load(config_path)
    methods_path = repo / 'src/g05/utils/training/fm_training_methods.py'
    spec = importlib.util.spec_from_file_location('fm_methods_codec_probe', methods_path)
    methods = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = methods
    spec.loader.exec_module(methods)
    # The inherited FM processor config never uses this AR path. Its noop
    # option drops constant gripper groups; a complete-control codec probe
    # must explicitly disable that option (as the existing AR v9 did).
    # Do not remove the missing-group guard or replace absent outputs by GT.
    codec_config = deepcopy(cfg.tokenizer.vq_config)
    inherited_noop_dropout = bool(codec_config.get('dropout_noop_parts', False))
    codec_config.dropout_noop_parts = False
    codec = VQActionTokenizer(codec_config, action_dim=27, device='cpu')
    codec.action_tokenizer.eval()
    batches = torch.load(INPUT / 'actual_cpu_batches.pt', map_location='cpu', weights_only=False)
    if len(batches) != len(receipt['batches']):
        raise RuntimeError('Cached batch/source receipt count differs')
    publish(args.output / 'manifest.json', dict(commit=commit, source=str(SOURCE),
        source_sha256=SOURCE_SHA, input_sha256=INPUT_SHA, methods_sha256=sha(methods_path),
        codec_sha256=sha(cfg.tokenizer.vq_config.ckpt_dir), train_only=True,
        inherited_dropout_noop_parts=inherited_noop_dropout, dropout_noop_parts=False,
        codec_config=OmegaConf.to_container(codec_config, resolve=True),
        inference_model_calls=0, optimizer_updates=0, simulator_controls=0,
        start_time=datetime.now(timezone.utc).isoformat(), max_original_rows=10))

    def reconstruct(action, mask, horizon):
        indices = codec._encode_action_indices(action, encode_kwargs={'action_dim_is_pad': mask})
        decoded = codec.decode_token_ids_to_actions(indices[0], time_horizon=horizon, action_dim=27,
            decode_kwargs={'is_action_token_space': True})
        tensor = decoded.action.float()
        if tuple(tensor.shape) != (horizon, 27) or not torch.isfinite(tensor).all():
            raise RuntimeError('Invalid decoded action shape/numerics')
        if decoded.absent_keys:
            raise RuntimeError('Missing action groups: ' + repr(decoded.absent_keys))
        return tensor, [int(x) for x in indices[0]]

    groups = {'left_arm': list(range(7)), 'left_gripper': [9],
              'right_arm': list(range(10, 17)), 'right_gripper': [19],
              'trunk': [20, 21, 22, 23], 'base': [24, 25, 26]}
    rows = []
    with torch.inference_mode():
        for bi, batch in enumerate(batches):
            sources = receipt['batches'][bi]['sources']
            if len(sources) != batch['action'].shape[0]:
                raise RuntimeError('Source batch count mismatch')
            for ri, source in enumerate(sources):
                action = batch['action'][ri:ri + 1].float().clone()
                mask = batch['action_dim_is_pad'][ri:ri + 1].bool()
                temporal = batch['action_is_pad'][ri:ri + 1].bool()
                if action.shape != (1, 32, 27):
                    raise RuntimeError('Expected original 32x27 actions')
                valid = ~temporal[0, :16]
                padded = methods.executed_prefix_codec_input(action, action_is_pad=temporal)
                if mask[0].nonzero().flatten().tolist() != [7, 8, 17, 18]:
                    raise RuntimeError('Unexpected R1Pro padding mask')
                locator = batch['sample_meta'][ri]['dataset_locator']
                if int(locator.rsplit('local_idx=', 1)[-1]) != source['requested_index']:
                    raise RuntimeError('Original sample locator differs from audited source')
                variants = {'original32': action, 'prefix16_holdpad32': padded}
                original, _ = reconstruct(action, mask, 32)
                row = dict(source=source, input_shape=list(action.shape), variants={},
                           valid_prefix_steps=int(valid.sum()),
                           full_execution_window=bool(valid.all()))
                for name, codec_input in variants.items():
                    if not torch.equal(codec_input[:, :16][:, valid], action[:, :16][:, valid]):
                        raise RuntimeError('Experimental representation changed executable target')
                    decoded, indices = reconstruct(codec_input, mask, 32)
                    error = decoded[:16][valid] - action[0, :16][valid]
                    row['variants'][name] = dict(supported=True, token_count=len(indices),
                        action_indices=indices, normalized_rmse_by_group={
                            name: float(error[:, dims].square().mean().sqrt()) for name, dims in groups.items()},
                        normalized_valid_rmse=float(error[:, ~mask[0]].square().mean().sqrt()),
                        reconstructed_prefix=decoded[:16].tolist())
                # Only the unseen suffix is changed; the proposed representation
                # must be completely independent of it. This is not new BC data.
                changed = action.clone()
                changed[:, 16:, 24:27] = -changed[:, 16:, 24:27] + .25
                alternate, _ = reconstruct(changed, mask, 32)
                row['future_suffix_effect_on_original_base_prefix_max'] = float(
                    (alternate[:16, 24:27][valid] - original[:16, 24:27][valid]).abs().max())
                if not torch.equal(padded,
                                   methods.executed_prefix_codec_input(changed, action_is_pad=temporal)):
                    raise RuntimeError('Unused future affected executable-prefix codec input')
                row['prefix16_padding_future_invariant'] = True
                try:
                    direct, indices = reconstruct(action[:, :16], mask, 16)
                    error = direct[valid] - action[0, :16][valid]
                    row['variants']['direct16'] = dict(supported=True, token_count=len(indices),
                        normalized_valid_rmse=float(error[:, ~mask[0]].square().mean().sqrt()),
                        normalized_rmse_by_group={name: float(error[:, dims].square().mean().sqrt())
                                                  for name, dims in groups.items()})
                except (ValueError, AssertionError, RuntimeError) as exc:
                    row['variants']['direct16'] = dict(supported=False,
                        error_type=type(exc).__name__, error=str(exc)[:1200])
                rows.append(row)
                publish(args.output / f'row_{len(rows):02d}.json', row)
                print(json.dumps(dict(completed_rows=len(rows), source=source,
                    metrics={k: {n: v for n, v in value.items()
                                 if n in {'supported', 'normalized_valid_rmse', 'error_type'}}
                             for k, value in row['variants'].items()})), flush=True)
    if len(rows) != 10 or {int(row['source']['task']) for row in rows} != set(range(5)):
        raise RuntimeError('Expected all ten audited rows covering five tasks')
    summary = {}
    for name in ('original32', 'prefix16_holdpad32', 'direct16'):
        values = [row['variants'][name] for row in rows if row['variants'][name]['supported']]
        summary[name] = dict(supported_rows=len(values),
            mean_normalized_valid_rmse=sum(x['normalized_valid_rmse'] for x in values) / len(values)
                if values else None)
    publish(args.output / 'result.json', dict(complete=True, n_original_train_rows=len(rows),
        n_full_execution_windows=sum(row['full_execution_window'] for row in rows),
        valid_executed_target_steps=sum(row['valid_prefix_steps'] for row in rows),
        summary=summary, rows=rows, source_sha256=SOURCE_SHA, input_sha256=INPUT_SHA,
        commit=commit, no_new_training_data=True, no_policy_or_success_rate_claim=True,
        limitations=['Ten original train inputs: engineering gate, not representative method efficacy.',
                     'Errors are normalized, not physical joint/base errors.',
                     'Partial end-of-trajectory windows score only real, unpadded prefix targets.',
                     'Direct16 shape support does not establish temporal/deployment compatibility.',
                     'Hold padding is an experimental codec representation, not real future actions.']))
    print(json.dumps({'complete': True, 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
