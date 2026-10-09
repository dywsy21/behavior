"""Bounded ONE-GPU parent/gradient smoke using accepted original expert rows.

No optimizer, SFT run, new labels, rollout, or shared environment modification.
Proves exact SFT parent restoration and the new trainability profile only.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_corpus import file_sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--component', choices=('H0', 'L0'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Use a clean frozen Git worktree')
    started = time.monotonic()
    recipe = json.loads(args.recipe.read_text())
    root = Path(recipe['root'])
    if recipe['preferred_node'] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise ValueError('Wrong A800 node')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Diagnostic is limited to the verified idle lc1 GPU2')
    processes = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('GPU2 is occupied; do not interrupt or compete with another job')
    branch = 'high' if args.component == 'H0' else 'low'
    parent = recipe['parents'][branch]
    path = root / parent['path']
    if file_sha(path) != parent['sha256']:
        raise ValueError('Wrong SFT parent; old B1500/A42500 must not silently initialize recovery')
    release = root / recipe['expert_release']
    accepted = json.loads((release / 'acceptance.json').read_text())
    if (accepted['status'] != 'ACCEPTED' or not all(accepted['gates'].values())
            or file_sha(release / 'manifest.json') != accepted['manifest_sha256']):
        raise ValueError('Original expert release no longer accepted')
    import numpy as np
    import torch
    from g05.data.memlite_stage1_dataset import Stage1Dataset, collate_stage1, to_device
    from g05.utils.training.stage1_model import configuration, restore_model, make_processor
    from g05.models.g05.qwen35 import vision
    # Match the already tested Stage-1/RL SDPA path, no package installation.
    vision._flash_attn_varlen = None
    vision._flash_attn_backend = None
    torch.set_num_threads(2)
    torch.manual_seed(recipe['seed'])
    np.random.seed(recipe['seed'])
    names = {}
    with (release / 'episodes.jsonl').open() as stream:
        for line in stream:
            ep = json.loads(line)
            names[ep['row']['task_index']] = ep['task_name']
    config = configuration(root, branch, names)
    config.update(initial_weights=str(path), initial_weights_sha256=parent['sha256'])
    if branch == 'low' and file_sha(config['stats_path']) != recipe['stats_sha256']:
        raise ValueError('Changed low action normalizer')
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if saved['state']['step'] != parent['step']:
        raise ValueError('Unexpected SFT checkpoint step')
    model, restoration = restore_model(config, branch, state=saved['model_state_dict'])
    del saved
    gc.collect()
    print(json.dumps(dict(phase='exact_parent_restored', component=args.component, **restoration)), flush=True)
    dataset = Stage1Dataset(release, config, branch, 'train')
    # Two different tasks, fixed before loading, not sampled by model loss.
    indices = [int(np.flatnonzero(dataset.task_ids == task)[0]) for task in sorted(names)[:2]]
    if branch == 'low':
        trainability = model.configure_recovery_expert_only()
        if {k:len(v) for k,v in model.coordination_trainable_parameter_groups().items()} != {'action_expert':322}:
            raise ValueError('L0 must only train the 322 existing action-expert tensors')
        model.cuda().train()
        batch = collate_stage1([dataset[i] for i in indices])
        identity = batch['source_identity']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = model(to_device(batch, 'cuda'))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite true FM loss')
        loss.backward()
        gradients = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
        frozen_grads = sum(p.grad is not None for p in model.parameters() if not p.requires_grad)
        if gradients != 322 or frozen_grads or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Wrong/nonfinite L0 gradient coverage')
        details = dict(loss=float(loss), action_shape=list(batch['action'].shape), source=identity,
            trainable_gradients=gradients, frozen_gradients=frozen_grads, trainability=trainability)
    else:
        from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
        from g05.utils.memlite_skill_protocol import canonical_json
        from g05.utils.memlite_causal_feedback import single_frame_member_prefix
        model.requires_grad_(False).eval().cuda()
        processor = make_processor(config, False)
        contexts, states, identity = [], [], []
        for index in indices:
            raw, ident = dataset.raw(index)
            raw.pop('action', None)
            raw.pop('action_is_pad', None)
            label = dict(raw['model_projection'])
            # Probe a member-conditioned observable prefix. This is not a
            # claim that that expert skill was already executed successfully.
            label['memory'] = canonical_json(dict(task_name=label['task_name'], issued_command_history=[], verified_world_facts=[]))
            prepared = processor._process_tensors(raw)
            sample = single_frame_member_prefix(processor.samples_builder, prepared,
                task_name=label['task_name'], parent_goal=label['parent_goal'],
                issued_bundle=label['active_skills_semantic_json'],member_index=0,
                memory=label['memory'],served_controls=0)
            pixels = {k:v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                context = model.outcome_context_from_prefix([sample], pixels)
            contexts.append(context.detach().float())
            states.append(sample['proprio']['value'].reshape(1, 27).cuda().float())
            identity.append(ident)
        context = torch.cat(contexts)[:,None,:]
        state = torch.cat(states)[:,None,:]
        observer = TemporalOutcomeObserver(context.shape[-1]).cuda()
        # Random, explicitly synthetic probe objective is NEVER saved as a
        # physical label or a trained checkpoint; no optimizer is constructed.
        logits = observer(context, state, torch.zeros(2,1,device='cuda',dtype=torch.long),
                          torch.ones(2,1,device='cuda',dtype=torch.bool))
        loss = logits.square().mean()
        loss.backward()
        if any(p.grad is not None for p in model.parameters()) or not torch.isfinite(loss):
            raise ValueError('H0 gradient reached the frozen high model or was nonfinite')
        details = dict(synthetic_graph_objective=float(loss), context_shape=list(context.shape), source=identity,
            observer_gradient_tensors=sum(p.grad is not None for p in observer.parameters()), high_frozen_gradients=0,
            result_scope='graph only; no physical target and no trained/calibrated observer')
    torch.cuda.synchronize()
    receipt = dict(schema='recovery_parent_graph_smoke_v1', component=args.component,
        parent=parent, recipe_sha256=file_sha(args.recipe), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        optimizer_steps=0, seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        exact_restore=restoration, **details)
    args.output.mkdir(parents=True)
    (args.output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
