"""Short, disposable 8-A800 full-gradient timing from verified champion weights.

Uses the pinned author's actual init_train_state/train_step, FM+FAST+stage
losses and AdamW/EMA. NO trained checkpoint is saved. A cached real mixed-task
batch is a compute measurement, not full-loader throughput or learning evidence.
"""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('reference','layout','patch','checkpoint','verified','probe','output'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--batch-size', type=int, choices=(32, 64, 128, 256), required=True)
    ap.add_argument('--flow-samples', type=int, choices=(1, 4, 15), default=4)
    ap.add_argument('--freeze-vision', action='store_true')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--measure', type=int, default=5)
    args = ap.parse_args()
    if args.output.exists() or args.warmup < 1 or not 1 <= args.measure <= 20:
        raise ValueError('Fresh output and an explicitly short timing window required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('This timing is explicitly single-node eight-GPU')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('All eight GPUs must be verified idle; never interrupt a teammate')
    args.output.mkdir(parents=True)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.reference, text=True).strip()
    delta = subprocess.check_output(['git','diff'], cwd=args.reference)
    if source != 'ca556f74a455cef7987a2be4537b5ac85cc56dd7' or delta != args.patch.read_bytes():
        raise ValueError('Unregistered champion source/patch')
    verified = json.loads(args.verified.read_text())
    if verified['status'] != 'all_selected_assets_verified' or verified['revision'] != '114902410a6439ad1a0b58a112b84159d3ab3f65':
        raise ValueError('Use the verified shared50 parent, not a final specialist silently')
    # Each immutable verified file must still have the same size. The separate
    # verifier hashes ALL files on this explicit root immediately before use.
    for row in verified['files']:
        path = args.checkpoint/row['path']
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError('Changed verified asset: '+str(path))
    sys.path.insert(0, str(args.reference/'src'))
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    import openpi.models.model as model_lib
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.shared.image_tools as image_tools
    import openpi.training.sharding as sharding
    from b1k.training import config as config_lib
    from b1k.shared import normalize
    from b1k.transforms_normalize import NormalizeWithPerTimestamp
    from b1k.transforms import TokenizeFASTActions
    from b1k.models.observation import Observation
    from behavior_champion.contracts import state23_from_proprio61, delta_actions, pad32, training_stage, append_embeddings
    if len(jax.devices()) != 8 or any(device.platform != 'gpu' for device in jax.devices()):
        raise ValueError('Expected eight actual CUDA devices')
    jax.config.update('jax_compilation_cache_dir', str(args.output/'jax-cache'))
    layout = json.loads(args.layout.read_text())['layout']
    assets = args.checkpoint/'assets/IliaLarchenko/behavior_224_rgb'
    tokenizer_file = assets/'fast_tokenizer/processing_action_tokenizer.py'
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != '6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20':
        raise ValueError('Unreviewed downloaded tokenizer code')
    stats = normalize.load(assets)
    normalizer = NormalizeWithPerTimestamp(stats, use_per_timestamp=True)
    tokenizer = TokenizeFASTActions(str(assets/'fast_tokenizer'), [(0,6),(7,23)], 200,
                                   norm_stats=stats, use_per_timestamp=True)
    native = np.load(args.probe/'native.npz', allow_pickle=False)
    probe_receipt = json.loads((args.probe/'result.json').read_text())
    with (args.probe/'native.npz').open('rb') as stream:
        h = hashlib.sha256()
        while block := stream.read(8*1024*1024): h.update(block)
    if h.hexdigest() != probe_receipt['native_npz_sha256'] or len(native['task_id']) != 100:
        raise ValueError('Use the exact real 100-TRAIN-task probe')
    examples = []
    for i, official_task in enumerate(native['task_id']):
        task = layout['dataset_to_model'][str(int(official_task))]
        state = state23_from_proprio61(native['state61'][i])
        row = normalizer(dict(state=state, actions=delta_actions(native['actions23'][i],state)))
        row = tokenizer(row)
        row['state'], row['actions'] = pad32(row['state']), pad32(row['actions'])
        row['image'] = {key: np.asarray(image_tools.resize_with_pad(native[key][i],224,224))
                        for key in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
        row['image_mask'] = {key: np.bool_(True) for key in row['image']}
        row['tokenized_prompt'] = np.array([task, training_stage(int(native['frame'][i]),int(native['length'][i]),layout['num_stages'][task])],dtype=np.int32)
        row['tokenized_prompt_mask'] = np.ones(2,dtype=bool)
        examples.append(row)
    batch = jax.tree.map(lambda *xs: np.stack(xs), *[examples[i%100] for i in range(args.batch_size)])
    observation = Observation.from_dict({k:v for k,v in batch.items() if k!='actions'})
    import_spec = importlib.util.spec_from_file_location('champion_train_core',args.reference/'scripts/train.py')
    core = importlib.util.module_from_spec(import_spec); import_spec.loader.exec_module(core)
    loader_receipt = {}

    class ExpandedLoader:
        def load(self, expected):
            loaded = flatten_dict(model_lib.restore_params(args.checkpoint/'params', restore_type=np.ndarray))
            shapes = flatten_dict(expected)
            if loaded.keys() != shapes.keys():
                raise ValueError('Checkpoint key coverage differs from actual constructed graph')
            grown = {}
            for key, old_rows in ((('task_embeddings','embedding'),50),(('task_stage_embeddings','embedding'),596)):
                old = loaded[key]
                shape = shapes[key].shape
                initializer = nnx.Embed(num_embeddings=shape[0],features=shape[1],rngs=nnx.Rngs(17+old_rows))
                fresh = np.asarray(initializer.embedding.value).astype(old.dtype)
                new = append_embeddings(old, fresh, old_rows)
                if new[:old_rows].tobytes() != old.tobytes():
                    raise ValueError('Changed pretrained embedding rows')
                grown['/'.join(key)] = dict(old=list(old.shape),new=list(new.shape),old_rows_byte_identical=True)
                loaded[key] = new
            for key, value in loaded.items():
                if value.shape != shapes[key].shape:
                    raise ValueError('Non-embedding shape mismatch: '+str(key))
            loader_receipt.update(dict(keys=len(loaded),expanded=grown,other_arrays_unchanged=True))
            return unflatten_dict(loaded)

    original = config_lib.get_config('pi_behavior_b1k_fast')
    config = dataclasses.replace(original,
        model=dataclasses.replace(original.model,num_tasks=100,task_num_stages=tuple(layout['num_stages'])),
        weight_loader=ExpandedLoader(),batch_size=args.batch_size,num_flow_samples=args.flow_samples,
        fsdp_devices=8,wandb_enabled=False,
        freeze_filter=nnx_utils.PathRegex('PaliGemma/img/.*') if args.freeze_vision else nnx.Nothing)
    mesh = sharding.make_mesh(8)
    data_sharding = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    batch = jax.device_put((observation,batch['actions']),data_sharding)
    with sharding.set_mesh(mesh):
        state, state_sharding = core.init_train_state(config,jax.random.key(17),mesh,resume=False,norm_stats=stats)
        jax.block_until_ready(state)
        step = jax.jit(functools.partial(core.train_step,config),
                       in_shardings=(replicated,state_sharding,data_sharding),
                       out_shardings=(state_sharding,replicated),donate_argnums=(1,))
        times = []
        for index in range(args.warmup+args.measure):
            started=time.monotonic()
            state, metrics=step(jax.random.key(19),state,batch)
            jax.block_until_ready((state,metrics))
            seconds=time.monotonic()-started
            if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics)):
                raise ValueError('Nonfinite training metrics')
            row=dict(step=index+1,seconds=seconds,warmup=index<args.warmup,
                     loss=float(metrics['loss']),grad_norm=float(metrics['grad_norm']))
            with (args.output/'steps.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if index>=args.warmup:times.append(seconds)
    result=dict(status='short_disposable_compute_timing_completed',batch_size=args.batch_size,
                flow_samples=args.flow_samples,true_vision_freeze=args.freeze_vision,
                seconds_per_step_median=float(np.median(times)),examples_per_second=args.batch_size/float(np.median(times)),
                warmup_steps=args.warmup,timed_steps=args.measure,optimizer_updates=int(state.step),
                trained_checkpoint_saved=False,source_commit=source,patch_sha256=hashlib.sha256(delta).hexdigest(),
                parent_revision=verified['revision'],probe_sha256=h.hexdigest(),checkpoint_migration=loader_receipt,
                warning='Cached real batch; no epoch ETA until shuffled full-loader/checkpoint/eval overhead is measured.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
