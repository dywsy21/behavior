"""CPU abstract model construction: verify ONLY the two embedding tables grow.

No checkpoint is read and no parameter is updated. This is a structural gate,
not proof of successful checkpoint migration or training throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--layout', type=Path, required=True)
    ap.add_argument('--patch', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('Explicit CPU-only JAX execution required')
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.reference, text=True).strip()
    if source != 'ca556f74a455cef7987a2be4537b5ac85cc56dd7':
        raise ValueError('Wrong champion revision')
    delta = subprocess.check_output(['git','diff'], cwd=args.reference)
    if delta != args.patch.read_bytes():
        raise ValueError('Unregistered patch or extra tracked changes')
    sys.path.insert(0, str(args.reference/'src'))
    import jax
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    from b1k.models.pi_behavior_config import PiBehaviorConfig
    from b1k.models.pi_behavior import PiBehavior
    layout = json.loads(args.layout.read_text())['layout']
    legacy = PiBehaviorConfig(use_fast_auxiliary=True, max_fast_tokens=200,
                              fast_loss_weight=.05, use_knowledge_insulation=False)
    expanded = PiBehaviorConfig(num_tasks=100, task_num_stages=tuple(layout['num_stages']),
                                use_fast_auxiliary=True, max_fast_tokens=200,
                                fast_loss_weight=.05, use_knowledge_insulation=False)
    trees = {}
    for name, cfg in (('legacy', legacy), ('expanded', expanded)):
        model = nnx.eval_shape(cfg.create, jax.random.key(17))
        trees[name] = flatten_dict(nnx.state(model).to_pure_dict())
        print(json.dumps(dict(model=name, leaves=len(trees[name]))), flush=True)
    if trees['legacy'].keys() != trees['expanded'].keys():
        raise ValueError('Changed parameter-key set')
    differences = {}
    for key, value in trees['legacy'].items():
        other = trees['expanded'][key]
        if value.dtype != other.dtype:
            raise ValueError('Changed dtype: '+str(key))
        if value.shape != other.shape:
            differences['/'.join(key)] = dict(old=list(value.shape), new=list(other.shape))
    wanted = {
        'task_embeddings/embedding': dict(old=[50, 2048], new=[100, 2048]),
        'task_stage_embeddings/embedding': dict(old=[596, 1024], new=[sum(layout['num_stages']), 1024]),
    }
    if differences != wanted:
        raise ValueError('Unexpected model changes: '+str(differences))
    result = dict(status='abstract_50_to_100_model_shape_gate_passed', differences=differences,
                  source_commit=source, patch_sha256=hashlib.sha256(delta).hexdigest(),
                  layout_sha256=hashlib.sha256(args.layout.read_bytes()).hexdigest(),
                  model_import=str(Path(sys.modules[PiBehavior.__module__].__file__).resolve()),
                  jax_version=jax.__version__, model_leaf_count=len(trees['legacy']),
                  weights_loaded=False, training=False, gpu_used=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
