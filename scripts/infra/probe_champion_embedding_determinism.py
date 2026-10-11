"""Repeated embedding-gradient microkernel, no policy/data/checkpoint training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('Fresh output and explicit eight-card test required')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Never overlap another GPU job')
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
    from behavior_champion.jax_transport import place_from_host
    if len(jax.devices()) != 8:
        raise ValueError('Eight real devices required')
    mesh = Mesh(np.asarray(jax.devices()), ('gpu',))
    weight_sharding, data_sharding = NS(mesh, P(None, 'gpu')), NS(mesh, P('gpu'))
    rng = np.random.default_rng(17)
    weights = place_from_host(np.zeros((1024, 128), np.float32), weight_sharding)
    indices_host = rng.integers(0, 1024, 32768, dtype=np.int32)
    coefficients_host = rng.normal(size=(32768, 128)).astype(np.float32)
    indices = place_from_host(indices_host, data_sharding)
    coefficients = place_from_host(coefficients_host, data_sharding)
    def loss(table, ids, vectors):
        return jnp.sum(table[ids] * vectors)
    gradient = jax.jit(jax.grad(loss), in_shardings=(weight_sharding, data_sharding, data_sharding),
                       out_shardings=weight_sharding)
    started = time.monotonic(); values = []
    for _ in range(32):
        current = np.asarray(gradient(weights, indices, coefficients))
        if not np.isfinite(current).all():
            raise ValueError('Nonfinite gradient')
        values.append(current)
    reference = np.zeros((1024, 128), np.float64)
    np.add.at(reference, indices_host, coefficients_host.astype(np.float64))
    row = dict(status='repeated_embedding_gradient_measured', policy_optimizer_updates=0,
        repetitions=32, identical_to_first=sum(np.array_equal(values[0], value) for value in values),
        unique_gradient_sha256=len({hashlib.sha256(value.tobytes()).hexdigest() for value in values}),
        largest_repeat_abs_delta=max(float(np.abs(value-values[0]).max()) for value in values),
        max_abs_error_against_fp64=float(max(np.abs(value-reference).max() for value in values)),
        seconds=time.monotonic()-started, xla_flags=os.environ.get('XLA_FLAGS'),
        jax_version=jax.__version__, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__': main()
