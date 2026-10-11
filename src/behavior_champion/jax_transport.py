"""Fail-closed host-origin placement for lc1's unreliable direct GPU copies.

NCCL_P2P_DISABLE does not control PJRT's own device-to-device copies. Never
create an unsharded RNG/data array on GPU0 and implicitly scatter it here.
Internal FSDP collectives use the independently checked NCCL transport.
"""
import numpy as np


def place_from_host(tree, sharding):
    import jax
    leaves = jax.tree.leaves(tree)
    if any(isinstance(x, jax.Array) for x in leaves):
        raise TypeError('Implicit GPU-to-GPU placement is forbidden on this cluster')
    placed = jax.device_put(tree, sharding)
    # Verify actual physical shards rather than one convenient replica only.
    for before, after in zip(leaves, jax.tree.leaves(placed), strict=True):
        before = np.asarray(before)
        for shard in after.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), before[shard.index])
    return placed


def check_collective_and_gradient(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding as NS, PartitionSpec as P
    replicate = NS(mesh, P())
    data = NS(mesh, P(('batch','fsdp')))
    weights = NS(mesh, P('fsdp', None))
    host = np.arange(8, dtype=np.float32)
    summed = jax.pmap(lambda x: jax.lax.psum(x, 'device'), axis_name='device')(host)
    np.testing.assert_array_equal(np.asarray(summed), np.full(8,28,dtype=np.float32))
    # MSE has an independent NumPy closed-form gradient; check actual FSDP
    # collectives, not only an isolated GPU kernel or all-reduce of a scalar.
    rng = np.random.default_rng(17)
    x = rng.normal(size=(256,128)).astype(np.float32) / 8
    w = rng.normal(size=(128,64)).astype(np.float32) / 8
    target = rng.normal(size=(256,64)).astype(np.float32) / 8
    expected_error = x @ w - target
    expected_loss = np.mean(expected_error**2)
    expected_grad = 2 * x.T @ expected_error / expected_error.size
    def loss(matrix, inputs, targets):
        return jnp.mean((jnp.matmul(inputs,matrix,precision=jax.lax.Precision.HIGHEST)-targets)**2)
    fn = jax.jit(jax.value_and_grad(loss), in_shardings=(weights,data,data),
                 out_shardings=(replicate,weights))
    actual_loss, grad = fn(place_from_host(w,weights),place_from_host(x,data),place_from_host(target,data))
    np.testing.assert_allclose(np.asarray(actual_loss),expected_loss,rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(np.asarray(grad),expected_grad,rtol=1e-4,atol=1e-7)
    updated = jax.jit(lambda a,b:a-.01*b, in_shardings=(weights,weights),out_shardings=weights)(
        place_from_host(w,weights),grad)
    np.testing.assert_allclose(np.asarray(updated),w-.01*expected_grad,rtol=1e-5,atol=1e-7)
    return dict(host_origin_psum=28,fsdp_gradient_max_abs_error=float(np.max(np.abs(np.asarray(grad)-expected_grad))),
                fsdp_loss=float(actual_loss), numpy_loss=float(expected_loss),update_checked=True,
                restriction='No implicit GPU0-origin resharding; direct D2D transport remains unsafe')
