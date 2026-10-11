import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('JAX_PLATFORMS') == 'cpu', 'Explicit isolated CPU JAX test required')
class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        if len(jax.devices()) != 8:
            raise unittest.SkipTest('Run with XLA_FLAGS=--xla_force_host_platform_device_count=8')
        cls.jax = jax
        cls.mesh = jax.make_mesh((1,8), ('batch','fsdp'))
        cls.sharding = jax.sharding.NamedSharding(cls.mesh, jax.sharding.PartitionSpec(('batch','fsdp')))

    def test_explicit_host_cast_then_exact_shard_equality(self):
        from behavior_champion.jax_transport import place_from_host
        x = np.arange(32, dtype=np.float64)/3
        actual = place_from_host(x,self.sharding)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(np.asarray(actual),x.astype(np.float32))

    def test_device_origin_is_rejected(self):
        from behavior_champion.jax_transport import place_from_host
        x = self.jax.device_put(np.arange(16,dtype=np.float32))
        with self.assertRaises(TypeError): place_from_host(x,self.sharding)

    def test_fsdp_loss_gradient_and_update_against_numpy(self):
        from behavior_champion.jax_transport import check_collective_and_gradient
        receipt = check_collective_and_gradient(self.mesh)
        self.assertTrue(receipt['update_checked'])
        self.assertLess(receipt['fsdp_gradient_max_abs_error'],1e-7)


if __name__ == '__main__': unittest.main()
