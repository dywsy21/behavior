import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('JAX_PLATFORMS')=='cpu','Explicit isolated CPU JAX test required')
class FrontendTests(unittest.TestCase):
    def test_native_rgb_matches_cached_uint8_resize_and_normalization(self):
        import jax
        from behavior_champion.frontend import resize_normalized_rgb, resize_uint8_rgb
        from openpi.shared.image_tools import resize_with_pad
        rng = np.random.default_rng(17)
        for height,width in ((720,720),(480,480),(480,640)):
            pixels=rng.integers(0,256,(1,height,width,3),dtype=np.uint8)
            expected=np.asarray(resize_with_pad(pixels,224,224)).astype(np.float32)/255.*2.-1.
            actual=np.asarray(jax.jit(resize_normalized_rgb)(pixels.astype(np.float32)/255.*2.-1.))
            np.testing.assert_array_equal(actual,expected)
            compact=np.asarray(jax.jit(resize_uint8_rgb)(pixels))
            np.testing.assert_array_equal(compact,expected)

    def test_compact_rgb_rejects_already_normalized_pixels(self):
        import jax
        from behavior_champion.frontend import resize_uint8_rgb
        with self.assertRaises(TypeError):
            jax.jit(resize_uint8_rgb)(np.zeros((1,480,480,3),np.float32))


if __name__=='__main__': unittest.main()
