"""Keep the cached probe's uint8 resize/rounding with raw sharded RGB inputs."""


def resize_normalized_rgb(image, size=224):
    import jax.numpy as jnp
    import numpy as np
    from openpi.shared.image_tools import resize_with_pad
    # Observation.from_dict normalized uint8 before placement. Reconstruct the
    # exact pixels, resize+round identically to the original cached transform,
    # then normalize. This executes under the eight-GPU training jit, never as
    # a GPU0 operation followed by an unsafe peer scatter.
    pixels = jnp.round((image + 1.) * 127.5).clip(0,255).astype(jnp.uint8)
    resized = resize_with_pad(pixels,size,size)
    # A constant LUT also preserves the old NumPy normalization bits instead
    # of introducing an FMA rounding difference in the compiled frontend.
    lut = np.arange(256,dtype=np.float32)/255.*2.-1.
    return jnp.asarray(lut)[resized.astype(jnp.int32)]
