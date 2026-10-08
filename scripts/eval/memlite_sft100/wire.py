"""Exact official byte-key msgpack ndarray representation, without Isaac imports."""
import functools
import msgpack
import numpy as np


def encode(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ('V','O','c'):
        raise ValueError('Object/void/complex arrays are not allowed on the wire')
    if isinstance(value, np.ndarray):
        return {b'__ndarray__': True, b'data': value.tobytes(),
                b'dtype': value.dtype.str, b'shape': value.shape}
    if isinstance(value, np.generic):
        return {b'__npgeneric__': True, b'data': value.item(), b'dtype': value.dtype.str}
    raise TypeError(type(value).__name__)


def decode(value):
    if b'__ndarray__' in value:
        dtype = np.dtype(value[b'dtype'])
        if dtype.kind in ('V','O','c'): raise ValueError('Unsupported wire array dtype')
        return np.ndarray(buffer=value[b'data'], dtype=dtype, shape=value[b'shape'])
    if b'__npgeneric__' in value:
        return np.dtype(value[b'dtype']).type(value[b'data'])
    return value


packb = functools.partial(msgpack.packb, default=encode)
unpackb = functools.partial(msgpack.unpackb, object_hook=decode)
