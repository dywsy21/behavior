"""Frozen NumPy/msgpack wire protocol for both simulator and model services.

Same protocol as g05.utils.websocket.msgpack; intentionally standalone since
the simulator Python need not import the VLA package. No live tools-path import.
"""
from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def _pack(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return dict(__ndarray__=True, data=obj.tobytes(), dtype=obj.dtype.str, shape=obj.shape)
    if isinstance(obj, np.generic):
        return dict(__npgeneric__=True, data=obj.item(), dtype=obj.dtype.str)
    return obj


def _bytes_to_str_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _bytes_to_str_keys(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bytes_to_str_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bytes_to_str_keys(v) for v in obj)
    return obj


def _unpack(obj):
    obj = _bytes_to_str_keys(obj)
    if '__ndarray__' in obj:
        return np.ndarray(buffer=obj['data'], dtype=np.dtype(obj['dtype']), shape=tuple(obj['shape']))
    if '__npgeneric__' in obj:
        return np.dtype(obj['dtype']).type(obj['data'])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)
