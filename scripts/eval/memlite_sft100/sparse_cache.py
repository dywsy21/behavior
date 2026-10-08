"""Equivalent finished-row cache copies without repeated CUDA nonzero calls."""
from contextlib import contextmanager


@contextmanager
def indexed_cache_freeze(helper, cache_type):
    """Use one index vector per token instead of one Boolean gather per layer.

    A finished high-level row must not advance its GatedDeltaNet recurrent or
    convolution state while other rows continue generating. Preserve that rule;
    only change how the same rows are copied and restored.
    """
    names=('_snapshot_sparse_states','_restore_sparse_states')
    previous={name:(name in helper.__dict__,helper.__dict__.get(name)) for name in names}

    def snapshot(cache, frozen_rows):
        if not isinstance(cache,cache_type):
            return None
        indices=frozen_rows.nonzero(as_tuple=False).flatten()
        if indices.numel()==0:
            return None
        saved={attr:{layer:state.index_select(0,indices).clone()
                     for layer,state in getattr(cache,attr).items() if state is not None}
               for attr in ('conv_states','recurrent_states')}
        return indices,saved

    def restore(cache, saved):
        if saved is None:
            return
        if not isinstance(cache,cache_type):
            raise TypeError('Sparse cache required')
        indices,states=saved
        for attr,by_layer in states.items():
            for layer,value in by_layer.items():
                state=getattr(cache,attr).get(layer)
                if state is not None:
                    state.index_copy_(0,indices,value)

    helper._snapshot_sparse_states=snapshot
    helper._restore_sparse_states=restore
    try:
        yield
    finally:
        for name,(had,value) in previous.items():
            if had:setattr(helper,name,value)
            else:delattr(helper,name)
