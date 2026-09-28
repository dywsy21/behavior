"""Preserve the native inferencer's CPU-only action postprocessing boundary."""
import torch


def float32_prefix(kv):
    from g05.models.kv_cache import SparseKVCache
    if kv.recurrent_states or kv.conv_states:
        raise ValueError('Expected cross-attention-only AE')
    return SparseKVCache({k:v.float() for k,v in kv.key_cache.items()},
                         {k:v.float() for k,v in kv.value_cache.items()})


def cpu_postprocess_inputs(action, batch):
    # Do not copy vision tensors/KV caches: the inverse transform needs only
    # these fields, and the processor's normalizer statistics live on CPU.
    result = {'action': action, 'proprio': batch['proprio'], 'selected_action_source': 'fm'}
    for key in ('action_dim_is_pad', 'proprio_dim_is_pad', 'action_op_mask'):
        if key in batch:
            result[key] = batch[key]
    return {key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in result.items()}
