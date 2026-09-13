"""Small, tied input/output deltas for actual codec markers, not whole vocab.

The original projections keep their parameter names and frozen shared weight.
The only new parameter is [number_of_required_markers, hidden_size]. Eager CE
must call decode(): fused linear CE reads the original weight directly and is
therefore NOT compatible with this adapter.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def required_marker_ids(tokenizer):
    serializer = tokenizer.serializer
    names = [f"<{key}_{level}>" if serializer.max_residuals > 1 else f"<{key}>"
             for level in range(serializer.num_residuals) for key in serializer.nn_key_names]
    names += [f"<{key}>" for key in serializer.rule_key_names]
    offset, end = int(tokenizer.action_token_begin_idx), int(tokenizer.action_token_end_idx)
    ids = [offset + int(serializer.group_marker_action_indices[name]) for name in names]
    if (not ids or len(set(ids)) != len(ids)
            or any(not offset + tokenizer._codebook_size <= token < end for token in ids)):
        raise ValueError("Required codec marker IDs must be unique and outside codebook payload")
    return tuple(sorted(ids))


class ActionMarkerRows(nn.Module):
    def __init__(self, token_ids, *, vocab_size, hidden_size, device, dtype):
        super().__init__()
        ids = tuple(token_ids)
        if (not ids or any(type(token) is not int for token in ids)
                or tuple(sorted(set(ids))) != ids or ids[0] < 0 or ids[-1] >= vocab_size
                or hidden_size < 1):
            raise ValueError("Need sorted unique in-vocabulary marker IDs and a positive hidden size")
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long, device=device))
        self.delta = nn.Parameter(torch.zeros(len(ids), hidden_size, device=device, dtype=dtype))

    def embed_delta(self, ids):
        # At most eight comparisons per position in the declared R1Pro recipe.
        matches = ids[..., None] == self.token_ids
        selected = matches.any(dim=-1)
        positions = matches.to(torch.int64).argmax(dim=-1)
        return F.embedding(positions, self.delta) * selected[..., None]


class MarkerEmbedding(nn.Embedding):
    def __init__(self, original, adapter):
        nn.Module.__init__(self)
        if (type(original) is not nn.Embedding or original.max_norm is not None
                or original.scale_grad_by_freq or original.sparse
                or original.weight.requires_grad):
            raise ValueError("Expected a frozen plain embedding without renormalization")
        self.num_embeddings, self.embedding_dim = original.weight.shape
        self.padding_idx = original.padding_idx
        self.max_norm, self.norm_type = None, original.norm_type
        self.scale_grad_by_freq, self.sparse = False, False
        self.weight = original.weight
        # Register the shared adapter ONCE on the model, not under each head.
        object.__setattr__(self, "_marker_rows", adapter)

    def forward(self, ids):
        original = F.embedding(ids, self.weight, self.padding_idx)
        return original + self._marker_rows.embed_delta(ids).to(original.dtype)


class MarkerOutput(nn.Linear):
    def __init__(self, original, adapter):
        nn.Module.__init__(self)
        if type(original) is not nn.Linear or original.bias is not None or original.weight.requires_grad:
            raise ValueError("Expected a frozen bias-free plain vocabulary projection")
        self.out_features, self.in_features = original.weight.shape
        self.weight = original.weight
        self.register_parameter("bias", None)
        object.__setattr__(self, "_marker_rows", adapter)

    def forward(self, hidden):
        logits = F.linear(hidden, self.weight)
        extra = F.linear(hidden, self._marker_rows.delta)
        return logits.index_add(-1, self._marker_rows.token_ids, extra.to(logits.dtype))


def install_marker_rows(vlm, token_ids):
    if (not isinstance(vlm.input_proj, nn.Embedding) or not isinstance(vlm.output_proj, nn.Linear)
            or vlm.input_proj.weight is not vlm.output_proj.weight
            or vlm.input_proj.weight.is_meta):
        raise ValueError("Marker adaptation requires loaded, truly tied input/output weights")
    weight = vlm.input_proj.weight
    # Freeze first; never allocate Adam or gradients for the entire vocabulary.
    weight.requires_grad_(False)
    adapter = ActionMarkerRows(token_ids, vocab_size=weight.shape[0], hidden_size=weight.shape[1],
                               device=weight.device, dtype=weight.dtype)
    embedding, output = MarkerEmbedding(vlm.input_proj, adapter), MarkerOutput(vlm.output_proj, adapter)
    vlm.input_proj, vlm.output_proj = embedding, output
    return adapter


def assert_marker_bindings(vlm, adapter):
    if (not isinstance(vlm.input_proj, MarkerEmbedding) or not isinstance(vlm.output_proj, MarkerOutput)
            or vlm.input_proj._marker_rows is not adapter or vlm.output_proj._marker_rows is not adapter
            or vlm.input_proj.weight is not vlm.output_proj.weight
            or vlm.input_proj.weight.requires_grad
            or adapter.delta.shape != (adapter.token_ids.numel(), vlm.input_proj.embedding_dim)
            or adapter.delta.device != vlm.input_proj.weight.device):
        raise RuntimeError("Marker adapter binding, shape, device, or frozen weight contract changed")


def restore_marker_rows(adapter, state, prefix="model.action_marker_rows."):
    if any("action_marker_rows." in key and not key.startswith(prefix) for key in state):
        raise RuntimeError("Unexpected marker adapter namespace; refusing silent zero initialization")
    available = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not available:
        if torch.count_nonzero(adapter.delta):
            raise RuntimeError("Fresh marker adapter must be zero initialized")
        return dict(mode="zero_init", restored=0, token_ids=adapter.token_ids.cpu().tolist())
    if (set(available) != {"token_ids", "delta"}
            or available["token_ids"].dtype != torch.long
            or not torch.equal(available["token_ids"].cpu(), adapter.token_ids.cpu())
            or available["delta"].shape != adapter.delta.shape
            or available["delta"].dtype != adapter.delta.dtype
            or not torch.isfinite(available["delta"]).all()):
        raise RuntimeError("Partial, reordered, or incompatible marker adapter checkpoint")
    adapter.load_state_dict(available, strict=True)
    return dict(mode="resume", restored=2, token_ids=adapter.token_ids.cpu().tolist())
