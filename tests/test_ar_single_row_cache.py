"""Single-row cache fast path must preserve stop and multi-row semantics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

target = Path(__file__).with_name("ar_helper.py")
if target.exists():
    spec = importlib.util.spec_from_file_location("g05.models.g05.helpers.ar_helper_candidate", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    import g05.models.g05.helpers.ar_helper as module
ARHelper = module.ARHelper
Cache = module._SKV

def decode(tokens, max_new_tokens=None, forbid_snapshot=False):
    batch = len(tokens[0]) if tokens else 1
    helper = object.__new__(ARHelper)
    sequence = iter(tokens)
    helper._sample = lambda logits, **kwargs: torch.tensor(next(sequence), dtype=torch.long)
    helper._assign_token_index = lambda ids: torch.full_like(ids, 6)
    if forbid_snapshot:
        helper._snapshot_sparse_states = lambda *a: pytest.fail("single-row snapshot must not be called")
    cache = Cache(key_cache={0: torch.zeros(batch, 1, 2, 1)}, value_cache={0: torch.zeros(batch, 1, 2, 1)})
    cache.conv_states[1] = torch.zeros(batch, 2, 1)
    cache.recurrent_states[1] = torch.zeros(batch, 1, 2, 2)
    class VLM:
        def decode(self, hidden):
            return torch.zeros(batch, 1, 8)
        def embed(self, token):
            return torch.zeros(batch, 1, 3)
        def __call__(self, **kwargs):
            cache.conv_states[1].add_(1)
            cache.recurrent_states[1].add_(1)
            cache.update(torch.zeros(batch, 1, 1, 1), torch.zeros(batch, 1, 1, 1), 0)
            return cache.conv_states[1][:, :1, :].expand(batch, 1, 3).clone(), cache
    model = SimpleNamespace(cfg=SimpleNamespace(eos_token_id=0), vlm=VLM(),
        attn_implementation="eager", build_causal_mask_and_position_ids=lambda *a, **kw: (None, None))
    result = helper._infer_batched(model, torch.zeros(batch, 3), torch.ones(batch, 2),
        {"head": torch.zeros(batch, 1)}, len(tokens) if max_new_tokens is None else max_new_tokens,
        return_kv_cache=True, past_key_values=cache, stop_token_ids=[7])
    return result, cache

@pytest.mark.parametrize("stop", [0, 7])
def test_single_row_stops_and_skips_empty_snapshot(stop):
    result, cache = decode([[2], [3], [stop]], forbid_snapshot=True)
    assert result["generated_ids"].tolist() == [[2, 3, stop]]
    assert torch.equal(cache.conv_states[1], torch.full((1, 2, 1), 2.0))
    assert cache.num_items() == 4

@pytest.mark.parametrize("stop", [0, 7])
def test_stop_first_token_never_forwards(stop):
    result, cache = decode([[stop]], forbid_snapshot=True)
    assert result["generated_ids"].tolist() == [[stop]]
    assert cache.num_items() == 2 and not cache.conv_states[1].any()

def test_budget_without_stop_keeps_all_forwards():
    result, cache = decode([[2], [3]], forbid_snapshot=True)
    assert result["generated_ids"].tolist() == [[2, 3]]
    assert cache.num_items() == 4

def test_multi_row_finished_cache_and_hidden_still_frozen():
    result, cache = decode([[0, 2], [4, 3], [4, 0]])
    assert result["generated_ids"].tolist() == [[0, 0, 0], [2, 3, 0]]
    assert cache.conv_states[1][0].sum() == 0
    assert torch.all(cache.conv_states[1][1] == 2)
    assert cache.recurrent_states[1][0].sum() == 0
    assert torch.all(cache.recurrent_states[1][1] == 2)
    assert result["last_hidden"][0].sum() == 0
    assert torch.all(result["last_hidden"][1] == 2)

def test_empty_budget_preserves_cache():
    result, cache = decode([], max_new_tokens=0, forbid_snapshot=True)
    assert result["generated_ids"].shape == (1, 0)
    assert cache.num_items() == 2
