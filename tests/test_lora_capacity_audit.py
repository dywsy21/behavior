from copy import deepcopy

import pytest
import torch

import audit_lora_capacity as audit


@pytest.mark.parametrize("rank", [1, 2, 8])
@pytest.mark.parametrize("zero", [False, True])
def test_skinny_qr_matches_direct_small_matrix_svd_and_leaves_inputs_rng_unchanged(rank, zero):
    torch.manual_seed(41)
    a, b = torch.randn(rank, 15), torch.randn(13, rank)
    if zero:
        b.zero_()
    before_a, before_b, rng = a.clone(), b.clone(), torch.get_rng_state().clone()
    result = audit.low_rank_spectrum(a, b, scaling=2.)
    dense = 2 * b.double() @ a.double()
    expected = torch.linalg.svdvals(dense)[:rank]
    torch.testing.assert_close(torch.tensor(result["singular_values"], dtype=torch.float64), expected,
                               rtol=1e-10, atol=1e-10)
    assert abs(result["update_frobenius_norm"] - float(dense.norm())) < 1e-10
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(a, before_a) and torch.equal(b, before_b)
    if zero:
        assert result["energy_rank95"] == 0 and result["leading_energy_fraction"] is None
    else:
        assert 1 <= result["energy_rank95"] <= rank
        assert 1 - 1e-10 <= result["energy_participation_rank"] <= rank + 1e-10


@pytest.mark.parametrize("damage", ["shape", "rank", "nan", "integer", "scale"])
def test_invalid_factors_stop_before_decomposition(damage):
    a, b, scale = torch.ones(2, 4), torch.ones(5, 2), 2.
    if damage == "shape":
        b = torch.ones(5, 3)
    elif damage == "rank":
        a, b = torch.ones(6, 4), torch.ones(5, 6)
    elif damage == "nan":
        b[0, 0] = float("nan")
    elif damage == "integer":
        a = a.long()
    elif damage == "scale":
        scale = 0.
    with pytest.raises(ValueError):
        audit.low_rank_spectrum(a, b, scaling=scale)


def example_state():
    return {"model.vlm.layers.0.q_proj.lora_A.skill_fm.weight": torch.ones(2, 4),
            "model.vlm.layers.0.q_proj.lora_B.skill_fm.weight": torch.ones(5, 2),
            "model.vlm.layers.0.q_proj.base_layer.weight": torch.ones(5, 4),
            "model.vlm.layers.1.in_proj_qkv.weight": torch.ones(8, 4)}


def test_coverage_uses_real_base_modules_and_distinguishes_untargeted_projections():
    state = example_state()
    pairs = audit.adapter_pairs(state)
    assert len(pairs) == 1 and pairs[0][0] == "model.vlm.layers.0.q_proj"
    inventory = audit.scope_inventory(state, {pairs[0][0]})
    assert inventory["by_module_leaf"] == {"in_proj_qkv": dict(total=1, with_lora=0),
                                           "q_proj": dict(total=1, with_lora=1)}


@pytest.mark.parametrize("damage", ["missing_a", "missing_b", "different_adapter", "no_base"])
def test_partial_or_unmapped_adapters_are_rejected(damage):
    state = deepcopy(example_state())
    if damage == "missing_a":
        del state["model.vlm.layers.0.q_proj.lora_A.skill_fm.weight"]
    elif damage == "missing_b":
        del state["model.vlm.layers.0.q_proj.lora_B.skill_fm.weight"]
    elif damage == "different_adapter":
        state["model.vlm.layers.0.q_proj.lora_B.other.weight"] = torch.ones(5, 2)
    else:
        del state["model.vlm.layers.0.q_proj.base_layer.weight"]
    with pytest.raises(RuntimeError):
        pairs = audit.adapter_pairs(state)
        audit.scope_inventory(state, {module for module, _, _ in pairs})
