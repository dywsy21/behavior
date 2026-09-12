"""Regression tests for observed v8 failures, not just tensor finiteness."""
from types import SimpleNamespace

import pytest
import torch

from g05.data_processor.processor.base_processor import FullProcessor
from g05.utils.data.normalizer import ActionNormalizationError, SingleFieldLinearNormalizer
from g05.utils.memlite_protocol import MemLiteProtocolError, parse_memlite_output


def test_three_camera_histories_are_not_flattened_or_relabelled():
    processor = SimpleNamespace(image_history_mode="preserve_cameras", num_obs_steps=6,
                                num_output_cameras=3, is_train=True)
    frames = {key: (torch.arange(6) + index * 100).view(6, 1, 1, 1).expand(6, 3, h, w)
              for index, (key, h, w) in enumerate((("head", 8, 8), ("left", 4, 6), ("right", 6, 4)))}
    for training in (True, False):
        processor.is_train = training
        actual = FullProcessor.build_pixel_values(processor, frames)
        assert list(actual) == ["head", "left", "right"]
        for key in frames:
            assert actual[key] is frames[key]
            assert actual[key].shape[0] == 6
    assert int(actual["head"][-1, 0, 0, 0]) == 5
    assert int(actual["left"][-1, 0, 0, 0]) == 105
    assert int(actual["right"][-1, 0, 0, 0]) == 205


def test_history_contract_rejects_missing_frames_and_camera_streams():
    p = SimpleNamespace(image_history_mode="preserve_cameras", num_obs_steps=6, num_output_cameras=3)
    with pytest.raises(ValueError, match="one output stream"):
        FullProcessor.build_pixel_values(p, {"head": torch.zeros(6, 3, 4, 4)})
    with pytest.raises(ValueError, match="chronological"):
        FullProcessor.build_pixel_values(p, {k: torch.zeros(1, 3, 4, 4) for k in ("h", "l", "r")})


def tail_stats(horizon=32):
    # Similar small-scale relative-joint distribution to the failed run.
    scalar = {"min": -.12, "max": .15, "q01": -.02, "q99": .023,
              "mean": .001, "std": .0085}
    return {k: torch.full((horizon, 2), v) for k, v in scalar.items()}


def test_bounded_tail_inverse_prevents_exponential_blowup_and_keeps_padding():
    norm = SingleFieldLinearNormalizer(tail_stats(), mode="z-score-tail-bounded")
    x = torch.full((2, 32, 3), 1e20)
    x[0, :, 1] = -1e20
    out = norm.backward(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[..., :2].amax(), torch.tensor(.15))
    assert torch.allclose(out[..., :2].amin(), torch.tensor(-.12))
    assert torch.equal(out[..., 2], x[..., 2])
    assert norm.last_backward_diagnostics["clipped_count"] == 128


def test_bounded_tail_keeps_training_forward_and_valid_roundtrip():
    stats = tail_stats()
    legacy = SingleFieldLinearNormalizer(stats, mode="z-score-tail")
    bounded = SingleFieldLinearNormalizer(stats, mode="z-score-tail-bounded")
    physical = torch.linspace(-.12, .15, 64).view(1, 32, 2)
    assert torch.equal(bounded.forward(physical), legacy.forward(physical))
    assert torch.allclose(bounded.backward(bounded.forward(physical)), physical, atol=1e-6)
    # Stepwise tail parameters, not only scale, must follow shortened horizons.
    short = physical[:, :7]
    assert torch.allclose(bounded.backward(bounded.forward(short)), short, atol=1e-6)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_bounded_tail_rejects_nonfinite_instead_of_silently_zeroing(invalid):
    norm = SingleFieldLinearNormalizer(tail_stats(), mode="z-score-tail-bounded")
    with pytest.raises(ActionNormalizationError):
        norm.backward(torch.full((1, 32, 2), invalid))


def test_protocol_preserves_valid_fields_without_truncation_or_field_shift():
    intent = "move cup " * 80
    out = parse_memlite_output(f"Intent: {intent}|Updated Memory: Task=0; Completed=none.|Status: CONTINUE|<HL_END>")
    assert out["intent"] == intent.strip() and len(out["intent"]) > 512
    assert out["memory"] == "Task=0; Completed=none."


@pytest.mark.parametrize("text", [
    "Intent: move cup|Status: CONTINUE|<HL_END>",  # actual can failure shape
    "Intent: move cup|Updated Memory: old|Status: CONTINUE|extra",
    "Updated Memory: old|Intent: move cup|Status: CONTINUE",
    "Intent: |Updated Memory: old|Status: CONTINUE",
    "Intent: move cup|Updated Memory: old|Status: probably",
    "Intent: Task complete|Updated Memory: old|Status: CONTINUE",
    "Intent: move cup|Updated Memory: old|Status: DONE",
    "Intent: Task complete|Updated Memory: old|Status: DONE|<HL_END>trailing",
])
def test_protocol_rejects_malformed_or_nonexecutable_proposals(text):
    with pytest.raises(MemLiteProtocolError):
        parse_memlite_output(text)
