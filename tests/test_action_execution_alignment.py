"""CPU regressions for decoded-action execution offsets.

These tests use ``FullProcessor.postprocess`` itself, while keeping the
normalizer / merger identity-shaped so row selection is directly observable.
The saved BEHAVIOR run is covered separately by its real processor regression.
"""

from __future__ import annotations

import pytest
import torch

from g05.data_processor.processor.base_processor import FullProcessor


class _Identity:
    def backward(self, data):
        return data


class _PerPartMerger:
    def backward(self, data):
        data = dict(data)
        action = data["action"]
        data["action"] = {"marker": action}
        return data


def _processor(start_index: int | None) -> FullProcessor:
    """Create the smallest object that can exercise real postprocess code."""
    processor = object.__new__(FullProcessor)
    processor.num_obs_steps = 6
    processor.action_execution_start_index = start_index
    processor.action_state_merger = _PerPartMerger()
    processor._normalizer = _Identity()
    processor.action_state_transforms = None
    processor.action_filter = _Identity()
    return processor


def _postprocess(processor: FullProcessor, *, source: str) -> dict:
    marker = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)
    data = {"proprio": torch.zeros(1, 6, 1)}
    if source == "fm":
        data["action_fm"] = marker
    elif source == "ar":
        data["action_ar"] = marker
    elif source == "fallback":
        data["action"] = marker
    else:
        raise AssertionError(source)
    return processor.postprocess(data)["action"]["marker"]


@pytest.mark.parametrize("source", ["fm", "ar", "fallback"])
def test_explicit_future_only_offset_zero_keeps_all_32_rows(source: str):
    action = _postprocess(_processor(0), source=source)
    assert action.shape == (1, 32, 1)
    assert action[0, :16, 0].tolist() == list(range(16))


def test_explicit_action_history_and_legacy_default_both_start_at_five():
    historical = _postprocess(_processor(5), source="ar")
    legacy = _postprocess(_processor(None), source="ar")
    assert historical.shape == legacy.shape == (1, 27, 1)
    assert historical[0, :16, 0].tolist() == list(range(5, 21))
    torch.testing.assert_close(historical, legacy)


def test_execution_offset_rejects_invalid_values_and_horizon_overrun():
    processor = _processor(None)
    with pytest.raises(TypeError, match="integer or None"):
        processor.set_action_execution_start_index(True)
    with pytest.raises(ValueError, match="non-negative"):
        processor.set_action_execution_start_index(-1)
    processor.set_action_execution_start_index(32)
    with pytest.raises(ValueError, match="outside decoded action horizon"):
        _postprocess(processor, source="fm")
