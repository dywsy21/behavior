from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_action_observations import ACTION_WIDTHS
from native_ar_actor_runtime import raw_actions_to_wire


def fixture():
    offset, raw = 0, {}
    for name, width in ACTION_WIDTHS.items():
        raw[name] = torch.arange(32 * width).reshape(1, 32, width).float() + offset
        offset += 1000
    def original_bridge(parts):
        return np.concatenate([parts[name] for name in ACTION_WIDTHS]).astype(np.float32)
    return raw, SimpleNamespace(grouped_raw_to_official23=original_bridge)


def test_every_raw_control_survives_in_official_order_and_execution_starts_at_zero():
    raw, bridge = fixture()
    result = raw_actions_to_wire(raw, bridge)
    expected = torch.cat([raw[name] for name in ACTION_WIDTHS], dim=-1).numpy()[0]
    assert result.shape == (32, 23) and np.array_equal(result, expected)
    assert np.array_equal(result[:16], expected[:16])
    assert not np.array_equal(result[:16], expected[5:21])


@pytest.mark.parametrize('damage', ['missing', 'nan', 'shape', 'absent', 'normalized27', 'wrong_bridge'])
def test_wire_has_no_omission_padding_slice_or_bridge_fallback(damage):
    raw, bridge = fixture()
    if damage == 'missing':
        del raw['base_qvel']
    elif damage == 'nan':
        raw['right_arm'][0, 0, 0] = float('nan')
    elif damage == 'shape':
        raw['left_arm'] = raw['left_arm'][:, 5:]
    elif damage == 'absent':
        raw['_absent_keys'] = {'lower_body'}
    elif damage == 'normalized27':
        raw = {'action': torch.zeros(1, 32, 27)}
    else:
        bridge.grouped_raw_to_official23 = lambda _: np.zeros(23, dtype=np.float32)
    with pytest.raises(ValueError):
        raw_actions_to_wire(raw, bridge)
