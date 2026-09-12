import pytest
import torch.nn as nn

from g05.utils.training.parameter_freezing import freeze_parameters_by_prefixes


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.vlm = nn.Linear(3, 3)
        self.model.action_expert = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3))


def test_freezes_requested_branch_and_reports_full_audit():
    policy = _TinyPolicy()
    audit = freeze_parameters_by_prefixes(policy, ["model.action_expert."])

    parameters = dict(policy.named_parameters())
    assert audit["frozen_parameter_tensors"] == 4
    assert audit["frozen_parameter_numel"] == sum(
        parameter.numel()
        for name, parameter in parameters.items()
        if name.startswith("model.action_expert.")
    )
    assert audit["frozen_parameter_names"] == [
        name for name in parameters if name.startswith("model.action_expert.")
    ]
    assert all(
        not parameter.requires_grad
        for name, parameter in parameters.items()
        if name.startswith("model.action_expert.")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in parameters.items()
        if name.startswith("model.vlm.")
    )


def test_unknown_freeze_prefix_fails_fast():
    with pytest.raises(ValueError, match="matched no parameters"):
        freeze_parameters_by_prefixes(_TinyPolicy(), ["model.not_a_real_branch."])
