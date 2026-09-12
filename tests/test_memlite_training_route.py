import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir

from g05.models.g05.g05_policy import G05Policy


class _Accumulator:
    def __init__(self):
        self.values = []
    def push(self, name, value):
        self.values.append((name, value))


class _Processor:
    def encode_train(self, samples, device, **kwargs):
        n = len(samples)
        # One valid target token per sample; route only needs shape/branch calls.
        ids = torch.zeros((n, 4), dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        labels[:, -1] = 1
        return ids, labels, torch.ones_like(ids), 2

class _Model:
    proprio_embedder = None
    proprio_encoder = None
    def __call__(self, **kwargs):
        branch = kwargs["embodiment_types"][0]
        value = 1.0 if branch == "high" else 3.0
        return {
            "ce_loss": torch.tensor(value, requires_grad=True),
            "fm_loss": torch.tensor(0.0),
            "overall_accuracy": value / 3.0,
            "action_accuracy": 1.0 if branch == "low" else 0.0,
            "cot_accuracy": 1.0 if branch == "high" else 0.0,
        }


def _policy(mode="mixed"):
    policy = object.__new__(G05Policy)
    policy.memlite_train_mode = mode
    policy.max_chunk_token_length = 128
    policy.max_pad_token_length = None
    policy.model = _Model()
    policy.processor = _Processor()
    policy.process_pixel_values = lambda x: x
    policy.language_loss_weight = 1.0
    policy.predict_cot = True
    policy.training = True
    policy._train_acc = _Accumulator()
    policy._fwd_step = 0
    return policy


def test_memlite_branch_validation_is_fail_fast():
    with pytest.raises(ValueError, match="memlite_branch"):
        G05Policy._memlite_branch_masks([{}], "mixed")
    with pytest.raises(ValueError, match="requires at least one high"):
        G05Policy._memlite_branch_masks([{"memlite_branch": "high"}], "mixed")
    assert G05Policy._memlite_branch_masks(
        [{"memlite_branch": "planner"}, {"memlite_branch": "action"}], "mixed"
    ) == {"high": [0], "low": [1]}


def test_memlite_mixed_route_trains_both_ce_branches_and_zero_fm():
    policy = _policy()
    samples = [
        {"memlite_branch": "high", "embodiment": "high"},
        {"memlite_branch": "low", "embodiment": "low"},
    ]
    loss, details = policy._forward_train_memlite(
        samples, torch.zeros((2, 1, 3, 4, 4)), actions=None, action_pad_masks=None
    )
    assert torch.isclose(loss, torch.tensor(2.0))
    assert torch.isclose(details["high_ce_loss"], torch.tensor(1.0))
    assert torch.isclose(details["low_ce_loss"], torch.tensor(3.0))
    assert torch.isclose(details["fm_loss"], torch.tensor(0.0))
    assert policy._fwd_step == 1
    assert {name for name, _ in policy._train_acc.values} == {"overall", "action_token", "cot"}


def test_memlite_task_is_step_bounded_on_the_non_dry_run_path(monkeypatch):
    """Compose the launch config and mirror finetune's normal scheduling gate."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("G05_OUTPUT_DIR", "/tmp/g05-test-output")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(project_root / "configs"),
        job_name="memlite-task-compose-test",
    ):
        cfg = compose(config_name="train", overrides=["task=r1pro_memlite_ar"])
    assert cfg.model.max_epochs is None
    assert cfg.model.max_steps == 5000

    # Keep this aligned with finetune.py's non-DRY_RUN scheduling branch:
    # when max_epochs is truthy it asserts that max_steps is absent.
    if cfg.model.max_epochs:
        assert not cfg.model.max_steps
    else:
        assert cfg.model.max_steps == 5000
