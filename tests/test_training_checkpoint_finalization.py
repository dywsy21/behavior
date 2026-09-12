from pathlib import Path

import pytest
import torch

import finetune
from g05.utils.checkpoint.checkpoint_utils import (
    checkpoint_save_reason,
    save_training_checkpoint,
)


def test_final_checkpoint_is_not_duplicated_when_it_is_periodic():
    assert checkpoint_save_reason(step=10, max_steps=10, checkpointing_steps=10) == "periodic"


def test_nonperiodic_final_step_gets_its_own_checkpoint():
    reasons = {
        step: checkpoint_save_reason(step=step, max_steps=11, checkpointing_steps=10)
        for step in range(1, 12)
    }

    assert {step: reason for step, reason in reasons.items() if reason is not None} == {
        10: "periodic",
        11: "final",
    }


def test_disabled_checkpoint_interval_still_saves_final_step():
    assert checkpoint_save_reason(step=11, max_steps=11, checkpointing_steps=None) == "final"
    assert checkpoint_save_reason(step=11, max_steps=11, checkpointing_steps=0) == "final"


@pytest.mark.parametrize("step", [0, 11])
def test_negative_checkpoint_interval_is_rejected(step):
    with pytest.raises(ValueError, match="checkpointing_steps must be non-negative"):
        checkpoint_save_reason(step=step, max_steps=11, checkpointing_steps=-1)


@pytest.mark.parametrize(("step", "max_steps"), [(0, 0), (10, 10)])
def test_completed_run_skips_optimizer_and_checkpoint_callbacks(step, max_steps):
    calls = []

    def optimizer_step():
        calls.append("optimizer")

    def save_checkpoint():
        calls.append("checkpoint")

    training_done = finetune._training_is_complete(step, max_steps)
    while not training_done:
        optimizer_step()
        step += 1
        if checkpoint_save_reason(step, max_steps, checkpointing_steps=10) is not None:
            save_checkpoint()
        training_done = finetune._training_is_complete(step, max_steps)

    assert calls == []


def test_finetune_has_no_inference_only_final_checkpoint_override():
    source = Path(finetune.__file__).read_text(encoding="utf-8")

    assert "optimizer=None" not in source
    assert source.count("save_training_checkpoint(") == 1
    assert "_save_resumable_training_checkpoint(" in source
    assert "training_done = _training_is_complete(step, max_steps)" in source


def test_resumable_save_keeps_optimizer_scheduler_and_action_batch_idx(monkeypatch, tmp_path):
    captured = {}

    def fake_save(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return tmp_path / "checkpoints" / "step_11.pt"

    monkeypatch.setattr(finetune, "save_training_checkpoint", fake_save)
    model = object()
    optimizer = object()
    scheduler = object()
    ema_model = object()

    finetune._save_resumable_training_checkpoint(
        tmp_path,
        step=11,
        epoch=3,
        batch_idx=7,
        action_batch_idx=9,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema_model=ema_model,
    )

    assert captured == {
        "output_dir": tmp_path,
        "step": 11,
        "epoch": 3,
        "batch_idx": 7,
        "action_batch_idx": 9,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "ema_model": ema_model,
    }


def test_training_checkpoint_and_last_pointer_are_resumable(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = save_training_checkpoint(
        tmp_path,
        step=11,
        epoch=3,
        batch_idx=7,
        action_batch_idx=9,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["action_batch_idx"] == 9
    assert (tmp_path / "last.pt").is_symlink()
    assert (tmp_path / "last.pt").resolve() == checkpoint_path.resolve()
