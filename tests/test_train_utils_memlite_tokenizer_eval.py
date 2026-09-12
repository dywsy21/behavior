"""Regression coverage for the first-batch VQ diagnostic MEM-Lite split."""

import pytest

from g05.utils.training.train_utils import _select_action_samples_for_tokenizer_eval


def _high():
    return {
        "memlite_branch": "high",
        "intent": "Intent: open the drawer",
        "memory_update": "Updated Memory: drawer is open",
        "intent_status": "Status: CONTINUE",
    }


def _low():
    return {
        "memlite_branch": "low",
        "action": {"value": object(), "parts_meta": {"lower_body": 7}},
    }


def test_first_batch_tokenizer_eval_high_fixture_skips_action_diagnostic():
    assert _select_action_samples_for_tokenizer_eval([_high()]) == ([], [0], True)


def test_first_batch_tokenizer_eval_low_fixture_keeps_action_target():
    assert _select_action_samples_for_tokenizer_eval([_low()]) == ([0], [], True)


def test_first_batch_tokenizer_eval_mixed_fixture_selects_only_low_action_target():
    assert _select_action_samples_for_tokenizer_eval([_high(), _low(), _low()]) == (
        [1, 2],
        [0],
        True,
    )


def test_first_batch_tokenizer_eval_rejects_malformed_low_sample():
    with pytest.raises(KeyError, match="missing samples.*action.*value"):
        _select_action_samples_for_tokenizer_eval([{"memlite_branch": "low"}])
