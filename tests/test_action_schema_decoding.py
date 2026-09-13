from types import SimpleNamespace

import pytest
import torch

from g05.models.g05.helpers.action_schema_decoding import ActionSchemaSampler, constrained_action_schema
from g05.utils.training.ar_training_methods import ActionTrainingSettings


def tokenizer():
    class Rule:
        vocab_size, num_tokens, num_valid_sequences = 3, 2, 5
        def decode(self, tokens, safe=True):
            assert not safe and len(tokens) == 2 and 0 <= tokens[0] * 3 + tokens[1] < 5
            return tokens
    return SimpleNamespace(action_token_begin_idx=10, action_token_end_idx=17, _codebook_size=3,
        action_tokenizer=SimpleNamespace(_rule_tokenizer=Rule()),
        serializer=SimpleNamespace(nn_key_names=["arm"], rule_key_names=["gripper"],
            num_residuals=1, max_residuals=1, code_len=2, rule_tokens_per_key=2,
            group_marker_action_indices={"<arm>": 3, "<gripper>": 4}))


def greedy(logits, **kwargs):
    return logits.argmax(dim=-1)


def test_static_markers_with_model_payloads_and_exact_termination():
    schema = ActionSchemaSampler(tokenizer(), 1)
    # Highest global logit is an illegal text token. Payload picks the best
    # legal neural score, NOT zero, a hard-coded hold or an expert token.
    logits = torch.zeros(1, 20)
    logits[0, 0], logits[0, 12], logits[0, 11] = 100., 8., 6.
    for _ in schema.slots:
        schema.sample(logits, greedy)
    assert schema.generated == [13, 12, 12, 14, 11, 11, 1]
    assert schema.require_complete()["action_tokens"] == 6
    assert schema.require_complete()["rule_safe_clamp"] is False
    assert all(row["unconstrained_argmax"] == 0 for row in schema.trace)
    with pytest.raises(RuntimeError, match="continued beyond"):
        schema.sample(logits, greedy)


@pytest.mark.parametrize("first,allowed", [(0, (10, 13)), (1, (10, 12))])
def test_gripper_joint_rank_respects_generated_prefix(first, allowed):
    schema = ActionSchemaSampler(tokenizer(), 1)
    schema.generated = [13, 10, 10, 14, 10 + first]
    assert schema.allowed_range() == allowed
    schema.generated[-1] = 12  # No legal sequence has rank >= 6.
    with pytest.raises(RuntimeError, match="no valid continuation"):
        schema.allowed_range()


def test_constraint_is_not_expert_forcing_payload_changes_with_logits():
    result = []
    for token in (10, 12):
        schema = ActionSchemaSampler(tokenizer(), 1)
        logits = torch.zeros(1, 20)
        logits[0, token] = 5.
        for _ in schema.slots:
            schema.sample(logits, greedy)
        result.append(schema.generated)
    assert result[0][0] == result[1][0] == 13
    assert result[0][1:3] == [10, 10] and result[1][1:3] == [12, 12]


def test_incomplete_illegal_or_unavailable_scores_fail_closed():
    schema = ActionSchemaSampler(tokenizer(), 1)
    with pytest.raises(RuntimeError, match="ended before"):
        schema.require_complete()
    logits = torch.full((1, 20), -torch.inf)
    with pytest.raises(RuntimeError, match="masks every"):
        schema.sample(logits, greedy)
    with pytest.raises(RuntimeError, match="codec-illegal"):
        schema.sample(torch.zeros(1, 20), lambda *a, **kw: torch.tensor([1]))
    with pytest.raises(RuntimeError, match="single-environment"):
        schema.sample(torch.zeros(2, 20), greedy)


def test_missing_rule_metadata_rejected_instead_of_guessing_binary_codes():
    codec = tokenizer()
    codec.action_tokenizer._rule_tokenizer = None
    with pytest.raises(ValueError, match="exact rule"):
        ActionSchemaSampler(codec, 1)


def policy():
    class Helper:
        block_wise_autoregressive = False
        _sample = staticmethod(greedy)
    return SimpleNamespace(action_training=ActionTrainingSettings(route="ar"),
        model=SimpleNamespace(ar_helper=Helper()), action_tokenizer=tokenizer(),
        processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1])),
        _get_action_stop_token_ids=lambda: [1, 2])


def test_context_restores_original_decoder_after_failure_and_rejects_nesting():
    instance = policy()
    with pytest.raises(ValueError, match="test failure"):
        with constrained_action_schema(instance):
            with pytest.raises(RuntimeError, match="one pure-AR"):
                with constrained_action_schema(instance):
                    pass
            raise ValueError("test failure")
    assert instance.model.ar_helper._sample is greedy
    assert "_sample" not in instance.model.ar_helper.__dict__
    assert not hasattr(instance.model.ar_helper, "_schema_sampling_active")
    with pytest.raises(RuntimeError, match="ended before"):
        with constrained_action_schema(instance):
            pass


def test_cot_and_wrong_delimiter_are_not_silently_applied_to_text_generation():
    instance = policy()
    instance.action_training = ActionTrainingSettings(route="ar", conditioning="native_subtask_cot")
    with pytest.raises(RuntimeError, match="non-CoT"):
        with constrained_action_schema(instance):
            pass
    instance.action_training = ActionTrainingSettings(route="ar")
    instance._get_action_stop_token_ids = lambda: [2]
    with pytest.raises(RuntimeError, match="delimiter"):
        with constrained_action_schema(instance):
            pass
