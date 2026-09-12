import torch
from types import SimpleNamespace

from g05.data_processor.processor.samples_builder import (
    IntentActionBuilder,
    MEMLiteHighLevelBuilder,
)
from g05.models.g05.g05_policy import G05Policy, MemLiteState
from g05.models.g05.inferencer import PolicyInferencer


DEFAULTS = dict(
    num_input_images=1,
    image_sizes={"cam": (224, 224)},
)


def test_high_level_builder_boundaries_and_fields():
    builder = MEMLiteHighLevelBuilder(**DEFAULTS)
    template = builder.template
    assert "<EOC>" in template and "<EOV>" in template and "<HL_END>" in template
    assert "<action_action>" not in template
    assert "<memory_text_!>" in template
    out = {}
    builder._populate_extra_samples(
        {"memory": "holding cup", "intent": "move cup", "memory_update": "at sink"}, out
    )
    assert out["intent"] == "Intent: move cup"
    assert out["memory_update"] == "Updated Memory: at sink"
    assert out["intent_status"] == "Status: CONTINUE"


def test_low_level_intent_is_masked_and_only_action_target():
    builder = IntentActionBuilder(**DEFAULTS)
    template = builder.template
    assert "<intent_text_!>" in template
    assert template.index("<intent_text_!>") < template.index("<EOC>")
    assert template.index("<EOV>") < template.index("<EOC>")
    assert "<action_action>" in template
    assert builder.can_handle({"intent": "grasp cup"})
    assert not builder.can_handle({})


def test_structured_parser_and_episode_state_are_bounded():
    parsed = G05Policy.parse_memlite_output(
        "Intent: grasp cup|Updated Memory: holding cup|Status: replan|<HL_END>"
    )
    assert parsed["intent"] == "grasp cup"
    assert parsed["memory"] == "holding cup"
    assert parsed["status"] == "REPLAN"
    state = MemLiteState(memory_text="old", intent_text="x", status="CONTINUE")
    state.reset()
    assert state.memory_text == "" and state.intent_text == "" and state.status == "INVALID"


def test_action_decode_receives_only_low_level_ids():
    policy = object.__new__(G05Policy)
    policy.discrete_action = True
    policy.continuous_action = False
    policy.model_config = SimpleNamespace(horizon_steps=1, action_dim=1)
    generated = torch.tensor([[101, 102]], dtype=torch.long)
    captured = {}

    class Model:
        def inference_ar(self, *args, **kwargs):
            return {"generated_ids": generated}

    class Processor:
        def decode_ar(self, ids, **kwargs):
            captured["ids"] = ids
            captured["action_only"] = kwargs.get("action_only")
            return [torch.zeros(1, 1)], [torch.tensor([101, 102])], [""], [set()]

    policy.model = Model()
    policy.processor = Processor()
    policy._get_action_stop_token_ids = lambda: None
    policy._get_action_generation_max_new_tokens = lambda: 8
    state = SimpleNamespace(
        kv_cache=[],
        attention_mask=torch.ones(1, 1),
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        last_hidden=torch.zeros(1, 2),
        pixel_values={},
        input_ids=torch.tensor([[7, 8]]),
        device=torch.device("cpu"),
        generated_ids=torch.tensor([[55, 56]]),
        check_invariants=lambda *args, **kwargs: None,
    )
    result = policy.generate_action(state, [{"embodiment": "r1"}], only_ar=True)
    assert torch.equal(captured["ids"][0], generated[0])
    assert captured["action_only"] is True
    assert result["selected_action_source"] == "ar"


def test_branch_local_templates_are_selected_and_isolated():
    policy = object.__new__(G05Policy)
    policy.num_input_images = 1
    base = {"image0": (224, 224), "embodiment": "r1", "template": "legacy"}

    high = policy._clone_memlite_samples(
        [base], branch="high", memory="old memory"
    )[0]
    assert "<HL_END>" in high["template"]
    assert "<action_action>" not in high["template"]
    assert high["memory"] == "old memory"

    low = policy._clone_memlite_samples(
        [base], branch="low", intent="grasp cup"
    )[0]
    assert "<intent_text_!>" in low["template"]
    assert "<action_action>" in low["template"]
    assert "<HL_END>" not in low["template"]
    assert low["intent"] == "grasp cup"
    assert high["template"] != low["template"]


def test_runtime_branch_data_is_copied_before_mixed_builder_preprocess():
    """Serving routing fields must be explicit and must not mutate official obs."""
    raw = {
        "memory": "stale raw memory",
        "intent": "stale raw intent",
        "atomic_task": "stale raw atomic task",
        "memory_update": "stale future update",
        "intent_status": "FAILED",
        "status": "FAILED",
        "state": {"joint": torch.tensor([1.0])},
    }
    high = PolicyInferencer._with_memlite_runtime_branch(
        [raw], branch="high", memory_texts=[""]
    )[0]
    assert "memlite_branch" not in raw
    assert high["memlite_branch"] == "high"
    assert high["memory"] == "No prior memory."
    assert high["intent"] == "__memlite_runtime_target_placeholder__"
    assert high["atomic_task"] == "__memlite_runtime_target_placeholder__"
    assert high["memory_update"] == "__memlite_runtime_target_placeholder__"
    assert high["intent_status"] == high["status"] == "CONTINUE"
    high["state"]["joint"][0] = 9.0
    assert float(raw["state"]["joint"][0]) == 1.0

    low = PolicyInferencer._with_memlite_runtime_branch(
        [raw], branch="low", intent_texts=["move the cup"]
    )[0]
    assert low["memlite_branch"] == "low"
    assert low["intent"] == low["atomic_task"] == "move the cup"
    try:
        PolicyInferencer._with_memlite_runtime_branch([raw], branch="low", intent_texts=[""])
    except ValueError as exc:
        assert "non-empty current intent" in str(exc)
    else:
        raise AssertionError("empty planner output must not receive a fake low-level intent")


def test_high_prefill_clone_removes_runtime_target_placeholders():
    policy = object.__new__(G05Policy)
    policy.num_input_images = 1
    base = {
        "image0": (224, 224),
        "embodiment": "r1",
        "template": "legacy",
        "memory": "No prior memory.",
        "prompt": "Output the next intent, updated memory, and status:",
        "intent": "__memlite_runtime_target_placeholder__",
        "memory_update": "__memlite_runtime_target_placeholder__",
        "intent_status": "CONTINUE",
    }
    high = policy._clone_memlite_samples([base], branch="high", memory="")[0]
    assert high["memory"] == ""
    assert high["prompt"] == base["prompt"]
    for key in ("intent", "memory_update", "intent_status"):
        assert key not in high


def test_memlite_template_uses_prepared_camera_count_not_temporal_model_frame_count():
    policy = object.__new__(G05Policy)
    # Saved Qwen MEM config: six historical frames x three cameras.  The
    # prepared sample nevertheless has three camera placeholders, with time
    # represented in pixel_values rather than image3..image17 metadata.
    policy.num_input_images = 18
    base = {
        "image0": (224, 224),
        "image1": (224, 224),
        "image2": (224, 224),
        "embodiment": "r1",
        "memory": "current memory",
        "prompt": "Output the next intent, updated memory, and status:",
    }
    high = policy._clone_memlite_samples([base], branch="high", memory="current memory")[0]
    assert high["template"].count("_image_!") == 3


def test_branch_template_fails_fast_without_image_metadata():
    policy = object.__new__(G05Policy)
    policy.num_input_images = 1
    try:
        policy._clone_memlite_samples([{"embodiment": "r1"}], branch="high")
    except ValueError as exc:
        assert "missing image0" in str(exc)
    else:
        raise AssertionError("missing image metadata must fail fast")


def test_high_level_generation_metadata_uses_untrimmed_ids():
    policy = object.__new__(G05Policy)
    policy.model = SimpleNamespace(cfg=SimpleNamespace(eos_token_id=99))
    metadata = policy._memlite_high_level_generation_metadata(
        torch.tensor([[11, 7, 99], [12, 13, 14], [99, 0, 0]]),
        max_new_tokens=3,
        hl_end_id=7,
    )
    assert metadata == [
        {
            "generated_token_count": 2,
            "max_new_tokens": 3,
            "budget_exhausted": False,
            "stop_token_id": 7,
            "stop_reason": "hl_end",
        },
        {
            "generated_token_count": 3,
            "max_new_tokens": 3,
            "budget_exhausted": True,
            "stop_token_id": None,
            "stop_reason": "max_new_tokens",
        },
        {
            "generated_token_count": 1,
            "max_new_tokens": 3,
            "budget_exhausted": False,
            "stop_token_id": 99,
            "stop_reason": "eos",
        },
    ]
