from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_policy_mem.py"
spec = importlib.util.spec_from_file_location("serve_policy_mem", SCRIPT)
serve = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = serve
spec.loader.exec_module(serve)


class _Processor:
    num_obs_steps = 1
    shape_meta = {
        "images": [{"key": "cam", "raw_shape": [3, 2, 2]}],
        "state": [{"key": "qpos", "raw_shape": 2}],
        "action": [{"key": "arm", "raw_shape": 1}],
    }


class _Policy:
    model_config = type("Cfg", (), {"predict_cot": True})()


class _Inferencer:
    def __init__(self, statuses=None):
        self.policy = _Policy()
        self.high_calls = []
        self.low_calls = []
        self.statuses = list(statuses or [])

    def infer_high_level(self, obs, memories, *, max_new_tokens=160):
        self.high_calls.append((obs[0].get("memory"), memories[0], max_new_tokens))
        n = len(self.high_calls)
        status = self.statuses[n - 1] if n <= len(self.statuses) else "CONTINUE"
        return [
            {
                "intent": f"intent-{n}",
                "memory": f"memory-{n}",
                "status": status,
                "raw": (
                    f"Intent: intent-{n}|Updated Memory: memory-{n}|"
                    f"Status: {status}|<HL_END>"
                ),
                "generation_metadata": {
                    "generated_token_count": 12,
                    "max_new_tokens": max_new_tokens,
                    "budget_exhausted": False,
                    "stop_token_id": 123,
                    "stop_reason": "hl_end",
                },
            }
        ]

    def infer_low_level_action(self, obs, intents):
        self.low_calls.append((obs[0].get("intent"), intents[0]))
        n = len(self.low_calls)
        return [{"arm": torch.tensor([[[float(n)], [float(n) + 1]]])}]


def _obs(event=None):
    out = {
        "images": {"cam": np.zeros((3, 2, 2), dtype=np.uint8)},
        "state": {"qpos": np.zeros(2, dtype=np.float32)},
        "task": "task",
    }
    if event:
        out["event"] = event
    return out


def test_memlite_state_machine_reuses_intent_and_replans():
    inf = _Inferencer()
    wrapper = serve.ChunkedPolicyWrapper(inf, _Processor(), action_steps=2)
    first, _ = asyncio.run(wrapper.get_action(_obs()))
    assert first["arm"].shape == (1,)
    assert len(inf.high_calls) == len(inf.low_calls) == 1
    assert wrapper.mem_state.intent_text == "intent-1"
    assert wrapper.need_obs is False

    asyncio.run(wrapper.get_action(_obs()))
    assert len(inf.high_calls) == len(inf.low_calls) == 1

    # Event invalidates the old chunk immediately and invokes HL on this frame.
    replanned, _ = asyncio.run(wrapper.get_action(_obs("done")))
    assert replanned["arm"][0] == 2
    assert len(inf.high_calls) == len(inf.low_calls) == 2
    assert wrapper.mem_state.intent_text == "intent-2"
    assert wrapper.mem_state.high_level_generation == 2


def test_memlite_reset_clears_semantic_state():
    inf = _Inferencer()
    wrapper = serve.ChunkedPolicyWrapper(inf, _Processor(), action_steps=2)
    asyncio.run(wrapper.get_action(_obs()))
    wrapper.mem_state.failure_count = 3
    wrapper.reset()
    assert wrapper.mem_state.memory_text == ""
    assert wrapper.mem_state.intent_text == ""
    assert wrapper.mem_state.status == "INVALID"
    assert wrapper.mem_state.memory_initialized is False
    assert wrapper.mem_state.task_id is None
    assert wrapper.mem_state.failure_count == 0
    assert wrapper.need_obs is True


def test_terminal_high_level_status_plans_next_request_after_current_chunk():
    """A terminal HL status must not skip the current LL action or loop inline."""
    inf = _Inferencer(statuses=["DONE", "CONTINUE"])
    wrapper = serve.ChunkedPolicyWrapper(inf, _Processor(), action_steps=2)

    action, _ = asyncio.run(wrapper.get_action(_obs()))
    assert action["arm"][0] == 1
    # Exactly one HL and one LL call happened in this request; the chunk is
    # available, while the next request is marked for fresh planning.
    assert len(inf.high_calls) == len(inf.low_calls) == 1
    assert wrapper._cached_chunk is not None
    assert wrapper._high_level_pending is True
    assert wrapper.need_obs is False
    assert wrapper._last_high_level_reason == "status:done"

    cached_action, _ = asyncio.run(wrapper.get_action(_obs()))
    assert cached_action["arm"][0] == 2
    assert len(inf.high_calls) == len(inf.low_calls) == 1
    assert wrapper.need_obs is True

    next_action, _ = asyncio.run(wrapper.get_action(_obs()))
    assert next_action["arm"][0] == 2
    assert len(inf.high_calls) == len(inf.low_calls) == 2
    assert wrapper._high_level_pending is False


def test_periodic_replan_happens_after_n_complete_chunks():
    inf = _Inferencer()
    wrapper = serve.ChunkedPolicyWrapper(
        inf,
        _Processor(),
        action_steps=2,
        memlite_replan_every_chunks=2,
    )

    actions = []
    for _ in range(4):
        action, _ = asyncio.run(wrapper.get_action(_obs()))
        actions.append(float(action["arm"][0]))
    assert actions == [1.0, 2.0, 2.0, 3.0]
    assert len(inf.high_calls) == 1
    assert len(inf.low_calls) == 2

    action, _ = asyncio.run(wrapper.get_action(_obs()))
    assert float(action["arm"][0]) == 3.0
    assert len(inf.high_calls) == 2
    assert len(inf.low_calls) == 3
    assert wrapper.mem_state.high_level_generation == 2


def test_memlite_trace_contains_high_level_protocol_and_low_level_intent():
    events = []
    wrapper = serve.ChunkedPolicyWrapper(
        _Inferencer(),
        _Processor(),
        action_steps=2,
        trace_hook=events.append,
    )
    asyncio.run(wrapper.get_action(_obs()))
    high = next(event for event in events if event["event"] == "high_level")
    low = next(event for event in events if event["event"] == "low_level_chunk")
    served = next(event for event in events if event["event"] == "served_action")
    assert high["input_memory"] == ""
    assert high["output_intent"] == "intent-1"
    assert high["memory_update"] == "memory-1"
    assert high["format_valid"] is True
    assert high["generation_metadata"] == {
        "generated_token_count": 12,
        "max_new_tokens": 160,
        "budget_exhausted": False,
        "stop_token_id": 123,
        "stop_reason": "hl_end",
    }
    assert high["trigger"] == "start"
    assert low["intent"] == "intent-1"
    assert served["low_level_intent"] == "intent-1"


def test_initial_memory_is_once_per_reset_and_high_cap_reaches_inferencer():
    events = []
    inf = _Inferencer()
    wrapper = serve.ChunkedPolicyWrapper(
        inf,
        _Processor(),
        action_steps=2,
        memlite_high_level_max_new_tokens=768,
        trace_hook=events.append,
    )
    assert wrapper.initialize_memlite_memory(
        task_id=3,
        canonical_memory="Task=3; Completed=none.",
    )
    assert wrapper.mem_state.memory_text == "Task=3; Completed=none."
    assert wrapper.initialize_memlite_memory(
        task_id=3,
        canonical_memory="must-not-overwrite",
    ) is False
    action, _ = asyncio.run(wrapper.get_action(_obs()))
    assert action["arm"][0] == 1
    assert inf.high_calls == [("Task=3; Completed=none.", "Task=3; Completed=none.", 768)]
    high = next(event for event in events if event["event"] == "high_level")
    assert high["generation_metadata"]["max_new_tokens"] == 768
    assert high["raw"].endswith("<HL_END>")

    # A model may later output an empty memory; the initialization flag prevents
    # a hidden mid-episode replacement with the canonical prior.
    wrapper.mem_state.memory_text = ""
    assert wrapper.initialize_memlite_memory(
        task_id=3,
        canonical_memory="must-not-reappear",
    ) is False
    wrapper.reset()
    assert wrapper.initialize_memlite_memory(
        task_id=1,
        canonical_memory="Task=1; Completed=none.",
    )
    assert wrapper.mem_state.memory_text == "Task=1; Completed=none."
    try:
        wrapper.initialize_memlite_memory(
            task_id=0,
            canonical_memory="Task=0; Completed=none.",
        )
    except ValueError as error:
        assert "identity changed" in str(error)
    else:
        raise AssertionError("task switch without reset was silently accepted")


def test_chunk_cache_serves_decoded_rows_zero_through_fifteen_after_alignment():
    """The serving cache must not introduce an extra temporal offset."""

    class _AlignedInferencer(_Inferencer):
        def infer_low_level_action(self, obs, intents):
            self.low_calls.append((obs[0].get("intent"), intents[0]))
            base = 100.0 * (len(self.low_calls) - 1)
            rows = torch.arange(base, base + 32.0).reshape(1, 32, 1)
            return [{"arm": rows}]

    events = []
    processor = _Processor()
    # This is the saved BEHAVIOR run's future-only action-label convention.
    processor.action_execution_start_index = 0
    inferencer = _AlignedInferencer()
    wrapper = serve.ChunkedPolicyWrapper(
        inferencer, processor, action_steps=16, trace_hook=events.append
    )

    first_chunk = [
        float(asyncio.run(wrapper.get_action(_obs()))[0]["arm"][0]) for _ in range(16)
    ]
    assert first_chunk == list(range(16))
    assert wrapper.need_obs is True

    # The next real observation starts a new chunk; it must serve its own row
    # zero, not a stale row 16 from the prior decoded horizon.
    next_action, _ = asyncio.run(wrapper.get_action(_obs()))
    assert float(next_action["arm"][0]) == 100.0
    assert len(inferencer.low_calls) == 2
    low_chunks = [event for event in events if event["event"] == "low_level_chunk"]
    assert len(low_chunks) == 2
    assert low_chunks[0]["action_execution_start_index"] == 0
    assert low_chunks[0]["decoded_action_horizon"] == 32
    assert low_chunks[0]["postprocess_action_horizon"] == 32
    assert low_chunks[0]["executed_action_steps"] == 16


def test_predict_cot_template_validation_accepts_real_memlite_mixed_builder():
    from g05.data_processor.processor.samples_builder import MEMLiteMixedBuilder

    builder = MEMLiteMixedBuilder(
        num_input_images=1,
        image_sizes={"cam": (2, 2)},
        embodiment_type="galaxea_r1pro",
    )
    processor = type("Processor", (), {"samples_builder": builder})()
    # The mixed builder itself intentionally raises when its universal
    # template is read; serving must inspect only its high-level branch.
    serve._validate_predict_cot_templates(processor)


def test_strict_memlite_rejects_high_level_failure_without_legacy_fallback():
    class _BrokenInferencer(_Inferencer):
        def infer_high_level(self, obs, memories):
            raise ValueError("synthetic high-level failure")

        def infer(self, obs):
            raise AssertionError("legacy infer() must not be used in strict mode")

    wrapper = serve.ChunkedPolicyWrapper(
        _BrokenInferencer(), _Processor(), action_steps=2, strict_memlite=True
    )
    try:
        asyncio.run(wrapper.get_action(_obs()))
    except RuntimeError as error:
        assert "strict mode" in str(error)
    else:
        raise AssertionError("strict MEM-Lite route silently accepted a high-level failure")
