import numpy as np
import pytest

from g05.utils.memlite_progress import MotionProgressMonitor, REPEATED_ROTATION_FEEDBACK


def monitor(**kwargs):
    return MotionProgressMonitor({"enabled": True, "fps": 30., "min_rotation_rad": 6.2,
        "min_seconds": 15., "window_seconds": 45., **kwargs})


def feed(m, step, velocity=(0., 0., .3), image=None, intent="face target"):
    return m.observe(executed_steps=step, velocity=np.array(velocity),
        head_rgb=np.zeros((3, 48, 48), np.uint8) if image is None else image, intent=intent)


def test_one_normal_search_turn_does_not_trigger_and_full_loop_does():
    m = monitor()
    for step in range(0, 600, 16):
        assert feed(m, step) is None
    events = [feed(m, step) for step in range(600, 800, 16)]
    assert sum(e is not None for e in events) == 1
    assert m.feedback == REPEATED_ROTATION_FEEDBACK


def test_translation_or_different_intent_prevents_false_loop():
    m = monitor()
    for step in range(0, 1000, 16):
        assert feed(m, step, velocity=(.1, 0, .3)) is None
    m = monitor()
    for step in range(0, 1000, 16):
        assert feed(m, step, intent=f"intent_{step//400}") is None


def test_signed_rotation_cancellation_is_not_monotonic_spin():
    m = monitor()
    for step in range(0, 1200, 16):
        velocity = (0, 0, .3 if step % 320 < 160 else -.3)
        assert feed(m, step, velocity=velocity) is None


def test_returning_to_an_old_intent_does_not_join_separate_intervals():
    m = monitor()
    for step in range(0, 1000, 16):
        intent = "initial" if step < 400 or step >= 600 else "different"
        assert feed(m, step, intent=intent) is None


def test_repeated_requests_and_reset_do_not_carry_elapsed_motion():
    m = monitor()
    for _ in range(1000):
        assert feed(m, 16) is None
    assert m.yaw_integral == 0
    feed(m, 32)
    with pytest.raises(ValueError, match="backwards"):
        feed(m, 0)
    m.reset()
    assert feed(m, 0) is None and m.feedback == "none"


def test_feedback_survives_brake_then_clears_after_real_translation():
    m = monitor()
    for step in range(0, 800, 16):
        feed(m, step)
    assert m.feedback != "none"
    feed(m, 800, velocity=(0, 0, 0))
    assert m.feedback != "none"
    for step in range(816, 880, 16):
        feed(m, step, velocity=(.3, 0, 0))
    assert m.feedback == "none"


def test_near_target_recovery_ack_requires_planner_exit_and_observed_stop():
    m = monitor()
    for step in range(0, 800, 16):
        feed(m, step)
    kwargs = dict(previous_intent="stop rotating, face [radio], and approach [radio]",
                  next_intent="pick up [radio]", status="CONTINUE")
    assert not m.acknowledge_recovery_exit(**kwargs)  # still spinning
    feed(m, 800, velocity=(0,0,0))
    assert not m.acknowledge_recovery_exit(**kwargs)  # only first stopped sample
    feed(m, 816, velocity=(0,0,0))
    assert not m.acknowledge_recovery_exit(**{**kwargs, "status": "REPLAN"})
    assert not m.acknowledge_recovery_exit(**{**kwargs, "previous_intent": "pick up [radio]"})
    assert m.acknowledge_recovery_exit(**kwargs)
    assert m.feedback == "none" and not m.history


def test_runtime_pause_replans_and_passes_feedback_without_low_action_override():
    from test_memlite_causal_runtime import Inferencer, wrapper, obs, step
    inf = Inferencer()
    w = wrapper(inf)
    w._progress_monitor = monitor(min_rotation_rad=.1, min_seconds=.1, window_seconds=5.)
    # Fixture has a dummy camera; monitor always gets the serving head key.
    w._progress_monitor.image_signature = lambda image: np.zeros((3, 24, 24), np.float32)
    original_observe = w._progress_monitor.observe
    w._progress_monitor.observe = lambda **kw: original_observe(**{**kw, "head_rgb": np.zeros((3, 24, 24), np.uint8)})
    o = obs()
    o["images"]["head_rgb"] = o["images"]["cam"]
    o["state"]["base_qvel"][:] = [0, 0, .3]
    holds = []
    for _ in range(24):
        action = step(w, o)
        holds.append(w._chunk_is_hold)
    assert any(holds) and w._progress_monitor.trigger_count == 1
    assert any(x.get("execution_feedback") == REPEATED_ROTATION_FEEDBACK for x in inf.high_obs)
    w.reset()
    assert w._progress_monitor.feedback == "none" and not w._progress_hold_pending


def test_runtime_exit_planner_sees_alert_but_subsequent_low_sees_acknowledged_feedback():
    from copy import deepcopy
    from test_memlite_causal_runtime import Inferencer, wrapper, obs, step, proposal

    inf = Inferencer([proposal("pick up [radio]")])
    low_observations = []
    infer_low = inf.infer_low_level_action

    def capture_low(observations, intents):
        low_observations.append(deepcopy(observations[0]))
        return infer_low(observations, intents)

    inf.infer_low_level_action = capture_low
    w = wrapper(inf)
    recovery_intent = "stop rotating, face [radio], and approach [radio]"
    w.mem_state.intent_text = recovery_intent
    w._progress_monitor = monitor()
    feed(w._progress_monitor, 0, velocity=(0, 0, 0), intent=recovery_intent)
    w._progress_monitor.feedback = REPEATED_ROTATION_FEEDBACK
    w._served_action_count = 16
    o = obs()
    o["images"]["head_rgb"] = np.zeros((3, 24, 24), np.uint8)
    o["state"]["base_qvel"][:] = 0
    step(w, o)
    assert inf.high_obs[0]["execution_feedback"] == REPEATED_ROTATION_FEEDBACK
    assert low_observations[0]["execution_feedback"] == "none"
    assert w._progress_monitor.feedback == "none"
