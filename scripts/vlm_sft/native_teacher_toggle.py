"""Scoped read-only observation of ONE target's real OG toggle updates.

Never patches a class, sets a state, steps physics, or changes an update result.
Measurement failures are deferred until readout, so the original update still
runs exactly once. This module is only for privileged TRAIN teacher processes.
"""
from contextlib import contextmanager
from types import MethodType

TOGGLE_SOURCE="/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson/object_states/toggle.py"
TOGGLE_SHA="a7a88f4a55242816ab930fe5336cfd94b26e2aaf6e2d33aa0239ed47eca2fd3b"


def verify_installed_dependency():
    from common import sha
    if sha(TOGGLE_SOURCE)!=TOGGLE_SHA:raise ValueError("Installed toggle dependency changed before reset")
    return TOGGLE_SHA


class ToggleObserver:
    def __init__(self, state, measure, max_events=32):
        self.state, self.measure, self.max_events = state, measure, max_events
        self.events, self.error = [], None
        self.index = 0
        self.active = False

    @contextmanager
    def installed(self):
        if self.active: raise RuntimeError("Toggle observer already installed")
        sentinel = object()
        previous = self.state.__dict__.get("_update", sentinel)
        original = self.state._update
        observer = self
        def observed(this, *args, **kwargs):
            before = None
            try:
                before = {"value": this.value, "counter": this.robot_can_toggle_steps,
                          "measurement": observer.measure()}
            except Exception as exc:
                observer.error = observer.error or exc
            # Preserve return value AND the original exception object.
            result = original(*args, **kwargs)
            observer.index += 1
            try:
                event = {"update_index": observer.index, "before": before,
                         "after": {"value": this.value, "counter": this.robot_can_toggle_steps}}
                if len(observer.events) >= observer.max_events:
                    raise RuntimeError("Too many state updates between control readouts")
                observer.events.append(event)
            except Exception as exc:
                observer.error = observer.error or exc
            return result
        self.state._update = MethodType(observed, self.state)
        self.active = True
        try:
            yield self
        finally:
            if previous is sentinel: del self.state.__dict__["_update"]
            else: self.state._update = previous
            self.active = False

    def drain(self):
        if not self.active: raise RuntimeError("Toggle observer not active")
        if self.error is not None: raise RuntimeError("Missing actual toggle update evidence") from self.error
        events, self.events = self.events, []
        return events


class PressCausality:
    """Five actual target updates, not five assumed 30 Hz controls."""
    def __init__(self, arm):
        self.arm, self.last_index, self.streak = arm, 0, 0
        self.previous_after = None

    def consume(self, events):
        caused = False
        for event in events:
            if event["update_index"] != self.last_index+1:
                raise ValueError("Missing/repeated toggle update sequence")
            before, after = event["before"], event["after"]
            if before is None or type(before["value"]) is not bool or type(after["value"]) is not bool:
                raise ValueError("Unknown toggle transition")
            if any(type(x["counter"]) is not int or x["counter"] < 0 for x in (before, after)):
                raise ValueError("Unknown real toggle counter")
            if self.previous_after is not None and {k:before[k] for k in ("value","counter")} != self.previous_after:
                raise ValueError("Toggle state changed outside observed updates")
            m = before["measurement"]
            if (set(m["arm_marker_overlap"]) != {"left","right"} or
                    set(m["arm_target_contact"]) != {"left","right"} or
                    any(type(v) is not bool for v in [*m["arm_marker_overlap"].values(),
                                                     *m["arm_target_contact"].values(),m["other_robot_marker_overlap"]])):
                raise ValueError("Unknown arm-specific marker/contact evidence")
            other = "right" if self.arm == "left" else "left"
            own = (m["arm_marker_overlap"][self.arm] and m["arm_target_contact"][self.arm]
                   and not m["arm_marker_overlap"][other] and not m["other_robot_marker_overlap"])
            self.streak = self.streak+1 if own and after["counter"] == before["counter"]+1 else 0
            if before["value"] is False and after["value"] is True:
                if not own or self.streak < 5 or after["counter"] != 5:
                    raise ValueError("Toggle edge not exclusively attributable to the designated arm")
                caused = True
            self.last_index = event["update_index"]
            self.previous_after = after
        return caused
