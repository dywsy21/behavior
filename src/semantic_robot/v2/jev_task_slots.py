"""Required task outcomes with Jev-selected order and hand, never scene truth.

One typed next_goal answer selects a complete slot/hand pair. Pick/place pairs
share the chosen hand. The existing executor supports only one carried object
at a time, so an unfinished transport blocks another pick or door operation.
This is a capability constraint, not a scripted choice between eligible goals.
"""
from dataclasses import asdict
import hashlib
import json
import re

from .harness import Goal, parse_plan
from .jev_client import choice

SCHEMA = "jev-task-slots-v4"
HANDS = ("left", "right", "both")


def load_slots(value, raw, task_id):
    fields = {"schema", "task_id", "official_instruction", "strategy_provenance",
              "slots", "dependencies", "target_references"}
    if (not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA
            or type(value["task_id"]) is not int or value["task_id"] != task_id):
        raise ValueError("Exact task-bound slot contract required")
    for name in ("official_instruction", "strategy_provenance"):
        if not isinstance(value[name], str) or not value[name].strip() or len(value[name]) > 4000:
            raise ValueError("Official text and truthful decomposition provenance required")
    slots, deps, refs = value["slots"], value["dependencies"], value["target_references"]
    if not isinstance(slots, list) or not 1 <= len(slots) <= 16:
        raise ValueError("1..16 required slots")
    n = len(slots)
    if (not isinstance(deps, list) or len(deps) != n or not isinstance(refs, list)
            or len(refs) != n or any(r != "world" for r in refs)):
        raise ValueError("Slot v4 supports independently observed world targets only")
    for i, parents in enumerate(deps):
        if (not isinstance(parents, list) or any(type(j) is not int or not 0 <= j < n or j == i for j in parents)
                or len(set(parents)) != len(parents)):
            raise ValueError("Invalid dependency IDs")
    ancestors = [set() for _ in slots]
    done = set()
    while len(done) < n:
        ready = [i for i in range(n) if i not in done and set(deps[i]) <= done]
        if not ready:
            raise ValueError("Dependency cycle")
        for i in ready:
            ancestors[i] = set(deps[i])
            for j in deps[i]:
                ancestors[i].update(ancestors[j])
        done.update(ready)
    picks, places = {}, {}
    for i, slot in enumerate(slots):
        if (not isinstance(slot, dict) or set(slot) != {"kind", "target", "done_when", "level", "hand_options", "binding"}
                or not isinstance(slot["hand_options"], list) or not slot["hand_options"]
                or any(hand not in HANDS for hand in slot["hand_options"])
                or len(set(slot["hand_options"])) != len(slot["hand_options"])):
            raise ValueError("Exact bounded slot fields and unique hand options required")
        Goal(slot["kind"], slot["target"], slot["hand_options"][0], slot["done_when"], slot["level"])
        binding = slot["binding"]
        if slot["kind"] in ("pick", "place"):
            if not isinstance(binding, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", binding):
                raise ValueError("Each transported object needs a stable semantic binding")
            group = picks if slot["kind"] == "pick" else places
            if binding in group:
                raise ValueError("Exactly one pick and one place per object binding")
            group[binding] = i
        elif binding is not None:
            raise ValueError("Only pick/place slots use an object binding")
    if set(picks) != set(places):
        raise ValueError("All required object transports need both pick and place")
    for binding, pick in picks.items():
        place = places[binding]
        if (pick not in ancestors[place] or slots[pick]["hand_options"] != slots[place]["hand_options"]
                or slots[pick]["level"] != slots[place]["level"]):
            raise ValueError("Place must depend on its pick with the same hand/level domain")
        if ancestors[place] - {pick} != ancestors[pick]:
            raise ValueError("All non-pick prerequisites for place must be satisfied before loading the hand")
    identity = {**value, "source": "required_task_outcomes_not_observed_state",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "official_instruction_sha256": hashlib.sha256(value["official_instruction"].encode()).hexdigest()}
    # Validate existence of a full executable ordering without making it the
    # policy choice. Only Jev's later answers can become the deployed plan.
    prefix, preview = [], []
    while len(prefix) < n:
        available = slot_options(identity, prefix)
        if not available:
            raise ValueError("Dependency graph deadlocks the executor's carry resources")
        key = next(iter(available))
        preview.append(available[key][1]); prefix.append(key)
    parse_plan(json.dumps([asdict(g) for g in preview]))
    return preview, identity


def _available(identity, selected, held):
    result = {}
    for i, slot in enumerate(identity["slots"]):
        if i in selected or not set(identity["dependencies"][i]) <= selected:
            continue
        if held is not None:
            if slot["kind"] != "place" or slot["binding"] != held[0]:
                continue
            hands = [held[1]]
        elif slot["kind"] == "place":
            continue
        else:
            hands = slot["hand_options"]
        for hand in hands:
            goal = Goal(slot["kind"], slot["target"], hand, slot["done_when"], slot["level"])
            result[f"goal_{i}_{hand}"] = (i, goal)
    return result


def slot_options(identity, prefix):
    selected, held = set(), None
    for key in prefix:
        available = _available(identity, selected, held)
        if not isinstance(key, str) or key not in available:
            raise ValueError("Invalid slot/hand prefix, repeated object or resource conflict")
        index, goal = available[key]
        selected.add(index)
        if goal.kind == "pick":
            held = (identity["slots"][index]["binding"], goal.hand)
        elif goal.kind == "place":
            held = None
    return _available(identity, selected, held)


def materialize_slots(identity, prefix, *, complete=True):
    goals, indices, visited = [], [], []
    for key in prefix:
        available = slot_options(identity, visited)
        if not isinstance(key, str) or key not in available:
            raise ValueError("Model chose an ineligible slot/hand")
        index, goal = available[key]
        goals.append(goal); indices.append(index); visited.append(key)
    if complete:
        if len(goals) != len(identity["slots"]):
            raise ValueError("Incomplete required task plan")
        parse_plan(json.dumps([asdict(g) for g in goals]))
    return goals, indices


def slot_request(identity, instruction, prefix):
    from .jev_client import JevError
    if instruction != identity["official_instruction"]:
        raise JevError("Official task instruction differs from the qualified task contract")
    available = slot_options(identity, prefix)
    if not available:
        raise ValueError("No eligible required slot remains")
    goals, indices = materialize_slots(identity, prefix, complete=False)
    state = {"phase": "ORDER_FUTURE_PLAN_NOT_EXECUTION",
             "official_task_instruction_verbatim": instruction,
             "decomposition_provenance": identity["strategy_provenance"],
             "required_slots": identity["slots"], "dependency_order": identity["dependencies"],
             "hypothetical_plan_prefix": [{"choice": key, "slot": i, "goal": asdict(g)}
                                          for key, i, g in zip(prefix, indices, goals)],
             "eligible_next_choices": {key: {"slot": i, "goal": asdict(g)} for key, (i, g) in available.items()},
             "current_execution_evidence": "No step has executed. No holding, visibility or success is asserted.",
             "execution_contract": "Order all required future outcomes. The executor permits one transported object at a time. "
                 "Pick/place must use the same Jev-selected hand. Slot dependencies and carry resources only constrain eligibility; "
                 "YOU choose between eligible tasks and hands. Real execution still requires fresh perception, independent grasp "
                 "verification, safety preflight and a separate Jev command for EVERY action."}
    options = {key: "Append future task slot with this hand: " + json.dumps(asdict(goal), ensure_ascii=False)
               for key, (_, goal) in available.items()}
    options["abstain"] = "The required future plan is inconsistent; stop without motion. Lack of current images alone is not inconsistency."
    return state, {"next_goal": choice("Choose ONE offered future slot/hand pair, not an immediate command. "
        "Preserve every task object, shared destination and final operation. Dependencies refer to the hypothetical prefix, "
        "not already executed actions. Hand selection and ordering among eligible branches are yours. "
        "Do not infer any current grasp or task success. Abstain if the future requirements conflict.", options)}
