"""Bind a complete task plan and each executed goal to durable Jev answers."""
import hashlib
import json

from .harness import parse_plan
from .jev_client import JevError


def validate_plan_proof(proof, ledger=None):
    fields = {"schema", "task_id", "task_plan_sha256", "selected_goal_ids", "slot_indices",
              "goals", "target_references", "planning_calls", "plan_sha256"}
    if (not isinstance(proof, dict) or set(proof) != fields
            or proof["schema"] != "jev-complete-plan-ownership-v1" or type(proof["task_id"]) is not int):
        raise JevError("Missing complete Jev plan ownership proof")
    body = {k: v for k, v in proof.items() if k != "plan_sha256"}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if proof["plan_sha256"] != sha:
        raise JevError("Changed Jev materialized plan digest")
    try:
        goals = parse_plan(json.dumps(proof["goals"]))
        n = len(goals)
        if (any(not isinstance(proof[k], list) or len(proof[k]) != n
                for k in ("selected_goal_ids", "slot_indices", "target_references", "planning_calls"))
                or any(not isinstance(k, str) or not k.startswith("goal_") for k in proof["selected_goal_ids"])
                or len(set(proof["selected_goal_ids"])) != n
                or any(type(i) is not int for i in proof["slot_indices"])
                or set(proof["slot_indices"]) != set(range(n))
                or proof["planning_calls"] != list(range(1, n+1))
                or any(type(i) is not int for i in proof["planning_calls"])
                or any(r not in ("world", "held_left", "held_right") for r in proof["target_references"])):
            raise ValueError("Incomplete or altered plan slots/call ranges")
    except (ValueError, TypeError, KeyError) as error:
        raise JevError("Invalid complete Jev task plan") from error
    if ledger is not None:
        valid = {r["call"]: r for r in ledger if r.get("event") == "validated"}
        for call, selected in zip(proof["planning_calls"], proof["selected_goal_ids"]):
            if valid.get(call, {}).get("selections", {}).get("next_goal") != selected:
                raise JevError("Materialized plan differs from durable Jev choices")
    return goals


def validate_goal_proof(row, plan):
    proof = row.get("jev_goal_authority") or {}
    index = proof.get("goal_index")
    if (type(index) is not int or not 0 <= index < len(plan["goals"])
            or proof != {"goal_index": index, "goal": plan["goals"][index],
                         "plan_sha256": plan["plan_sha256"],
                         "selected_option": plan["selected_goal_ids"][index]}):
        raise JevError("Executed goal/hand is not bound to the Jev task plan")


def validate_plan_contract(proof, path, task_id, ledger):
    """The launcher supplies the registered path, never a result-provided path."""
    from .jev_policy import load_task_plan, planning_request
    goals = validate_plan_proof(proof, ledger)
    required, identity = load_task_plan(path, task_id)
    if proof["task_id"] != task_id or proof["task_plan_sha256"] != identity["sha256"]:
        raise JevError("Plan belongs to another registered task/config")
    if identity.get("schema") == "jev-task-slots-v4":
        from .jev_task_slots import materialize_slots
        materialized, indices = materialize_slots(identity, proof["selected_goal_ids"])
    else:
        indices, materialized = [], []
        for key in proof["selected_goal_ids"]:
            _, questions = planning_request(required, identity, identity["official_instruction"], indices)
            if key == "abstain" or key not in questions["next_goal"]["criteria"]:
                raise JevError("Plan violates required goal order/resources")
            index = int(key.removeprefix("goal_"))
            indices.append(index); materialized.append(required[index])
    if (materialized != goals or proof["slot_indices"] != indices or len(indices) != len(required)
            or proof["target_references"] != [identity["target_references"][i] for i in indices]):
        raise JevError("Executed plan/hand differs from source requirements and choices")
    return {"task_plan_sha256": identity["sha256"], "plan_sha256": proof["plan_sha256"],
            "ordered_slots_and_hands_checked": True, "required_goals": len(goals)}


def validate_actor_plan(result, path, task_id, ledger=None):
    """New campaigns require a complete bound plan OR a legal recorded abstention."""
    if result.get('stop_reason') != 'JEV_PLAN_ABSTAINED':
        return validate_plan_contract(result.get('jev_plan_ownership'), path, task_id, ledger)
    if result.get('jev_plan_ownership') is not None:
        raise JevError('Abstention cannot carry an executed plan')
    from .jev_policy import load_task_plan, planning_request
    goals, identity = load_task_plan(path, task_id)
    proof = result.get('planning_abstention') or {}
    selected = proof.get('selected_goal_ids')
    if not isinstance(selected, list) or len(selected) >= len(goals):
        raise JevError('Invalid incomplete-plan abstention prefix')
    prefix = []
    valid = {} if ledger is None else {r['call']: r for r in ledger if r.get('event') == 'validated'}
    for index, key in enumerate(selected + ['abstain'], 1):
        _, questions = planning_request(goals, identity, identity['official_instruction'], prefix)
        if key not in questions['next_goal']['criteria']:
            raise JevError('Abstained prefix violates registered plan')
        if ledger is not None and valid.get(index, {}).get('selections', {}).get('next_goal') != key:
            raise JevError('Abstained prefix differs from Jev ledger')
        if key != 'abstain':
            prefix.append(key if identity.get('schema') == 'jev-task-slots-v4' else int(key.removeprefix('goal_')))
    return {'task_plan_sha256': identity['sha256'], 'plan_sha256': None,
            'legal_abstention_prefix_checked': True, 'required_goals': len(goals)}
