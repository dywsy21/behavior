"""Frozen six-case development campaign; no simulator/model imports or writes."""
import hashlib
import json
from pathlib import Path
import re

from .jev_policy import load_task_plan

REGISTRY = "configs/semantic_robot/jev_multitask_v1.json"
CASE_IDS = ("task0a", "task0b", "task1a", "task1b", "task3a", "task3b")
TASK_NAMES = {0: "turning_on_radio", 1: "picking_up_trash", 3: "cleaning_up_plates_and_food"}
BUDGET = dict(max_decisions=96, max_controls=3072, action_seconds=2400, total_seconds=3600,
              cleanup_seconds=60, jev_requests=288, observer_requests=215)
WINDOW_ROOT = Path("/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c1_windows_v2_matched")
ROBOT_SHA = "a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_registry(value):
    if (value.get("schema") != "jev-multitask-campaign-v1" or value.get("campaign_id") != "jev_multi_20260930"
            or value.get("budget") != BUDGET or value.get("split") != "train"
            or value.get("window_root") != str(WINDOW_ROOT) or value.get("robot_config_sha256") != ROBOT_SHA
            or any(type(value.get(k)) is not int or value[k] != 0 for k in
                   ("seed", "training_updates", "expert_prefix_controls", "diagnostic_replay_controls"))):
        raise ValueError("Unregistered Jev campaign identity/budget")
    cases, plans = value.get("cases"), value.get("plans")
    if (not isinstance(cases, list) or [c.get("id") for c in cases] != list(CASE_IDS)
            or not isinstance(plans, dict) or set(plans) != {"0", "1", "3"}):
        raise ValueError("Exactly the six preregistered cases and three plans required")
    seen = set()
    for c in cases:
        if (set(c) != {"id", "task_id", "task_name", "instance", "window_name", "window_sha256"}
                or type(c["task_id"]) is not int or c["task_id"] not in TASK_NAMES
                or c["id"][:-1] != "task"+str(c["task_id"])
                or c["task_name"] != TASK_NAMES[c["task_id"]]
                or type(c["instance"]) is not int or c["instance"] < 0
                or not re.fullmatch(r"c1v2-matched-t[013]-train-e\d+-f\d+-grasp", c["window_name"])
                or not c["window_name"].startswith(f"c1v2-matched-t{c['task_id']}-train-")
                or not re.fullmatch(r"[0-9a-f]{64}", c["window_sha256"])):
            raise ValueError("Invalid registered task/window identity")
        pair = (c["task_id"], c["instance"])
        if pair in seen:
            raise ValueError("Different original instances required, not repeated successes")
        seen.add(pair)
    for task, plan in plans.items():
        expected = f"configs/semantic_robot/jev_task{task}_" + ("plan.json" if task == "0" else "slots.json")
        if set(plan) != {"path", "sha256"} or plan["path"] != expected or not re.fullmatch(r"[0-9a-f]{64}", plan["sha256"]):
            raise ValueError("Only registered repository plan paths/hashes")
    return value


def load_registry(repo):
    repo = Path(repo)
    raw = (repo/REGISTRY).read_bytes()
    value = validate_registry(json.loads(raw))
    for task, row in value["plans"].items():
        path = repo/row["path"]
        if sha(path.read_bytes()) != row["sha256"]:
            raise ValueError("Registered task plan bytes changed")
        load_task_plan(path, int(task))
    return value, sha(raw)


def case_identity(repo, case_id, task_id=None):
    value, registry_sha = load_registry(repo)
    if case_id not in CASE_IDS:
        raise ValueError("Unknown registered episode")
    row = next(c for c in value["cases"] if c["id"] == case_id)
    if task_id is not None and (type(task_id) is not int or task_id != row["task_id"]):
        raise ValueError("Episode key does not match requested task")
    return {**row, "registry_sha256": registry_sha, "seed": 0, "split": "train",
            "window_path": str(WINDOW_ROOT/row["window_name"]/"window.json"),
            "robot_config_sha256": ROBOT_SHA, "task_plan": value["plans"][str(row["task_id"])],
            "budget": dict(BUDGET)}


def validate_window(case, raw):
    if sha(raw) != case["window_sha256"]:
        raise ValueError("Registered original-start window bytes changed")
    value = json.loads(raw)
    expected = dict(kind="native_oracle_low_window", immutable=True, window_id=case["window_name"],
                    task_name=case["task_name"], instance_id=case["instance"], official_mode="train", seed=0,
                    robot_config_sha256=ROBOT_SHA)
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError("Registered window task/instance/seed/robot mismatch")


def resolve_case(repo, case_id, task_id):
    case = case_identity(repo, case_id, task_id)
    validate_window(case, Path(case["window_path"]).read_bytes())
    return case
