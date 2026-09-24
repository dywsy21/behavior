"""Reproduce the public Zetta artifact blocker on CPU, without model/simulator.

Run from the repository root:
  python -m experiments.zetta_behavior.preflight --zetta-source /path/to/Zetta

Exit 2 means the audited release is not deployable; it is not a task failure.
The synthetic rule trace demonstrates upstream temporal evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

from .bridge import inspect_rule_requirements


UPSTREAM_COMMIT = "1fee179644d52c32fa5a7728751cf0853a29b9c0"
ARTIFACT = "scripts/experiments/critic-recovery-v3.json"
ARTIFACT_SHA256 = "3b90203f33058b36d3e2b6efc3794e16649a3b282080376b9863aef85265cac0"
PUBLIC_FEATURES = frozenset({
    "command.available", "episode.terminated", "episode.truncated",
})


def load_upstream(root: Path):
    """Import the reviewed immutable source, not a lookalike installed module."""
    root = root.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise ValueError("use a clean Zetta checkout at the pinned upstream commit")
    sys.path.insert(0, str(root))
    modules = {}
    for name in ("zetta", "zetta.evolution", "zetta.evolution.jsonio",
                 "zetta.evolution.models", "zetta.evolution.critic"):
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        expected = root.joinpath(*name.split("."))
        expected = expected / "__init__.py" if expected.is_dir() else expected.with_suffix(".py")
        if actual != expected:
            raise ValueError(f"wrong imported source for {name}")
        modules[name] = module
    return modules["zetta.evolution.models"], modules["zetta.evolution.critic"]


def run_preflight(root: Path) -> dict:
    models, critics = load_upstream(root)
    raw = (root / ARTIFACT).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ARTIFACT_SHA256:
        raise ValueError("public critic artifact bytes changed")
    payload = json.loads(raw)
    requirements = inspect_rule_requirements(payload, public_features=PUBLIC_FEATURES)
    try:
        models.CandidateBundle.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        bundle_error = {"type": type(error).__name__, "message": str(error)}
    else:
        bundle_error = None
    rules = tuple(models.CriticRule(**{
        **rule,
        "evidence_ids": tuple(rule["evidence_ids"]),
        "activation_conditions": tuple(
            models.CriticPredicate(**condition)
            for condition in rule["activation_conditions"]
        ),
    }) for rule in payload["critic_rules"])
    critic = critics.TemporalCritic(rules)
    public_only = {
        "command.available": True,
        "episode.terminated": False,
        "episode.truncated": False,
    }
    try:
        critic.evaluate(public_only, step_index=0)
    except (KeyError, TypeError, ValueError) as error:
        public_error = {"type": type(error).__name__, "message": str(error)}
    else:
        public_error = None

    # Explicitly synthetic values, NOT BEHAVIOR observations or a policy test.
    critic.reset()
    synthetic = {
        **public_only,
        "privileged.available": True,
        "privileged.task.semantic_available": True,
        "privileged.task.goal.progress_available": True,
        "privileged.task.goal.progress": 0.0,
        "privileged.entity.main_table_plate_region.distance_to_eef_m": 0.2,
        "privileged.joint.wooden_cabinet_1_middle_level.normalized": 1.0,
        "privileged.task.success": False,
    }
    trace = [critic.evaluate(synthetic, step_index=step) for step in range(4)]
    if [len(proposals) for proposals in trace] != [0, 0, 1, 0]:
        raise ValueError("unexpected upstream temporal-critic behavior")
    return {
        "status": "BLOCKED_RELEASE_AND_EMBODIMENT",
        "upstream_commit": UPSTREAM_COMMIT,
        "artifact": ARTIFACT,
        "artifact_sha256": digest,
        "artifact_candidate_id": payload["candidate_id"],
        "referenced_but_unavailable_promoted_bundle_sha256": payload["source_promoted_bundle_sha256"],
        "artifact_suite": payload["suite"],
        "critic_rule_count": len(rules),
        "recovery_rule_count": len(payload.get("recovery_rules", [])),
        "upstream_bundle_load_error": bundle_error,
        "public_only_critic_error": public_error,
        "feature_requirements": requirements,
        "synthetic_rule_test": {
            "label": "SYNTHETIC_UPSTREAM_CRITIC_ONLY_NOT_BEHAVIOR",
            "proposal_counts": [len(items) for items in trace],
            "triggered_rule_id": trace[2][0]["rule_id"],
            "environment_write": trace[2][0]["environment_write"],
        },
        "behavior_recovery_executor": "NOT_IMPLEMENTED_NO_RELEASED_PROGRAM",
        "g05_model_loads": 0,
        "model_inference_calls": 0,
        "simulator_resets": 0,
        "training_steps": 0,
        "success_rate": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zetta-source", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.zetta_source.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
