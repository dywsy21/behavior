"""Four or fewer fixed saved observations, new perception context, zero control."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.grounding import localize_target
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.vision import VisualBundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.state) <= 4:
        raise ValueError("At most four preselected states, no retry")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Clean fixed client source required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    policy = GroundedPolicy(args.uri, args.revision, max_calls=len(args.state))
    rows = []
    for index, value in enumerate(args.state):
        source = Path(value)
        old = json.loads((source / "observation.json").read_text())
        old_context = json.loads(old["request"]["text"])["harness"]
        model = RobotModel(json.loads((source.parent / "robot_calibration.json").read_text()))
        proprio = json.loads((source / "proprio.json").read_text())
        state = model.state(np.asarray(proprio["q"]), np.asarray(proprio["gripper"]), np.zeros(3))
        harness = GroundedHarness([Goal(**old_context["goal"])])
        harness.index = old_context["goal_index"]
        harness.stage = old_context["stage"]
        harness.held = old_context["held_target_claims"]
        history = old_context["recent_executed"]
        if history:
            harness.last_action = Action(**history[-1]["action"])
            harness.feedback = history[-1]["feedback"]
        labels = [entry["label"] for entry in old["request"]["images"]]
        images = [Image.open(source / (label + ".png")).convert("RGB") for label in labels]
        raw = {view: np.asarray(Image.open(source / ("CURRENT_" + view.upper() + "_RAW.png")))
               for view in ("head", "left_wrist", "right_wrist")}
        bundle = VisualBundle(images, labels, proprio["geometry"], raw)
        observation, call = policy.observe(harness, state, bundle)
        call["request"]["images"] = [{"label": label} for label in labels]
        row = {"source": str(source), "old_result": old["result"], "new_call": call,
               "old_text_chars": len(old["request"]["text"]), "new_text_chars": len(call["request"]["text"]),
               "new_evidence": asdict(observation),
               "new_grounding": localize_target(observation, dict(np.load(source / "depth.npz")), model, state.q),
               "exact_same_input_pixels": old["result"]["images"] == call["result"]["images"],
               "controls": 0, "training_updates": 0}
        if not row["exact_same_input_pixels"]:
            raise ValueError("Observer comparison changed image inputs")
        rows.append(row)
        (output / f"case_{index:02d}.json").write_text(json.dumps(row, indent=2))
        print(json.dumps({"source": str(source), "old": old["result"]["text"],
                          "new": call["result"]["text"], "same_pixels": row["exact_same_input_pixels"]}), flush=True)
    result = {"code_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
              "model_identity": policy.identity, "model_calls": policy.calls, "controls": 0,
              "manual_point_review_required": True, "rows": rows}
    (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
