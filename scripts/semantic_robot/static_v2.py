"""15 preselected development observations, 30 calls/model, zero actuation.

The first subgoal is supplied for a controlled local comparison; this is not an
autonomous-planning benchmark. Human labels/review must remain separate from
model inputs. Existing failed frames do not establish positive-grasp sensitivity.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
from semantic_robot.v2.harness import Goal, TaskHarness
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.policy import VLMPolicy, OBSERVE_SYSTEM, ACTION_SYSTEM, observation_context
from semantic_robot.v2.protocol import Action, Evidence
from semantic_robot.v2.vision import prepare_views
from serve_v2 import normalize_json_transport

CASES = [(run, step) for run, steps in (
    ("radio_4b_v2", (0,8,12,16,32)), ("radio_2b_v2", (0,8,16,32,47)),
    ("plates_4b_v2", (0,8,16,32,47))) for step in steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--uri", default="http://127.0.0.1:8907")
    p.add_argument("--revision")
    p.add_argument("--reuse-run", help="Completed original static result.json; never rerun cached generations")
    p.add_argument("--reuse-service", help="Original service dir with identity and raw call ledger")
    args = p.parse_args()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():
        raise RuntimeError("Use fixed clean source")
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    model = RobotModel(json.loads(Path(args.calibration).read_text()))
    cached, original, reused, original_calls = None, None, 0, 0
    if bool(args.reuse_run) != bool(args.reuse_service) or (args.prepare_only and args.reuse_run):
        raise ValueError("Both cache identities required for inference-only transport repair")
    if args.reuse_run:
        original = json.loads(Path(args.reuse_run).read_text())
        identity = json.loads((Path(args.reuse_service)/"identity.json").read_text())
        ledger = [json.loads(line) for line in (Path(args.reuse_service)/"calls.jsonl").read_text().splitlines()]
        cached = [r for r in ledger if r.get("kind")=="observe"]
        original_calls = original["calls"]
        if (identity["revision"] != args.revision or original["status"] != "complete" or
                len(cached) != len(CASES) or len(ledger) != original_calls or
                original["calibration_sha"] != model.sha or original["cases"] != len(CASES)):
            raise ValueError("Cache revision/calibration/case/call identity mismatch")
    remaining_calls = 30-original_calls
    policy = None if args.prepare_only else VLMPolicy(args.uri,args.revision,max_calls=remaining_calls)
    rows = []
    for index,(run,step) in enumerate(CASES):
        source = root/run/f"decision_{step:03d}"
        proprio = json.loads((source/"proprio.json").read_text())
        state = model.state(np.asarray(proprio["q"]),np.asarray(proprio["gripper"]),np.zeros(3))
        images = {v+"_rgb":np.asarray(Image.open(source/(v+"_rgb.png")).convert("RGB")) for v in ("head","left_wrist","right_wrist")}
        previous = None
        if step > 0:
            earlier = root/run/f"decision_{step-1:03d}"
            previous = {v:np.asarray(Image.open(earlier/(v+"_rgb.png")).convert("RGB")) for v in ("head","left_wrist","right_wrist")}
        bundle = prepare_views(images,model,state.q,previous)
        case = out/f"case_{index:02d}_{run}_{step:03d}"; case.mkdir()
        for label,image in zip(bundle.labels,bundle.images): image.save(case/(label+".png"))
        goal = Goal("pick","radio on the coffee table","right","radio moves with gripper") if run.startswith("radio") else Goal("pick","a pizza on its own plate","right","plate and pizza move together",True)
        manager = TaskHarness([goal])
        row = {"index":index,"source":str(source),"run":run,"step":step,"goal":asdict(goal),
               "calibration_sha":model.sha,"proprio":proprio,"calls":[],"acted":False,
               "scope":"human-specified first pick subgoal; not full task planning"}
        # Local contact sheet for HUMAN review, never included in model input.
        sheet = Image.new("RGB",(960,352),(16,20,28))
        for j,v in enumerate(("head","left_wrist","right_wrist")):
            sheet.paste(Image.fromarray(images[v+"_rgb"]).resize((320,320)),(320*j,32))
        ImageDraw.Draw(sheet).text((8,8),f"{index:02d} {run} decision {step:03d} | HUMAN REVIEW, not a training label",fill="white")
        sheet.save(case/"human_contact_sheet.jpg")
        if policy:
            try:
                if cached is None:
                    obs, call = policy.observe(manager,state,bundle)
                else:
                    stored = cached[index]
                    if stored.get("hit_token_cap"):
                        raise ValueError("Cached generation was truncated; never repair into a valid action")
                    expected_text = observation_context(manager,state,bundle)
                    if stored["prompt_sha256"] != hashlib.sha256((OBSERVE_SYSTEM+expected_text).encode()).hexdigest():
                        raise ValueError("Cached observation prompt drift")
                    expected_images = []
                    for label,image in zip(bundle.labels,bundle.images):
                        image = image.copy(); image.thumbnail((640,640))
                        expected_images.append({"label":label,"size":list(image.size),
                            "pixels_sha256":hashlib.sha256(image.tobytes()).hexdigest()})
                    if stored["images"] != expected_images:
                        raise ValueError("Cached observation pixels drift")
                    normalized, wrapper = normalize_json_transport(stored.get("raw_text",stored["text"]))
                    result = dict(stored, text=normalized, raw_text=stored.get("raw_text",stored["text"]),
                                  transport_wrapper_removed=wrapper, reused_generation=True)
                    obs = Evidence.parse(normalized)
                    call = {"result":result,"request":{"system":OBSERVE_SYSTEM,"text":expected_text}}
                    reused += 1
                row["calls"].append({"kind":"observe","result":call["result"],"system":call["request"]["system"],"text":call["request"]["text"]})
                row["evidence"] = obs.as_dict()
                manager.observe(obs,state,bundle.geometry)
                old = [] if original is None else [c for c in original["rows"][index]["calls"] if c["kind"]=="act"]
                if old:
                    expected_text = observation_context(manager,state,bundle)
                    expected_text += "\nVisible evidence: "+json.dumps(asdict(manager.observation))
                    expected_text += "\nChoose exactly one of these complete commands:\n"+"\n".join(a.text() for a in manager.palette())
                    stored = old[0]
                    if stored["system"] != ACTION_SYSTEM or stored["text"] != expected_text:
                        raise ValueError("Cached action prompt drift")
                    action = Action.parse(stored["result"]["text"]); manager.authorize(action)
                    call = {"result":dict(stored["result"],reused_generation=True),
                            "request":{"system":stored["system"],"text":stored["text"]}}
                    reused += 1
                else:
                    action, call = policy.act(manager,state,bundle)
                row["calls"].append({"kind":"act","result":call["result"],"system":call["request"]["system"],"text":call["request"]["text"]})
                row["action"], row["stage"] = asdict(action), manager.stage
            except Exception as exc:
                row["error"] = repr(exc)  # no hidden retry or actuator fallback
        (case/"result.json").write_text(json.dumps(row,indent=2))
        rows.append(row)
        print(json.dumps({k:row[k] for k in ("index","run","step","evidence","action","error") if k in row}),flush=True)
    result = {"status":"complete","code_commit":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
              "cases":len(rows),"calls":policy.calls if policy else 0,"actuations":0,"training_updates":0,
              "reused_calls":reused,"original_calls":original_calls,
              "total_unique_generations":original_calls+(policy.calls if policy else 0),
              "model_identity":policy.identity if policy else None,
              "reuse_run":args.reuse_run,"reuse_service":args.reuse_service,
              "calibration_sha":model.sha,"rows":rows,"human_review_pending":True}
    (out/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
