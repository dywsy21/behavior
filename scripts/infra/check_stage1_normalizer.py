"""CPU all-task action round trips through the exact deployment postprocessor.

Use the same locatable stratified windows as RGB/token QA. This neither fits
statistics to evaluation data nor changes the model's state coordinates.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch

from g05.data.memlite_stage1_dataset import Stage1Dataset
from g05.utils.training.stage1_model import configuration, make_processor
from g05.utils.training.stage1_runtime import atomic_json, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    started = time.monotonic()
    manifest = json.loads((args.release/"manifest.json").read_text())
    qa = json.loads((args.qa/"result.json").read_text())
    if qa["manifest_sha256"] != sha256(args.release/"manifest.json"):
        raise ValueError("QA and data identity differ")
    rows = json.loads((args.qa/"rows.json").read_text())
    cfg = configuration(args.root,"low",manifest["task_names"])
    processor = make_processor(cfg,training=False)
    datasets = {s:Stage1Dataset(args.release,cfg,"low",s) for s in ("train","eval")}
    results = []
    for row in rows:
        raw, identity = datasets[row["split"]].raw(row["candidate"],images=False)
        if any(identity[k] != row[k] for k in identity):
            raise ValueError("QA source locator changed")
        transformed = processor.action_state_transform(processor.action_filter.forward(deepcopy(raw)))
        # Check the independent streaming-bounds formulas against the processor.
        expected = {
            "left_arm": raw["action"]["left_arm"]-raw["state"]["left_arm"],
            "right_arm": raw["action"]["right_arm"]-raw["state"]["right_arm"],
            "left_gripper": raw["action"]["left_gripper"],
            "right_gripper": raw["action"]["right_gripper"],
            "lower_body": torch.cat((raw["action"]["trunk_qpos"],raw["action"]["base_qvel"]),dim=-1),
        }
        expected["lower_body"][:,:3] -= raw["state"]["trunk_qpos"][:,:3]
        if any(not torch.equal(transformed["action"][k],v) for k,v in expected.items()):
            raise ValueError("Streaming bounds and actual embodiment transform differ")
        norm = processor.preprocess_action_state(deepcopy(raw))
        decoded = processor.postprocess(dict(action_fm=norm["action"][None],proprio=norm["state"][None],
            _raw_state_anchor={k:v[None] for k,v in raw["state"].items()}))["action_fm"]
        valid = identity["valid_action_steps"]
        error = {k:float((decoded[k][0,:valid]-v[:valid]).abs().max()) for k,v in raw["action"].items()}
        results.append(dict(**identity,split=row["split"],error_by_part=error,max_error=max(error.values()),
                            max_abs_normalized_target=float(norm["action"][:valid].abs().max())))
    bad = [r for r in results if r["max_error"] > 1e-4]
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA in CPU normalization audit")
    result = dict(status="PASS" if not bad else "FAIL",manifest_sha256=qa["manifest_sha256"],
        stats_sha256=sha256(cfg["stats_path"]),qa_rows_sha256=sha256(args.qa/"rows.json"),
        windows=len(results),tasks=sorted({r["task"] for r in results}),
        max_error=max(r["max_error"] for r in results),failures=len(bad),
        raw_state_anchor=True,actual_transform_equals_streaming_bounds=True,
        seconds=time.monotonic()-started,rows=results)
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}),flush=True)
    if bad:
        raise ValueError("Valid action roundtrip exceeds tolerance; evidence retained")


if __name__ == "__main__":
    main()
