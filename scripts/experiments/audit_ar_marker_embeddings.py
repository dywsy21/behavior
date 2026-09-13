"""Read-only CPU audit of actual codec markers and frozen AR vocabulary rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from probe_ar_execution_codec import sha, publish
from train_fm_method_probe import BASE
from native_action_initialization import NATIVE_PARENT, NATIVE_PARENT_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.parent != BASE or args.output.exists():
        raise ValueError("Use a new audit directory directly under the experiment root")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    read = lambda path: json.loads(Path(path).read_text())
    grammar_path = BASE / "ar_native_omission_audit_v1/result.json"
    if sha(grammar_path) != "429d8fa2e9d09e63477dd94433b58ca166442c4ca8b07e29ea1f0b456590b09e":
        raise RuntimeError("Actual codec grammar audit changed")
    grammar = read(grammar_path)["identity"]["grammar"]
    inspection = read(BASE / "ar_a4_fulltrain_v2/formal/checkpoint_inspection.json")
    if not inspection["passed"] or inspection["actual_updates"] != 500:
        raise RuntimeError("AR500 did not complete verified training")
    sources = [(NATIVE_PARENT, NATIVE_PARENT_SHA),
               (Path(inspection["checkpoint"]), inspection["checkpoint_sha256"])]
    markers = {int(key) + grammar["offset"]: tuple(value) for key, value in grammar["markers"].items()}
    expected = {("left_control", 0), ("left_control", 1), ("right_control", 0), ("right_control", 1),
                ("lower_body", 0), ("lower_body", 1), ("left_gripper", 0), ("right_gripper", 0)}
    if {(value[0], value[1]) for value in markers.values()} != expected:
        raise RuntimeError("Actual five-group/two-residual grammar changed")
    body = [key for key, value in markers.items() if value[0] == "lower_body"]
    args.output.mkdir(exist_ok=False)
    state_rows, reports = [], []
    for path, digest in sources:
        if sha(path) != digest:
            raise RuntimeError("Declared immutable checkpoint changed")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["model_state_dict"]
        matrices = {key.rsplit(".", 2)[-2]: value for key, value in payload.items()
                    if "vlm" in key and (key.endswith("input_proj.weight") or key.endswith("output_proj.weight"))}
        if set(matrices) != {"input_proj", "output_proj"}:
            raise RuntimeError("Expected exactly the VLM input and output vocabulary matrices")
        selected = {}
        for name, weight in matrices.items():
            if weight.ndim != 2 or weight.shape[1] != 2048 or max(markers) >= weight.shape[0]:
                raise RuntimeError("Vocabulary shape changed")
            rows = weight[list(markers)].detach().clone()
            if not torch.isfinite(rows).all():
                raise RuntimeError("Nonfinite marker vocabulary weights")
            selected[name] = rows
            vectors = weight[body].float()
            reports.append(dict(checkpoint=str(path), checkpoint_sha256=digest, matrix=name, shape=list(weight.shape),
                body_row_cosine=float(torch.nn.functional.cosine_similarity(vectors[:1], vectors[1:])[0]),
                body_row_l2_distance=float((vectors[0] - vectors[1]).norm()),
                input_output_share_storage=matrices["input_proj"].data_ptr() == matrices["output_proj"].data_ptr(),
                rows=[dict(token_id=key, group=group, level=level, payload_length=count,
                    norm=float(weight[key].float().norm()), std=float(weight[key].float().std()))
                    for key, (group, level, count) in markers.items()]))
        state_rows.append(selected)
    identical = {name: bool(torch.equal(state_rows[0][name], state_rows[1][name])) for name in state_rows[0]}
    result = dict(complete=True,
        commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), grammar_result_sha256=sha(grammar_path), matrices=reports,
        native_to_ar500_marker_rows_byte_equal=identical,
        optimizer_updates=0, vlm_forwards=0, simulator_controls=0,
        conclusions=["Actual grammar maps 252175/252178 to lower_body residual 0/1, not grippers.",
            "Vocabulary-row statistics cannot prove a token was never pretrained or establish sole failure causality.",
            "Selective row adaptation still needs a controlled training and free-generation comparison."])
    publish(args.output / "result.json", result)
    print(json.dumps(dict(complete=True, result_sha256=sha(args.output / "result.json"), marker_rows_exact=identical)), flush=True)


if __name__ == "__main__":
    main()
