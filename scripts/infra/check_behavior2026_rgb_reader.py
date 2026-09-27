"""One official episode through the real v3 reader with NO depth visible in its view."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/workspace/wsy/behavior2026"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Use a new output path")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.data.base_lerobot_dataset import BaseLerobotDataset
    from g05.data.lerobot import lerobot_dataset_v3 as reader
    from download_behavior2026 import RGB_KEYS, REVISION, verify_file

    register_default_resolvers()
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)  # oc.load resolves relative paths from the project.
    config_path = project / "configs/data/behavior2026_r1pro_rgb.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    emb = config["embodiment_datasets"]["galaxea_r1pro"]
    selected = emb["video_keys"]
    if set(selected) != RGB_KEYS or not emb["local_files_only"]:
        raise ValueError("The real training profile must select exactly RGB and disable downloads")
    # Reuse the actual training configuration's offset builder, without loading
    # 20k incomplete episodes or constructing train/eval splits during a smoke test.
    query_plan = object.__new__(BaseLerobotDataset)
    query_plan.shape_meta = emb["shape_meta"]
    query_plan.obs_size = config["obs_size"]
    query_plan.obs_stride_second = config["obs_stride_second"]
    query_plan.obs_stride = max(1, round(query_plan.obs_stride_second * 30))
    query_plan.image_meta = query_plan.shape_meta["images"]
    query_plan.state_meta = query_plan.shape_meta["state"]
    query_plan.action_meta = query_plan.shape_meta["action"]
    deltas = query_plan._build_delta_timestamps(30, 0, config["action_size"])
    if set(deltas).intersection(RGB_KEYS) != RGB_KEYS:
        raise ValueError("Configured training query omits an RGB stream")
    manifest_bytes = (args.root / "manifests/hf_official_files.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != "502c11870eb0bce85e2ad94a75f32d30aad5ae2963480f7b5f73f6c8cabf5b22":
        raise ValueError("Official manifest SHA mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["revision"] != REVISION:
        raise ValueError("Revision mismatch")
    files = {f["rfilename"]: f for f in manifest["files"]}
    dataset = args.root / "datasets/2026-challenge-demos"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    view = args.output.parent / (args.output.stem + "_rgb_only_view")
    view.mkdir()  # New isolated symlink view; never rewrite the official dataset.
    (view / "meta").symlink_to(dataset / "meta", target_is_directory=True)
    paths = ["data/chunk-000/file-000.parquet"]
    paths += [f"videos/{key}/chunk-000/file-000.mp4" for key in selected]
    for name in paths:
        f = files[name]
        identity = dict(path=name, size=f["size"], blob_id=f["blob_id"], sha256=f["lfs"]["sha256"])
        if not verify_file(dataset, identity):
            raise ValueError(f"Sample source not hash-verified: {name}")
        target = view / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(dataset / name)
    with patch.object(reader, "snapshot_download", side_effect=AssertionError("Unexpected Hub access")) as hub:
        ds = reader.LeRobotDataset(
            "behavior-1k/2026-challenge-demos", root=view, episodes=[0],
            revision=REVISION, delta_timestamps=deltas, video_keys=selected,
            local_files_only=True, tolerance_s=0.4/30,
        )
        sample = ds[0]
        hub.assert_not_called()
    if set(ds.meta.video_keys) == set(selected):
        raise AssertionError("The fixture must retain the original depth-declaring metadata")
    if any("depth" in key for key in sample) or list((view / "videos").glob("*depth*")):
        raise AssertionError("Depth unexpectedly used or made visible")
    if tuple(sample["action"].shape) != (32, 23):
        raise AssertionError("Raw control dimensions or prediction horizon changed")
    if not torch.isfinite(sample["action"]).all():
        raise AssertionError("Nonfinite action")
    for key in selected:
        if sample[key].shape[0] != config["obs_size"] or not torch.isfinite(sample[key]).all():
            raise AssertionError(f"Invalid image sample: {key}")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU reader check initialized CUDA")
    report = dict(status="passed", host=socket.gethostname(), revision=REVISION,
                  manifest_sha256=manifest_sha, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  reader_source=reader.__file__, profile=str(config_path), view=str(view), episode=0, sample_index=0,
                  selected_videos=selected, original_video_keys=ds.meta.video_keys,
                  shapes={key: list(value.shape) for key, value in sample.items() if isinstance(value, torch.Tensor)},
                  download_calls=0, cuda_initialized=False, utc=datetime.now(timezone.utc).isoformat())
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
