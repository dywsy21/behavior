"""CPU-only shared-environment/data smoke test; no weights or training are loaded."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/workspace/wsy/behavior2026"))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output already exists; use a new receipt path")
    # In particular, this check must not allocate a CUDA context on occupied nodes.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from download_behavior2026 import REVISION, verify_file

    packages = ("torch", "torchvision", "torchaudio", "transformers", "datasets",
                "accelerate", "peft", "av", "torchcodec", "flash-linear-attention")
    versions = {name: version(name) for name in packages}
    for module in ("torchvision", "torchaudio", "transformers", "datasets", "accelerate",
                   "peft", "av", "torchcodec", "g05.data.mixture_lerobot_dataset",
                   "g05.models.g05.g05_policy", "g05.models.g05.g05_policy_qwen35"):
        importlib.import_module(module)
    import g05
    from g05.data.lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec

    manifest_bytes = (args.root / "manifests/hf_official_files.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != "502c11870eb0bce85e2ad94a75f32d30aad5ae2963480f7b5f73f6c8cabf5b22":
        raise ValueError("Official manifest SHA256 differs from the registered source")
    manifest = json.loads(manifest_bytes)
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected official dataset revision")
    by_path = {}
    for entry in manifest["files"]:
        item = dict(path=entry["rfilename"], size=entry["size"], blob_id=entry["blob_id"])
        if entry.get("lfs"):
            item["sha256"] = entry["lfs"]["sha256"]
        by_path[item["path"]] = item
    dataset = args.root / "datasets/2026-challenge-demos"
    video = "videos/observation.rgb.zed_link_camera_0/chunk-000/file-000.mp4"
    parquet = "data/chunk-000/file-000.parquet"
    for name in ("meta/info.json", video, parquet):
        if not verify_file(dataset, by_path[name]):
            raise ValueError(f"Official hash/size validation failed: {name}")
    info = json.loads((dataset / "meta/info.json").read_text())
    if (info["total_tasks"], info["total_episodes"]) != (100, 20000):
        raise ValueError("Unexpected dataset scope")
    batch = next(pq.ParquetFile(dataset / parquet).iter_batches(
        batch_size=32, columns=["action", "observation.state"]))
    arrays = {key: np.asarray(batch.column(key).to_pylist())
              for key in ("action", "observation.state")}
    for key, width in (("action", 23), ("observation.state", 61)):
        if arrays[key].shape != (32, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid sample batch: {key}")
    codec = get_safe_default_codec()
    if codec != "torchcodec":
        raise ValueError("Expected actual torchcodec decoding, not an implicit fallback")
    frames = decode_video_frames(dataset / video, [0.0, 1/30, 2/30], tolerance_s=1/30)
    if (tuple(frames.shape) != (3, 3, 720, 720) or not torch.isfinite(frames).all()
            or frames.min() < 0 or frames.max() > 1):
        raise ValueError("Invalid decoded frames")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke test unexpectedly initialized CUDA")
    report = dict(status="passed", host=socket.gethostname(), python=sys.version,
                  executable=sys.executable, versions=versions, cuda_build=torch.version.cuda,
                  g05_source=list(g05.__path__), dataset_revision=REVISION,
                  manifest_sha256=manifest_sha,
                  sample_hashes={name: by_path[name] for name in ("meta/info.json", video, parquet)},
                  parquet_rows=32, action_shape=list(arrays["action"].shape),
                  state_shape=list(arrays["observation.state"].shape),
                  codec=codec, frames_shape=list(frames.shape), cuda_initialized=False,
                  utc=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
