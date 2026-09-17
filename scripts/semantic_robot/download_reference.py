"""One pinned public reference model, with disk/size guards; no inference."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    repo = "Qwen/Qwen3.8-27B"
    revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    out = Path(args.output)
    if out.exists():
        raise RuntimeError("Use a new model directory; do not overwrite a prior download")
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [f for f in info.siblings if "/" not in f.rfilename and
             f.rfilename.endswith((".json", ".safetensors", ".jinja", ".txt"))]
    size = sum(f.size or 0 for f in files)
    if info.sha != revision or not 50e9 < size < 60 * 1024**3:
        raise RuntimeError(f"Unexpected reference identity/size: {info.sha}, {size}")
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < size + 80 * 1024**3:
        raise RuntimeError("Would leave less than 80 GiB free; download refused")
    out.mkdir()
    receipt = {"repo": repo, "revision": revision, "expected_bytes": size,
               "files": [f.rfilename for f in files], "state": "downloading"}
    (out / "download_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    snapshot_download(repo, revision=revision, local_dir=out,
                      allow_patterns=receipt["files"], max_workers=4)
    for f in files:
        if f.size is not None and (out / f.rfilename).stat().st_size != f.size:
            raise RuntimeError(f"Size mismatch: {f.rfilename}")
    receipt["state"] = "complete"
    (out / "download_receipt.json").write_text(json.dumps(receipt, indent=2))
    print("PINNED_DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
