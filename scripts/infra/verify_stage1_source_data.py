"""Content-verify downloaded RGB/action/meta/annotations against official SHA.

Read-only to the corpus; never downloads depth/raw. A failed/truncated result
does not become a release. Non-LFS blobs use Git's blob SHA1 convention.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from g05.utils.training.stage1_runtime import atomic_json, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4, choices=(1, 2, 4, 8))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.manifest.read_text())
    if source["revision"] != "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2" or source["video_mode"] != "rgb":
        raise ValueError("Wrong registered official source manifest")
    files = [f for f in source["files"] if f["path"].startswith(("annotations/", "data/", "meta/", "videos/observation.rgb."))]
    def check(record):
        path = args.root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            return dict(path=record["path"], status="SIZE_OR_MISSING")
        if record.get("sha256"):
            actual, expected, algorithm = sha256(path), record["sha256"], "sha256"
        else:
            digest = hashlib.sha1(f"blob {record['size']}\0".encode())
            digest.update(path.read_bytes())
            actual, expected, algorithm = digest.hexdigest(), record["blob_id"], "git-blob-sha1"
        return dict(path=record["path"], status="PASS" if actual == expected else "HASH_MISMATCH",
                    algorithm=algorithm, actual=actual, expected=expected, bytes=record["size"])
    started = time.monotonic()
    good = total_bytes = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.output / "checks.jsonl").open("x") as stream:
        for index, result in enumerate(pool.map(check, files)):
            stream.write(json.dumps(result) + "\n")
            if result["status"] == "PASS":
                good += 1
                total_bytes += result["bytes"]
            else:
                failures.append(result)
            if index % 250 == 0:
                stream.flush()
                atomic_json(args.output / "progress.json", dict(checked=index+1, expected=len(files), passed=good,
                    verified_bytes=total_bytes, failures=len(failures), seconds=time.monotonic()-started))
    result = dict(status="PASS" if good == len(files) else "FAILED", files=len(files), passed=good,
        failures=failures, verified_bytes=total_bytes, manifest_sha256=sha256(args.manifest),
        source_revision=source["revision"], checks_sha256=sha256(args.output/"checks.jsonl"), seconds=time.monotonic()-started)
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
