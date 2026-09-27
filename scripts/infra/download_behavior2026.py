"""Pinned, resumable public dataset download (RGB by default); no credential storage."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import threading

REPO = "behavior-1k/2026-challenge-demos"
REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
RGB_KEYS = frozenset(("observation.rgb.zed_link_camera_0",
                      "observation.rgb.left_realsense_link_camera_0",
                      "observation.rgb.right_realsense_link_camera_0"))


class DiskAdmission:
    """Conservatively account full in-flight files in addition to current free bytes."""
    def __init__(self, free_bytes, reserve):
        self.free_bytes, self.reserve = free_bytes, reserve
        self.lock = threading.Lock()
        self.inflight = 0

    @contextmanager
    def admit(self, size):
        with self.lock:
            if self.free_bytes() - self.inflight < self.reserve + size:
                raise RuntimeError("Disk reserve reached; running downloads may finish")
            self.inflight += size
        try:
            yield
        finally:
            with self.lock:
                self.inflight -= size


def validate_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe repository path: {name!r}")
    return p


def include_file(name, video_mode):
    validate_path(name)
    if video_mode not in ("rgb", "all"):
        raise ValueError("Unknown video selection")
    return (video_mode == "all" or not name.startswith("videos/")
            or name.split("/")[1] in RGB_KEYS)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(root, item):
    path = root / str(validate_path(item["path"]))
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
        return False
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset file escaped destination")
    algorithm = "sha256" if item.get("sha256") else "sha1"
    return digest(path, algorithm) == item.get("sha256", item["blob_id"])


def write_json(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def download_retry(operation, retryable, sleep=time.sleep):
    """Bounded retry of broken HTTP streams; preserve .incomplete range-resume state."""
    for attempt in range(5):
        try:
            return operation()
        except retryable as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if (status is not None and status != 429 and status < 500) or attempt == 4:
                raise
            print(json.dumps(dict(stage="network_retry", attempt=attempt+1,
                                  error_type=type(error).__name__)), flush=True)
            sleep(2**(attempt+1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reserve-gib", type=int, default=1024)
    ap.add_argument("--video-mode", choices=("rgb", "all"), default="rgb",
                    help="rgb keeps all non-video files but excludes depth videos")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.workers <= 8 or args.reserve_gib < 512:
        ap.error("Use 1..8 workers and at least 512 GiB free-space reserve")
    args.root.mkdir(parents=True, exist_ok=True)
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.root / ".behavior-download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from huggingface_hub import HfApi, hf_hub_download
        from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout, HTTPError

        # Manifest hashes always come from the official Hub, not a mirror.
        info = HfApi(endpoint="https://huggingface.co", token=False).dataset_info(
            REPO, revision=REVISION, files_metadata=True, timeout=90)
        if info.sha != REVISION:
            raise ValueError("Repository revision mismatch")
        items = []
        for entry in info.siblings:
            validate_path(entry.rfilename)
            if entry.size is None or not entry.blob_id:
                raise ValueError("Missing authoritative file metadata")
            item = dict(path=entry.rfilename, size=entry.size, blob_id=entry.blob_id)
            if entry.lfs:
                item["sha256"] = entry.lfs.sha256
            if include_file(item["path"], args.video_mode):
                items.append(item)
        # Metadata and the first chunk of each modality are useful for early I/O checks.
        def priority(item):
            name = item["path"]
            if name.startswith("meta/") or "/chunk-000/" in name:
                return (0, name)
            return (1, name)
        items.sort(key=priority)
        total = sum(x["size"] for x in items)
        remaining = sum(max(0, x["size"] - (args.root / x["path"]).stat().st_size)
                        if (args.root / x["path"]).is_file() else x["size"] for x in items)
        free = shutil.disk_usage(args.root).free
        manifest = dict(repo=REPO, revision=REVISION, video_mode=args.video_mode,
                        files=items, total_bytes=total,
                        created_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run / "manifest.json", manifest)
        print(json.dumps(dict(stage="manifest", video_mode=args.video_mode,
                              files=len(items), total_bytes=total,
                              remaining_bytes=remaining, free_bytes=free)), flush=True)
        if free < remaining + args.reserve_gib * 2**30:
            raise RuntimeError("Insufficient disk space including reserve; nothing downloaded")
        if args.manifest_only:
            return
        started = time.monotonic()
        verified_count = verified_bytes = 0
        admission = DiskAdmission(lambda: shutil.disk_usage(args.root).free,
                                  args.reserve_gib * 2**30)

        def download(item):
            target = args.root / item["path"]
            if target.is_symlink() or not target.resolve().is_relative_to(args.root.resolve()):
                raise ValueError("Refusing symlink/escaped download target")
            if verify_file(args.root, item):
                return item
            with admission.admit(item["size"]):
                download_retry(lambda: hf_hub_download(
                    REPO, filename=item["path"], repo_type="dataset",
                    revision=REVISION, local_dir=args.root, token=False,
                    force_download=target.exists()),
                    (ChunkedEncodingError, ConnectionError, Timeout, HTTPError))
                if not verify_file(args.root, item):
                    raise ValueError(f"Hash/size mismatch: {item['path']}")
            return item

        with (args.run / "verified.jsonl").open("a") as receipt:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(download, item): item for item in items}
                try:
                    for future in as_completed(futures):
                        item = future.result()
                        verified_count += 1
                        verified_bytes += item["size"]
                        receipt.write(json.dumps(item) + "\n")
                        receipt.flush()
                        if verified_count % 100 == 0 or item["size"] > 2**30:
                            status = dict(stage="downloading", video_mode=args.video_mode,
                                          verified_files=verified_count,
                                          verified_bytes=verified_bytes, total_files=len(items),
                                          total_bytes=total, elapsed_s=time.monotonic()-started)
                            write_json(args.run / "status.json", status)
                            print(json.dumps(status), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        write_json(args.run / "complete.json", dict(
            repo=REPO, revision=REVISION, video_mode=args.video_mode,
            verified_files=verified_count,
            verified_bytes=verified_bytes, elapsed_s=time.monotonic()-started,
            completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
