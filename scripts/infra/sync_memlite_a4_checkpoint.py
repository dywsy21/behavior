"""Copy the registered A4 checkpoint through local SSH, then verify on lc3.

This transfers binary assets, not code or credentials. Source is immutable;
neither a partial download nor a matching byte count is considered verified.
Completed chunks are resumable. Never removes source/old partial files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

SOURCE = "/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt"
SIZE = 16581363550
SHA256 = "6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269"
DESTINATION = "/data/workspace/wsy/behavior2026/models/memlite-a4-20260912"


def chunks(size: int, chunk_size: int):
    if size <= 0 or chunk_size <= 0:
        raise ValueError("Sizes must be positive")
    return [(i, start, min(chunk_size, size - start))
            for i, start in enumerate(range(0, size, chunk_size))]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def publish_json(path: Path, content: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def download_chunk(entry, directory: Path):
    index, start, length = entry
    target = directory / f"{index:05d}.bin"
    receipt = target.with_suffix(".json")
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if (saved["source_sha256"] == SHA256 and saved["offset"] == start
                and saved["bytes"] == length and target.stat().st_size == length
                and digest(target) == saved["sha256"]):
            return {"index": index, "bytes": length, "resumed": True}
        raise RuntimeError(f"Existing chunk identity failed: {index}")
    if target.exists():
        raise RuntimeError(f"Unreceipted completed chunk; preserve and inspect: {target}")
    partial = target.with_suffix(".partial")
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "robo",
               f"timeout 180 dd if={shlex.quote(SOURCE)} bs=1048576 "
               f"iflag=skip_bytes,count_bytes skip={start} count={length} status=none"]
    # Retry at most once, and only this owned incomplete segment. Completed
    # segments and the existing rsync partial are never overwritten.
    for attempt in range(2):
        with partial.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, timeout=210)
        if result.returncode == 0 and partial.stat().st_size == length:
            sha = digest(partial)
            # Receipt first: after a crash, a .partial remains resumable by
            # re-download; a published .bin always has its expected identity.
            publish_json(receipt, {"source_sha256": SHA256, "offset": start,
                                   "bytes": length, "sha256": sha})
            partial.rename(target)
            return {"index": index, "bytes": length, "resumed": False}
        if attempt:
            raise RuntimeError(f"Chunk {index} failed: {result.stderr.decode(errors='replace')[:200]}")
    raise AssertionError("Unreachable")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--lc3-control", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8, choices=range(1, 9))
    ap.add_argument("--chunk-mib", type=int, default=16, choices=(8, 16, 32, 64))
    args = ap.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / "sync.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parts = work / "chunks-v1"
    parts.mkdir(exist_ok=True)
    status = work / "sync-status.json"
    started = time.monotonic()
    record = dict(source="robo:" + SOURCE, destination="lc3:" + DESTINATION + "/step_2500.pt",
                  source_bytes=SIZE, source_sha256=SHA256, pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(), state="preflight",
                  workers=args.workers, chunk_mib=args.chunk_mib)
    def update(**values):
        record.update(values, updated_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        publish_json(status, record)
        print(json.dumps(record), flush=True)

    try:
        ssh = ["ssh", "-T", "-S", str(args.lc3_control), "-o", "BatchMode=yes", "lc3"]
        subprocess.run(ssh + ["true"], check=True, timeout=20)
        if shutil.disk_usage(work).free < 2 * SIZE + 1024**3:
            raise RuntimeError("Insufficient local staging space")
        shape = subprocess.check_output(
            ["ssh", "-T", "-o", "BatchMode=yes", "robo", f"stat -Lc %s {shlex.quote(SOURCE)}"],
            timeout=30, text=True).strip()
        if int(shape) != SIZE:
            raise RuntimeError("Source checkpoint size changed")
        entries = chunks(SIZE, args.chunk_mib * 1024**2)
        update(state="downloading", completed_chunks=0, local_bytes=0, total_chunks=len(entries))
        completed = total = 0
        last_report = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_chunk, entry, parts) for entry in entries]
            try:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    completed += 1
                    total += row["bytes"]
                    if time.monotonic() - last_report >= 15 or completed == len(entries):
                        update(completed_chunks=completed, local_bytes=total)
                        last_report = time.monotonic()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        assembled = work / "A4-step2500.verified.pt"
        if assembled.exists():
            if assembled.stat().st_size != SIZE or digest(assembled) != SHA256:
                raise RuntimeError("Existing verified-name file has wrong identity")
        else:
            update(state="assembling_and_hashing")
            partial = work / "A4-step2500.assembled.partial"
            h = hashlib.sha256()
            with partial.open("wb") as target:
                for index, _, _ in entries:
                    with (parts / f"{index:05d}.bin").open("rb") as source:
                        for buf in iter(lambda: source.read(8 * 1024**2), b""):
                            target.write(buf)
                            h.update(buf)
            if partial.stat().st_size != SIZE or h.hexdigest() != SHA256:
                raise RuntimeError("Full source SHA256 mismatch: do not publish")
            partial.rename(assembled)
        update(state="uploading", local_verified_sha256=SHA256)
        subprocess.run(ssh + [f"mkdir -p {shlex.quote(DESTINATION)}"], check=True, timeout=30)
        remote_partial = DESTINATION + "/step_2500.pt.partial"
        with (work / "upload.log").open("ab") as log:
            subprocess.run(["rsync", "--partial", "--append-verify", "--info=progress2",
                            "--timeout=180", "-e",
                            shlex.join(["ssh", "-S", str(args.lc3_control), "-o", "BatchMode=yes"]),
                            str(assembled), "lc3:" + remote_partial], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        update(state="remote_hash_verification")
        # Hash before publication; never overwrite a different existing model.
        verify = (
            "import hashlib,json,pathlib; "
            f"p=pathlib.Path({remote_partial!r}); q=p.with_name('step_2500.pt'); "
            f"assert p.stat().st_size=={SIZE}; h=hashlib.sha256(); "
            "f=p.open('rb'); "
            "[h.update(b) for b in iter(lambda:f.read(8388608),b'')]; f.close(); "
            f"assert h.hexdigest()=={SHA256!r}; "
            "assert not q.exists(), 'Refuse overwriting an existing published checkpoint'; "
            "p.rename(q); print(json.dumps({'path':str(q),'bytes':q.stat().st_size,'sha256':h.hexdigest()}))"
        )
        result = subprocess.check_output(ssh + ["python3 -c " + shlex.quote(verify)],
                                         text=True, timeout=300)
        remote = json.loads(result)
        update(state="complete", remote_verification=remote)
    except BaseException as error:
        update(state="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
