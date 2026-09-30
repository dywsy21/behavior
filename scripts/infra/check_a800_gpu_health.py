"""Bounded, non-destructive A800 memory/health gate; never resets a GPU.

Run on an explicitly allocated idle node. Existing aggregate ECC history is
retained and compared, not cleared. This is a short engineering check, not a
replacement for NVIDIA field diagnostics or a guarantee against future faults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import xml.etree.ElementTree as ET


def inventory(xml):
    result = []
    for gpu in ET.fromstring(xml).findall("gpu"):
        counters = {}
        for scope in ("volatile", "aggregate"):
            node = gpu.find(f"ecc_errors/{scope}")
            if node is None:
                raise ValueError("Missing ECC counters")
            for field in node:
                if "correctable" in field.tag:
                    counters[f"{scope}/{field.tag}"] = int(field.text)
        result.append(dict(uuid=gpu.findtext("uuid"), bus=gpu.get("id"),
            name=gpu.findtext("product_name"), recovery=gpu.findtext("gpu_recovery_action"),
            ecc_mode=gpu.findtext("ecc_mode/current_ecc"), counters=counters,
            remap_pending=gpu.findtext("remapped_rows/remapped_row_pending"),
            remap_failure=gpu.findtext("remapped_rows/remapped_row_failure"),
            pids=[int(p.findtext("pid")) for p in gpu.findall("processes/process_info")]))
    return result


def snapshot():
    return inventory(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=30))


def validate(rows, *, allowed_pids=(), baseline=None):
    if len(rows) != 8 or len({r["uuid"] for r in rows}) != 8:
        raise ValueError("Require exactly eight identifiable GPUs")
    prior = {r["uuid"]: r for r in baseline} if baseline is not None else None
    if prior is not None and set(prior) != {r["uuid"] for r in rows}:
        raise ValueError("GPU identity changed")
    for row in rows:
        if not {"volatile/dram_uncorrectable", "aggregate/dram_uncorrectable"} <= row["counters"].keys():
            raise ValueError("Missing required DRAM ECC counters")
        if "A800" not in row["name"] or row["ecc_mode"] != "Enabled":
            raise ValueError("Expected ECC-enabled A800")
        if (row["recovery"] != "None" or row["remap_pending"] != "No"
                or row["remap_failure"] != "No"):
            raise ValueError("GPU needs maintenance: " + row["uuid"])
        if set(row["pids"]) - set(allowed_pids):
            raise ValueError("Another GPU client is present; do not compete")
        for key, count in row["counters"].items():
            if key.startswith("volatile/") and "uncorrectable" in key and count:
                raise ValueError("Nonzero volatile uncorrectable ECC: " + row["uuid"])
        if prior is not None and row["counters"] != prior[row["uuid"]]["counters"]:
            raise ValueError("ECC counters changed during probe: " + row["uuid"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected-host", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--memory-gib", type=int, default=68)
    args = ap.parse_args()
    if socket.gethostname() != args.expected_host or args.expected_host != "a800-1":
        ap.error("This recovery probe is authorized only on a800-1")
    if not 16 <= args.memory_gib <= 72:
        ap.error("Memory probe must remain within 16..72 GiB per GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0,1,2,3,4,5,6,7"):
        ap.error("Do not silently remap physical GPU identities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a unique receipt before touching CUDA; never overwrite evidence.
    with args.output.open("x") as output:
        report = dict(status="RUNNING", host=socket.gethostname(), pid=os.getpid(),
                      started_utc=datetime.now(timezone.utc).isoformat(), gpus=[])
        started = time.monotonic()
        try:
            before = snapshot()
            validate(before)
            report["before"] = before
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            if torch.cuda.device_count() != 8:
                raise ValueError("CUDA device count differs from inventory")
            # Repaired GPU first; retain the same buffers across all patterns.
            for index in (1, 0, 2, 3, 4, 5, 6, 7):
                if time.monotonic() - started > 600:
                    raise TimeoutError("Memory-probe ten-minute budget exhausted")
                validate(snapshot(), allowed_pids=(os.getpid(),), baseline=before)
                torch.cuda.set_device(index)
                free, _ = torch.cuda.mem_get_info()
                if free < (args.memory_gib + 3) * 1024**3:
                    raise RuntimeError("Insufficient safe memory headroom")
                gpu_started = time.monotonic()
                buffers = [torch.empty(1024**3 // 4, dtype=torch.int32, device="cuda")
                           for _ in range(args.memory_gib)]
                for pattern in (0, -1, 0x55555555, -1431655766):
                    for buf in buffers:
                        buf.fill_(pattern)
                    for buf in buffers:
                        if not torch.all(buf == pattern).item():
                            raise RuntimeError(f"GPU {index} pattern mismatch")
                torch.cuda.synchronize()
                del buf, buffers
                gc.collect()
                torch.cuda.empty_cache()
                checked = snapshot()
                validate(checked, allowed_pids=(os.getpid(),), baseline=before)
                row = dict(index=index, tested_gib=args.memory_gib, patterns=4,
                           status="PASS", seconds=time.monotonic()-gpu_started)
                report["gpus"].append(row)
                print(json.dumps(row), flush=True)
            report.update(status="PASS", after=checked)
        except BaseException as error:
            report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report.update(seconds=time.monotonic()-started,
                          ended_utc=datetime.now(timezone.utc).isoformat())
            json.dump(report, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()
