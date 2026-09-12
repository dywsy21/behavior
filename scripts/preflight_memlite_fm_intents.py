"""Scan every unique original/recovery intent for the FM text-context contract."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
from transformers import AutoTokenizer

from memlite_fm_v11_common import config, sha, SIDECAR, RECOVERY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_arch.hf_processor_path, local_files_only=True)
    texts, rows = set(), 0
    parquet = pq.ParquetFile(SIDECAR)
    for batch in parquet.iter_batches(columns=["intent"], batch_size=65536):
        rows += len(batch)
        texts.update(x.strip() for x in pc.unique(batch.column(0)).to_pylist() if x and x.strip())
    recovery = json.loads((RECOVERY / "recovery_manifest.json").read_text())
    for split in ("train", "eval"):
        texts.update(r["intent"].strip() for r in recovery["samples"][split] if r["intent"].strip())
    strings = sorted(texts)
    encoded = tokenizer(strings, add_special_tokens=False, truncation=False)["input_ids"]
    lengths = [len(ids) for ids in encoded]
    limit = int(cfg.model.model_arch.fm_intent_adapter.max_tokens)
    result = {"passed": max(lengths) <= limit, "rows_scanned": rows, "unique_intents": len(strings),
        "max_tokens": max(lengths), "configured_limit": limit, "sidecar_sha256": sha(SIDECAR),
        "recovery_manifest_sha256": sha(RECOVERY / "recovery_manifest.json"),
        "longest": sorted(zip(lengths, strings), reverse=True)[:10]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "longest"}), flush=True)
    if not result["passed"]:
        raise ValueError("Source intent exceeds configured context; do not silently truncate")


if __name__ == "__main__":
    main()
