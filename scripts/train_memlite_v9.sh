#!/usr/bin/env bash
# New causal MEM-Lite only. Does not invoke or wait for a baseline.
set -Eeuo pipefail

mode=${1:?usage: bash scripts/train_memlite_v9.sh smoke|train REVIEW_APPROVAL_JSON}
review_receipt=${2:?human review receipt is required}
case "$mode" in smoke|train) ;; *) exit 2 ;; esac
repo=/mnt/sdc1/robodojo/GalaxeaVLA
sidecar=/mnt/sdc1/robodojo/datasets/memlite_annotations_task0_4_causal_v9_final_20260906/meta/memlite_annotations.parquet
cd "$repo"

# This receipt is written by the primary agent only AFTER personally reading
# the audit packet and viewing its original-video contact sheets. The data
# generation/audit programs deliberately never issue approval themselves.
"$repo/.venv/bin/python" - "$review_receipt" "$sidecar" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
assert receipt["status"] == "approved_by_primary_after_manual_review"
assert receipt["structural_errors"] == 0
assert receipt["text_cases_inspected"] >= 150
assert receipt["visual_moments_inspected"] >= 30
assert Path(receipt["sidecar"]).resolve() == Path(sys.argv[2]).resolve()
digest = hashlib.sha256()
with open(sys.argv[2], "rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == receipt["sidecar_sha256"], "Sidecar changed after review"
print("Manual review receipt and exact sidecar hash verified", flush=True)
PY

if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 > 1000 || $2 > 0 {busy=1} END {exit !busy}'; then
    printf '%s\n' 'A GPU is occupied; refusing to interfere with another job.' >&2
    exit 3
fi
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
test "$gpu_count" -eq 4

source "$repo/activate_g05.sh"
cd "$repo"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
export EXP_NAME="behavior5_memlite_ar_v9_${mode}_${run_stamp}"
export G05_OUTPUT_DIR=/mnt/sdc1/robodojo/outputs/g05
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
if [[ "$mode" == smoke ]]; then export DRY_RUN=1; else export DRY_RUN=0; fi

run_dir="$G05_OUTPUT_DIR/r1pro_memlite_ar_v9/$EXP_NAME"
test ! -e "$run_dir"
control_dir=$(mktemp -d /mnt/sdc1/robodojo/behavior_dev/memlite_v9_train_control.XXXXXX)
printf '%s\n' "$run_dir" > "$control_dir/run_dir.txt"
printf '%s\n' "$$" > "$control_dir/launcher.pid"
cp "$review_receipt" "$control_dir/manual_review_receipt.json"
trap 'rc=$?; printf "%s\n" "$rc" > "$control_dir/exit_code.tmp"; mv "$control_dir/exit_code.tmp" "$control_dir/exit_code.txt"' EXIT
printf 'RUN_DIR=%s\nCONTROL_DIR=%s\nMODE=%s\n' "$run_dir" "$control_dir" "$mode"

# 8 samples/rank keeps the branch sampler at exactly one HL + seven LL.
# VLM/vision checkpointing reduces activation memory without dropping MEM.
"$repo/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/finetune.py \
    task=r1pro_memlite_ar_v9 \
    "data.embodiment_datasets.galaxea_r1pro.memlite_sidecar=$sidecar" \
    model.max_steps=5000 model.max_epochs=null resume_ckpt=null \
    model.batch_size=8 model.grad_accumulation_steps=1 \
    model.num_workers=8 model.prefetch_factor=2 \
    batch_size_val=8 checkpointing_steps=500 eval_steps=100 \
    logger.mode=offline
