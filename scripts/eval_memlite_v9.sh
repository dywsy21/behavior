#!/usr/bin/env bash
# Evaluate one completed causal MEM-Lite v9 5k run on the fixed five-task BEHAVIOR
# public-test protocol. This launcher deliberately has no baseline or gate-file
# dependency: its only model source is --train-output.
set -Eeuo pipefail

REPO="/mnt/sdc1/robodojo/GalaxeaVLA"
BRIDGE="/mnt/sdc1/robodojo/behavior_bridge_staging"
EVAL_ROOT="/mnt/sdc1/robodojo/behavior_eval"
G05_OUTPUT_DIR="/mnt/sdc1/robodojo/outputs/g05"
OG_ROOT="/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson"
OG_PYTHON="/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python"
ROBOT_CONFIG="$OG_ROOT/omnigibson/eval/r1pro.yaml"
EVAL_RUNNER="$EVAL_ROOT/run_behavior_eval_chunked.py"
MEM_SERVER="$BRIDGE/serve_behavior_policy_mem.py"
MEM_SUMMARY_SCRIPT="$REPO/scripts/summarize_memlite_v9.py"
TASKS="/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos/meta/tasks.jsonl"
GLU_LIB="/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib"
GLOBAL_LOCK="$G05_OUTPUT_DIR/r1pro_memlite_ar/memlite_official_eval_chain.lock"

TRAIN_OUTPUT=""
REPLAN_EVERY_CHUNKS=8
# Match the v9 causal planner's 128-frame cadence and validated text budget.
MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS=1024
A100_PROFILE=0
WRITE_VIDEO=0
RUN_TAG=""
CPU_BUDGET_RUNNER="$REPO/scripts/run_with_cpu_budget.py"
A100_PROFILE_SCRIPT="$REPO/scripts/a100_eval_profile.py"
SHARED_A100_CACHE="$EVAL_ROOT/a100_cache/robo_a100_v1"
declare -a CPU_AFFINITIES=()

usage() {
    cat <<'EOF'
Usage: eval_memlite_v9.sh --train-output PATH [--a100] [--write-video] [--run-tag TAG] [--memlite-replan-every-chunks N] [--memlite-high-level-max-new-tokens N]

Runs the fixed official public-test condition: split index 0 (instance 301),
seed 0, one rollout per task, RGBDFullRes, action_steps=16, and the evaluator's
own default maximum step limit. The v9 cadence is 8 chunks x 16 actions =
128 simulator actions; high-level cap is 1024 tokens. Both must match the saved
causal runtime configuration. Initial memory uses only the official task_id.
No baseline, ground-truth planner, or early model-declared episode termination.

--a100 enables robo's validated CPU/NUMA budget (sim=1, policy=2 Torch threads,
BLAS/OpenCV/interop=1), and enables video by default. It does NOT change the
physics backend, camera rendering, observation cadence or model precision.
Only compiled-asset/shader cache directories are reused per GPU; episode state,
logs and GUI/settings data remain isolated under each evaluation run.
--run-tag appends a safe identifier to the output path; existing runs are never
overwritten. With --a100 the default tag is a100_cpu1_policy2_video1.
EOF
}

while (( $# )); do
    case "$1" in
        --a100)
            A100_PROFILE=1
            WRITE_VIDEO=1
            shift
            ;;
        --write-video)
            WRITE_VIDEO=1
            shift
            ;;
        --no-write-video)
            WRITE_VIDEO=0
            shift
            ;;
        --run-tag)
            [[ $# -ge 2 ]] || { echo "--run-tag needs a value" >&2; exit 2; }
            RUN_TAG="$2"
            shift 2
            ;;
        --train-output)
            [[ $# -ge 2 ]] || { echo "--train-output needs a path" >&2; exit 2; }
            TRAIN_OUTPUT="$2"
            shift 2
            ;;
        --memlite-replan-every-chunks)
            [[ $# -ge 2 ]] || { echo "--memlite-replan-every-chunks needs an integer" >&2; exit 2; }
            REPLAN_EVERY_CHUNKS="$2"
            shift 2
            ;;
        --memlite-high-level-max-new-tokens)
            [[ $# -ge 2 ]] || { echo "--memlite-high-level-max-new-tokens needs an integer" >&2; exit 2; }
            MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$TRAIN_OUTPUT" ]] || { usage >&2; exit 2; }
[[ -z "$RUN_TAG" || "$RUN_TAG" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$ ]] || {
    echo "--run-tag must be a safe identifier, at most 80 characters" >&2; exit 2;
}
if (( A100_PROFILE )); then
    affinity_text="$("$REPO/.venv/bin/python" "$A100_PROFILE_SCRIPT" --affinities-only)"
    mapfile -t CPU_AFFINITIES <<< "$affinity_text"
    [[ ${#CPU_AFFINITIES[@]} -eq 4 && -f "$CPU_BUDGET_RUNNER" ]] || exit 2
    [[ -n "$RUN_TAG" ]] || RUN_TAG="a100_cpu1_policy2_video${WRITE_VIDEO}"
fi
[[ "$REPLAN_EVERY_CHUNKS" =~ ^[0-9]+$ ]] || {
    echo "--memlite-replan-every-chunks must be a non-negative integer" >&2
    exit 2
}
[[ "$MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--memlite-high-level-max-new-tokens must be a positive integer" >&2
    exit 2
}

TRAIN_OUTPUT="$(readlink -f "$TRAIN_OUTPUT")"
CHECKPOINT="$TRAIN_OUTPUT/checkpoints/step_5000.pt"
LAST_PT="$TRAIN_OUTPUT/last.pt"
if [[ ! -s "$CHECKPOINT" || ! -e "$LAST_PT" ]]; then
    echo "Expected non-empty $CHECKPOINT and $LAST_PT" >&2
    exit 20
fi
if [[ "$(readlink -f "$LAST_PT")" != "$(readlink -f "$CHECKPOINT")" ]]; then
    echo "last.pt does not resolve to step_5000.pt for $TRAIN_OUTPUT" >&2
    exit 20
fi


# Reject old/fallback configurations before starting any policy or simulator.
(
cd "$REPO"
"$REPO/.venv/bin/python" - "$TRAIN_OUTPUT" "$REPLAN_EVERY_CHUNKS" "$MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS" <<'PY'
import hashlib, math, sys
from pathlib import Path
import yaml
cfg = yaml.safe_load((Path(sys.argv[1]) / ".hydra/config.yaml").read_text())
runtime = cfg["memlite_runtime"]
assert runtime["schema_version"] == 5
assert runtime["action_steps"] == 16
assert runtime["replan_every_chunks"] == int(sys.argv[2]) == 8
assert runtime["high_level_max_new_tokens"] == int(sys.argv[3]) == 1024
assert runtime["require_explicit_grippers"] is True
arch = cfg["model"]["model_arch"]
assert arch["discrete_action"] is True and arch["continuous_action"] is False
assert arch["predict_cot"] is True and arch["num_obs_steps"] == 6
assert cfg["model"]["processor"]["image_history_mode"] == "preserve_cameras"
assert cfg["data"]["obs_size"] == 6
assert cfg["data"]["past_action_size"] == 0
assert math.isclose(cfg["data"]["obs_stride_second"], 16 / 30)
assert cfg["tokenizer"]["vq_config"]["dropout_noop_parts"] is False
assert cfg["tokenizer"]["vq_config"]["require_group_markers"] is True
sidecar = Path(cfg["data"]["embodiment_datasets"]["galaxea_r1pro"]["memlite_sidecar"])
assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == "a5f1026110cac3ed57682d8177d2dabcb2f30d6572791ebb27c2800100f5c5d2"
print("Verified saved v9 causal, vision, pure-AR and reviewed-data contracts.")
PY
)

FINAL_EXP="$(basename "$TRAIN_OUTPUT")"
RUN_ROOT="$EVAL_ROOT/memlite_public301_${FINAL_EXP}_replan${REPLAN_EVERY_CHUNKS}_hl${MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS}_taskmem_actionalign0_v9causal"
[[ -z "$RUN_TAG" ]] || RUN_ROOT="${RUN_ROOT}_${RUN_TAG}"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN_ROOT/orchestrator.log"
}

prepare_a100_cache() {
    local gpu kind target link
    for gpu in 0 1 2 3; do
        for kind in local global; do
            target="$SHARED_A100_CACHE/gpu$gpu/${kind}_cache"
            link="$RUN_ROOT/appdata/gpu$gpu/$kind/cache"
            mkdir -p "$target" "$(dirname "$link")"
            if [[ -L "$link" ]]; then
                [[ "$(readlink "$link")" == "$target" ]] || {
                    echo "Refusing unexpected cache symlink: $link" >&2; return 2;
                }
            elif [[ -e "$link" ]]; then
                echo "Refusing existing non-symlink cache: $link" >&2; return 2
            else
                ln -s "$target" "$link"
            fi
        done
    done
}

training_processes() {
    pgrep -af 'scripts/finetune\.py|torchrun.*finetune\.py' || true
}

cuda_processes() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true
}

all_gpu_memories_below_limit() {
    local -a memories
    mapfile -t memories < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ ${#memories[@]} -eq 4 ]] || return 1
    local memory
    for memory in "${memories[@]}"; do
        memory="${memory//[[:space:]]/}"
        (( memory < 500 )) || return 1
    done
}

resource_gate_clear() {
    local training cuda
    training="$(training_processes)"
    if [[ -n "$training" ]]; then
        log "GATE_WAIT training process(es) remain: ${training//$'\n'/,}"
        return 1
    fi
    cuda="$(cuda_processes)"
    if [[ -n "$cuda" ]]; then
        log "GATE_WAIT CUDA compute process(es) remain: ${cuda//$'\n'/,}"
        return 1
    fi
    if ! all_gpu_memories_below_limit; then
        log "GATE_WAIT one or more GPUs remain >= 500 MiB"
        return 1
    fi
    log "GATE_CLEAR no finetune/CUDA compute process and four GPUs < 500 MiB"
    return 0
}

wait_for_resource_gate() {
    while true; do
        if resource_gate_clear; then
            log "GATE_FIRST_CLEAR sleeping 30s before independent confirmation"
            sleep 30
            if resource_gate_clear; then
                return 0
            fi
        fi
        sleep 60
    done
}

declare -a server_pids=()
declare -a eval_pids=()
stop_eval_process_groups() {
    # ``run_eval`` records only its own setsid leader.  Killing the process
    # group prevents an interrupted launcher from leaving an OmniGibson child
    # reparented under PID 1, while the cmdline check prevents touching a
    # reused/unrelated PID.
    local pid_file pid command
    local -a pid_files=()
    shopt -s nullglob
    pid_files=("$RUN_ROOT"/tasks/*/eval_pgid.txt)
    shopt -u nullglob
    for pid_file in "${pid_files[@]}"; do
        pid="$(<"$pid_file")"
        [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
        command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
        if [[ "$command" != *"$RUN_ROOT/tasks/"* ]]; then
            log "CLEANUP_SKIP unexpected eval leader pid=$pid file=$pid_file"
            continue
        fi
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
}
cleanup() {
    local pid
    stop_eval_process_groups
    for pid in "${eval_pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${eval_pids[@]:-}" "${server_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM

mkdir -p "$(dirname "$GLOBAL_LOCK")"
exec 9>"$GLOBAL_LOCK"
if ! flock -n 9; then
    echo "Another MEM-Lite official evaluation chain already holds $GLOBAL_LOCK" >&2
    exit 22
fi

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite an existing MEM-Lite evaluation root: $RUN_ROOT" >&2
    exit 21
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tasks" "$RUN_ROOT/appdata" "$RUN_ROOT/traces"
if (( A100_PROFILE )); then prepare_a100_cache; fi
exec > >(tee -a "$RUN_ROOT/orchestrator.log") 2>&1

cat > "$RUN_ROOT/manifest.txt" <<EOF
train_output=$TRAIN_OUTPUT
checkpoint=$CHECKPOINT
last_pt_resolved=$(readlink -f "$LAST_PT")
protocol=official_public_test
public_test_index=0
instance_id=301
seed=0
rollouts_per_task=1
action_steps=16
env_wrapper=omnigibson.eval.wrappers.RGBDFullResWrapper
max_steps=official_default_unchanged
write_video=$WRITE_VIDEO
a100_performance_profile=$A100_PROFILE
performance_cpu_affinities=${CPU_AFFINITIES[*]:-legacy_unrestricted}
performance_shared_cache=$([[ "$A100_PROFILE" == 1 ]] && printf '%s' "$SHARED_A100_CACHE" || printf 'disabled')
performance_contract=physics_and_rendering_unchanged; no_NVENC_assumption; no_resolution_or_observation_reduction
memlite_replan_every_chunks=$REPLAN_EVERY_CHUNKS
memlite_replan_reason=v9_causal_planner_stride_128_frames; fixed_eval_condition=8_chunks_x16_actions=128_simulator_actions
memlite_high_level_max_new_tokens=$MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS
memlite_initial_memory=official_payload_task_id_only; task_ids_0_to_4_map_to_canonical_Task_equals_id_semicolon_Completed_equals_none; explicit_nonempty_memory_has_priority; unknown_task_id_without_explicit_memory_is_refused; no_goal_predicate_or_future_trajectory_is_read
action_execution_start_index=0
decoded_action_horizon=32
postprocess_action_horizon=32
executed_action_steps=16
action_alignment_provenance=saved_data_past_action_size_0; decoded_rows_0_to_31_are_future_only; execute_rows_0_to_15
inference_config_attempt=taskmem_hl${MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS}_actionalign0
vision_contract=preserve_cameras; each_of_3_cameras_has_6_chronological_frames_at_t_minus_80_64_48_32_16_0; training_obs_stride=16_over_30_seconds; real_temporal_MEM_path; startup_history_repeats_first_observation
memlite_contract=schema5; LL_and_HL_share_current_primitive_intent; memory_target_uses_current_observed_annotation_time; previous_intent_prompt; strict_atomic_HL_parser; DONE_hold_and_recheck_not_LL_Task_complete; explicit_all_action_groups; bounded_train_only_tail_inverse; infrastructure_errors_propagate
known_memlite_limit=demo_annotation_supervision_and_synthetic_text_corruption_are_not_real_failure_recovery; hidden_past_state_not_uniquely_observable_after_memory_erasure; 1_rollout_per_task_is_diagnostic_not_leaderboard; no_nonMEM_baseline_or_ablation
EOF

sha256sum "$REPO/scripts/serve_policy_mem.py" "$REPO/src/g05/models/g05/inferencer.py" "$REPO/src/g05/utils/memlite_protocol.py" "$REPO/src/g05/data_processor/processor/base_processor.py" "$REPO/src/g05/utils/data/normalizer.py" "$MEM_SERVER" "$EVAL_RUNNER" "$MEM_SUMMARY_SCRIPT" > "$RUN_ROOT/runtime_sources.sha256"
if (( A100_PROFILE )); then
    sha256sum "$CPU_BUDGET_RUNNER" "$A100_PROFILE_SCRIPT" >> "$RUN_ROOT/runtime_sources.sha256"
    "$REPO/.venv/bin/python" "$A100_PROFILE_SCRIPT" > "$RUN_ROOT/a100_profile.json"
fi

start_policy_server() {
    local gpu="$1"
    local port="$2"
    local trace_root="$RUN_ROOT/traces/policy_gpu${gpu}"
    local log_file="$RUN_ROOT/logs/mem_policy_gpu${gpu}_port${port}.log"
    local -a cpu_budget=()
    if (( A100_PROFILE )); then
        cpu_budget=("$CPU_BUDGET_RUNNER" --threads 2 --cpu-affinity "${CPU_AFFINITIES[$gpu]}")
    fi
    (
        # Saved OmegaConf files use relative ``oc.load:configs/...`` paths.
        # Resolve those from the training repository, not the evaluator cwd.
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
            "$REPO/.venv/bin/python" "${cpu_budget[@]}" "$MEM_SERVER" \
            --ckpt_path "$CHECKPOINT" \
            --tasks_path "$TASKS" \
            --host 127.0.0.1 \
            --port "$port" \
            --device cuda:0 \
            --action_steps 16 \
            --pure-ar \
            --predict-cot \
            --require-memlite \
            --memlite-replan-every-chunks "$REPLAN_EVERY_CHUNKS" \
            --memlite-high-level-max-new-tokens "$MEMLITE_HIGH_LEVEL_MAX_NEW_TOKENS" \
            --trace-root "$trace_root" \
            model.model_weights_to_bf16=true \
            model.use_torch_compile=false \
            eval_embodiment=galaxea_r1pro
    ) >"$log_file" 2>&1 &
    local pid=$!
    server_pids+=("$pid")
    printf '%s\n' "$pid" > "$RUN_ROOT/logs/mem_policy_gpu${gpu}.pid"
}

await_policy_servers() {
    local port ready
    for port in 8765 8766 8767 8768; do
        ready=0
        for _ in $(seq 1 120); do
            if curl -fsS --max-time 3 "http://127.0.0.1:${port}/healthz" >/dev/null; then
                ready=1
                break
            fi
            sleep 10
        done
        if (( ready == 0 )); then
            echo "MEM-Lite policy server on port $port did not become healthy" >&2
            exit 10
        fi
    done
}

run_eval() {
    local task="$1"
    local gpu="$2"
    local port="$3"
    local task_root="$RUN_ROOT/tasks/$task"
    local appdata="$RUN_ROOT/appdata/gpu${gpu}"
    local rc
    local eval_pgid
    local -a cpu_budget=() video_args=(--no-write-video)
    if (( A100_PROFILE )); then
        cpu_budget=("$CPU_BUDGET_RUNNER" --threads 1 --cpu-affinity "${CPU_AFFINITIES[$gpu]}")
    fi
    if (( WRITE_VIDEO )); then video_args=(--write-video); fi
    mkdir -p "$task_root" "$appdata"
    log "START task=$task gpu=$gpu port=$port"
    _run_eval_stop_group() {
        [[ -n "${eval_pgid:-}" ]] && kill -TERM -- "-$eval_pgid" 2>/dev/null || true
    }
    trap _run_eval_stop_group INT TERM
    set +e
    (
        cd "$OG_ROOT"
        export OMNI_KIT_ACCEPT_EULA=YES
        export OMNIGIBSON_HEADLESS=1
        export OMNIGIBSON_GPU_ID="$gpu"
        export OMNIGIBSON_NO_OMNI_LOGS=1
        export OMNIGIBSON_APPDATA_PATH="$appdata"
        export BEHAVIOR_ACTION_STEPS=16
        export MEMLITE_SIM_TRACE_PATH="$task_root/sim_trace.jsonl"
        export PYTHONHASHSEED=0
        export LD_LIBRARY_PATH="$GLU_LIB:${LD_LIBRARY_PATH:-}"
        export PYTHONUNBUFFERED=1
        # Each official rollout owns a session.  The launcher can therefore
        # stop its exact group if interrupted without broad process matching.
        exec setsid /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kb=%M' -o "$task_root/time.txt" \
            "$OG_PYTHON" "${cpu_budget[@]}" "$EVAL_RUNNER" \
            --task-name "$task" \
            --robot-config "$ROBOT_CONFIG" \
            --mode public_test \
            --instance-indices 0 \
            --num-rollouts 1 \
            --host 127.0.0.1 \
            --port "$port" \
            --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
            --output-dir "$task_root/output" \
            "${video_args[@]}" \
            --headless
    ) >"$task_root/eval.log" 2>&1 &
    eval_pgid=$!
    printf '%s\n' "$eval_pgid" > "$task_root/eval_pgid.txt"
    wait "$eval_pgid"
    rc=$?
    trap - INT TERM
    if (( rc == 0 )); then
        local -a result_files=()
        shopt -s nullglob
        result_files=("$task_root"/output/json/*.json)
        shopt -u nullglob
        if (( ${#result_files[@]} != 1 )); then
            log "INVALID_RESULT task=$task expected_one_json got=${#result_files[@]}"
            rc=70
        fi
    fi
    set -e
    printf '%s\n' "$rc" > "$task_root/exit_code.txt"
    log "FINISH task=$task gpu=$gpu rc=$rc"
    return "$rc"
}

log "WAITING_FOR_FREE_GPUS train_output=$TRAIN_OUTPUT checkpoint=$CHECKPOINT"
wait_for_resource_gate
log "EVALUATION_CONDITION_READY instance=301 seed=0 rollout=1 action_steps=16 replan_chunks=$REPLAN_EVERY_CHUNKS"

start_policy_server 0 8765
start_policy_server 1 8766
start_policy_server 2 8767
start_policy_server 3 8768
log "LOADING_POLICY_SERVERS"
await_policy_servers
log "POLICY_SERVERS_READY"

log "EVALUATION_STARTED"
(run_eval putting_away_Halloween_decorations 0 8765) & eval_pids+=("$!")
(run_eval cleaning_up_plates_and_food 1 8766) & eval_pids+=("$!")
(run_eval can_meat 2 8767) & eval_pids+=("$!")
(
    first_rc=0
    second_rc=0
    run_eval turning_on_radio 3 8768 || first_rc=$?
    run_eval picking_up_trash 3 8768 || second_rc=$?
    (( first_rc == 0 && second_rc == 0 ))
) & eval_pids+=("$!")

overall_rc=0
for pid in "${eval_pids[@]}"; do
    wait "$pid" || overall_rc=1
done

set +e
"$REPO/.venv/bin/python" "$MEM_SUMMARY_SCRIPT" \
    --mem-root "$RUN_ROOT" \
    --output-json "$RUN_ROOT/memlite_success_rate.json" \
    --output-markdown "$RUN_ROOT/memlite_success_rate.md"
summary_rc=$?
set -e
if (( summary_rc != 0 )); then
    overall_rc=1
fi
log "EVALUATION_FINISHED rc=$overall_rc summary_rc=$summary_rc summary=$RUN_ROOT/memlite_success_rate.md"
exit "$overall_rc"
