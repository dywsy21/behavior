#!/usr/bin/env bash
set -euo pipefail
source /run/ti/behavior_stage3_20260930/tools/activate_sim.sh
export OMNIGIBSON_DATA_PATH="${RL_SIM_DATA_ROOT:-$OMNIGIBSON_DATA_PATH}"
unset CUDA_VISIBLE_DEVICES
export OMNIGIBSON_GPU_ID="$RL_GPU"
export RL_PILOT_RUN="$1"
RL_REPAIR_SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
exec python -u "$RL_REPAIR_SOURCE/tools/train_sim_pilot.py" --task-name "$2" --policy local --mode train --instance-indices "$3" "$4" --num-envs 2 --max-steps "$RL_EPISODE_STEPS" --num-rollouts 1 --headless --no-write-video --output-dir "$1/eval_$2"
