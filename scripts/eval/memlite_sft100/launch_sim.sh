#!/usr/bin/env bash
set -euo pipefail
source /run/ti/behavior_stage3_20260930/tools/activate_sim.sh
unset CUDA_VISIBLE_DEVICES
export OMNIGIBSON_GPU_ID="${EVAL_GPU:?}"
export OMNIGIBSON_DATA_PATH=/run/ti/rl_memlite_stage1_20261006/assets_readiness/sim_data
export PYTHONPATH="/run/ti/rl_memlite_stage1_20261006/code/official_behavior_a8247a8/OmniGibson:${EVAL_SOURCE:?}${PYTHONPATH:+:$PYTHONPATH}"
exec /run/ti/behavior_stage3_20260930/envs/sim/bin/python "$@"
