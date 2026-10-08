#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${RL_G05_SOURCE:-/run/ti/rl_memlite_stage1_20261006/code/behavior}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
exec /home/tione/notebook/baselines/GalaxeaVLA/.venv/bin/python "$@"
