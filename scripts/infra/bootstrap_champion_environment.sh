#!/usr/bin/env bash
# New, isolated inference/training dependency environment. No optimizer or GPU job.
set -euo pipefail
champion_root=${1:?Usage: bootstrap_champion_environment.sh /data/workspace/wsy/behavior2026}
[[ "$champion_root" == /data/workspace/wsy/behavior2026 ]] || { echo 'Unexpected infrastructure root' >&2; exit 2; }
champion_source="$champion_root/src/champion2025-ca556f74-v1"
champion_env="$champion_root/envs/champion-openpi-py311-v1"
champion_receipt="$champion_root/runs/champion_assets_20261011/env-v1"
[[ ! -e "$champion_source" && ! -e "$champion_env" && ! -e "$champion_receipt" ]] || {
  echo 'Existing bootstrap target: inspect first; never overwrite or hot-sync an environment' >&2; exit 2;
}
mkdir -p "$champion_receipt"
export UV_CACHE_DIR="$champion_root/tools/champion-uv-cache-v1"
export UV_PYTHON_INSTALL_DIR="$champion_root/tools/champion-python-v1"
export UV_PROJECT_ENVIRONMENT="$champion_env"
export GIT_LFS_SKIP_SMUDGE=1
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=''
git clone --no-checkout https://github.com/IliaLarchenko/behavior-1k-solution.git "$champion_source"
git -C "$champion_source" checkout --detach ca556f74a455cef7987a2be4537b5ac85cc56dd7
git -C "$champion_source" submodule update --init openpi
[[ $(git -C "$champion_source/openpi" rev-parse HEAD) == 01177e0242a1c7e8fad2547caa0e987def614cda ]]
"$champion_root/tools/uv" sync --project "$champion_source/openpi" --locked --no-dev --python 3.11
"$champion_env/bin/python" -c 'import sys,jax,flax,orbax.checkpoint; print(sys.version); print("jax",jax.__version__,"flax",flax.__version__,"orbax",orbax.checkpoint.__version__); assert jax.__version__=="0.5.3"; assert flax.__version__=="0.10.2"; assert orbax.checkpoint.__version__=="0.11.13"'
"$champion_root/tools/uv" pip freeze --python "$champion_env/bin/python" > "$champion_receipt/pip-freeze.txt"
git -C "$champion_source" rev-parse HEAD
git -C "$champion_source/openpi" rev-parse HEAD
echo 'CHAMPION_DEPENDENCIES_READY; model/data compatibility and GPU acceptance still required'
