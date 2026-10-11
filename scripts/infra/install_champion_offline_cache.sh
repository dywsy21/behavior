#!/usr/bin/env bash
# Run only after SHA-verified cache archives were fully extracted, and all
# previous installers on this owned environment have exited. No GPU work.
set -euo pipefail
champion_root=/data/workspace/wsy/behavior2026
integration_source=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
champion_source="$champion_root/src/champion2025-ca556f74-v1"
champion_env="$champion_root/envs/champion-openpi-py311-v1"
champion_cache="$champion_root/tools/champion-uv-transferred-cache-v1"
champion_extra="$champion_root/tools/champion-extra-transferred-cache-v1"
grep -qx CACHE_EXTRACTED "$champion_root/runs/champion_assets_20261011/cache-extract-v1.log"
if pgrep -af "$champion_root/tools/uv (sync|pip).*" | grep -F "$champion_source"; then
  echo 'Other installer still running' >&2; exit 2
fi
"$champion_root/envs/download/bin/python" "$integration_source/scripts/infra/relocate_champion_uv_cache.py" \
  --root "$champion_cache" --old-root /home/wsy/behavior/artifacts/champion-wheelcache-v1
"$champion_root/envs/download/bin/python" "$integration_source/scripts/infra/relocate_champion_uv_cache.py" \
  --root "$champion_extra" --old-root /home/wsy/behavior/artifacts/champion-extra-cache-v1
export CUDA_VISIBLE_DEVICES='' UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$champion_env"
export UV_CACHE_DIR="$champion_cache"
"$champion_root/tools/uv" sync --offline --locked --no-dev --project "$champion_source/openpi" \
  --python "$champion_root/tools/champion-python-transfer-v1/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
UV_CACHE_DIR="$champion_extra" "$champion_root/tools/uv" pip install --offline --no-deps \
  --python "$champion_env/bin/python" -r "$integration_source/configs/champion2026/runtime-extra-locked.txt"
"$champion_root/tools/uv" pip check --python "$champion_env/bin/python"
"$champion_env/bin/python" -c 'import jax,flax,orbax.checkpoint; print(jax.__version__,flax.__version__,orbax.checkpoint.__version__); assert (jax.__version__,flax.__version__,orbax.checkpoint.__version__)==("0.5.3","0.10.2","0.11.13")'
"$champion_root/tools/uv" pip freeze --python "$champion_env/bin/python" > "$champion_root/runs/champion_assets_20261011/env-v1/pip-freeze.txt"
echo CHAMPION_OFFLINE_DEPENDENCIES_READY
