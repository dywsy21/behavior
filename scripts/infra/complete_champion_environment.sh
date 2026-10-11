#!/usr/bin/env bash
# Resume an OWNED, INACTIVE environment using the upstream lock's exact versions
# and wheel hashes. Only the package transport mirror changes, not the recipe.
set -euo pipefail
champion_root=/data/workspace/wsy/behavior2026
champion_source="$champion_root/src/champion2025-ca556f74-v1"
champion_env="$champion_root/envs/champion-openpi-py311-v1"
champion_run="$champion_root/runs/champion_assets_20261011"
[[ $(git -C "$champion_source" rev-parse HEAD) == ca556f74a455cef7987a2be4537b5ac85cc56dd7 ]]
[[ $(git -C "$champion_source/openpi" rev-parse HEAD) == 01177e0242a1c7e8fad2547caa0e987def614cda ]]
[[ -x "$champion_env/bin/python" ]]
if pgrep -af "$champion_root/tools/uv (sync|pip).*" | grep -F "$champion_source"; then
  echo 'Another dependency installer still active; do not overlap' >&2; exit 2
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="$champion_root/tools/champion-uv-cache-v1"
export UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES='' GIT_LFS_SKIP_SMUDGE=1
"$champion_root/tools/uv" export --project "$champion_source/openpi" --frozen --no-dev --no-emit-workspace \
  --format requirements-txt --output-file "$champion_run/openpi-requirements-locked.txt" >/dev/null
"$champion_root/tools/uv" pip install --python "$champion_env/bin/python" --no-deps \
  --index-url https://mirrors.aliyun.com/pypi/simple/ -r "$champion_run/openpi-requirements-locked.txt"
"$champion_root/tools/uv" pip install --python "$champion_env/bin/python" --no-deps \
  --index-url https://mirrors.aliyun.com/pypi/simple/ "$champion_source/openpi" "$champion_source/openpi/packages/openpi-client"
"$champion_root/tools/uv" pip check --python "$champion_env/bin/python"
"$champion_env/bin/python" -c 'import sys,jax,flax,orbax.checkpoint; print(sys.version); print("jax",jax.__version__,"flax",flax.__version__,"orbax",orbax.checkpoint.__version__); assert jax.__version__=="0.5.3"; assert flax.__version__=="0.10.2"; assert orbax.checkpoint.__version__=="0.11.13"'
"$champion_root/tools/uv" pip freeze --python "$champion_env/bin/python" > "$champion_run/env-v1/pip-freeze.txt"
echo 'CHAMPION_DEPENDENCIES_READY; GPU and model acceptance still required'
