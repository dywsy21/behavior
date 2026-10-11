# Source only; starts no job and changes no shared environment files.
# Independent pinned Python/JAX env, existing read-only FFmpeg runtime.
export BHV_CHAMPION_ROOT=/data/workspace/wsy/behavior2026
export PATH="$BHV_CHAMPION_ROOT/envs/champion-openpi-py311-v1/bin:$BHV_CHAMPION_ROOT/tools/ffmpeg/usr/bin:$PATH"
export LD_LIBRARY_PATH="$BHV_CHAMPION_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu:$BHV_CHAMPION_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu/pulseaudio:$BHV_CHAMPION_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu/blas:$BHV_CHAMPION_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_use_memcpy_local_p2p=false"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
# Caller must select GPU(s), frozen source, run path and authorized experiment.
