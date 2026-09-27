# Source this from an interactive shell; it starts no job and writes no files.
# Temporary NCCL P2P workaround is specific to the measured lc1/lc2/lc3 cluster.
export BEHAVIOR_CLUSTER_ROOT=/data/workspace/wsy/behavior2026
export VIRTUAL_ENV="$BEHAVIOR_CLUSTER_ROOT/envs/g05-py310-cu128"
export PATH="$VIRTUAL_ENV/bin:$BEHAVIOR_CLUSTER_ROOT/tools/ffmpeg/usr/bin:$PATH"
export LD_LIBRARY_PATH="$BEHAVIOR_CLUSTER_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu:$BEHAVIOR_CLUSTER_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu/pulseaudio:$BEHAVIOR_CLUSTER_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu/blas:$BEHAVIOR_CLUSTER_ROOT/tools/ffmpeg/usr/lib/x86_64-linux-gnu/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$BEHAVIOR_CLUSTER_ROOT/tools/hf-cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=2
# Deliberately do not set NCCL_ALGO, CUDA_VISIBLE_DEVICES, ranks, or training budgets.
# Choose a clean frozen source checkout and an approved run config before torchrun.
