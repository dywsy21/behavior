#!/usr/bin/env bash
source "/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/activate"
source "/mnt/sdc1/robodojo/GalaxeaVLA/.env.g05-a100"

# TorchCodec's CUDA video decoder needs NVIDIA NPP at runtime. The NPP wheel
# installs the library inside the virtual environment, so expose that directory
# to the Linux dynamic linker whenever the G0.5 environment is activated.
G05_NPP_LIB="/mnt/sdc1/robodojo/GalaxeaVLA/.venv/lib/python3.10/site-packages/nvidia/npp/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$G05_NPP_LIB:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$G05_NPP_LIB"
fi
unset G05_NPP_LIB

cd "/mnt/sdc1/robodojo/GalaxeaVLA"
printf "G0.5 环境已激活：%s\n" "$VIRTUAL_ENV"
