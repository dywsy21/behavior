# robo：四张 A100 的 MEM-Lite 评测运行配置

主仓库：`/mnt/sdc1/robodojo/GalaxeaVLA`。CPU/NUMA/录像/缓存配置为显式 opt-in，旧命令保留原配置；另有一个不改变计算结果的单样本AR缓存优化，对所有单样本调用生效。

```bash
bash scripts/eval_memlite_v9.sh \
  --train-output /mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v9/behavior5_memlite_ar_v9_train_20260906T031051Z \
  --a100 --write-video --run-tag a100_eval_01
```

每次重跑使用新的`--run-tag`；启动器拒绝覆盖已有结果。该命令会跑完五个任务的官方原始timeout，不是256步性能检查。

## 配置实际改变什么

- 仿真PyTorch线程1，推理线程2；BLAS、OpenCV和interop线程各1。
- 读取`nvidia-smi topo -m`，将每个仿真和对应推理进程绑定到GPU附近的CPU核：当前GPU0→0–23，GPU1→24–47，GPU2→48–71，GPU3→72–95。
- 保持每GPU一个仿真和一个policy服务，三个长任务与radio→trash的排程不变。
- 开启CPU H.264录像。A100没有NVENC，不尝试`h264_nvenc`或依赖NVENC的串流。
- 按GPU复用`behavior_eval/a100_cache/robo_a100_v1`中的编译资产/着色器缓存。仅链接local/cache和global/cache；日志、GUI设置和episode数据继续按run隔离。缓存首次使用仍需生成。
- 保存`a100_profile.json`及运行源文件SHA256。检查GPU型号、80GB显存、CPU亲和性与四卡拓扑，遇到不匹配拒绝猜测。

## 不改变什么

不改变官方CPU物理模式、接触/任务条件、seed、timeout、相机画质/分辨率、每16动作渲染的既有时序、3×6帧历史、HL每128动作的节奏、bf16权重、纯AR或动作维度。不会为了提高GPU百分比而开启不同物理后端。

录像逐动作写入，但既有chunk渲染每16动作刷新一次，因此中间画面重复；保存为30fps不等于30个独立渲染帧/秒。

## 已验证的性能范围

2026-09-07，同一段64步记录动作的仿真回放：默认96线程3.40步/秒；限制为1线程22.95步/秒，4线程24.64步/秒；全部CPU线程池限1且录像18.23步/秒。五种配置的64×61机器人proprio逐值一致。这不证明所有场景都逐位确定。

四卡真实MEM-Lite闭环、每任务256步、前16步热身后、开启录像：装饰4.04、餐盘4.00、装罐3.87、收音机3.94步/秒。每个结果都检查`complete=true`且有256个动作，不能仅依赖进程exit0。它们是吞吐验证，不是完整episode成功率。

同权重单一真实输入，推理CPU线程96/1/2/4分别测试，生成动作与高层文本一致，低层均约2.45秒/次；线程限制的主要收益来自仿真侧。

## 单样本AR缓存优化

`src/g05/models/g05/helpers/ar_helper.py`：batch=1时，若解码循环仍未退出，该样本必然尚未结束，无需每个token都保存空的“已结束样本”缓存。跳过这段空索引/复制，不改多样本缓存冻结、停止条件或数值计算。

同真实输入隔离前后测试，低层均值2.364秒→2.224秒；恢复旧实现再测2.404秒。正式部署后重新加载同一权重，无临时替换，热身后的低层均值2.222秒、高层1.574秒；33条动作和高层原文与旧实现逐值/逐字一致。只覆盖该固定输入，不是全任务输出一致性证明。四任务256步闭环测速在这个小优化部署前完成，不把两个测试的收益相乘。

主仓库71项相关回归通过，包括单样本停止、多样本已结束行缓存冻结、CPU配置、启动器和MEM-Lite接口。GPU profiler仍显示大量小kernel启动与同步，当前不声称持续跑满A100；未新增FlashAttention依赖、CUDA Graph或多环境动态批处理。

诊断证据：`/mnt/sdc1/robodojo/behavior_dev/memlite_perf.wm4qXp/`中的`sim_replay_v1`、`policy_v2`、`live_v2`、`policy_cache_v3`和`policy_deployed_v1`。`live_v1`的自定义计时脚本递归错误已修正，不计入有效结果；profiler采集行不用于延迟对比。缓存链接的幂等性、隔离性与拒绝覆盖已测试；新共享缓存路径尚未跑完整五任务启动流程，首次填充仍有冷启动开销。

## 硬件边界

A100没有RT Core和NVENC：[NVIDIA架构说明](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。Isaac Sim对无RT Core显卡不提供标准支持保证；当前机器能实际运行，不等于所有RTX功能可用：[Isaac Sim要求](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)。本配置保留当前可工作的渲染后端，没有把A100改造成RTX硬件。
