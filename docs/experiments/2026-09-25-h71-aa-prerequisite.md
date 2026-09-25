# H71：非DLSS抗锯齿的配套选项修复

2026-09-25 14:54北京时间，owner Codex。H70实际source3642dd7，运行508.514s后在同一原session初始化失败；已捕获第一台相机clipping_range更新中的native app.update将`/rtx/post/aa/op`从0改3，其他9个登记PT值不变。其Python栈不是C++backtrace，H70没有读到limitedOps旧值，不能提前宣称该flag因果性已证实。

依据：服务器安装Replicator1.12.27的`core/scripts/settings.py:140–143`在非DLSS/DLAA选择时先清`/rtx-transient/post/aa/limitedOps`；[NVIDIA1.11.4变更记录](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/CHANGELOG.html)也注明此修复。[NVIDIA设置API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/API.html)将post AA0定义为None、3为DLSS。旧profile只设置aa/op=0而漏了配套限制。

唯一变化/假设：在原两个空app/空simulator边界，先设置limitedOps=false，再设置原aa/op=0，能避免首相机render自动回退。继续原PathTracing/4spp/16total/OptiX、三原720/480相机、物理、机器人和reset顺序；不改默认AA值、不设置多项候选、不在观察者/每帧自动写回。加入该flag读回/变更追踪/最终检查，若仍变或另有漂移则失败。`a100_full_v1`仍表示原目标图像配置，具体修复由新commit/digest绑定，旧源不改且不能用旧gate放行。

安装依据文件SHA `e4901268a16048062fe5c33bb3507752a907cb17fa7f4b6797835d8687816006`，精确路径为`/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim/extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/settings.py`；本票不修改该文件、不调用会切换为RT的set_render_rtx_realtime，只复用其非DLSS配置前置条件。

- task0 TRAIN138/seed0，reset→load138→reset，0专家/旧policy前缀、0模型/训练；H64资产SHA保持。
- 单次scene/session，24现有基础gate动作、≤1536controls；动作阶段≤1200s，总外墙钟≤1200s＋30自有清理。H69/H70已知冷启动约510s，本票为后续真实24动作留时间，并非延长原已结束run。失败停止，不自动重试。
- GPU3≤24576MiB、启动free≥32768/运行每卡free≥8192；自有SID在0/1/2辅助分别≤512MiB，0/1队友正常启停；2/3启动无外部进程。4CPU72–75，原资源门完全保持。
- run `/mnt/nvme_tmp/robodojo_agentic_20260925/h71_aa_prerequisite_v1`、runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h71_aa_prerequisite_v1`；新固定Git源码，无共享安装或活跃目录写入，不借旧shader缓存。
- 验收必须实际无设置漂移、原完整gate_ok/24动作/测量门、原reset计数以及人工原三视角；只越过H70断言不足以称harness通过。若输出为黑图/深度缺失则按真实失败处理。不验证press/VLM语义，更不称完整任务SR。

当前本地一行配套设置＋新单次入口已实现，CPU/独审/新source固定后再运行。

14:59独审要求关闭来源漂移：安装Replicator/settings.py的精确路径/SHA现已进入共用PT与compatible profile的DEPENDENCIES，父launcher与worker各自identity均检查，24安装文件缺失/改字节拒绝。新增对应CPU反例16目标过0.055s，原完整回归703/698pass5skip29.912s在此依赖门delta前；最终邻接/独审待，0launch。

15:03最终增量：113邻接0.970s，独立16 renderer、11 PT、3 compatible、18 native-gate和diff通过，无剩余实质代码阻塞；父/worker来源硬门确认有效，未修改安装文件，0H71 GPU。下一固定运行源、robo实际24依赖及资源预检后唯一原生运行。
