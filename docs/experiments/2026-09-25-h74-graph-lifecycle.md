# H74：同步传感图的原生生命周期修复

2026-09-25，北京时间；唯一owner Codex，源commit待固定。H73尚未验证同步相机，其首render因ReferenceTime attach的延迟USD guard失败；不重跑旧源，不关闭原生保护。

主要假设：遵循安装OmniGibson的非嵌套editing_usd事务，且以固定render-product路径detach，能消除同步I/O的初始化/卸载错误，继而实际检验H73控制/时间戳方案。不是一次新模型/策略改进实验。

- 仅附加时间annotator；attach、首次orchestrator建图、detach各自独立事务。上下文exit完成Fabric同步后才验时钟；错误保留primary，部分attach先退出原context再清理，不嵌套。
- 新initialize日志与唯一close纳入连续时钟验收；仍逐control恰好4physics ticks、capture/初始化/清理零物理推进，不改速度、相机、原24动作或目标门。
- Replicator ReferenceTime文档只承诺对应render/annotations，不直接证明等于SimulationContext累积physics时间。额外记录原生timeline before/after用于辨明时间域，原严格匹配暂保留；不猜offset、不以像素变化或人为改时间凑匹配。若不匹配，保存真实三路值并停，不能放行actor。
- 监管优先读通用failure.json与native_failure.json，保留两份及损坏回执，避免SDK exit0和缺result掩盖根错误。新CPU反例覆盖实际HydraTexture路径、非嵌套、context exit推进/异常、失败清理及exit0/1。

## 唯一原生预算与验收

- task0 TRAIN138/seed0，官方reset→load→reset，0专家/旧策略前缀、0模型调用/训练更新。
- 单次1200s总墙钟＋30s清理，最多24gate动作/1536controls，任何时钟/原gate失败即停；不自动追加同配置。
- GPU3自有≤24576MiB，辅助0/1/2各≤512MiB、每卡≥8192MiB余量，CPU72–75；保留队友GPU0/1及所有原产物。
- 新run `/mnt/nvme_tmp/robodojo_agentic_20260925/h74_graph_lifecycle_v1`、runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h74_graph_lifecycle_v1`，固定Git独立worktree。先双端CPU/独审/安装依赖和资源门，再唯一启动。
- 验收是三路同步证据、每control实际时间、原24动作及原native profile全部通过；不是完整任务SR。task3 gate/actor另登记，未过不能拿该次数据做SFT。

安装依据：OmniGibson simulator.py SHA d800c2832f24962c440c4781ebfdb4ea78c74aac37d2c74ee7894ae430f1e2a9；Replicator1.12.27 annotators/orchestrator沿H73三源pin。[同版本官方API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html)。无共享SDK修改。

16:52本地最终回归/独审：132邻接0.940s、完整semantic726/721pass5skip28.388s，独立46/46过，H73归档19件/SHA独立重算一致。仅放行新固定源下预登记一次原生复验；实际ReferenceTime时钟域和graph副作用仍待，未称任何SR/新训练收益。准备Git固定与robo同源预检，0launch。

16:54真实唯一启动：固定4781860399a25d67693e0d6b754cf28a5b696ba5，新Git worktree graph_lifecycle_4781860；robo132 CPU2.024s/24＋3安装源/资产/资源门全过，digest86702d8af9df0fb93eadfdc4af950da9b5997d3c68c8a76bfdab77195bc1bb0c。launch UTC08:53:53.496210/监管3597389，run/runtime已建；worker和实际初始化待核，原0模型训练/前缀，GPU0队友3564916保留。源冻结，不重提或热改。
