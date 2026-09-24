# H56：保留提前配置的相机，改用PathTracing/OptiX

2026-09-24 23:04北京时间登记，owner Codex，分支`feat/semantic-agent-grounded-20260918`。71目标CPU通过，独审待；尚未部署/启动GPU，准确commit由首次launch固定。

## 主要假设与对照

H55在600s内完成scene导入并进入相机渲染，但未完成Evaluator/reset/RGB-D，末仍出现A100 DLSS-RR不支持警告。743样本自有峰3317MiB，四训练保留；不是最终峰值/图像通过。全量负例见[H55](2026-09-24-h55-preconfigured-cameras.md)。

本票唯一主要假设：**在相机首次创建就是最终RGB-D尺寸的条件下，换为原生PathTracing/OptiX兼容profile，能在原共享资源预算取得完整原场景观测。** 不声称RR警告已证明超时原因，也不预设PT更快或显存更低。相对于H55只改renderer整套兼容profile，不再回退到1080初始化。旧H54b继续未运行；不以旧H54早期前置断言失败当作已测过真实PT场景。

兼容性依据/具体十项设置来自已独审[H54/H54b](2026-09-24-h54-pathtracing-optix.md)：[NVIDIA特性表](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_architecture_support.html)列A100支持OptiX、不支持DLSS-RR；这不等于官方支持全部Isaac/OG功能。使用PathTracing/后处理AA0/PT-DLSS关闭/4spp/累计16/4bounce/specular-transmission6/OptiX on/temporal off/blend0。变更图像噪声和光照积分分布，不能混称和RT2像素完全相同；必须人工审。

## 固定边界

- 新独立入口`probe_scene_compatible_cameras.py`。只此入口显式允许camera提前配置＋PT；原H55/H54单项入口仍拒绝未经登记的组合并在configure时清除组合标志。
- 保留H55三camera元数据映射、head720/双腕480、RGB-D、只读wrapper/延后space reload、初始/current尺寸和sensor/render-product身份校验；不减少actor视图、删物体或改变physics/材质/光源/23D控制。
- 保留H54b原生empty app与原og.launch→empty sim两个边界的profile重应用，以对抗SimulationApp覆盖totalSpp及原OG覆盖mode；接受实际观测到的两个合法实时前值，unknown仍拒绝。实际scene/capture之后只读核十项PT设置，不能在live camera上切renderer。
- 原TRAIN138/seed0/window/robot/Isaac5.1.0.0/OG3.9.1开发运行时，原OfficialSession reset→load138→reset，无演示/旧策略前缀。全部23依赖SHA启动时核验，独立Git源码，不写共享安装或热改训练环境。
- **另一次600s活动监管＋最多30s自有清理**，1Session/一次三RGB-D/4render-only，0actor、前缀、模型、训练。仍GPU3新增及自有≤4096MiB、辅助每卡≤512MiB、运行各卡free≥3072MiB/启动≥7168MiB；四训练原PID/UUID严格保持，未知进程/资源/时间越界只停自有child。不是给H55临时加时或提高显存额度。
- 同CPU72–75/Kit4线程/texture0.01和16MiB、新冷私有runtime；不拷贝旧shader缓存、不改变冷启动条件。全量失败也归档，不能反复加码启动直到碰巧过门。

## 验收

71目标CPU包括全部H52–H55回归、明确组合profile预算/依赖/恢复、真实H56 mock构造链，同时检验两阶段原生renderer覆盖、camera配置、原reset链、捕获/身份和所有strict success字段。还需独审、服务器同测/23依赖及资源门。CPU通过不替代实际观测；本票完整通过需真实3RGB-D、物理状态未动、准确camera身份和PT设置、资源/退出、全部RAW人工审。

不是actor评测，不计SR分母，不完成goal；后续仍是原reset/0专家及旧策略前缀、官方物理成功。若失败，先审真实原因，不自行动训练、扩大显存或延时。

- run `/mnt/nvme_tmp/robodojo_agentic_20260924/h56_compatible_cameras_v1`
- runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h56_compatible_cameras_v1`
- 本地全包 `artifacts/agentic-vlm-goal-20260918/h56_compatible_cameras_bundle_v1`

以上目录尚未创建；旧H55/H54/H53源码、缓存与证据全部保留。

23:06独立窄审通过，复跑71/71/diff过，确认两套终验/组合默认关闭/23依赖并集与原资源门无缺失；准备Git固定及服务器同门，尚未运行。单次通过也只能说明本组合能否在这次冷启动额度内完成，不做强因果/速度归因。

23:09实际提交：固定`d7ed21efcad7af313a6b0f72f7c5155d3230ab49`/`git_worktrees/shared_scene_d7ed21e`，服务器71 CPU/23依赖/原YAML真实转换/Hydra target与资源门过。此前仅同步命令SSH超时（不含launch），只读核源码/run均不存在后才续接，没有重复提交。唯一launch UTC15:09:30.708713，supervisor3496762，原目录/预算；真实scene/三RGB-D/renderer终验及资源结果待，不算实际OptiX执行或相机成功。
