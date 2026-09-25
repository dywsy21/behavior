# H72：相机与控制时钟的只读原生审计

2026-09-25 15:30北京时间，owner Codex，源待固定。H71初始化已通过，原底盘后退60mm在RGB-D中只有30.914mm；219→225两head depth逐bit一致。无真实物理位移/时间，不把速度积分57.272mm当真值。

主要假设：当前PathTracing/原生更新链中的观测与物理不同步，导致看似高置信的零位移。替代解释包括真实物理运动不足、时钟偏差和视觉求解误差；本票同时记录以区分，不预设根因。

- 对照H71，task0 TRAIN138/seed0/原reset→load→reset/零专家与旧策略前缀，原24基础动作、23维控制、4render屏障、RGB-D里程计和位移门全保留；不改变PT累积、物理参数或执行步长。
- 独立诊断入口只记录：真实sim time/index与三个dt、机器人从初始位姿起的相对位移、每次render前后的head RGB/depth指纹。私有位姿仅落审计文件，不进入actor、运动估计或控制回执；没有actor/VLM/训练。
- 模块digest额外绑定诊断代码并明确不是policy gate。诊断完成和正常gate通过分开；即使诊断跑通也不能授权actor。
- 单次总1200s＋30清理，最多24决策/1536controls，原GPU3≤24576MiB、自有辅助0/1/2每卡≤512MiB、全卡free≥8192MiB；4CPU72–75。失败不自动重提。CPU每次≤120s，独审后Git固定/robo复检。
- run `/mnt/nvme_tmp/robodojo_agentic_20260925/h72_observation_clock_v1`；runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h72_observation_clock_v1`。新source worktree待定，原所有目录/队友进程保留。

安装源已核：SensorBase.get_obs调用VisionSensor._get_obs直接读取annotator，无Python帧缓存；OG宏默认physics120/render30/action30；Isaac5.1 SimulationContext.render会force_update Fabric并在playSimulations=False下app.update。它们不能证明当前run真实时钟/图像已同步，需要本票实测。当前尚未启动，无新效果或SR证据。

15:34实现：新增独立probe_observation_clock.py与launch_h72.py；state时钟变化及每capture前/每原4render后/返回时记审计，日志≤8192条并绑定SHA，原result写入前封存。返回原公开对象、不增加physics/render、不改变失败门；诊断摘要和真实gate_ok分开，digest另外绑定两新文件，不能放行普通actor。10新用例＋邻接55过0.178s，初次测试模块路径错误已纠正重跑；独审待，0launch。

15:41重要更正/实现：安装R1Pro get_position_orientation确选base_footprint，但底层XFormPrim→get_world_pose读取Fabric hierarchy，不能当独立物理真值。因此改为从原PhysX articulation view直接get_link_transforms读named base_footprint，不改OG缓存/不强制Fabric更新，同时记录Fabric位姿；两者相对同一PhysX起始frame。SimulationContext、usd_utils、xform三安装源精确SHA均父/worker硬验。

独审两缺口已修：必须验证decision11真执行和完整base链对应的每snapshot六条前/后render序列及RGB-D SHA；不能靠“12决策”或两行日志放行。失败时在native context退出前封存partial摘要，secondary保存失败不覆盖原错。诊断完成仍不等于gate通过，0launch；最终回归/独审待。

观测者效应限定：额外get_obs和PhysX张量读取可能同步GPU，虽没有新增render/physics，异步时序仍不与H71完全相同。现每事件先读head/取指纹，再读PhysX/Fabric，并记录两部分monotonic耗时和实际张量backend/device。若H72不再出现重复图，不能单凭它排除H71的发布滞后；需考虑额外读引入的等待。摘要明确instrumented_timing_not_identical_to_H71=true。

15:45最终审结：父118邻接过0.991s、完整semantic703中698过/5skip28.842s；独立14目标0.059s，三项已关闭、无新实质阻塞。固定运行源b59498f834bfb86e79f920ddf0f9708b0fb4764e已push，robo新worktree/CPU预检中，仍0launch。私有native API/真实backend和物理诊断尚待。

15:46唯一原生提交：robo clean `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/observation_clock_b59498f`，118 CPU1.910s/24＋3依赖/资产/资源门过。原normal digest6e50eb1b…0661e保持、diagnostic digest ab1306a356945c872a1ea0a7fe1187f4cecd1a26bb8663c4b8d3c6a8ccd83fdc。launch UTC07:46:36.616892/supervisor3585374；GPU0队友3564916/12548MiB保持、1/2/3空。预算不变，真实初始化/终态待，不重提。

15:51只读候选核对：[NVIDIA Isaac5.1官方文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html#async-rendering-and-frame-skipping)记录throttling把asyncRendering切true时可能丢帧，但服务器安装`isaacsim.core.throttling/config/extension.toml:27`已经默认enable_async=false；不据文档假定本run命中，不热改。另从已安装SimulationContext.step看render=True依赖app.update，而render=False显式physics step；默认dt正确不能代替实测每control物理推进。H72时钟用于区分这项替代解释，原输入/控制/预算不改。

## 实际终态与初步根因（16:05北京时间）

738.518454s，监管completed仅表示诊断完整；普通gate_ok=false，12决策/232controls，末1安全保持。supervisor3585374/worker3585381均已退出；GPU只剩队友3564916/12548MiB。0模型/训练/前缀，不是完整SR样本。

443事件journal SHA `0f8cde334a22613d1bd8b9fe05a078b05ad1015a11f2d6dc4d53ac41531c4999` 已在本地匹配。capture_return47–50对应base开始与三个6control段，初步数值：

| snapshot | physics index | sim time(s) | PhysX相对起点X(mm) | head depth SHA前缀 |
| --- | ---: | ---: | ---: | --- |
| 47 | 553 | 4.608333574 | 0 | 11ad4e4a20bf |
| 48 | 569 | 4.741666914 | -5.8223 | d0b9bc05f9d3 |
| 49 | 585 | 4.875000254 | -30.9960 | d0b9bc05f9d3 |
| 50 | 597 | 4.975000259 | -34.8881 | 770baae43d90 |

- 18次控制应推进0.6s/72physics ticks，实际0.366667s/44ticks；不是单靠默认dt作推断。现有journal只在变化时记录state，不能据此给每次未推进调用编号，须在修复中记录每次before/after。
- 48→49 PhysX移动25.174mm而depth完全重复，Fabric全443条的位移与PhysX相同；每次capture内time/index未变。故当前证据支持物理推进不足与观测buffer滞后并存，不支持“全部是里程计比例错”或“只是Fabric没更新”。
- 142条new_state_clock相邻差都是4ticks，实际原生tensor为CPU torch.Tensor，仍承认额外读的观测者效应。
- 后继候选：显式physics-only env.step并逐调用断言4ticks/1/30s；按原生render完成与reference timestamp取三路RGB-D，而非用4次调用或图像差分冒充freshness。安装API需先核，再单独冻结测试；不改物理参数/目标门、不泄露诊断真值。

完整归档/逐段receipt绑定和独立审查正在进行，尚未新launch或宣布修复成立。

16:07归档与绑定完成：529件210241629B，规范排序path/bytes/SHA清单聚合`3df519b44ff2fe2fd093dde6f33de50eb7a954c1952c9a7c1d680ec8e77e95f9`双端全同；本地`artifacts/agentic-vlm-goal-20260918/h72_observation_clock_bundle_v1`。四receipt分别绑定control213/219/225/231与snapshot47/48/49/50，action_motion三个原区间对应，无额外控制。独审继续，0新launch。
