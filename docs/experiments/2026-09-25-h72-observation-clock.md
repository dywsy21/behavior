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
