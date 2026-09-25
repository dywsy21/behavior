# H70：定位原生renderer设置覆盖

## 终态（2026-09-25 14:55北京时间）

source `3642dd77867cb39f743397771e3d016aa516c115`；唯一launch UTC06:43:02.583122，supervisor3572777/worker3572784均已退出。508.5141857s、worker exit0、监管failed。原错误在SDK关闭前写出，监管引用真实AA差异，不再只报缺result。reset1/load138各1 completed、最终reset0/controls0/model0/training0；诊断目的达成，完整gate和SR未通过。

5个真实Carb通知/0丢失/0记录错误。第1个在订阅后367.060845s：`/rtx/post/aa/op:0→3`，堆栈为`VisionSensor._post_load→clipping_range→og.sim.render→SimulationContext.render→app.update`。之后4个为Replicator`_restore_settings`对spp/totalSpp原值的通知，未改变其他9个注册目标。真正native C++setter不在Python栈中；没有读取旧limitedOps值，不能单凭这个trace证明后继修复因果性。

209资源采样：主GPU3自有峰5199MiB、余量最低75918MiB，辅助0/1/2峰456/416/416MiB；退出后1/2/3空，GPU0队友3564916/12548MiB保留。12件4553758B完整取回`artifacts/agentic-vlm-goal-20260918/h70_renderer_trace_bundle_v1`，下列12个SHA逐件双端一致。无图像或动作产物。

| 相对文件 | SHA256 |
| --- | --- |
| launch.json | d8d0de4469ca1405f170bd90d46798ceb779f442d941caa319603fae7f1a6e25 |
| supervisor.log | 75eb7ff4bb9f964e0544ac054e4c673634a9a48a2916cc3eef10edfb197040bb |
| supervisor.claim.json | 826d7f13952605f618da1df50dd4ad048a4ab664e875549834b8b1fea25267cb |
| worker.log | 5e0018ecfac84d26768ea970e11d4c93932d042ae628d2aa4e3e5d1a85813dc5 |
| supervisor.json | 58ed60200c2dd0ab555fe4dee9636a5ab16076931c1262e943c7df3d5c41e499 |
| gate/manifest.json | 218ccfc3523469fada9c8aa66774b947715477464e734a0b1aabc6060f5f1c98 |
| gate/steps.jsonl | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| gate/worker.json | 41e3b9635c90aedb44d43e5451f493353293cdf53e7a0e3ea68995653c6a5d72 |
| gate/native_profile.json | 5d8f695b242a3066165a5be305541d7f9c0603d9e357aa5b8a42736d97051e78 |
| gate/native_failure.json | 5d8f695b242a3066165a5be305541d7f9c0603d9e357aa5b8a42736d97051e78 |
| gate/pathtracing_changes.json | 531b0b47b9573b9f9d628c799da10d6d11593ee7a33c3ab08f1da25fa605bbab |
| gate/kit.log | eb28cae992f13192203da2cf612a0617388c11b100ad2946a5d84084547493ee |

以下原预登记/执行史保留，不覆盖终态。

2026-09-25 14:35北京时间，owner Codex；当前基线5e4ce75/H69，运行commit经修审后固定。

唯一假设：H69 empty-simulator边界到原scene/reset/load期间存在可定位的渲染设置覆盖；完整的actual/diff与Carb变更通知能区分SDK主动改写、场景恢复和检查错误。旧H69只存早期通过快照，不能直接判定哪个字段变了。此票不预设或修正设置值，不放宽门。

- 原task0 TRAIN138/seed0、原机器人/window，原reset→load138→reset；0专家/旧policy前缀，0模型/训练。只在已验证empty-simulator边界订阅10个设置变更，最多64条栈/值记录。订阅仅只读；超限/记录错误仍失败。
- 唯一新session，外部总900s＋30s仅自有进程清理；沿H69至多24 gate决策/1536控制/动作1200s（总900s优先）。如果同一初始化断言再次触发，立即保存后退出，不能忽略异常继续控制。不自动重试。
- GPU3主≤24GiB、各卡8GiB余量、启动空余≥32GiB；其他卡自有SID上下文合计≤512MiB，0/1其他训练正常启停，2/3无其他进程。4CPU72–75，无新依赖安装、无共享文件改写，原三RGB-D尺寸/物理不变。
- 新源码worktree和run `/mnt/nvme_tmp/robodojo_agentic_20260925/h70_renderer_trace_v1`、runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h70_renderer_trace_v1`；不复用H69 launch或live cache。H69只读完整归档已有10件。
- 验收：报错时具体actual/diff与最初覆盖事件、Python调用栈（不冒充C++栈），异常在SDK __exit__前持久化；监管引用真实native_failure而非仅缺失result。如果没有漂移，仍必须原gate通过，不能仅因exit0标成功。无论如何都不是完整任务SR。
- CPU回归/独审先行，固定新源后单次物理复验。具体根因出来后另提最小修复，不能为赶结果在观察者里自动改回设置。

14:39独审修正：`native_failure`只归enter/初始化检查/二次reset及成功路径shutdown前最后检查，不把调用者动作异常重新归因为native。SDK shutdown前再核actual及完整事件，包含曾漂移后恢复也拒绝；动作原异常不被最后检查覆盖。13新目标回归通过0.040s，最终复审待，0launch。

14:41最终独审通过：独立13新目标/18native gate/11PT均过，父110邻接过1.047s，两项原阻塞关闭。同步fake不能证明真实Carb线程/unsubscribe行为；只批准本票唯一900s真实诊断。源即将固定，尚未launch。
