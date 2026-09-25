# H70：定位原生renderer设置覆盖

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
