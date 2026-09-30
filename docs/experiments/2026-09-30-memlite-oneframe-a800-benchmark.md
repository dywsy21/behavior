# 单帧 MEM-Lite / 八卡 batch64、128 短测

2026-09-30，Codex / BENCH-MEM1F，分支 `bench/memlite-oneframe-a800-20260930`。用户授权短测速，不是全量训练。

## 预登记

- 假设：单帧与更大的每卡 microbatch 能提高旧 A4 低层的观察样本吞吐；不据测速判断任务效果。
- 范围：单节点 lc3、8×A80080GB PCIe，先 global64=8×8×1，再 global128=8×16×1。保留三相机256、32未来动作、23真实控制/27表示、FM四噪声、FM→VLM LoRA r8＋动作专家；高层不参与低层更新。
- 数据：固定官方 2026 revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`，ModelScope snapshot 的 RGB/动作/meta/annotation，不下 depth/raw。只读取有限代表样本，不构造/发布百任务正式监督集，不保存部署权重。
- **12:04北京时间用户覆盖：不使用随机初始化，优先同步旧模型，缺失才下载G0.5。** robo原A4完整checkpoint已找到，16,581,363,550B；本轮重新计算SHA256为`6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`，与历史一致。完整权重/LoRA迁至共享盘并验SHA后才允许测速；原随机初始化方案取消，0GPU更新。
- 实施：独立测速入口复用实际 G0.5 神经计算图和 A4 LoRA/冻结/技能前缀配置，区别缓存输入计算吞吐与真实视频/动作读取吞吐。若不能完整贯通原 SkillFM 的数据合同，明确标注性能代理与未验证项，不用伪造合格标签/削掉梯度换速度。
- 短测预算：每个 batch 至多40更新（包含预热与计时），两臂80；若128 OOM可一次 micro8×accum2 的128替代，额外≤25更新。单臂GPU墙钟≤20分钟、总GPU运行≤50分钟；CPU准备≤30分钟，有限读取≤10,000观察窗口。内存不足/非有限loss/梯度缺失/数据不匹配/外部占卡即停止当前臂，保留失败证据，不启动长训。
- 对照：两臂同 seed73、同模型/输入协议/训练参数集合；报告每卡batch、累积、真实样本/s、时间、峰值显存、数据等待和包含/排除的开销。不能只按更新/s比较batch，也不能把四次FM噪声算四个观察样本。
- 代码：启动前 commit/push，服务器独立固定 worktree；共享env与原infra源不改。单节点没有队友GPU进程才启动，绝不停止队友任务。结果和源SHA补入本文及plan。
- 全量外推：原始210,916,774帧与约95% train两种口径分开；原逐task留出、技能有效区间未完成正式发布，外推不冒充实际唯一全覆盖。

## 当前状态

13:18（北京时间）GPU计算探针修复初始化后CPU/GPU权重比较问题，源`7af393b`、`batch64-v2`已八rank全部1138状态/192LoRA逐字节恢复；预热未完成、吞吐待。原`2193368`/v1失败0更新，约68s计入原预算，v2仅1100s超时、30更新。CPU七回归实际通过。百任务I/O采用独立只读代理：100任务×一episode×四窗口×两pass=800，32CPU worker/≤600s，须计算臂全部退出后才运行；保留“缓存计算≠完整训练、grouped读盘≠正式shuffle”的限制。

13:02（北京时间）**A4权重与14配套资产迁移完成**：主权重`models/memlite-a4-20260912/step_2500.pt`，16,581,363,550B，源/本地/目的完整SHA一致（`61867047…32f269`）。真实checkpoint已就位，旧原件和中转保留。新增单帧入口/六项CPU合同测试待服务器执行，0GPU更新；旧五任务十行TRAIN缓存仅用于计算吞吐，不当百任务端到端I/O证据。

12:43（北京时间）主A4在本地完整SHA通过后自动上传共享盘中，最终名尚未发布；`models/action_tokenizer.pt`源/本地/目的三个完整SHA已一致。完整14件配套资产已就位（原13件＋ActionCodec），原配置仅作档案，含robo绝对路径；后续须显式覆盖模型、processor、codec路径，不能原样开训。高层B-final不是本轮低层测速依赖，未迁移。还没有神经前向、梯度或batch吞吐结果。

12:31（北京时间）配套资产13件已完成源/目的逐文件SHA验收：A4五份配置/统计/回执＋Qwen3.5 processor八件；本地`assets/`、共享`models/memlite-a4-20260912/`及`models/qwen3_5_2b_base_processor/`。CPU内存解析单帧配置通过，但未做模型恢复/前向。加载器还需独立ActionCodec（506,886,775B，SHA `5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace`），已开始同步；该分支在纯FM执行中不生成动作token，不因此改训练目标。主A4同步仍进行，0GPU更新。

12:15（北京时间）单连接慢，转为固定`7960bb0`的校验式8连接/16MiB分块中转，PID193613、989块；旧rsync315MB partial保留。本地状态`weights/sync-status.json`，12:14:33累计503,316,480B/33.7s。脚本在完整SHA通过后才上传，远端再次完整SHA通过才发布最终文件；测试3项/7组边界、py_compile/diff通过，0GPU启动。源CPU mmap证1138模型条目/192 LoRA、全FP32；完整checkpoint保留optimizer/RNG，不改变数值精度。

12:04（北京时间）旧A4完整权重同步准备/传输中：源`robo:/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt`，本地可续传暂存`artifacts/a800-memlite-oneframe-bench-20260930/weights/A4-step2500.pt.partial`，计划目的`lc3:/data/workspace/wsy/behavior2026/models/memlite-a4-20260912/step_2500.pt`。robo不能直连lc3，使用本地受控中转；完整大小/SHA验收前不放行。共享盘余约4TiB、本地600GiB，原文件不移动/删除；无需G0.5替代。

11:55（北京时间）只读确认 lc3 八卡均0MiB/0计算进程，复用现有VPN连接，无重认证。小processor资产已同步，tokenizer.json与tokenizer_config.json双端SHA一致；完整资产清单验收待。尚未启动GPU训练。
