# MEM-Lite v9：本人复查、改进与五任务 5000 步重训

> 当前状态索引（2026-09-06 23:24 UTC）：v9训练已5000步正常结束；官方五任务评测完成2/5，收音机和垃圾均超时失败、进程exit0，其余三项未结束。最新阶段结论见`MEMLITE_V9_EVAL_REPORT.md`，未部署的后续候选见`MEMLITE_V10_CANDIDATE.md`（v2补丁918行/10文件，145项测试通过）。下方按实施过程保留历史记录，其中“尚未训练/尚未部署”等旧句不代表此刻状态。主仓库18项源码哈希23:22再次一致，本轮评测未中途改动。

目标：在 robo 的 `/mnt/sdc1/robodojo/GalaxeaVLA` 改进完整的高低层 MEM-Lite，保留五任务、逐任务 5% episode holdout、纯 AR 和 5000 optimizer steps；不启动已取消的基线。本文是执行计划，不是完成声明。旧 v8 权重、数据、评测原件不覆盖。分析和实现由主 agent 自己完成，不使用 subagent。

## 1. 本人从当前源码确认的失效链

1. `vq_base.py::_derive_noop_keys` 只根据 future action 恒定省略 rule-based gripper，不比较当前状态或上一条命令。恒定 close 与恒定 open 都可能丢失。
2. `FullProcessor.build_pixel_values` 将 3×6 帧展平后随机取三帧（训练）或取最早三张 head（运行），再平均分回三个 camera key。因此 `G05ModelQwen35._forward_vision` 实际收到每 camera `n_k=1`，`has_mem` 为 false：不仅相机/时间不匹配，多帧 MEM 路径本身也没进入。
3. 数据端 history stride 默认为 1/30 秒；server 只在 16 步 chunk 边界 append。当前三路完整历史应为 `[t-80,t-64,t-48,t-32,t-16,t]`，训练必须匹配。
4. 旧高层标签在 `t` 输入 `State(t)`，却以 `State(next high event)` 为 memory_update；server 随即提交。因此未完成的动作会提前进入 Completed。
5. 高层使用 primitive 意图，低层使用 skill 意图，同一规划和控制条件不一致。低层从未训练 `Task complete`，server 却在 DONE 后继续调用它。
6. 高层 parser 补缺字段，格式检查只记日志。缺 Updated Memory 时 Status 被错写入 memory，无效意图进入 LL。
7. AR codec 输出无界；tail normalizer backward 中的 `expm1` 无界。有限 normalized 偏差可放大成异常动作。旧日志不足以唯一确定 Halloween spike 的来源，不能假装已证明；新实现必须同时避免放大并记录中间量。

## 2. 具体改进设计

### A. 视觉契约：保留真实多帧 MEM

- 增加显式 camera-history-preserving 模式，不随机跨 camera 抽帧，不把 head 冒充其他 camera 的数组。
- 三路各六帧保持 camera-major、每路按时间升序；当前帧必须在末尾。使用现有 causal spacetime vision 和 token drop，最后仍仅输出三个 camera 的视觉特征，不把 18 张图当成 18 个独立语言占位符。
- 新任务配置训练 `obs_stride_second=16/30`，运行 `action_steps=16`，启动时验证两者一致。保留多帧视觉，不能通过改成单帧来让测试容易通过。
- 验证 marker 图像/不同分辨率/顺序、真实 processor 到 vision 的 n_k=6、改变历史会改变输出、最新三路存在，训练和运行同帧条件一致。

### B. 夹爪：始终监督明确的目标命令

- 新训练配置关闭 noop part dropout，保留左右夹爪的显式 rule-based token；不删除 rule-based 二值编码本身。
- 对真实 train open→close、close→open 和持续持物窗口走完整 processor→codec→decode，检查命令仍在。
- 新运行对缺失必需夹爪 group 进行拒绝/诊断，不静默把它当合法预测。安全等待时沿用最近有效的夹爪命令，区别于重新设为当前指宽。

### C. 数据：回顾式记忆、同粒度意图和未完成样本

- 复用原始 annotation 的严格区间解析、并行任务/缺口处理，生成一个新的版本化 sidecar，不改旧 v8。
- 每帧 LL 条件改为其当前 canonical primitive 意图；高层在该帧输出同一文本。保留原 skill 边界用于截断 action supervision，不能让 action horizon 穿越无标签区间。
- 高层锚点覆盖意图变化及每 128 帧的中间状态，而不是只有 primitive 开始和终点。高层输入为上一轮已观察的 memory 加 `previous_intent`；输出 memory_update 只包括当前观察时刻之前已结束的 primitives。
- 输出当前可执行意图与当前任务状态，不能在执行前把新意图加入 Completed。高层固定周期相应改为 8 chunks = 128 actions，周期内保持 LL 控制。
- 构造可追溯的错误文本记忆样本：只对上一轮已经知道但尚未完成的 intent 提前宣称完成，或删除历史条目；目标仍是当前真实 annotation 状态。禁止把完整未来 primitive 列表/对象身份作为负例输入，这仍会暴露未来信息。明确标注 synthetic memory corruption，不捏造视觉失败、奖励、FAILED 标签或实机恢复轨迹；另保留干净评估分类。
- DONE 只能放在实际存在的、原始 annotation 完成边界之后的观察帧。若 `terminal_end` 已超出 episode，记录缺少 post-action terminal observation，不能退到 `end-1` 再把未来 State(end) 当现状。
- 检查每 task 的样本量、分支比例、时间因果性、HL/LL intent 完全匹配、terminal 观察可用性、坏记忆可纠正性。本人抽查各任务的开始/中间/切换/终点/并行与坏记忆样本，并看对应原始图像；有错先修数据。

### D. 运行时：显式控制状态机

- 统一严格三字段解析，拒绝缺失/重复/额外字段、非法 status、空内容和超长内容，不通过截断/补字段假装修复。
- 解析、任务身份和状态检查全部成功后才原子更新 memory/intent；失败保持上一份有效状态并记录拒绝原因。
- `CONTINUE` 且意图可执行才调用 LL。`DONE` 只是模型的完成声明，不是官方 success；进入安全等待与低频高层复查，不把 `Task complete` 喂给 LL，也不篡改官方终止条件。
- 对 schema/action 安全拒绝使用明确 hold+replan 路径，不悄悄退回普通 VLA；基础设施错误仍报错，不能把 CUDA/IO 故障伪装成正常策略失败。
- 完整 reset 连接内 memory、active intent、缓存、历史、重试计数及最近有效夹爪命令。

### E. 数值：限制逆变换并保留可定位证据

- 增加显式 bounded tail normalization 模式，训练正向保持现有表示；逆变换在指数运算前限制到 train-only action 统计范围，避免先 overflow 再 nan_to_num。不能只夹 normalized ±5 就宣称物理安全。
- 统计边界按 horizon/维度匹配，不使用 eval 数据。保持正常数据 round-trip，并验证极端值、退化维、padding、短 horizon 和非有限输入。
- 运行侧记录 decoded normalized、denormalized delta、当前 state、最终 target 的范围与裁剪数量；对异常输出拒绝执行并触发重新规划，避免长时间饱和动作被隐藏。
- 机械关节限位只是一层 guard，不是学习成功的证明；不能把 target clamp 等同于测得的实际关节运动。

## 3. 实施与验证门槛

1. 保存相关远端原文件哈希；本地 staging 使用 apply_patch 编辑，提交远端前检查文件未被其他人改动。保留已有 dirty worktree。
2. 用真实 saved config/stats/tokenizer、marker 视觉输入和异步 mock server 对上述契约做回归；测试不仅检查 finite 或 shape，还检查条件、时序和状态转换。
3. 新 sidecar 全量结构验证与本人抽样审阅后才允许训练。先真实 GPU smoke：高低层都产生 loss/梯度，真实六帧 vision 分支运行，训练/运行都保留 gripper token。
4. 从原 G05 base 重新训练，不续接带旧监督问题的 v8 checkpoint。五任务、原 task-stratified 5% holdout、5000 optimizer steps、每 500 步 checkpoint；显存不足只能调整 microbatch/梯度累积，不能静默丢历史或训练分支。
5. 监控真实进程与日志直至 step5000 和正常退出；检查 checkpoint metadata、last 指向、分支 loss、eval 与数值/视觉诊断。不是看见 launcher/PID 文件就算已经开训。
6. 用新 checkpoint 做闭环进展检查，明确样本范围与成功/失败，不用 teacher-forcing loss 替代成功率。必要的官方 rollout 保持原始目标与 timeout，不运行已取消的基线。

## 4. 完成条件

详细方案、实际代码和数据变更、上述真实路径验证、新五任务 5000 步训练完成以及新模型的进展证据缺一不可。当前仅完成源码复查和设计，实施/训练/新结果仍待完成。

## 5. 实施验证记录（2026-09-06，尚未完成）

- 修复在独立 staging 实现，尚未写入远端主仓库，也未启动重训。准备以原文件 SHA256 校验和增量 patch 部署，保留用户已有改动。
- 60 项当前版本核心回归通过，另有 39 项先前兼容/sidecar/sampler 回归通过（两批有重叠，不相加充当独立测试数）。
- 使用真实 train-only stats、ActionCodec 权重和 episode 0 的 9 个动作窗口验证，包含不同夹爪目标及开合切换，左右夹爪均未缺失。
- 使用真实 G05 base 视觉权重，在 GPU 0 验证每相机六帧经真实 `_forward_vision` 路径；输出 `[1,192,2048]`。只改历史、保持当前帧不变，最大特征变化为 61.174；这不是训练效果指标，只证明历史确实影响视觉输出。
- 1000 个原始 annotation 的保守长度预检查未发现超出 1024 输出 token 或 schema 字符上限的 episode；最长候选输出约 638 token。仍需对生成 sidecar 的全部高层样本精确审计。新训练 token 上限 4096，并对 causal prompt 超限显式报错，不能静默截断 memory/proprio。开启 VLM checkpointing 控制显存。
- Markerless AR 序列在新配置下显式拒绝，不再允许 decoder 的合成零动作被当成正常完整输出。
- 新 sidecar 正在生成，尚未完成全量审计与本人手工审阅；因此训练仍未启动。

### 源标注人工复核新发现

官方/数据集 task ID 的准确映射为 `0=radio, 1=trash, 2=Halloween, 3=plates, 4=can_meat`。不得按旧报告的展示顺序反推 ID。

本人逐条阅读原始 JSON 后确认：`can_meat` 的 episode 886、964、998（raw IDs 41510、42490、42960）中，primitive 1 将第二罐的 skill 16 导航错误挂到第一罐。该 skill 的目标香肠分别为 `bratwurst_231/232/231`，都属于 primitive 3 的第二罐 `hinged_jar_235`，而非 primitive 1 的 `hinged_jar_236`。primitive 1 因此出现 `[start,[substart,end]]` 嵌套结束时间，并延伸到第一罐放回柜子之后。

不猜测修改 ground truth：新版 causal builder 隔离带嵌套 primitive 区间的 episode，记录 `ambiguous_primitive_interval`。原始 JSON、视频和旧 sidecar 不修改。预计有效 996/1000 episodes（另外已有 544 严重截断）；需以完整 manifest 最终核实。初版候选 `memlite_annotations_task0_4_causal_v9_20260906` **不得训练**；修订候选为 `memlite_annotations_task0_4_causal_v9_reviewed_20260906`，其中 `reviewed` 仅是版本名，人工审批仍为待完成。

重新构建加入不改变文本语义的 episode 内缓存，避免在每帧重复渲染同一 primitive/memory；回归测试和全量审计必须确认结果。

### 最终候选与验证门槛

- 当前 87 项相关 CPU 回归通过。真实 ActionCodec 的 9 个示范窗口进一步通过完整逆变换和运行时动作准入，输出全部六组原始动作，最大 normalized 越界仅 0.02035（拒绝阈值为 1），没有误拒绝这些真实编码样本。
- 缓存改写已做全量 JSONL 校验：排除 886/964/998 后，两次构建的 9,191,858 行逐字节 SHA256 相同（`33bba4e90e885af15666756d01e87350665b9f5ca3dfccf0f331d35e7220e690`）。
- 隔离异常 episode 后的候选通过全部 9,191,858 行结构审计及 74,803 行高层目标与原始 primitive 事件时间的独立逐条核对；有效 996 episodes，801 个 DONE 锚点，其余末尾没有可用的完成后观测，不伪造 DONE。
- 最后收紧了错误记忆增强：只能提前宣称 **此前已知 intent** 完成，不能输入真实未来全部对象和步骤。此前两个候选均不可用于正式训练；正式目标为 `memlite_annotations_task0_4_causal_v9_final_20260906`，须重新完成全量审计、本人至少 150 条文本样本及 30 个三相机观察时刻的审阅后才能签发精确 SHA256 审批。训练入口已仅指向该最终候选。
- 本节是进展，不是训练完成或人工审批。新 5000 步训练尚未启动。

### 02:04 UTC 部署状态

20 个文件的增量 patch 已写入 `robo:/mnt/sdc1/robodojo/GalaxeaVLA`；11 个被修改原文件全部通过原始 SHA256 校验，`git apply --check` 通过后才应用，未覆盖其他已有变更。主仓库直接运行 87 项相关测试全部通过，训练脚本通过 `bash -n`。最终 sidecar 已生成 9,191,858 行，正在写出 parquet；人工审阅和训练仍未完成。

### 最终数据人工准入审阅完成

以上各节的“尚未部署/生成/审阅”是当时进展，不代表当前状态。最终 parquet 已写完并通过全量审计，SHA256 为 `a5f1026110cac3ed57682d8177d2dabcb2f30d6572791ebb27c2800100f5c5d2`。本人已逐条读完最终 164 条文字样本，并查看 30 个不同观察时刻、90 路相机图像，记录见 `memlite-v9-review-20260906/PRIMARY_REVIEW.md`。已签发精确工件的训练准入 receipt；明确保留源弱标注、隐藏状态不可仅由 RGB 验证、擦除记忆不等于真实恢复数据等限制。没有把人工抽样等同于全数据绝对无误。下一步为真实四卡 smoke，再正式 5000 步；新训练尚未开始。

### 四卡 smoke 与 eval 统计修复

第一轮 `behavior5_memlite_ar_v9_smoke_20260906T022210Z` 完成 1 次 optimizer update + 1 次 validation，launcher exit_code=0，四卡均正常释放。真实每相机 pixel_values 为 `[8,6,3,256,256]`；每卡 1 high + 7 low。离线 W&B 实测：high CE=9.06017、low CE=17.08358、混合 CE=16.08066、裁剪前 grad_norm=167.66435；从基座初始一步所得，不能解释为收敛效果。

补查发现已有 `PeriodicEvaluator` 的 teacher-forcing 仍在 train mode，因而验证时会随机丢历史；验证 forward 还会污染下一次 train accuracy 累计。已以原文件 SHA256 保护追加部署 `scripts/utils/train_eval.py`/`metric.py` 和新测试：验证范围显式进入 eval mode；成功或异常均恢复各 module 原模式、训练准确率及 accumulator；额外输出真正的 high/low CE，保留旧混合 CE key 兼容。主仓库 9 项相关测试通过（含已有 sampler 4 项，不与先前 87 项直接相加）。第二轮 smoke 已启动，正式 5000 步仍待其通过。

下一轮闭环启动器须使用 v9 实际参数 `replan=8, high_cap=1024`。旧 `behavior_eval/run_memlite_first5_after_train.sh` 虽可传参数，但 manifest 的固定解释仍写 v8 视觉缺陷和 skill/primitive 不一致；不能直接复用这些过时结论，需创建版本化 v9 启动器/准确 manifest，保持同一 official public301/seed0/16actions 协议和原 timeout，不动旧评测工件。

### 正式训练已启动（02:35:14 UTC）

第二轮 smoke `behavior5_memlite_ar_v9_smoke_20260906T022948Z` 已正常 exit0；真实 MEM forward `num_frames=6, bsz=21`，high eval CE=9.47537、low eval CE=16.79880、混合 eval CE=15.88337（仅第 1 步，非收敛指标）。因此启动正式五任务 5000 步：

- run：`robo:/mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v9/behavior5_memlite_ar_v9_train_20260906T023514Z`
- launcher log：`/mnt/sdc1/robodojo/behavior_dev/memlite_v9.KWefx2/train_launch_20260906T0235.log`
- control：`/mnt/sdc1/robodojo/behavior_dev/memlite_v9_train_control.BEJ2b8`；launcher PID=2984226（只作为查证线索，不以 PID 文件证明存活/完成）。
- 只读进度工具：本地 `memlite_v9_progress.py`，可通过 SSH stdin 交给远端 G05 Python 读取离线 W&B history；不上传日志、不触碰权重。
- 新 `scripts/eval_memlite_v9.sh` 已部署主仓库，bash syntax、help、真实已保存 v9 config + 数据 SHA256 guard 通过；尚未启动模拟器。它保持旧 official protocol，但准确记录 v9 参数与限制，不覆盖旧 launcher/结果。其生成的模型/bridge/runner 源哈希用于运行时溯源。

当前仍需确认正式 optimizer 更新、监控至 step5000 正常退出，并做闭环进展评测；不能将“两轮 smoke 通过”说成训练或方法验证完成。

### 首次正式运行在 step100 验证时失败，修复验证路径

`behavior5_memlite_ar_v9_train_20260906T023514Z` 完成 100 次更新，但第一次周期验证 rank2 抛出 `ActionDecodeError: Missing AR action group markers`，launcher 已 exit1，四卡释放。无 checkpoint（首个保存点为500），保留失败运行及日志，不能声称已续训。

本人从调用链确认：`PeriodicEvaluator → rollout_and_calculate_metrics → model(..., inference_mode=True) → forward_inference` 使用普通 G05 的 CoT→Action 两阶段，而 MEM-Lite 的低层模板监督 action-only、已把 intent 放在输入中。旧路径可能把本应输出的动作消耗在 CoT 阶段，随后动作解码为空；它与实际部署的 `generate_low_level_action` 不一致。v9 的通用 inference entry 对 MEM-Lite low rows 改为显式调用同一个低层 API，保留逐样本原 intent，拒绝 high rows/空 intent；普通 G05 路径不变。验证仍是以标注意图为条件的开环 LL 检查，不是完整 HL+LL 闭环效果。

同时，验证仅捕获明确的 `ActionDecodeError`：保留已经计算的 high/low teacher-forcing CE，该 rank 整个 action batch 记为未评分，不补零，不把它当成“所有行都确实无效”。汇总记录 attempted/scored/unscored sample counts 和 failed batch counts；只对有真实输出的样本报告动作误差，CUDA/IO/其他异常继续抛出。严格 server/codec 准入规则未放松。第一轮 smoke 通过不足以暴露模型训练100步后这一协议差异，须追加真实四卡 smoke，并观察新正式训练通过第一次100步验证。

另外，v9 独立成功率汇总器已部署且17项测试通过：必须有正常 exit0、恰好一个 official JSON、正确 task/instance301/rollout0、布尔 success 与有限 q-score。缺失/崩溃/非法结果不缩小五任务分母，也不能输出有效成功率。旧汇总器/旧结果不改。

### 03:10:51 UTC：验证路径修复后，正式训练重新启动

三个追加修改文件通过原始 SHA256 guard 后增量部署，主仓库56项相关测试通过。第三轮真实四卡 smoke `behavior5_memlite_ar_v9_smoke_20260906T030517Z` 完成训练更新和正确低层 AR 路径验证，exit0；4个rank共28个LL样本均未触发解码异常（不等于28个样本通过部署动作准入，更不是模拟器成功）。各卡原始输出经实际codec解码，没有补零替代解码异常。high/low eval CE=9.47545/16.79877，仅初始一步。

新的正式run：`/mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v9/behavior5_memlite_ar_v9_train_20260906T031051Z`。
control：`/mnt/sdc1/robodojo/behavior_dev/memlite_v9_train_control.ukJ7O1`，launcher PID2994566。
log：`/mnt/sdc1/robodojo/behavior_dev/memlite_v9.KWefx2/train_launch_llroute_20260906T031049Z.log`。
不续接失败run或smoke；同一G05基座、同一已审批sidecar、同一五任务5%holdout、纯AR、四卡8样本/rank、5000更新。须实际监控通过step100 eval、step500 checkpoint及最终5000正常退出。

只读完成核验工具 `memlite_v9_verify_checkpoint.py` 可通过SSH stdin读取CPU mmap checkpoint，核验控制目录对应run、exit0、last指向、serialized step及optimizer/scheduler/action_batch_idx；不把文件名存在视为训练完成，不声称已扫描所有权重数值或验证成功率。下一轮模拟器eval root随新run更改为 `memlite_public301_behavior5_memlite_ar_v9_train_20260906T031051Z_replan8_hl1024_taskmem_actionalign0_v9causal`，目前尚未启动。

### 11:32 UTC 阶段检查：已通过100步验证，完成约3000/5000

再次核验真实launcher PID2994566及四个训练GPU PID2994589–2994592，均在运行，四卡占用约74GB。日志已推进到3012步，500/1000/1500/2000/2500/3000 checkpoint均已写出。本人以CPU mmap实际读取step3000：serialized step、scheduler.last_epoch、action_batch_idx均为3000，optimizer有623项参数状态，model state有946项，文件29,244,490,893 bytes，last已指向它。没有将其标为训练正常结束：launcher尚无exit_code。

18个关键源码/config哈希与重启时记录逐项一致；磁盘余量约1.8TB。已刷新的29轮周期验证（step100–2900）全部零ActionDecodeError，116个rank-batches共812个LL样本完成解码评分；此计数不等于完整动作准入或成功率。第100步eval high/low CE=0.38833/3.81733；第2900步=0.00005864/2.27137。最近五个eval点low CE均值2.28496，早期五点均值2.93770；后期趋于平台而非持续单调下降。变化batch、每轮仅4个HL+28个LL，不能用这些点代替全eval集或证明无过拟合。

阶段曲线由原始离线W&B记录导出，未做模拟器测试：本地 `memlite-v9-progress-step3000.json` / `.png`，绘图源 `plot_memlite_v9_progress.py`，均对应新031051训练run。后续仍需真实5000完整checkpoint+exit0和同协议五任务闭环结果；本目标保持active。

### 用户扩展目标：本轮评测后还要改进并实际继续训练

新的active objective是：监控至本轮训练结束，评测success rate有无长进，深入研究新暴露的问题，实施改进；必要时重训，否则用更有效的方法接着训练。**不能再以“v9评测完成”作为整个目标完成条件。** 本轮训练条件保持固定，先取得闭环证据，再决定下一轮的数据、模型/推理修复、采样和训练策略；不预先为了让指标漂亮而改变当前评测协议。分析/实现仍由本人完成，不使用subagents，不启动已取消的非MEM基线。

14:27 UTC真实状态：同一launcher和四GPU训练进程仍存活，stdout4084/5000；4000checkpoint已落盘且last已切换；最近4000步periodic eval混合/low/high CE=1.784745/2.038657/0.007359，28个LL样本零解码异常。尚未有完整5000 checkpoint或官方success结果。下一步仍是等待该run正常结束，再使用v9 launcher做同official public301/rollout0条件的五任务比较。后续改进必须针对这次实际的失败证据，完成实际新训或续训及相应验证；不能仅写出计划就标记目标完成。

### 为本轮闭环证据分析准备的只读工具

新增主仓库 `scripts/analyze_memlite_rollout_traces.py` 及7项测试，先通过staging和真实旧v8日志校验，再以新文件不存在检查/`git apply --check`增量部署，主仓库7项测试通过。它不进入训练或策略代码，不重算/查询仿真隐藏状态，只读已有JSONL并生成独立分析报告。

工具按明确的hold_chunk/low_level_chunk归类实际送出的动作，避免把hold时携带的`Task complete`文本误算为“用terminal intent生成动作”；proposal与accepted normalization分别累计，避免同一动作两份trace被双算；记录拒绝原因、动作目标极值及原记录定位、intent/status切换、关键帧索引、官方goal_status变化和每次DONE之后第一条env_step的真实success。保留缺失/破损trace警告，不把没有记录解释为没有问题。目标命令≠实测运动/抓取成功，DONE下一步未确认仅作为人工复核候选，goal predicate数量≠官方q-score。

只读旧v8完整日志核验结果：五task各有1个policy trace，official_action数分别3225/7902/20683/20545/17771，与sim env_step逐task完全一致，trace无解析错误；旧run有38195条`Task complete`条件下真正生成的动作（比早期人工统计的“至少37528条”更完整），Halloween左臂命令目标极值1959.840576再次可定位。不是重新跑基线，也不是本轮v9的结果。旧日志分析输出位于stage的`past_v8_trace_analysis.json`，新v9完成后另存新报告，并由本人结合关键帧核查，不能仅凭自动计数下因果结论。

### 16:59 UTC：5000 步训练正常完成；17:05 UTC 启动闭环评测

同一 031051 run 已完成 5000 次更新，四 rank 和 launcher 正常退出，control/exit_code.txt=0。本人实际 CPU mmap 读取 step_5000.pt（29,244,490,893 bytes）：serialized step、scheduler.last_epoch、action_batch_idx 均为5000，946个model state项、623个optimizer state项，last.pt正确提交到该文件。18项训练关键源码哈希均与启动时一致，四GPU已释放。核验不是全部tensor的数值扫描，也不代表模拟器成功。

完整离线W&B导出在本地 `memlite-v9-progress-step5000.json`。最后一次稀疏periodic eval（step5000，4个HL+28个LL）混合CE=1.945466、low CE=2.223387、high CE=0.00002105，4个rank-batch零解码失败、28个LL均评分；仍是变化的小批量评估，不是完整eval集或闭环成功率。

17:05:23 UTC 已启动主仓库 `scripts/eval_memlite_v9.sh`，launcher PID3102539，独立日志 `behavior_dev/memlite_v9.KWefx2/eval_launch_031051_ckpt5000_attempt1.log`。root为 `behavior_eval/memlite_public301_behavior5_memlite_ar_v9_train_20260906T031051Z_replan8_hl1024_taskmem_actionalign0_v9causal`。空记忆初始化和过短高层生成上限已在当前版本解决：官方task ID初始化 `Task=id; Completed=none.`，HL cap1024；使用该最终权重与此前声明的固定协议，不启动基线。此时仅验证了启动器/资源门，尚待策略健康、实际动作、全部official结果及本人失败分析；不能提前报告成功率。

### 18:00 UTC：闭环仍在运行；本人新增动作链路诊断

四策略服务17:06:56健康就绪，四路模拟器先后完成初始化并实际发送动作；trash依既定协议等待radio结束后复用GPU3。18:00时radio约1623步、Halloween1707、plates1632、can_meat1386；无任务正常结束，不能报告success rate。只读进度脚本为本地 `memlite_v9_eval_progress.py`（SSH stdin），有界tail缺少某event不等于从未发生。

已保存两个已写入日志快照 `memlite-v9-trace-live-1718.json` / `memlite-v9-trace-live-1739.json`；不是最终结果。本人实际看过初始/128动作后的三task头部视图、can_meat初始视图（7张），以及三task768动作后的头部视图、can_meat640动作后的头部视图、radio左右腕部图（6张），本地 `memlite-v9-initial-images`。图像显示反复改变视角，但尚未看到有效抓取/开门。17:39完整已写入trace窗口的四task均只有初始intent、CONTINUE，无goal_status变化、夹爪目标切换、proposal_rejected、缺失action group；不是提前判定最终失败，更不能单独归因MEM高层。

CPU隔离重建检查：本地 `audit_memlite_action_reconstruction.py`，远端stage保留每个版本。v1的覆盖点选择误用了action下标读取state，结果不得作为覆盖/运动量证据；发现后v2起按真实shape_meta分别取action/state下标。真实action为23D，state为61D；右臂state是28:35，不能根据action位置猜。v2/v3/v4均为15个demo（每task3个）、105个起点/大动作/中点/夹爪切换窗口，不是随机总体指标。只运行processor与ActionCodec，CUDA_VISIBLE_DEVICES为空、CPU两线程，没有加载VLM、改变模拟器或训练。

当前最完整工件为 `memlite-v9-action-reconstruction-v4.json`，远端 `behavior_dev/memlite_v9.KWefx2/action_reconstruction_audit_20260907_v4.json`。结果：全部105窗口无缺失group、无部署动作准入拒绝、两夹爪重建误差严格0。动作编码不是把所有机械臂动作压成零：可重建约2 rad的真实姿态。然而还发现三类不同误差，必须区分：

1. 原有action forward统一clip±5使部分大增量有损；不可简单在推理端改尺度、假装恢复丢失信息。
2. v9使用抽样train统计min/max作为inverse硬边界，会剪掉部分真实示范尾部。例ep950/frame1907左臂max误差0.792237；CPU仅关闭inverse边界后降至0.00000161。尚未改运行边界，不把取消所有安全保护当修复。
3. **确定的后处理结构缺陷：从有损归一化proprio恢复原始关节位置作为relative-action anchor。** ep875/frame1163右臂第二关节原值−2.75474286，state往返变成−1.41068184，误差1.34406102；关闭action inverse边界也完全不消失。ep800/frame1393同类误差1.16791713。应在serving保留真实观测原始state，用其作为恢复绝对动作的anchor，模型输入及已训练action表示不变。这两例来自真实demo的CPU路径，不等于证明它是本轮初始导航停滞的原因。

已排除一个源码注释造成的假线索：本次saved config的vlm_input_action_norm_default_mode=null；实际AR监督slot逐窗口断言等于sample[action]，没有“AR学原始delta但被误当z-score多反归一化”这一配置错位。

下一步在独立candidate中准备raw-state anchor修复和回归，不能改正在跑的主repo策略源码/中途变评测条件。最终成功率、全部失败分析仍待本轮五任务正式结束；后续训练策略也未决定。

### 18:25 UTC：隔离 raw-state anchor 修复已验证，未部署到运行中的主仓库

候选源码位于 `robo:/mnt/sdc1/robodojo/behavior_dev/memlite_anchor_fix.GbAIUp`，本地变更文件及测试副本为 `memlite-anchor-fix-20260907.WGKN4Q/`。不是主仓库，也不是第二轮正在运行的评测。四个修改点为 `inferencer.py`、`base_processor.py`、`relative_action.py`、`serve_policy_mem.py`：预处理之前克隆真实最后一帧左右臂和躯干状态，只在动作后处理用于恢复绝对目标；私有锚点不进入模型、训练样本或动作协议。保留既有 action 表示、归一化、codec、动作边界和纯 AR 路径。缺失/非法锚点报错，不静默错配 batch。服务端单独记录锚点误差诊断，随后从动作缓存/回包中移除元数据。

本人新增的16项测试及58项既有相关测试，共74项通过；之前的72项是同一套测试的较早版本，不能相加。覆盖真实数值回归、最后观测与预处理就地修改、不同 batch 的锚点身份、错误 shape/NaN/缺失值拒绝、其他 embodiment 的旧行为、AR/FM/legacy 后处理分支、MEM 低层多样本路由和服务缓存不泄露元数据。测试不等于真实 VLM 或模拟器效果验证。

CPU真实示范复核使用实际候选 `PolicyInferencer._prepare` 捕获锚点，逐窗口断言模型输入和 action target 没有被更改，并验证 import 确实来自候选目录。15个episode的同一105个覆盖窗口，全部六组动作齐全、准入通过、夹爪指令重建误差为0。ep800/frame1393右臂不经codec的最大重建误差从1.16791713降至0；ep875/frame1163从1.34406102降至7.45e-9。加上实际codec之后，这两例最大误差分别0.00670969与0.03291583，执行前16动作的MAE分别0.00055308与0.00141016。它修复了已复现的后处理错误，不代表解决了原始 action clip、抽样 min/max 边界或闭环导航停滞。

完整证据为本地 `memlite-v9-action-reconstruction-raw-anchor.json`，远端候选 `action_reconstruction_with_raw_anchor.json`；审计脚本版本为 stage 中 `audit_memlite_action_reconstruction_v5.py`，带 `--use-raw-anchor`。尚未加载候选真实 VLM，也未部署到主仓库/本轮评测，不需要为这一单纯后处理修复重训权重。

18:24:59 UTC只读评测状态：radio2371、Halloween2734、plates2457、can_meat2134，trash仍待radio完成；正式完成0/5，runtime crash0，success_rate=null而不是0%。四路仍然停留在初始 intent/CONTINUE；日志有界tail没有拒绝/hold/bridge异常，不能据此宣称全程绝无这些事件。继续保持本轮权重、代码、原始官方timeout，不中途替换候选。

### 18:40–18:46 UTC：补充人工视觉复核及评测性能检查

新完整已写入trace快照 `memlite-v9-trace-live-1836.json` 记录radio2708、Halloween3184、plates2832、can2473个sim步骤。四task仍仅初始intent/CONTINUE，分别22/25/23/20个HL事件，无proposal rejection。can右夹爪目标有2次变化，不代表抓取。plates的初始已满足predicate0在2170/2225步短暂变false、下一步恢复；没有新达成的top-level goal。不能继续引用早期17:39的“全程没有夹爪切换/goal变化”来概括后续运行。

本人新查看12张原始camera图（四task最近已保存时刻各3相机），记录和原图在 `memlite-v9-late-images-1836/`。radio的2689步头部视角面向厨房，腕部主要机身/地板；Halloween3073步见地面装饰物/cauldron形物，腕部机身/地板；plates2817步见两份pizza及两bowl仍在桌上、冰箱关闭；can2433步head朝天花板、腕部也强烈倾斜，疑似姿态失稳或倒地，但缺少实测base orientation，不能从图片确定物理原因。高层仍重复初始开柜门。这12张与早期已看的13张互不重复，本轮共已亲看25张原始camera图；复制但没有打开的其他图片不计。

动作锚点修复已导出可增量应用的 `memlite-anchor-fix-20260907.WGKN4Q/raw-state-anchor.patch`（4实现+1测试文件）；在主仓库 `git apply --check` 通过、候选 `git apply --reverse --check` 通过，均仅检查没有应用。主仓库18关键源码hash于18:29左右再次全部匹配。

完整50轮稀疏periodic eval（step100–5000）共200个rank-batch、1400个LL样本，decode_failed_batches总和0、scored_samples1400；最后五点LL CE均值2.101175、范围1.908567–2.241785。不是全eval集、不是成功率、不能证明没有过拟合。

资源只读检查：服务器96在线CPU，vmstat连续采样user97–98%、idle1–2%、run queue410–437；四OG进程NLWP约620/623/623/684，总CPU平均约88核，GPU快照0/0/0/5%。说明当前慢主要在模拟器CPU侧，但仅这些指标不能证明具体哪类线程是瓶颈。没有修改运行中进程的affinity、线程配置、物理设置、权重或源代码。下一轮才单独验证线程设置/评测吞吐，不将未经基准验证的调参宣称为已加速。

### 18:49 UTC：原始采样器的5000步安排已重放核验

本地 `audit_memlite_v9_training_allocation.py`、结果 `memlite-v9-training-allocation.json` / `.md`；远端stage保留同脚本及 `training_allocation_replay_20260907.json`。CPU两线程、无模型/图像/模拟器加载，使用实际源码sidecar branch index、实际元数据及task-stratified split方法、实际源码sampler，saved seed7、4 ranks、batch8、每rank 1HL+7LL，重放epoch0前5000 batches；Dataset总长度8,898,502与训练日志一致。

五task（radio/trash/Halloween/plates/can）安排的LL样本分别6138/14359/42113/41542/35848，占4.38%/10.26%/30.08%/29.67%/25.61%；HL分别972/2155/5984/5817/5072。LL合计140000、HL20000，所有安排的index均唯一。有效LL pool8,612,874，故安排覆盖约1.625%，不能仅凭这一比例判定欠拟合。高层pool71232。以上是**采样器安排**，不是逐条训练回执，Dataset异常重采样可能改变实际送入GPU的样本；没有据此声称训练丢数据或某任务失败的因果关系。

结论限于采样设计：branch均衡不等于task均衡，长示范任务分到更多样本，而当前诊断SR对五task各给一个rollout。因此任务均衡/混合任务与帧均衡是下一轮合理候选，目前尚未修改sampler、重训或验证收益。推理锚点候选与本轮运行继续保持分离。

### 18:53:45 UTC：radio正式失败；trash已自动接续

收音机 official public301/rollout0 在3225步正常超时结束，JSON bool success=false、q_score.final=0、exit_code=0；本人读取原始JSON及最后sim记录核对。原始结果本地副本 `memlite-v9-radio-final.json`。与旧v8同任务0/1相比，本任务成功结果没有提升。不能据单task推断全部五task结果，当前总success_rate仍null。

完整radio trace分析 `memlite-v9-trace-radio-final-1854.json`：sim和policy均3225条实际动作，26次HL全部CONTINUE并只用最初pickup intent，202个LL动作块，零proposal rejection、零非法动作、零夹爪目标切换，没有DONE或hold替代动作。radio-on官方谓词从未满足。202个32步proposal的左右臂最大相对增量分别0.00649015/0.00697158 rad；这些是模型动作目标增量，不是实测关节移动。3225条官方输入action中，yaw方向指令全部为正（min0.010386/max0.300000/mean0.287432），其中3223条>0.05，支持持续同方向旋转指令的退化模式。不得把速度目标积分当实际机器人转角。

本人额外打开radio3201时刻三相机原图：head见红色radio仍在茶几上，腕部机身/地板；目前共28张亲看图，记录在 `memlite-v9-late-images-1836/README.md`。终态SR按官方JSON，不按图片判分。trash在18:53:45自动开始，19:03已执行149动作，表明不是只启动了空进程。

### 19:14 UTC：追加小规模同权重离线LL诊断，非基线/非新官方rollout

为区分低层本身未学会与闭环偏离示范，新增本地 `probe_memlite_v9_low_level.py`。独立目录 `robo:/mnt/sdc1/robodojo/behavior_dev/memlite_ll_probe.phvVYF`，从五个holdout episode190/390/590/790/990各选起步、大动作、较大姿态、中点及夹爪切换附近，共25个合法LL窗口；并非随机全集指标。

CPU准备已完成：使用真实eval Dataset/processor，禁止异常后随机替换样本，逐样本核对三相机各6帧、32动作无padding，归一化action target和最后proprio与独立raw parquet构造逐值匹配。初版诊断断言混淆了eval视图local idx与Dataset返回的原始global idx（如ep190/frame1是local1、global413192）；修正的是诊断脚本的比较坐标，不是发现训练取帧错误。随后25样本全部通过。manifest本地 `memlite-v9-ll-probe-manifest.json`，远端每sample独立CPU tensor及manifest。没有修改数据、标注或主仓库。

19:14启动独立推理进程，使用同一v9 step5000、原有AR低层API、GT intent条件、不调用高层、无optimizer或模拟器。物理GPU3启动前约55GB空闲，PyTorch allocator限制25%单卡显存，CPU两线程/nice10；不触碰现有policy连接或它的episode/KV状态。只生成25个离线动作并算teacher-forced低层CE与动作误差，完成后释放资源。它与官方评测共享GPU的空余资源，因此会短暂影响墙钟吞吐，但不改变任务、seed、动作频率、timeout、权重或模型源码；不能把它当新官方SR或被取消的非MEM基线。此时仅已启动，结果尚未确认。
### 19:41 UTC：25 个真实低层诊断全部完成，正式评测仍 1/5

### 20:32 UTC续记：只准备下一阶段候选，当前评测仍保持原条件

20:54补充：v10候选配置已实际Hydra组合验证，arch/processor/tokenizer解析后与v9完全相同。raw-anchor修复已在`behavior_dev/memlite_train_strategy.IofJdZ`联合candidate中整合，本地副本同步；141项联合回归通过。联合105窗口真实重建与原独立anchor报告逐record完全一致，缺组/拒绝/夹爪重建错误均0。874行/9文件`combined-v10-candidate.patch`对主repo apply-check及候选reverse-check通过，均只是检查，主repo18源码hash仍全匹配。主仓库未部署、无新训练/新模拟器。后续需等当前五task正式结束后再走部署和四卡smoke准入；现有训练launcher仍是旧v9脚本，不存在已启动的v10run。监控cell533已经自然结束并关闭，最近单次正式进度20:44 radio3225失败、trash2960/Halloween8718/plates6821/can6236 pending。

最新独立训练策略及证据见 `MEMLITE_V10_CANDIDATE.md`，不是已经部署/开训。真实950 train episode统计及旧sampler回放发现140000LL中夹爪切换2669（1.91%），task0仅6138LL/134切换，task3为41542/517；机械臂活动窗口并不少。已实现任务均衡+每卡1HL/1切换LL/6其他LL的opt-in sampler，候选91项相关回归通过，真实Dataset四rank×5000批回放LL全部唯一，15条真实样本索引/分支/相机形状对上。原标签、图像结构、损失权重与主仓库未变；无新训练。

新增正式trace已写入窗口分析 `memlite-v9-trace-live-2018.json`：radio3225、trash2273、Halloween7537、plates5973、can5474；依旧仅radio正式超时失败。全窗口初始intent/CONTINUE、高层26/18/59/47/43，无拒绝/非法动作，can右夹爪4次命令变化。各task没有新满足的顶层目标；Halloween原predicate3在4524变false，plates predicate0多次false/true，不等于q-score或自动“退步”。读取任务bddl第四项为全部柜门关闭，但仍需确认runtime索引映射；开柜门也可能是必要中间过程。

本人又看过15张未改动的原始bridge相机图（trash初始和2177各3，其余task2/3/4为7425/5889/5377各3），见 `memlite-v9-late-images-2018/README.md`；当前个人正式闭环视觉复核总43张。多路仍无已确认收纳/操作，can贴近柜体且姿态视角明显倾斜，不能仅凭图片宣布跌倒。当前read-only监控cell533计划20次45秒间隔，约20:33 UTC自然结束；不要把它与正式launcher混淆。

后续同25窗口的实际 codec + InputPreprocessor.encode_train CPU审计 exit0：左右夹爪全部32步无损；train/eval两模式50次完整动作目标逐token与codec一致，没有夹爪mask遗漏。每行63监督token，只有4个夹爪payload（6.35%），三组连续动作各16payload；该计数可解释平均CE对稀有关键错误不敏感，但不是调权成功的证明。报告 `memlite-v9-probe-supervision-audit.json`。主仓库仍未修改。

独立 held-out 低层探针 v3 已 exit0，25/25 实际生成、无缺失 group，报告 `memlite-v9-ll-probe-results.json` / `memlite-v9-ll-probe-report.md`。本人已逐行阅读。首次尝试仅因 fused CE 的 hidden/head dtype 不匹配退出；v3 在独立进程中以现有 serving 权重数值的 fp32 线性 CE 进行诊断，AR 生成前恢复 helper，不改变正式服务、训练或参数。此 CE 不是训练时 fp32 权重的逐位复算。峰值 CUDA reserved 12,585,009,152 bytes，19:41 核验进程释放显存，四正式策略和 launcher 均存活。

GT intent + 真实示范观测下，ep190/f1105 右臂前16步 target 最大增量 0.739576 rad vs pred 0.005937；ep990/f10926 为0.987871 vs0.003293。ep190/f915 右夹爪15/16指令错误但 CE1.380898。也存在较大/过冲输出，不能说模型全部输出零。五个首次夹爪转换窗口均8/16步不匹配；继续核实这些实际窗口的 codec 重建与训练监督 mask，避免误把编码分辨率当模型学习问题。不能仅归咎高层，但未辨识唯一原因。

正式评测19:41:14 UTC：radio已3225步正常超时失败；其余trash1230、Halloween5941、plates4852、can4451仍运行。总completed1/5，runtime crash0，success_rate=null。代码、权重、原timeout保持不变。当前只进行隔离分析/后续候选准备，未部署 raw-anchor 候选、未启动新训练。
