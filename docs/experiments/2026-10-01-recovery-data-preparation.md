# P107：10.7 负例/恢复数据准备的实施与验收台账

2026-10-01 21:53（北京时间）。协调任务 `P107-COORD`；active-goal owner 是 Codex。**父代理是唯一的最终验收者和分层视觉抽样审查者。** 本文把[训练设计 §10.7](2026-09-29-memlite-gradient-training-design.md#107-阶段1未结束时可以并行准备什么2026-10-01)转为可执行台账；不表示数据已构造、质量已通过或阶段3已运行。

## 授权边界与当前基线

用户授权完整推进 §10.7 的负例/恢复数据准备和质量工程，包括小预算 metadata、schema、候选和仿真 readiness 工作；该授权仅覆盖此范围，覆盖旧的“尚未获数据采集授权”表述。它**不**授权阶段2/3正式训练、长时搜索或任何新的大训练，也不允许改运行中的阶段1冻结源码、共享环境或 v4 release，或把新标签热接入阶段1。depth/raw 仍不下载；仿真特权状态不可进入部署 actor/高层。

现有阶段1和队友的活跃工作保留。本协调分支由 `origin/main` `33677bd` 建立，并 fast-forward 纳入阶段1前置 `3db716d`；这只是 Git 基线，不是任何数据、节点或模拟器访问结论。

## 规模、独立性与完成口径

第一交付是将所有 **eligible v4** 来源索引为 `100 tasks × 35 skills` 的可定位表。每项保留 source instance/episode、不可变 split、技能/目标/手臂或 bundle、完整 observation/action 时钟及 source hash；不能用目录存在、普通覆盖统计或标注区间结束代替索引/成功证据。

约 200 个候选只是一道早期人工筛查门，不是目标封顶。通过后向下列实际、可审计规模推进：

| 视图 | 目标 | 禁止替代 |
| --- | --- | --- |
| goal/outcome temporal windows | `>=10,000` 经审核的不同时间窗，来自 `>=1,000` 个 source episodes；100 个任务均须有可定位索引与纳入/缺失结论，并报告 distinct episode/task/skill 覆盖 | 同事件相邻帧、同窗重采样、整任务末标签回填全段 |
| genuine corrective-action windows | 另计 `>=1,000` 个经审核纠正动作窗；逐项报告 distinct source episode/task/skill/event-type/teacher-or-controller 覆盖 | 失败动作作正向 FM、正常动作改名恢复、未重执行的旧后缀 |
| TRAIN 候选 | 仅从以上原子记录派生，按 source episode/instance 固定 split | public_test、原5% holdout或任一 sibling/近重复回灌 TRAIN |

独立性主计数永远是 `source_episode`；同一 episode 的多窗只能作附属计数，不能补来源不足。若真实数据没有足量、可证实的纠正例，必须报告 task/skill/episode 缺口和停止理由，绝不伪造目标或用重复窗凑数。父代理审看原始片段和 provenance 前，任何记录不得称为 `audited`。

## 数据合同：分视图、可回放、因果正确

每个原子记录至少含：`record_id`、`source_hash`、`source_instance`、`source_episode`、`immutable_split`、`task_id`、`skill_id`、`goal_target_binding`、`arm_or_bundle_member`、`policy_version`、`schema_version`、`controller_source`、`annotation_provenance`、`policy_clock_observation`、`policy_clock_action`、`action_start`、`actual_executed_length`、`event_interval`、`evidence_kind`、`evidence_available_time`、`causal_available_time`、`attempt_id`、`parent_attempt_id`、`attempt_outcome`、`outcome_validity`、`observability`、`recovery_start`、`recovery_end`、`recovery_success`、`restore_identity`、`branch_seed`、`review_status`、`reviewer_id_or_agent_id`、`review_artifact`、`disagreement_reason`。

`label_kind` 严格分为四个不可混训视图：

1. `attempt_outcome`：结果/事实或校准；`IN_PROGRESS`、`SUCCEEDED`、`FAILED`、`UNKNOWN` 与 `valid_result_mask` 独立，缺标是 mask 而非确定 UNKNOWN。
2. `recovery_decision`：高层 `RETRY`/`REPLAN`/实际技能/记忆 CE；只准使用 `causal_available_time` 前的合法观察与历史。
3. `corrective_action`：低层 FM；只接收同一偏离状态、实际执行且经局部后置条件/稳定性/后续可行性验证的纠正动作。跨技能 32-action 窗必须切分或 mask。
4. `goal_satisfaction_counterfactual`：目标满足/部分完成判别；不冒充真实尝试结果，不可与旧动作拼成 action-SFT。

R1Pro 动作始终保存真实 23D。模型导出才补齐 27D 的 `[7,8,17,18]`，必须保存逐维/逐时刻 mask；底盘/躯干不可屏蔽。`actual_executed_length` 来自真实执行，预测但未执行的未来 32 动作不得成为 transition。

禁止把 segment end、夹爪闭合、超时、模型自报或“尚未完成”单独标成 FAILED/SUCCEEDED。记录 `evidence_kind`、`evidence_available_time`；遮挡/边界不清保留区间和 mask。仿真真值可用于标签与诊断，不能进入部署输入；未来帧/结果不倒灌给早时刻监督。

## 阶段、责任、预算与停止条件

| 阶段 | 唯一责任与交付 | 初始预算/停止条件 | 放行条件 |
| --- | --- | --- | --- |
| A：infra只读 | infra 负责人核 lc1–lc3 连通性、挂载、阶段1占用 | 无作业；连接故障即记录并停止 | 仅在获准恢复连接后只读复核；不将9/30快照当当前 |
| B：本地schema/index | schema owner 核 v4 metadata、映射、split/hash、23D→27D、clock/mask | 不读 RGB；只产生小型代码/测试证据 | 映射、空值、sibling-split 反例通过 |
| C：robo/local readiness | sim owner 核 restore、状态日志、物理事实接口 | 首次物理 readiness 至多4 resets、20分钟；不采集GPU模型数据 | restore前后观察/物体/控制器/抓持/必要粒子、时钟和后续动作一致 |
| D：metadata inventory | data pipeline worker 建 eligible episode/skill/event 定位索引 | CPU≤30min、1 worker、无RGB | 100×35覆盖/缺口表、source hash、immutable split验证通过 |
| E：候选与标注分片 | annotation workers 提取候选并保存 agent provenance/unknown | CPU≤60min、4 workers、2GiB小 sidecar、约200候选；不是最终封顶 | 父代理分层看片，裁定真实纠正/正常/未知；GPU标注需readiness与父代理ticket |
| F：扩展/发布候选 | data workers 依覆盖缺口扩至上述规模；独立 reviewer 做最终代码审 | 每批先登记source/version/时间/I/O/人工预算；到预算即评估 | 所有质量门、父审、独立代码审与source split审计通过；不等于批准阶段3训练 |

阶段 A–C 可以并行但仅限只读/有限物理准备；D–F 必须在独立 worktree，绝不改根目录或运行源。可用标注模型为 `gpt-5.6-luna/max`；其输出的 `annotation_provenance=agent`，不可称 human review 或单独决定训练准入。

## 必经质量门和父审证据

- **来源/切分：** source-group split 先冻结；5% holdout、public_test、派生 sibling 不进 TRAIN。每条记录可回溯 source hash。
- **时钟/动作：** observation/action policy clock、真实执行长度、23D→27D补齐/mask 需正反例回归；未知必须拒绝或隔离。
- **真值：** label_kind 与 truth/evidence/available-time 一致；不以 end/close/timeout/模型文本冒充物理结果；UNKNOWN 和缺失都有显式 mask。
- **同状态纠正：** 记录任务约束、剩余预算、restore identity/seed及验证证据；一次 branch 成功不推成原决策必错或普遍可恢复。
- **标注：** agent/human/provenance 分开存；遮挡、边界模糊、冲突保留，不强制确定。抽样覆盖自动接受/拒绝、成功近失误、自然恢复/恢复失败、双手和技能族。
- **父代理：** 父代理亲自完成分层图像/视频—标签对应抽样并保留 review artifact；annotation agent 不能替代。未参与同一实现的 reviewer 另查代码、manifest和split。

所有实质状态和轻量 manifest/summary 写入 `docs/plan.md`；大 sidecar、RGB、视频、权重与模型输出在受管 artifact/data 路径，用路径+SHA关联，绝不进Git。

## 当前状态、阻塞与交接

截至本台账创建时，metadata inventory、候选提取、标注、GPU标注、仿真 reset 和训练都**未启动**，没有数据质量结论。infra 报告 `ssh lc1/lc2/lc3` 均为 `exit 255: Connection closed UNKNOWN 65535`，且未见 `127.0.0.1:1080` SOCKS listener / SSH ControlMaster；此报告等待获准恢复连接后的只读复核。不得自行改 VPN、凭据或连接配置，亦不得称9/30训练为当前。robo/sim映射待核。

台账、总计划和团队板已通过 `git diff --check`，作为仅文档提交 `b291196` 在30秒上限内推送到 `origin/feat/memlite-recovery-prep-20261001`。实施者现使用下列干净隔离目录，均从精确 `b29119688279751f5c59f28e2da5aa4df699ce5b` 创建：

- `/home/wsy/behavior-worktrees/p107-data` → `feat/p107-data-20261001`
- `/home/wsy/behavior-worktrees/p107-sim` → `feat/p107-sim-20261001`

根工作区仅供实时计划更新和集成；不覆盖他人 dirty/untracked 文件。未完成项包括：连通性/挂载复核、100×35索引、200候选门、schema回归、restore readiness、分片标注、父代理视觉审、独立最终代码审及所有规模化质量验收。
