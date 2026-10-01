# P107：10.7 负例/恢复数据准备的实施与验收台账

2026-10-01 21:53（北京时间）。协调任务 `P107-COORD`；active-goal owner 是 Codex。**父代理是唯一的最终验收者和分层视觉抽样审查者。** 本文把[训练设计 §10.7](2026-09-29-memlite-gradient-training-design.md#107-阶段1未结束时可以并行准备什么2026-10-01)转为可执行台账；不表示数据已构造、质量已通过或阶段3已运行。

### 2026-10-02 00:09:33：P107-LC3-CAL40 仅进入受限 setup/run，未完成 decode

`prep_contract_review` 是唯一 operator。renderer `a068404` 已 final-review clear（10/10 focused tests及actual-40 local create/resume），并在 lc3 reachable/idle、目标输出缺失的状态开始冻结 Git deployment/decode setup。输入固定为 full-v3、protocol snapshot `0693b93c` / `7f4f…aab0`和 queue seal `78eda9c87b18e806e00e4820c172b02002d806e7c77fe9911368d42711de227c`：40条 `annotation_calibration`、每条10个请求 slot、392个 distinct episode-local frame，最多计划1,176张 camera-native PNG及40张仅review contact sheet。

远端仅写`/data/workspace/wsy/behavior2026/p107/runs/calibration40-temporal-v1`，本地目标仅为`/home/wsy/behavior-annotations/p107/calibration40-temporal-v1`；setup≤30分钟、decode≤30分钟、1 CPU/`nice 19`/RLIMIT 4 GiB、output≤2 GiB、无GPU、不改shared/active env或其他job。seal/receipt不匹配、episode/PTS/camera绑定失败、actor/future leakage、预算/I/O影响任一即停止。此刻实际 decoded packet、native RGB、annotation、outcome/action、FM positive与release均为**0**，须等operator日志和终态receipt。MINING TOCTOU hardening已独审9 tests clear，sealed source queue未改。

## 授权边界与当前基线

用户授权完整推进 §10.7 的负例/恢复数据准备和质量工程，并明确 **DART 数据为必需交付**，不是可选讨论或只做 candidate-only 的替代品；仍须按下述全规模、来源/模型/agent provenance 和父代理视觉 QA 门推进。用户最新硬件范围为使用空闲 `lc*`，**不再进行任何 robo 工作**。授权仍只覆盖此范围，覆盖旧的“尚未获数据采集授权”表述；它**不**授权阶段2/3正式训练、长时搜索或任何新的大训练，也不允许改运行中的阶段1冻结源码、共享环境或 v4 release，或把新标签热接入阶段1。depth/raw 仍不下载；仿真特权状态不可进入部署 actor/高层。

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

## DART：必需数据流的动作语义与验收边界

DART 实际数据目前为 **0**；`prep_dart_design` 正在进行原始代码研究，尚无可发布 DART 样本。原始实现的语义是：在 `REACHED` 状态执行 `a_sampled_noise`，但 BC target 是同一观察状态下的干净教师 `a_intended`。每条 DART 记录因此必须分开、不可覆盖地保存 `a_intended`、`a_sampled_noise`、`a_executed_raw23`、对应 observation/policy clock、noise spec、`label_kind` 及 source/model/agent provenance；只允许把已验证的干净教师建议作为 expert target，不能把注入噪声、被扰动的实际动作或普通 offline-data noise 改称 DART expert。

DART 仍须满足 source-group split、可回放时间、23D→27D mapping/mask、immutable source hash、独立 publisher quality gate 和父代理分层图像/视频 QA。它与 recovery/outcome/corrective-action 视图分开计数和发布，不能以 DART candidate、合成噪声或重复窗口填充本台账的 `>=10k` / `>=1k` 真值目标；DART quality 通过也不将 formal stage3 training authorization 由 `false` 改为已授权。

### 2026-10-01：DART 合同实现已推送；严格区分原始 clean-feedback 与有界实际恢复候选

隔离 `feat/p107-dart-20261001` 已推送 `569200e74e52932952ea99d535abaf6d398a9fcb`，但实际 DART record、live sim、数据 release、positive FM mask 和 formal stage3 training 均仍为 **0 / false**。该实现只提供日后经独立审查的 adapter 所需合同；本地 CPU 的 `test_dart_noise.py` 与 `test_dart_collection.py` 已通过，当前环境未安装 pytest，因此测试保留 pytest 兼容和标准库 fallback。它没有安装环境、建立 SSH、启动 GPU/sim 或进行 robo 工作。

| 候选 view | fresh-state teacher / 实际控制语义 | 绝不能写成 |
| --- | --- | --- |
| `dart_clean_supervisor_feedback`（原始 Gaussian） | 每个已到达的当前 RGB/proprio+intent state 都重新询问 teacher；保存单步 `clean_intended23`，采样并实际执行 `requested_noisy23`/`applied23`。exact original mode 要求 applied 等于 sampled request；clean label 并非 actual execution。 | 离线 jitter、旧 demo action replay、或已执行 clean corrective action。frozen one-pass covariance 也不能冒称已完成 full iterative DART 实证。 |
| `dart_inspired_noisy_injection` 与 `dart_inspired_actual_clean_recovery` | named native23 physical parts 的 bounded/correlated/capped noise 是显式 DART-inspired；noisy requested/applied 与随后每个 reached state 重新查询、实际 applied 的 clean recovery 分开。每个 clean target 仅一动作，32-predict/从0最多执行16 的 contract 不制造跨未来 state 的32步 expert tail。 | original unbounded DART、把 injected noise 当 FM positive、或把 unexecuted clean recommendation 当 actual action。 |

每条候选需保留 fresh teacher query、当前 observation/state/clock 和 applied-action pre-state digest、teacher/source/model/runtime/noise provenance、actual action bytes 和 delayed outcome evidence；任何 future/stale receipt、abort、非有限值、超过16控制 chunk 或 budget-end 均 fail-closed/UNKNOWN。它们一律 `candidate_only=true`、`training_eligible=false`、`low_action_supervision_positive=false`、`authority_minted=false`；外部 data authority 和父代理分层视频/标签审查仍是独立门。

冻结 v4 fixture 与两条本地样本说明 source metadata 能绑定 immutable TRAIN task/task-instance，但没有 serialized simulator state、restore、controller/grasp/particle/RNG state。因此 online DART 只能从 official TRAIN task-instance 的 fresh reset 开始，并记录当前 session/runtime/asset/config/reset receipt；同状态 branch 若日后需要 restore，必须是该 live session 新捕获且可验证的 snapshot，**绝不能**以 demo video 或 actions replay 代替。

SIM `c556…` 现报31个 sim＋5个 protocol 测试通过，仍待独立复审，故 real recovery/DART release 继续 blocked。DATA temporal renderer、artifact/member semantics 修复尚在进行；mining `ebf3c70` candidate review 未批准，首40条 temporal images/annotation batch 仍为0，unknown skill-ID gap 也尚未证实为 v3 source defect。local4080 compatibility 路线仍待用户选择，禁止 robo。

独立 reviewer 对 `569200e` 的结论是 **NEEDS FIX，不是候选合同批准**：CLI 仅拒绝两个 per-record training flags，错误接受 `candidate_only=false`、`ready_for_training=true` 或 `authority_minted=true` 的已哈希 record；original profile 的 caller string 可虚写 `empirical_faithfulness=true`，实际并没有 full iterative covariance evidence；`CandidateSourceReceipt` / `CalibrationReceipt` 的 caller-supplied group strings 和 digest shape 未证明 canonical split/protected membership，也未将 covariance 绑定 teacher/learner/calibration trajectories/alpha/horizon artifacts。P2 是 raw23 named layout 还缺 immutable embodiment/layout/projection manifest SHA。下一修复一律 fail-closed：所有 record flags 精确匹配 candidate-only；经验 DART claim 固定 false；使用 DATA owner 提供的一次验证 canonical SHA-pinned membership boundary 与 explicit calibration-noise binding；加入 eval/protected spoof、covariance-mismatch 和 flag adversarial tests。新 DART 文件之后才移植到 SIM consumer `56e33bd`（在`c556`上、仍待独立复审），禁止带回旧 bridge；其 future postcondition 必须有结束于/晚于 actual raw23 end 的 physical-evidence observation，不能以 evidence availability、clean action receipt 或 pre-action proof 单独生成 corrective-positive；仍0 live data/release/training。

## 阶段、责任、预算与停止条件

| 阶段 | 唯一责任与交付 | 初始预算/停止条件 | 放行条件 |
| --- | --- | --- | --- |
| A：infra只读 | infra 负责人核 lc1–lc3 连通性、挂载、阶段1占用 | 无作业；连接故障即记录并停止 | 仅在获准恢复连接后只读复核；不将9/30快照当当前 |
| B：本地schema/index | schema owner 核 v4 metadata、映射、split/hash、23D→27D、clock/mask | 不读 RGB；只产生小型代码/测试证据 | 映射、空值、sibling-split 反例通过 |
| C：lc*/local sim compatibility（禁止 robo） | sim owner 仅核官方可兼容路线、状态日志/物理事实接口合同 | 未获用户在 local4080+lc3 与 lc-only defer 间的明确选择前，0安装、0GPU启动、0物理reset | 已证实官方路径、环境/assets、RTX兼容性与时钟合同；否则记录缺口并停止 |
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

## 2026-10-01 独立审查纠正：协议/仿真交付暂停

独立审查在 exact data `a8e14f`（protocol `b697097bd57bf94ab6f6c7e1e3e7bdb2b78f2480`）和 sim `8968327` 上发现8项阻断，尽管已交付测试声称23个唯一测试/28次执行均通过。此前22:18记录中的“fake/模拟回执不得训练”或“正例 fail-closed”只是未验证的设计意图，现已在这些 SHA 上被**证伪**；旧记录保留为历史，不能再当工程事实。

- **HIGH 1：** `eval`、未审 `PROPOSED` 与 `recovery_verified` bool 可接受 `mask=true`。
- **HIGH 2：** 嵌套 privilege 引用可泄漏 `object_pose`。
- **HIGH 3：** 同 group 的不同 episode/event receipt 可被接受。
- **HIGH 4：** 将 fake backend 改名为 live 即可通过并产生正例。
- **HIGH 5：** future evidence `999` 可在 final clock `2` 得到认证。
- **HIGH 6：** 正常的两条分支均 SUCCESS 仍可称 recovery，未要求初始偏离的肯定证据。
- **MED 7：** 接受重复 `episode_index`。
- **MED 8：** 接受 `resume files={}`。

因此所有代码集成和数据 release **PAUSED**，等待修复和独立复审；尚未采集任何实际 recovery-data positive，亦无已知 live bug impact。data owner 负责1/6/7/8及coverage，sim owner负责2/3/4/5，packager已获知。真实 recovery proof 必须先验证**初始偏离**，不能只以 baseline fail 代替；baseline也可能恢复，且“发生偏离”的证明与“恢复有效性”是两项独立判据。

### 2026-10-01 22:41：sim `88563b5` 复审更正—仍不可 release

本次复审确认原 receipt cross-episode 问题已修，基本 fake/future 测试也在该 SHA fail-closed；但这不能外推为 release 已安全。22个 sim 和5个 protocol 测试通过也不是证明。仍有4个新证实孔洞：

1. **Critical：** 自构 `LiveReadinessAttestation` 加普通 `NotOmniGibson` object 可成为 live positive，roundtrip hash 尚未解决。
2. **Critical：** snapshot backend A 与 action backend B 在 branch-start clocks `0/1` 时仍可接受 `same_state=true`；同 ID/session 的跨-wrapper capture 也可绕过绑定。
3. **High：** 在 restore snapshot `0` 前、CURRENT clock `7` 评估 fault 仍可生成 positive。
4. **High：** `evidence_kind=TIMEOUT` 可被接受为 physical success/fault。

sim owner 正修复：无外部 candidate 的 trusted receipt registry、所有 provider 的 runtime-capabilities binding、restore-before-fault-observation 以及封闭的 physical-evidence taxonomy。实际 OmniGibson proof 仍为零，release 继续 **BLOCKED**，上述 blocker 不得标 resolved。

### 2026-10-01 22:49：data `8a683fe` 复审—20测试通过仍有6个发布阻断

`8a683fe` 的20个复审测试通过，但不构成 release 证明。以下6项仍使真实 data release **BLOCKED**：

1. 可自行 mint 为 `unboundexternal` 的 authority；
2. 可变 map；
3. action 与 artifact bytes 尚未解析绑定；
4. pre-action evidence 可验证后续 action；
5. future typed actor reference；
6. decoded PNG 与 input index 没有 sealed digest。

`af0e43d` 已解决 coverage Cartesian 问题；早先的 eval-calibration、nested-key、duplicate-episode 和 resume-file-inventory 问题亦已修复，但均不解除上述 release gate。架构职责现固定为：raw collector 只能产生 `CANDIDATE_ONLY`，无权设置任何 positive mask；经独立审查的 publisher 是唯一 dataset-quality authority。dataset quality 与正式 stage3 training authorization 严格分离，后者仍为 `false`。

### 2026-10-01 23:06：全量诊断索引、DART/仿真范围与最新 authority 复审

冻结`af0e43d`的本地 full diagnostic index 已在`/home/wsy/behavior-worktrees/p107-index-run/index-validation/full-diagnostic-v1`完成：1 CPU、32.03秒、1.018GiB。raw/published/groups 各有20,000个唯一 source ID；19,889个 source candidate-bearing、403,257个唯一 event、覆盖全部100 tasks。19,000 student groups/1,000 eval groups 是**原始分组**，不是 eligible 结论；eligible 仍为18,895 TRAIN、994 eval、111 quarantined。它 accepted-all=0 且 `training=false`，并由 seal `3e2371a57d4e5b6685630104779ea7e6cdb280ba216a677f4bf01f2ce3948bf5`、manifest `469eef7b11ca2f7747d09565908707cd8d5d6c91045e43305db4fc58a97ce4cc` 绑定，不能据此发布数据。

该诊断发现官方35个 lowercase description 与现有 uppercase verb 词表有语义错配；已提出35/35映射至既有 Stage1 `SKILL_MAPPING` 的`skill-vocabulary-map.v1.json`，`prep_queue_review`正独立审计。mining 修复`8293c90`自有4测试通过，复审仍在运行，尚未产生实际队列。

仿真兼容性以 NVIDIA Isaac 5.1 的无 RTX GPU 不受支持、BEHAVIOR 要求 RTX 为基线：空闲 lc3 A800 没有已验证的官方仿真路线。用户已异步被询问是否允许 **local 4080 sim + lc3 expert/data（无 robo）**，或只使用 lc 节点并暂缓实际仿真；答复前不得安装或启动 GPU。raw SIM `df9c1fa` 仍严格 candidate-only、无 positive authority；其 same-runtime/ref-clock/physical-enum 的26个 sim +5个 protocol 测试只是自测，独立最终审查仍 pending。实际 DART、live data 与 DART/recovery release 全为0/NOT DONE。

最新 data authority review `544b010` 仍报告 mutable capability、artifact bytes、roles、self-sealed packet/resume、index-artifact proof 的未决门；较早 checkpoint 所列某些关闭项不得外推为全部修复。owner 正修复；packager 已适配新的 external root，所在分支提交仍 pending。

### 2026-10-01 23:21：v3 candidate index取代旧诊断作当前metadata证据；三项data和三项sim复审门仍阻断release

`/home/wsy/behavior-annotations/p107/index-validation/full-v3-candidate-index` 已完成metadata-only v3 candidate index，inventory seal为`7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479`：20,000个source episodes、403,257 events、19,889个candidate-bearing source episodes；100个tasks及35个官方`skill_id`+`skill_description`均出现，missing=0。输出为**1.018GiB**（不是MB）、30.16秒；usage roles为18,884 `student_candidate`、1,000 `evaluation_only`、116 `annotation_calibration`，后者精确为16个既有pilot group与每task一组100的预留并集。accepted labels=0、所有`training_eligible=false`、仅metadata；这确认官方35项映射合同，不宣称每个task都要求或观测35个skill pair，更不构成release或训练准入。

mining的已知问题在修复中。`0..39`只完成locator preflight；首40条temporal camera-native images**尚未提取**，所以不存在可称为diverse annotation batch的实际标注。DART隔离实现worktree已开始，但实际DART record仍为0；DART的clean teacher与noise/actual-action分离、source/model/agent provenance和父审要求不变。

SIM `df9c1fa`独审的剩余三项为false-genuine判别、actor input blacklist与stale validator，sim owner正在修复。独立DATA seam review在冻结protocol `89f0462`与renderer `4e2631f`上确认的剩余三项是实际sealed-index membership、typed artifact的role-specific contents以及packet semantic resume；data owner正在修复。上述复审结论覆盖任何较早“测试通过”表述：不得发布real recovery data、不得导出positive FM mask、不得启动stage3。

23:21新鲜infra observer结论为本地SOCKS未监听且lc3不可达。此前用户许可下唯一进行的reconnect由`prep_contract_review` sole operator执行，状态为RUNNING；未取得新鲜receipt前，禁止写成已连接或据此启动任何远端任务。local4080 sim+lc3 expert/data与lc-only defer的用户选择保持pending；当前仍不使用robo，不改远端/共享环境，不启动GPU sim或formal training。

### 2026-10-01 23:48：首个有界 calibration40 queue 完成；仅 metadata 候选，不能替代 temporal 标注或恢复数据

MINING `33e1a5b` 经独立复审后在`/home/wsy/behavior-annotations/p107/index-validation/calibration40-v1`完成：固定 full-v3 inventory seal `7ed1b2c5cbb50c8af042a2b9dca8529a6d133de5fc0fbe224601854235c31479`，create 18.14 秒 / 1 CPU / peak RSS 125,352 KiB / exit 0，queue seal `78eda9c87b18e806e00e4820c172b02002d806e7c77fe9911368d42711de227c`。它有40条 calibration、0条 student；40个 distinct task / source group / episode，32/35 official skills。103不在 calibration pool；8、100在pool但没被固定 policy cap 选中，不能误报为全源缺失或不可能覆盖。create 后的 sealed resume 也 exit 0（17.61 秒、123,956 KiB）。

队列 policy 为 candidate budget 0、calibration budget 40、每 source group 1、每 episode 2；每项仍是`CANDIDATE_MISSING_EVIDENCE`，整个 queue `training_eligible=false`、accepted labels=0。40 条 temporal requests 尚未提取 camera-native images，尚无人工/agent temporal annotation、outcome truth、verified recovery 或 corrective action；因此不计入本文件的10k/1k目标，也不触发任何训练数据发布。

此前 protocol artifact 缺失的 review claim 已更正：commit `0693b93c29cf92e831b3cbc2edec40b69f4877e9` 的 immutable snapshot 在`/home/wsy/behavior-annotations/p107/protocol-snapshots/0693b93/src/g05/data/memlite_event_protocol.py`，SHA `7f4f9fbf18fb4ba6ec97a304f0d787ebadb4f84c7184dd041c99027ad84aab0c`，与 full-v3 manifest 完全一致；无 index corruption、无重封或重建。SIM `56f7452`已独立复审为 candidate code approved（36 recovery + 5 protocol tests），但 live OmniGibson/physical proof仍为0；DATA renderer `e112` 的 creation nested-context、resume identity/PTS/manifest、distinct timestamp 问题和 DART `569200e` fail-closed fixes 均仍在处理，未合入/未 release。

## 当前状态、阻塞与交接

2026-10-01 22:49（北京时间）为避免不相关的协议代码门阻碍有界的实际候选准备，infra owner 将在独立、干净且冻结于`af0e43d`的`/home/wsy/behavior-worktrees/p107-index-run`运行**本地 candidate-only** ≤200 pilot：1 CPU、≤30分钟、≤4GiB RAM、≤2GiB输出；必须以官方全局100-task/35-skill词表为合同，对无对应 task-skill pair 显式写 `null`，不得伪造覆盖。它不发布数据、不能设置正例或越过`8a683fe` gate；data owner 独立继续 protocol 与 PNG seal 修复。该运行已由上方23:06全量诊断索引取代为已完成状态；`p107-mining`状态同上，尚未生成实际队列。

本地物理盘点确认 WSL2 RTX 4080 16GB、driver 610.47 存在，但已记录的 OmniGibson/Isaac 环境与 assets 均不存在。该历史 robo probe 已被用户最新的“不要再用 robo”范围撤销；不再请求其路由/endpoint，也不得任意网络重启、安装、创建隧道。离线数据准备继续，P107不是全局 blocked；仍为0 GPU训练/仿真、0数据 release。

2026-10-01 22:41（北京时间）data P1 fix/API仍在进行。candidate pack/audit 用3个实际 pilot header 作为 AUX receipt 演练，仅得到 `PASS_WITH_STAGE3_TRAINING_BLOCKED` / `CANDIDATE_ONLY`；它不是 source-resolved 的真实 release。infra owner 下一步仅可做 targeted read-only local GPU/OmniGibson availability 与 robo route probe，以穷尽 live-physics 替代路径；禁止 install、restart 或另建 tunnel。metadata index 本地范围不变，queue picker 实现必须 task-skill-first、不得按排序 N。无实际数据release、GPU sim或新训练启动。

2026-10-01 22:37（北京时间）状态覆盖：exact v4 metadata 已复制并哈希核验至`/home/wsy/behavior-annotations/p107/frozen-v4-metadata`，总403,777,381B；`episodes`为399,905,443B、SHA `c62fe885143bcdc07a9dcb302a5af294afb355db98d078a838c587f9efcc16ca`，manifest SHA `90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23`，fixture-schema SHA `b7d22723ed1b2a9a22862adb7fef5333ae7c0f07f343bf38ae3c61c70c254ae7`。这是一次49秒、7.9MiB/s、≤512MiB预算的传输；disk busy pre31%/post29.8%，没有 raw/RGB/depth/Parquet读取、远端source写入、环境或job变动。infra 最新有界tail为high12102/low20504、median6.864s/2.200s，只作运行状态记录而非新训练结论。远端认证为infra operator一次性操作，凭据不入仓库。

P1 fixes后，data owner获准执行本地 metadata index shape 与≤200 episode validation：1 CPU、≤30分钟、≤4GiB RAM、≤2GiB输出；扩展前必须先交 full-index size projection。`prep_annotation_mining`拥有隔离`/home/wsy/behavior-worktrees/p107-mining`（`feat/p107-mining-20261001`），只做queue picker/tests/metadata candidates，且要求stratified normal/retry query control、不得造 FAILED 标签。

sim blocker fix `88563b5` 自有22测试及5个protocol测试通过，独立复审**正在运行**，故八项 blocker 不得标为已解决；data fix与packager仍待。所有24条 auxiliary calibration仍非训练数据，尚无实际 recovery data 或大规模 event labels。P107继续进行，未完成。

2026-10-01 22:25（北京时间）新鲜服务器只读状态：lc1 high RUNNING step12045（atomic12000），lc2 low RUNNING step20380（atomic20000），两节点各8卡均忙；lc3 GPU idle。系统 load 0.61、可用内存996GiB、ext4 free3.91TB。v4 manifest `90ff0fa...85d6f23` 已核为18895 TRAIN/994 eval/100 tasks/35 skills，数据源 `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`。未改阶段1。

下一 infra 门仅是只读 I/O gate；只有在利用率<50%且无重写入时，才允许一次≤10分钟、≤8MiB/s、≤512MiB的 metadata 复制到本地外部 `frozen-v4-metadata`，禁止 RGB/Parquet 解码与任何远端 source 写入。该门尚未启动。P107仍进行中、非 blocked，且不因当前可执行的只读检查解除上述集成/release暂停。

2026-10-01 22:18（北京时间）状态覆盖：父代理已亲自查看全部24/24原始 pilot 图；三个 shard 修订后均接受为 **AUX visual_relation_calibration_only**，并保持 `human_reviewed=false`、`APPROVED_AFTER_REVISION`、`CANDIDATE_ONLY`，绝不称 human review 或阶段3数据。最终工件为：shard0 pilot/reviews SHA `3b98705eae688cc93ff8e6735a515fa8aab38c61ceca470495b499f5bf683ad3` / `25d259d3d8651893bc106eaacd1c907071de86d392dba2ed093fbd42d379ceed`；shard1 `1d2b53f528bf6de3b9e7dcc07baeea140c21dcd4b2816f17910e3e48a87b05b0` / `5638834a114294e5e47dd067aa1762884ddc23dfd00c0c1cf9ce2f4b3e9ceb6c`；shard2 `869bd60f2c9a4115f1a9f511e1753a2a5c3ff6885614518ba867bbeb49f030bf` / `64b5e62323c2d030aadaba30b43d6f86a513e88a3f7d90ce05773ace4de80ad9`。这些记录仍无 outcome/action 正标签、无 final-scale release，不能训练或计入规模目标。

canonical data protocol 已提交为 `b697097bd57bf94ab6f6c7e1e3e7bdb2b78f2480`，五个标准库测试通过；sim 实现为 `e368f98`，12个标准库测试、`py_compile`、CLI help 和 diff 检查通过，正接入 canonical protocol。上述只证明本地工程合同，所有真实 OmniGibson restore/物理真值仍**未验证**；fake/模拟回执不得训练。代码在独立审查结论前不合入协调分支。

新的 packager worker 拥有`/home/wsy/behavior-worktrees/p107-package`（`feat/p107-package-20261001`），负责 publisher/audit/explicit-view dataset；独立 `prep_contract_review` 正审 protocol+sim。下一批仍不得排队，直到 protocol/infra gates 通过；它必须 task/skill 分层，不能重用当前按前序行偏置的24项。P107仍进行中，未完成。

网络状态更新：VPN loopback 已恢复、lc1–lc3路由可达，但先前 SSH authentication 失败；获授权的 SSH auth handoff 正在进行，尚无新服务器状态。robo `127.0.0.1:23117` 仍 closed。无凭据写入本文，无新服务器job/env变动。

2026-10-01 22:13（北京时间）状态覆盖：用户现已明确授权“现在可以重连 ec cli”。infra owner 正仅以既有已认证 profile 重连；连接尚未确认，未启动作业、未改环境/数据，也不在文档记录任何凭据。原 PUBLIC metadata/source-version 只读 fallback 保留至连通性实核。

父代理已亲自看完24/24原始 pilot 图。修订后的 shard0 八项已接受为 **AUX visual_relation_calibration_only**，其元数据固定为 `human_reviewed=false`、`parent_root_model_review_completed=true`、`APPROVED_AFTER_REVISION`；pilot SHA `3b98705eae688cc93ff8e6735a515fa8aab38c61ceca470495b499f5bf683ad3`，reviews SHA `25d259d3d8651893bc106eaacd1c907071de86d392dba2ed093fbd42d379ceed`。它仍是 `CANDIDATE_ONLY`，等待外部 provenance/split release，且没有 recovery/outcome/action masks，不能作为恢复或训练数据。

其余最终 QA 未完成：shard1 ordinal19 将由夹爪持有误作 on-counter，ordinal7须分清地面橙色物和 bin 内不同橙色物，ordinal4的支撑关系为 uncertain；shard2 ordinal20的 on-counter 无支持证据，ordinal23混淆机器人夹爪与 jar 自身 clasp，必须删除该 query。不得扩展此单时刻 pilot；后续候选必须 task/skill 分层，不能再按排序前几行取样（当前24条只覆盖旧前5个任务）。P107仍在进行，未完成。

2026-10-01 22:08（北京时间）状态覆盖：data 与 sim 实现已分派到上述隔离 worktree，且已请求 canonical data protocol 的早期交接；本地代码/QA 可继续，但未将任何输出发布为恢复数据。三个标注 shard（0/1/2）均已返回：输入严格是266个 **LOCAL TRAIN QA** 单时刻、三相机JPEG，经 `train_ordinal % 3` 后每片前8项、总共24项。父代理已直接查看 shard0和shard2的全部16个原图，**拒绝当前pilot，待修正**；shard1的父代理视觉审仍为 PENDING。24条均未发布、不得扩大、不得用于任何 recovery/outcome/action训练或计入10k/1k目标。

已核实且必须保留 review audit 的错误为：shard0 ordinal3 将被支撑于桌面的关系误标成 `ON`，实际为桌面上方 held；ordinal12 将橙色、瓣状的地面物体与浅色、圆形、靠近夹爪的物体混同；shard2 ordinal8 将同一 bin 的三相机重复视角数成多个物理 bin；ordinal23 将悬在板上方的 jar 标成 supported on board。两名标注者正在修订原值及审查记录；`is_unambiguous/proven` 的 meta 问题也改为 UNKNOWN semantic issue。该拒绝说明本pilot至少需要 camera-native 图像、无 ground-truth footer、跨视角物体去重，以及“物理支撑关系不等于图像重叠”的明确规则；不从非随机、极小24条推导模型准确率。

这24条仅为辅助 goal/visual-relation QA，**零条 verified corrective action**。`memlite-event-recovery-v1` 的 early-handoff schema 代码尚未提交，约定隔离 `goal_satisfaction_counterfactual`、`attempt_outcome`、`recovery_decision`、`corrective_action` 四视图，sim owner 正协调；代码存在或标注返回都不构成该四视图的已发布数据。

用户请求的 `gpt-6-luna/max` 不在可用工具列表；已明确使用且只可记为 `gpt-5.6-luna/max` 的 agent 标注回退，绝不改称人工或 gpt-6。各条仍须写入 `annotation_provenance=agent` 及精确模型版本，经过父审后才可能进入后续候选门。

真实数据/提取与 restore roundtrip 在连接确认前仍阻塞：此前 lc1–lc3 本地缺 `127.0.0.1:1080` SOCKS，robo `127.0.0.1:23117` closed；上方 ec CLI 重连尚无成功回执。infra fallback owner 仅可做 PUBLIC dataset metadata availability/cache 的只读检查，缓存上限100MiB并核精确 source version；这不是数据提取或连接恢复。阶段1当前进度也未重新核验，禁止把9/30步数当作当前状态。无 GPU 标注、仿真 reset、真实 extraction、restore 或训练启动，亦无数据质量结论。

台账、总计划和团队板已通过 `git diff --check`，作为仅文档提交 `b291196` 在30秒上限内推送到 `origin/feat/memlite-recovery-prep-20261001`。实施者现使用下列干净隔离目录，均从精确 `b29119688279751f5c59f28e2da5aa4df699ce5b` 创建：

- `/home/wsy/behavior-worktrees/p107-data` → `feat/p107-data-20261001`
- `/home/wsy/behavior-worktrees/p107-sim` → `feat/p107-sim-20261001`

根工作区仅供实时计划更新和集成；不覆盖他人 dirty/untracked 文件。未完成项包括：连通性/挂载复核、100×35索引、200候选门、schema回归、restore readiness、分片标注、父代理视觉审、独立最终代码审及所有规模化质量验收。
