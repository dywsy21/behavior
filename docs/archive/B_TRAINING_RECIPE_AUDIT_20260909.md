# B 阶段 `high_planner_only` 训练配方核对（只读）

日期：2026-09-09。范围仅限隔离 model/runtime 配置与 R2 sidecar；未改代码、未启动 GPU 或训练。

## 已确认的模型与优化器边界

配置：

`/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/model/configs/task/r1pro_memlite_high_planner_only.yaml`

其规范为 `B / high / high_planner_only / 6`：6 帧、3 相机（18 图），高层 `coordination_v6` 采样，学习率 `2e-5`、100 step warmup、默认 5,000 steps。

`G05PolicyMEMLitePlannerOutcome.configure_coordination_trainability()` 在此 profile 下只开启：

- `model.vlm.*`
- `model.multi_modal_projector.*`
- `model.proprio_embedder.*`

它们的唯一优化器组是 `planner_vlm`。以下逐 tensor 冻结并且不进 AdamW：视觉塔、`model.action_expert.*`、`outcome_head.*`。高层 `forward_train()` 丢弃 action 输入，只走 AR CE；planner-only 分支不调用 outcome head，且 `outcome_loss_weight=0`。

这意味着 B 训练不会改变独立低层 SkillFM 的动作专家，也不会训练低层 LoRA 或 FM objective。当前 B 配方并没有高层 LoRA：它训练完整高层 VLM / projector / proprio embedding，而不是低层 VLM adapter。

## CE 的实际字段和权重

模板 EOC 后的固定字段顺序是：

1. `outcome_target`
2. `next_decision`
3. `current_parent_goal`
4. `active_skills_semantic_json`
5. `memory_update`
6. `task_complete`

EOC 前的 task、previous parent、memory、previous intent、执行反馈、图像与 proprio 都是输入，不直接承受 token CE。

- `outcome_target` 在 `outcome_supervision_mask=false` 时逐 token mask；B planner-only 要求它是 `UNKNOWN` 且永远 mask。
- `current_parent_goal` 在 `parent_goal_supervision_mask=false` 时逐 token mask。
- `high_planner_only` 中的 `task_complete` 保留在输出语法中，但整字段
  是 **format-only**：不进入 CE。R2 的 `false` 是格式默认值，不是有物理
  依据的“任务未完成”负例。运行时也仍只信任物理 evaluator 的
  `done.success`，不信任模型生成的 terminal claim。
- `next_decision`、`current_parent_goal`、`active_skills_semantic_json` 和
  `memory_update` 在其各自监督条件满足时仍是普通 CE token；这一修复不改变
  低层 FM、outcome head、参数可训练集合或加载键。

### B-only memory-copy 加权（初始配方）

`high_planner_only` 显式 opt-in `memory_update_ce_weight=0.25`。先以 b5
规则将 UNKNOWN outcome、format-only `task_complete`、以及未监督 parent
span 置为 `IGNORE_INDEX`；再以真实 Qwen tokenizer 的既有
`memory_update` token span 赋权 0.25，其他仍有效 token 权重为 1。AR label
shift 不变，loss 是 `sum(valid_token_ce * weight) / sum(valid_weight)`。

任何被 mask 的 token 都不会被恢复为 CE。A、low FM、`high_planner_outcome`
和默认权重 1 的旧 CE 路径保持原行为。运行时只额外记录 raw unweighted CE、
weighted-token CE、实际 weighted training CE 及 detached 的字段 token
count/mean；不增加 loss gate。

历史 b5 全量 46,710 行 token accounting 显示 memory update 占有效 CE token
的 67.07%，中位样本 effective-suffix share 为 70.30%。0.25 的初始配方令
聚合 memory 权重约为 33.7%、bundle + parent 约为 54.4%。这是降低复制目标
主导风险的初始值，不是最优权重宣称；data 产出 event/token 分布后才能调整。

### 已审计的 planner-only terminal 掩蔽

模型侧的 `high_planner_only` 专用掩蔽已通过独立 CPU 审核。它只在真实
`forward_train` 已先完成严格 high/planner-only 校验之后，将
`task_complete` 的 target token span 标为 ignore；模板字段、推理格式和
其他 EOC 字段均不变。现有入口已经要求 `UNKNOWN` outcome，拒绝伪造的
`STOP` / terminal 目标，因此不会让非法 terminal 文本绕过语义校验。

对正式 Qwen tokenizer、B fixture 的 initial / same-bundle hold / parallel
switch 三类行，`Task complete: false` 均精确为 4 个 token，且这 4 个
target token 都被掩蔽。`next_decision`、parent、semantic bundle、
`memory_update` 的可监督 token 保持不变；`UNKNOWN` outcome 继续按既有
outcome mask 掩蔽，低层 action mask 不受影响。

实现 / 证据：

- policy：`src/g05/models/g05/g05_policy_memlite_planner_outcome.py`
  terminal-mask 基线：
  `b5b71fd1c9de07a49efd0ccb7865d54fb09a5ebf8af7b1f3e7a79b5362aaf3b6`；
  当前 weighted-CE candidate：
  `a1cca87e481516796adbeb420a7e567070f5f9b66926d9c3793b9fb6ebe451a2`
- shared AR helper：`src/g05/models/g05/helpers/ar_helper.py`
  `53f20318356af93bb2762fe40ed142a2e0f0779bfda111dbdf12a762a45c8ec5`
- B config：`configs/task/r1pro_memlite_high_planner_only.yaml`
  `af303a0b60743b892c6d1ce247fb49b494d0f6c011d586484505286c6fb5273f`
- regression：`tests/test_memlite_skill_models.py`
  `30365a0e72958026be60f72ebedecef639f3b69497f688b8852c657788160bb5`
- CPU evidence：full contract suite 44 passed，log
  `/tmp/memlite_b_weighted_full_pytest_20260909_v2.log`
  `1ddbf017e374298154f28e451fd143583e693723f480252023aec9e129fdf4e9`；
  覆盖 weight=1 loss/gradient exact compatibility、0.25 normalized reduction、
  actual initial/hold/switch builder+Qwen spans 和 forward→AR helper seam。
  独立 training 审核已重新执行六项定向 CPU 测试（6 passed，8.32s），确认
  b5 mask→weight→shift/valid-gather 顺序、padding/IGNORE 不复活、默认 1
  等价、0.25 span/denominator 及 B-only 范围。

这不是新的运行 gate，也不需要为 4 个掩蔽 token 额外做一次 GPU
graph-load。加权候选不改变参数图、optimizer 分组或 checkpoint 加载键；
既有 e95 graph-load 仍可用于权重图核验。最终 B source 中仍须按正常
loader + audit 复验。

## R2 当前可用数据的事实

输入 sidecar：

`/mnt/sdc1/robodojo/datasets/memlite_skill_annotations_task0_4_v6_clean_r2_20260909/meta/memlite_skill_annotations_v6.parquet`

实际 high 行数为 **17,660**：

| 字段 | 分布 |
| --- | --- |
| `next_decision` | 全部 `EXECUTE` |
| `task_complete` | 全部 `false` |
| `outcome_target` | 全部 `UNKNOWN` |
| `outcome_supervision_mask` | 全部 `false` |
| `parent_goal_supervision_mask` | true 5,200；false 12,460 |

因此它能训练语义 bundle 和记忆格式，但不能证明训练了 `REPLAN`、`RETRY`、`STOP` 或真实 outcome / terminal 行为。planner-only 本身也会拒绝 `UNKNOWN` 加 STOP / terminal 的伪监督。

字符串长度（字符，不可替代最终 tokenizer 统计）显示 memory copy 的 CE 风险是真实的：

| EOC 相关值 | p50 | p90 | p99 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| `memory`（EOC 前输入） | 147 | 255 | 256 | 279 |
| `previous_intent`（EOC 前输入） | 140 | 145 | 252 | 292 |
| `memory_update`（EOC 后 CE） | 147 | 255 | 256 | 279 |
| `active_skills_semantic_json`（EOC 后 CE） | 147 | 153 | 244 | 319 |
| `parent_goal` / target parent | 45 | 152 | 166 | 184 |

`memory_update` 与新 bundle 的目标长度量级相同甚至略大，decision / task-complete 则只有少数 token；若无采样平衡，复制型输出可能在梯度预算中压过切换信号。最终 B causal overlay 必须先实际统计每个字段的 **token** 数和 `EXECUTE/REPLAN/RETRY` 事件数，不能仅按行数假定均衡。

## v10 初始化与 schema-6 内容

候选 parent checkpoint：

`/mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v10/behavior5_memlite_ar_v10_train_20260907T094716Z/checkpoints/step_5000.pt`

SHA-256：`7109d20b8054fc59d33ed236fbfbf55457d1eac46fe2bfd38b746d9aad1ab43d`。

启动应以该绝对路径绑定 `MEMLITE_HIGH_PLANNER_V10_INIT_CKPT`，不从随机 high model 或旧 optimizer resume 启动。

需要从 v10 加载并允许更新的是 VLM、multimodal projector、proprio embedder。schema-6 的 causal K3 memory 是数据 / 模板 / canonical parser 语义，不是一组新可训练的 memory tensors；其学习发生在 VLM 对新 structured output 的 CE 中。新加的 outcome head 在 v10 中没有可靠权重：在 planner-only 里应保持随机但逐 tensor 冻结、且不入优化器，而非假装从 v10 恢复或训练它。

尚未执行真实 v10-to-planner load，因此开始前必须由正式 loader 记录 load coverage：VLM / projector / proprio 覆盖应通过；新 outcome-head missing keys 只能在 `high_planner_only` 被显式白名单。

## 可执行但保守的 B 配方

只在以下条件同时满足时进入 B：

1. runtime 实现并测试独立的 `planner_only → high_planner_only` 入口；
2. B causal overlay 的 exact-19 schema、K=3 recurrence、train/eval split、canonical parser 及 row hashes 已冻结；
3. 输出预检给出 decision、bundle-change / periodic-hold、parent-mask、EOC field-token 的 train / eval 统计；
4. v10 load coverage 和 `planner_vlm` 真实 grad / update receipt 都通过。

配置保持：`task=r1pro_memlite_high_planner_only`、`data=behavior5_r1pro_memlite`、high-only `coordination_v6`、obs=6、`outcome_loss_weight=0`、coverage=0、validated=false、UNKNOWN_ONLY、`resume_ckpt=null`，且 checkpoint 使用上面的固定 SHA。不要以 `high_planner_outcome` 代替它。

如果最终 overlay 仍只有 EXECUTE 行，则将该阶段明确命名为 **bundle-BC / causal-memory formatting**，而不是 replan/retry policy。不要用无标签的 periodic copies 或 memory copies 扩量；只有 data 明确给出语义正确的 periodic-hold / REPLAN / RETRY 行时，才在 sampler 层按事件类别受控平衡。当前实现只提供上述 B-only memory span 权重；不做 event/token sampler 重权、不训练无物理标签的 RETRY/STOP，也不把 UNKNOWN outcome 或 terminal 格式字段变回监督。

在 data 发布 fixed-80 的因果 overlay 统计前，不决定 event / task / field
重权重。届时统计口径固定为：用真实 `PlannerOutcomeBuilder` 与官方 Qwen
`InputPreprocessor.encode_train` 形成 token span，再应用同一
`_mask_unsupervised_ar_fields`；按 task × initial/switch/hold 报告每个 EOC
字段的原始 token、实际 CE token、memory-update CE share、prefix memory /
previous-intent token，以及 outcome / task-complete mask 率和 parallel
degree。当前三行的 124 / 167 / 216 个实际 CE token 仅是接口校准样本，不能
外推为最终 B 分布。

## 当前训练入口缺口

`runtime_train/src/g05/utils/training/coordination_runtime.py` 已承认 `B/high/high_planner_only/6`；但 `runtime_train/scripts/finetune.py:_coordination_train_config()` 仍只接受 `low_ae`、`low_ae_lora_history`、`high_planner_outcome`，且 stage 只接受 `skill_fm_stage_a`、`skill_fm_stage_b`、`planner_outcome`。`coordination_launcher.py` 也尚无 planner-only role。

所以 **当前 B YAML 会在真实训练入口、模型加载前失败**。这是 runtime/training owner 的独立 B ticket，不应触碰已经冻结的 A 流程。

## paired obedience diagnostic 的现有边界

`SkillFMActionBuilder.build(canonical_19_projection, tensor_sample)` 可严格构造 low 条件；SkillFM `forward_train` 可复用同一 actions / masks / pixels。因此理论上可比较 original bundle 与 canonical、非等价 bundle 下的 FM loss。

但 `GalaxeaCoTProcessor.preprocess()` 会在一次调用中完成 tensor processing 和 builder，当前没有公开的“认证 tuple 后、builder 前”的 diagnostic-only hook；唯一现成拆分入口是私有 `_process_tensors()`。FMHelper 也没有显式 flow-time / noise 参数。

因此不要把已有训练回调冒充 paired audit。最小安全做法是后续由 training owner 提供一个 **DIAGNOSTIC_ONLY** 工具：先让 tuple resolver 认证原始 obs/action/mask；一次官方 tensor processing 后深拷贝临时 sample；仅替换 strict canonical 19-field semantic condition 并再次走 `SkillFMActionBuilder.build()`；actions、masks、pixels 字节保持原样；错误 condition 绝不进训练集。固定比较只能采用已批准的官方 FM sampler fixed-seed replay，并记录 spy 证明两次采到同一 t/noise。
