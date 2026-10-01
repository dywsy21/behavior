# P107 首40条时序视觉校准标注指南

状态（2026-10-02，北京时间）：本指南只适用于 sealed `calibration40-v1` 的 40 条 `annotation_calibration` 候选。它们仍是 `CANDIDATE_MISSING_EVIDENCE`、`training_eligible=false`：尚无 native RGB、人工标签、outcome、recovery action、FM 正例或数据 release。本指南是计划和审核规程，不是标签文件或训练视图定义。

## 范围与输入

- 只接收 renderer 从 sealed index/queue 解出的三路 **camera-native RGB PNG**。原生图不得带 outcome、success、failure、recovery、label、evidence 或 review footer；带时间戳的 contact sheet 仅供 review，绝不作为 actor 输入。
- 每个 sidecar 只用 raw candidate packet 已有身份/时钟定位：`packet_id`、`event_id`、`actor_packet.observation_frame`、`audit.source`、`audit.usage_role`、`audit.temporal_schedule`、每个 sample 的 `sample_frame`/`timestamp_s`/三路 `video_locators`，以及存在时的 `audit.decoded_pts_receipts`。不要改写这些字段、生成训练 view 字段，或把 packet 改成可训练。
- 首40条必须全程为 `annotation_calibration`；不得混入 `student_candidate`、evaluation/protected source 或旧24张 auxiliary pilot。空字段一律明确写 `UNKNOWN` / `no_event_evidence`，不是强行补成正负例。

## 逐帧视觉关系

1. 先读明确的 `question_context` 和 target binding；每个回答都必须写清“哪个实体”和“相对哪个目标/关系”。source skill/segment 只描述**尝试的目标**，不是成功、失败或恢复的真值。
2. 对每个已解出的 `(sample_frame, exact camera reference)`，只记录可见的目标关系：`YES`、`NO` 或 `UNKNOWN`。遮挡、反光、边界、目标身份或支撑关系不清时用 `UNKNOWN`，并保留原因。
3. 元数据对象 ID 或名称不能单独完成 visual grounding。先在该相机画面中确认实体；同一物体在 head/left/right 三路相机中只算一个物体，不得按相机数重复计数。
4. 使用 renderer 实际记录的 frame/PTS/相机 locator；不要用 container duration、近似帧号或另一个 episode 的相邻画面替代。episode 边界被 clip、同一 frame 因 clamp 去重时，按实际 distinct frame 标注，不把重复 slot 当作多个观察。

## 时间线、因果与弃权

- 全部请求时间线都要看；必要时查看原生 PNG 的局部放大，而非只看单个 anchor。可选的 `timeline_findings` 可以为空：没有足够的可视证据就是有效结论。
- 只在视觉序列本身证明时，记录“观察到的 attempt 变化”或“观察到的 recovery 尝试”。不得从 metadata segment end、夹爪闭合、timeout、模型文本或“尚未完成”推断 FAILED/SUCCEEDED，也不得把 source skill 当 outcome。
- 每项可见关系/finding 绑定实际 `label_available_frame`。actor 可用上下文只可引用 `<= label_available_frame` 的 causal frame；严格未来的 offline audit frame 可帮助后续人工审阅，却永远不能倒灌为更早时刻的 actor/记忆/已知结果。
- image-only 证据不产生 actual-action BC、23D 控制、27D projection、attempt outcome 或 corrective-action 正标签。它们仍需独立的实际执行、时钟、验证和发布门。

## 审核、provenance 与发布门

- `parent_root_model_review_completed` 与 `human_reviewed` 分开记录；前者不能替代后者。当前未经人工审核的一律 `human_reviewed=false`。
- 可用的 agent fallback 是 `gpt-5.6-luna/max`，并必须保留 agent/model provenance；`gpt-6-luna/max` 当前不可用，不能冒充已使用或 human review。
- 父代理在任何声称的 failure/recovery、全部 ambiguous case 以及分层抽样上亲自检查完整 timeline 和所需原生局部图。第一批之后也不能仅靠 summary 或 contact sheet 发布正例。
- 直到该人工/父审、独立代码审、source-group/split、时钟/PTS 和正例 authority 门都通过前，首40条永远只是 calibration-sidecar 候选；不得发布为训练数据或计入大规模 recovery/outcome/action 数量。
