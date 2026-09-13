# 双路线执行：FM训练改进与纯AR重新验证

2026-09-13，唯一owner Codex；分支`feat/dual-track-fm-ar-20260913`，起点main `dbc89c8`。本计划承接用户完整goal，非仅研究建议。当前仍五task方法准备，不启动50任务大训练，不调用subagent、不重复队友数据扩充/RL。

## 完成要求

1. 对已有训练候选池逐项给出实测结论或有证据的适用条件；有收益的兼容方法要组合并复测，不能默认收益相加。候选池包括分组LR/日程、Beta时间分层、执行段/动作组加权、LoRA容量、EMA、KI式CE+FM；任务梯度检查决定是否进一步试PCGrad，不能凭一个task退化直接上复杂方法。
2. 纯AR须重新做编码/目标/自由生成/动作执行检查，实际训练后闭环；不能把KI训练中的离散辅助分支当成纯AR推理验证。保留原生G0.5任务条件AR与MEM-Lite条件AR的区别，不能只复测已经失败的旧v10即宣布AR不行。
3. 各路线经原始23维、底盘/躯干、真实本体锚点、六帧历史/动作时钟、prefix无teacher-forcing泄漏检查；评测使用固定数据/实例/seed/预算。AR若调整编码horizon，明确声明，并保持物理执行频率与执行段对齐。
4. 最终分别交付FM组合与AR训练配方/权重身份、可比离线和局部/完整任务结果、视频、失败原因、面向50任务的选择建议。证据不足不得宣布方法有效或goal完成。

## 分阶段预算与下一动作

| ID | 唯一主要问题 | 初轮上限与对照 | 当前状态 |
| --- | --- | --- | --- |
| M-00 | 方法扩展是否保持原FM及评估口径 | 真实helper CPU前向/反向/随机数对照；不训练大模型 | 13项CPU测试通过；真实大模型门待执行 |
| AR-00 | codec执行段是否受未执行未来影响 | 既有10条五任务train缓存，原32步/执行段padding/原生16步输入对照；CPU、0VLM/0仿真 | v1因继承noop dropout导致缺夹爪而停止；v2显式全部组配置准备中 |
| M-01 | 分组LR能否提高有效学习 | 同A4父权重/相同优化状态处理与数据顺序，动作专家1e-5对2e-5，LoRA相同；两组各最多500更新 | 待M-00及真实GPU门 |
| M-02 | Beta分层/执行段加权各自效果 | 对同一对照一次只改一项，最多500更新/候选；固定train与原80eval，不改评估函数 | 待实施/训练 |
| M-03 | 容量、EMA、动作组和日程是否有额外收益 | 独立短对照，不与M-01/M-02同时改；EMA与原权重并列，可训练参数范围明确 | 待方法证据后选定具体配方 |
| M-04 | 离散动作监督及梯度隔离是否有效 | 区分纯FM、CE+FM无隔离、CE+FM有隔离；同数据来源，先梯度/无泄漏和可学习性门 | 待实现，非已有KI |
| AR-01 | 纯AR是否能学会且自由解码正确 | 原生/当前训练接口、codec/schema/完整动作组、自由生成而非仅teacher-forcing；短训后局部闭环 | 待编码门，后续冻结正式训练预算 |
| BOTH-01 | 有效方法组合是否保留收益 | 已验证候选才组合；相同新起点/seed的局部与完整任务对照，逐项消融有害交互 | 待前序结果 |

以上不是自动重跑许可：基础设施错误先修复/记录，同一失败机制无新假设不重复消耗整套闭环。500步筛选是方向证据，不是所有方法的最终容量上限；有可学习性但收敛不足时先写明确续阶段理由和预算，不无限延长五task调参。正式运行前固定真实source commit、checkpoint/normalizer/data SHA、GPU、有限步骤和停止条件；本文件的候选不冒称已经运行。

## 参考身份与实现边界

- A4父权重：`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt`，SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`。
- 原A3训练快照：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/a2_lora_history_candidate_v7_samplercoverage`，source SHA `356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281`。实验明确绑定该快照及新增Git扩展，不冒称main已经完成全部A3模型整合。
- 原train工程缓存：同上级目录`a3_samplercoverage_real_processor_v2_microbatch2/actual_cpu_batches.pt`，SHA `237acf01b29bd0d6806ed1a11d9033a747640b3ea9e0bf7246b7a62b4e92be81`，10条样本；不能将重复拟合此缓存作为泛化实验。
- AR旧v10：`docs/archive/MEMLITE_V10_RESULTS.md`第4节给出目标token仍刹不住的证据。执行前缀末值补齐只是codec内部表示候选，不将补齐的后16步当真实动作标签，不回灌任何旧eval派生数据。
- 原固定80诊断有source/seed身份约束；方法只在训练forward作用域内启用，评估必须恢复原helper并以真实同输入回放核验。不得通过删掉校验来“兼容”新loss。

## 实时状态

13:46（北京时间）：服务器`7d7a8cf`的13项CPU检查通过。AR编码v1按缺组保护停止，未训练/仿真；保留失败manifest，新v2显式关闭noop dropout，不填造缺失动作或放宽完整性检查。继承的FM配置不会执行codec，不把此探针配置错误当FM失败根因。

13:32（北京时间）：已建立feature、完成M-00扩展与测试草稿；未启动训练或仿真。robo GPU1空闲，其余GPU保留六个旧服务。下一步从Git独立worktree运行CPU门和真实codec门，再进入训练；整个goal active。
