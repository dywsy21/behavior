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
| M-00 | 方法扩展是否保持原FM及评估口径 | 真实helper CPU前向/反向/随机数对照；不训练大模型 | 15项CPU测试＋7项配方检查通过；真实大模型门待执行 |
| AR-00 | codec执行段是否受未执行未来影响 | 既有10条五任务train缓存，原32步/执行段padding/原生16步输入对照；CPU、0VLM/0仿真 | v3十行完成；直接16步不支持，holdpad32均值微降且3行退化，非策略收益 |
| M-01 | 分组LR能否提高有效学习 | 同A4父权重/相同优化状态处理与数据顺序，动作专家1e-5对2e-5，LoRA相同；两组各最多500更新 | control通过GPU/四卡保存门，14:14正式500进程运行；候选未启动 |
| M-02 | Beta分层/执行段加权各自效果 | 对同一对照一次只改一项，最多500更新/候选；固定train与原80eval，不改评估函数 | 待实施/训练 |
| M-03 | 容量、EMA、动作组和日程是否有额外收益 | 独立短对照，不与M-01/M-02同时改；EMA与原权重并列，可训练参数范围明确 | 待方法证据后选定具体配方 |
| M-04 | 离散动作监督及梯度隔离是否有效 | 区分纯FM、CE+FM无隔离、CE+FM有隔离；同数据来源，先梯度/无泄漏和可学习性门 | KI/joint真实梯度、两临时更新及suffix无泄漏通过；正式对照训练和闭环未完成 |
| AR-01 | 纯AR是否能学会且自由解码正确 | 原生/当前训练接口、codec/schema/完整动作组、自由生成而非仅teacher-forcing；短训后局部闭环 | 真实40视图含严格码块、54 CPU与两临时更新门通过，自由生成仍缺组；完整train入口通过，实际训练/闭环待完成 |
| BOTH-01 | 有效方法组合是否保留收益 | 已验证候选才组合；相同新起点/seed的局部与完整任务对照，逐项消融有害交互 | 待前序结果 |

以上不是自动重跑许可：基础设施错误先修复/记录，同一失败机制无新假设不重复消耗整套闭环。500步筛选是方向证据，不是所有方法的最终容量上限；有可学习性但收敛不足时先写明确续阶段理由和预算，不无限延长五task调参。正式运行前固定真实source commit、checkpoint/normalizer/data SHA、GPU、有限步骤和停止条件；本文件的候选不冒称已经运行。

## 参考身份与实现边界

- A4父权重：`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt`，SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`。
- 原A3训练快照：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/a2_lora_history_candidate_v7_samplercoverage`，source SHA `356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281`。实验明确绑定该快照及新增Git扩展，不冒称main已经完成全部A3模型整合。
- 原train工程缓存：同上级目录`a3_samplercoverage_real_processor_v2_microbatch2/actual_cpu_batches.pt`，SHA `237acf01b29bd0d6806ed1a11d9033a747640b3ea9e0bf7246b7a62b4e92be81`，10条样本；不能将重复拟合此缓存作为泛化实验。
- AR旧v10：`docs/archive/MEMLITE_V10_RESULTS.md`第4节给出目标token仍刹不住的证据。执行前缀末值补齐只是codec内部表示候选，不将补齐的后16步当真实动作标签，不回灌任何旧eval派生数据。
- 原固定80诊断有source/seed身份约束；方法只在训练forward作用域内启用，评估必须恢复原helper并以真实同输入回放核验。不得通过删掉校验来“兼容”新loss。

## 实时状态

15:38（北京时间）：7edf644、独立`ar_blocks_20260913`的54项CPU测试passed；`ar_input_gate_v2`十原train/40视图均60-token/8完整码块，SHA `f4fe54428154821af39a4c53062d959edd8d618c13094cb1fbc6bd3324f361ea`。0模型/更新/仿真，不是自由AR效果。control最新438/500；完成后依既定单变量预算做AE×2，AR正式训练入口仍待接入。

15:35（北京时间）：joint真实门complete（SHA `fb48b231c918a9bd7412bff65713837414afd14f5250329c6afb4c5debfe16ad`），FM→322 AE+182 LoRA、CE→192 LoRA；两更新后FM略升，不作收益结论。原完整loader CPU门complete（SHA `1bb36c7aa62d911bc1e6901846dcd85f5908791ee613194232bfd0c26428869c`），10真train行/5 eval身份窗口、无优化/仿真。新AR严格码块校验待54项CPU及真实输入v2；只拒绝错误预测，不用专家动作修补。control实际417/500，step400固定80=0.1983362647。正式AR/KI/joint训练、候选FM收益和组合仍未完成。

15:19（北京时间）：`ki_gpu_gate_v1`真实梯度/无teacher-suffix泄漏/两临时更新通过，result SHA `eb5c6366051d736df45983c815818b3a7d09ac63da975cd4b2cbe11fc47efec6`；不是方法收益，FM固定输入略升。下一独立`joint_gpu_gate_v1`同9045cf7/A4/两更新；新`action_training_data.py`拟CPU`ar_loader_gate_v1`验证真正原train loader与独立eval身份，10 train行/5 eval窗口、0模型/0优化，不回灌eval或发布新数据。

15:13（北京时间）：`ar_gpu_gate_v2`真实CE→LoRA和两Adam更新通过，固定输入CE15.07449→14.84605，FM恒0，结果SHA `50215961993d57088fdfa2460a30308d47f6b03e9161ca70eb58df13558fa5ae`；自由生成仍缺lower_body/两个gripper，正确60-token目标没有缺组。只完成工程门，不是正式训练、SR或AR被否定。下一独立`ki_gpu_gate_v1`同A4/原train/单GPU1、两临时更新和真实无泄漏/梯度路由检查，无checkpoint发布。

15:01（北京时间）：3807531在独立`ar_inputs_20260913`完成41项CPU测试与`ar_input_gate_v1`真实输入门，结果SHA `5a6ca932949dabc4198e546045e8deb52d9c0977eed81f01fd96a7665a70ddbe`，十原始行/40视图全部通过，60动作tokens/行。拟`ar_gpu_gate_v1`单GPU1、A4恢复、单行原train、两临时更新及真实梯度/自由生成，不保存权重；先AR、其他路由另门。共享GPU期间不比较整轮墙钟，仍须完成真正训练/闭环。

14:51（北京时间）：AR/KI策略扩展及CPU回归、真实tokenizer探针已写，仅语法门通过。首轮tokenizer run拟`ar_input_gate_v1`：十条已有五任务train样本、四种视图40行、两CPU线程、无VLM/优化/物理。须真实验证prefix/目标/23D后再训练；task-only不是完整上游CoT复现。新代码用另一个Git worktree运行，不拉取活跃FM副本。

14:40（北京时间）：control日志实际119/500，step100原固定80为0.1971548254（结果SHA `ce9a6b8f8c6256c78f0f5d92a2f43c48645df3acb7854e88488ac21fb144830c`），较A4略高约0.107%，非最终效果。此次只读答疑，未改训练/启动新臂；AE×2及AR/KI真实策略训练和闭环仍未完成。

14:14（北京时间）：control四卡5步保存回读passed，临时checkpoint SHA `56cc992a81463180344c871a345368c368a15623bc01db17b6411947f4243c03`；正式500进程1508221开始从原A4初始化，不从smoke继续。正式完成和所有方法收益仍待测。

14:07（北京时间）：control单卡GPU门已passed（3次相同评估口径前向＋4 microbatch/2真实更新），四卡5步短测1499487运行中。该结果仅是默认方法工程门，非Beta/加权真实收益；正式500和AE×2对照均待后续。

14:00（北京时间）：fb40145服务器22项CPU门通过，control编排1499025启动，run `fm_control_v1`；GPU门→四卡5步保存回读→独立正式500步，当前尚无500步效果。`method_spec.json` SHA `569455fb9ca4c0417cae9a998c767e82cdf846c9703a7f64adb62fb7718a9d03`。

13:58（北京时间）：AR v3完整结果见服务器`dual_track_fm_ar_20260913/ar_codec_gate_v3/result.json`，8个完整执行窗＋两个末尾窗口共141有效目标步。原32/holdpad32逐行归一化RMSE均值0.0219491/0.0215759，7行改善/3行退化；直接16步全不支持。下一步AR仍须训练＋自由生成和闭环，不能以此替代。

M-01真实配方`train_fm_method_probe.py`已写，先control，A4父权重、新Adam/scheduler、seed41、global16、AE/LoRA均1e-5、warmup50、cosine500至0.1；后续ae_lr2x只将AE设2e-5、backbone multiplier=0.5使LoRA不变。两臂各500上限，原80评估不改；每臂先真实单卡门及四卡5步保存回读，不承诺单位GPU时间或SR改善。新扩展源码和配方hash写入原trainer config identity，不冒称原训练实现无扩展。

13:46（北京时间）：服务器`7d7a8cf`的13项CPU检查通过。AR编码v1按缺组保护停止，未训练/仿真；保留失败manifest，新v2显式关闭noop dropout，不填造缺失动作或放宽完整性检查。继承的FM配置不会执行codec，不把此探针配置错误当FM失败根因。

13:32（北京时间）：已建立feature、完成M-00扩展与测试草稿；未启动训练或仿真。robo GPU1空闲，其余GPU保留六个旧服务。下一步从Git独立worktree运行CPU门和真实codec门，再进入训练；整个goal active。
