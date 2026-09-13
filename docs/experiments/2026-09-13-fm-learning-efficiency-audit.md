# A4 FM学习效率：只读诊断与下一轮建议

2026-09-13，Codex / A-02。用户询问“loss下降慢但仍有效，如何学得更快”。本轮仅回读配置、完整固定诊断曲线、离线训练指标和既有10样本拟合结果；未启动训练/仿真、修改模型或数据、上传W&B记录。以下实验均为建议，尚未执行。

## 结论

现有证据支持局部控制有收益，不支持“已证明普遍有效”或“学习率太小就是根因”。固定80是留出诊断，不是训练集拟合速度；局部成功与完整成功分开看，最新自主收音机仍为0/3，见[完整评测](2026-09-13-a4-radio-full-eval.md)。

### 1. 不能只用两端点判断下降速度

全部25份A4诊断使用相同80窗口、固定噪声/时间采样身份。中途先变差、后改善：

| checkpoint | 固定80 FM |
| --- | ---: |
| 父A3-5000 | 0.1987768451 |
| A4-100 | 0.1979736799 |
| A4-600（25点中最高） | 0.2012262223 |
| A4-1500 | 0.2006362639 |
| A4-2000 | 0.1979806525 |
| A4-2500 | 0.1969439941 |

600步较父权重差约1.23%，最终比父权重好约0.92%。后半程改善与LR下降同时发生，不能仅凭相关性认定因果；至少不支持盲目增大全局LR。A4重新初始化Adam/scheduler，而非完整恢复A3优化状态，换了采样seed；早期波动也可能与这些变化、样本顺序和分布有关。

### 2. 不是完全学不动，也未发现频繁裁剪的证据

复查已完成、未保存发布权重的 `a3_original_train_capacity_100_v1`：固定10条原train样本、恒定LR1e-5、100次临时更新；同噪声评估FM从0.1077156056降至0.0179847352（约83.30%），归一化生成动作RMSE均值从0.4339661259降至0.1243986795。评估文件各20条记录是10个样本×2生成seed，不能当20个独立样本；FM批次值重复出现但每批重复次数相同。它只证明局部拟合能力/梯度链路能工作，不证明泛化、全分布容量足够或成功率提升。

只读解析A4离线W&B二进制，得到每10步一次的250个训练记录：

- clip前grad_norm中位0.289689，范围0.149531–0.715061，250个记录点均低于clip阈值1；未覆盖其余2250次更新，不能断言所有步骤从未裁剪。
- 记录FM范围0.003477–0.461938，中位0.098713；前四分之一均值0.111185，后四分之一0.114784。
- trainer的FM记录来自最后一个microbatch，并非固定窗口或全部梯度累积样本的严格均值；不能据此和固定80直接做过拟合判定，也不能宣称训练loss持续下降。
- 没有证据优先支持增大clip阈值；“梯度没有更新”也不符合既有504项训练状态更新回执。

### 3. 已有配置与潜在杠杆

实际A4为四卡global batch16（每卡2、累积2）；学习率100步warmup到1e-5，2500步cosine到1e-6，AdamW/betas0.9,0.95/weight_decay0.03。动作专家与VLM LoRA共同训练、其余基座冻结；LoRA rank8/alpha16/dropout0.05。optimizer继承分组接口，但本轮没有独立设置较小的LoRA学习率。

采样已经使用task轮转和episode_round_robin_v1，不能把“加task均衡/轨迹覆盖”当成尚未做的新方法；下一步应检查技能内部的接近、接触、闭爪、保持等阶段与有效动作分量的监督占比。不能一刀切删除静止：保持抓稳和运输时的稳定控制本身重要。

FM针对32步×23个有效控制维度的噪声条件速度回归，不是物理抓取误差；4个补齐位权重0。总体均值可能掩盖关键手臂/夹爪或接触阶段的改进；小误差变化可能跨过接触阈值，但这是机制解释，不是本次单次成功已证明的唯一因果。

## 推荐顺序（尚未实施）

1. **先补可比的train/eval诊断。** 从原训练split固定分层窗口，使用与固定80一致的评估模式/噪声口径，同时记录task、skill、动作组、生成动作误差。检查是train也不降、train降但eval不降，还是关键动作改善被均值掩盖；不为调参将留出轨迹或本轮失败评测回灌训练。
2. **只做一次有界、单变量学习率对照。** 例如同A4父checkpoint、同恢复Adam策略、同样本顺序/seed/批量/共同调度，两臂各最多500更新：动作专家LR1e-5对2e-5，LoRA保持相同1e-5，其余不变。这是待验证的筛选范围，不承诺翻倍收敛，也不是原A4末段1e-6的等价续训。若eval早期恶化或条件漂移明显，下一步才另立“降低LoRA LR”的单变量试验，不同时改rank、采样和loss。正式执行前还须固定代码版本、资源、停止条件和指标，当前没有授权/启动新训练。
3. **随后考虑监督效率，不只加步数。** 保留现有task/episode覆盖，对可信的关键运动/接触窗口适度重采样，或单独试验按动作组归一化的加权FM；两者不要一次全改。加权训练目标与原未加权固定80指标并列，避免把改标尺当进步。需要新标签/恢复数据时向数据owner提出需求。
4. **容量与吞吐后置。** 若代表性train集仍难拟合，再试LoRA rank16/32或有限解冻；10行可拟合不能排除更大分布的容量限制。墙钟优化另行测量数据加载、18张历史图像和FM成本：本轮torch_compile/fused_optimizer均关闭，每观察num_flow_samples=4；比较4→2时须同时看同时间真实样本数和泛化，不能把多噪声draw算成多条新状态。不得仅据vision的SDPA fallback警告断言未用GPU或承诺FlashAttention必然提速。

OpenPI官方DROID训练说明明确指出适当过滤idle动作chunk可以改善策略，但该结论针对其数据，不能直接照搬阈值或删掉本任务的保持动作。[官方数据过滤说明](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md#data-filtering)

OpenPI当前代码的默认cosine配方为peak2.5e-5、1000步warmup、30000步decay，说明应把峰值与日程一并考虑；它不是G0.5本任务最佳LR的证据。[官方优化器源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py)

## 证据位置

- A4实际配置：`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/.hydra/config.yaml`，SHA `4d45b4c2ae8872b8e4a88916c4143d922b8cf0e76eedaa6a4116d473d9ef2483`。
- 完整曲线：同run的 `fixed_diagnostic/`，本轮读取全部25份 `aggregate`；窗口/噪声身份见[原效果报告](2026-09-13-a4-training-effectiveness.md)。
- 训练指标：同run的 `wandb/wandb/offline-run-20260912_154843-zo51ulbf/run-zo51ulbf.wandb`，SHA `f9a780d12caef7c95b26c5a404bf63c56b485fba68a022614de1ef215c14d2f4`。只以rb模式扫描选定history指标，不读取输出秘密/环境配置，不上传。
- 实际源码：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/a2_lora_history_candidate_v7_samplercoverage/` 下 `scripts/finetune.py`、`src/g05/utils/training/get_scheduler.py`、`src/g05/utils/common/coordination_sampler.py`、`src/g05/models/g05/g05_policy.py` 与 `helpers/fm_helper.py`。
- 既有10样本拟合：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/a3_original_train_capacity_100_v1/` 的 `evaluation_step_000.json`、`evaluation_step_100.json` 与 `result.json`；本地副本在 `artifacts/local-archive-20260912/root/memlite-resume.L6rqZ6/a3_original_train_capacity_100_v1/`。

下一步仍由Codex负责训练机制与可比诊断，队友负责数据扩充/恢复与通用RL。本轮仅形成诊断与建议，不将其记作已验证加速、已完成新实验或goal完成。
