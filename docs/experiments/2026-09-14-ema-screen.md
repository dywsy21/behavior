# M-03-E：可训练参数EMA的独立有限对照

2026-09-14 17:31北京时间预登记，owner Codex，feature `feat/dual-track-fm-ar-20260913`。**17:46已实际排队1707751，精确等待原CoT，0GPU/0更新。** source `d28581a9b0bc68d7740adf875b8493fbea392cc8`、264 CPU通过；spec `d195fd580899e5d947df5daa8d768d6be44315ca61a7c441beab159e2d8fd2b7`。不是已经完成策略训练，不能重复提交。[当前机器摘要](results/2026-09-14-ema-screen.json)。已完成库时钟检查见[前置证据](2026-09-14-ema-preflight.md)，不重复该小例子。

## 唯一问题与对照

在当前正确的纯FM低层配方中，逐步平滑可训练权重能否改善留出动作预测，而不是把一次训练的loss波动当作学习率或模型容量问题。主要配对是本轮**同一更新轨迹的online与EMA**；已完成的M-04 FM500（`efce4dfe…`）仅作已有参照，不重新启动独立control500。EMA不进入反向/优化器，不回写成下一步在线参数。额外评估与共享硬件可能改变加载时序，因此不预先宣称与旧run训练像素/RNG逐位相等。

## 固定配方、身份与上限

- 唯一run为`robo:/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/fm_trainable_ema_v1`；从A4-2500 `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`重新初始化、新Adam，不接CoT或smoke权重。
- 原五任务950 train/50 eval，原固定80 SHA `a365370d81596cb720ba5df38e6935aa1e0a2bdcbf1fbc0c54dfd52bbf45b254`、train-only stats `846bcbeac181df5555cb5d40d4183d61743a556df5cf8c8a8fb1bc17e2a40b19`；不增数据/标签或回灌诊断集。source/data/processor的完整身份继续由原trainer记录。
- 4 rank × microbatch2 × accumulation2，全局16，seed41/各rank原规则，原LR1e-5、betas(0.9,0.95)、weight_decay0.03、50 warmup/cosine500/min ratio0.1、全局clip1、rank8；6帧/32未来/执行0:16、真实23维全部保留。高层不训。
- EMA beta=0.99，update_every=1，无power/额外warmup；第0步复制父可训练参数，之后每次成功optimizer update后执行一次`shadow += (1-beta)*(online-shadow)`。跟踪原514个可训练张量（322 AE+192 LoRA），即使本步无梯度也不伪造Adam状态；冻结权重/统计buffer不另复制到EMA。配方需真实合同核验，不按名字猜数量。
- 先CPU单元/序列化/异常还原/在线Adam与RNG门；之后四卡smoke最多5实际更新并验完整online+EMA+Adam+时钟+RNG存取。通过才从原A4独立最多500/8000draw；串行等待既有`ar_native_subtask_cot_fulltrain_v1`完整验收（spec `3dffcd78de30bff9d19c3e2b1658433bda29e0a215d4b48bf7630bf3d2d37863`）。每rank训练进程显存预设上限55%总显存，启动仍检查原实际余量；不停止旧服务或挤占队友任务。
- 正式step0只做原80参考，EMA当时与online逐位相同，不重复一份初始80；每100分别原80在线与额外80 EMA（五个时点共新增400窗），不更改原未加权指标。smoke只用原五task小诊断，不声称完整留出效果。不得将这个数学上相同的step0写成另一次实际EMA前向。
- 不设新增正式墙钟终止；保留原smoke安全上限。任何身份/切分/非有限/显存/完整状态/还原失败即停止并保留证据，无自动重试、额外5000或部署。本登记没有新增AR生成、动作十窗或物理回合预算；看到最终结果后再明确决定必要的动作检查。

## 必须验收的实现性质

1. 影子参数不替代在线参数、不参加优化器；创建/更新/评估不消耗随机数，CPU玩具在线梯度、参数和Adam状态与无EMA轨迹逐位相等。真实加载/增广时序另论，不将CPU门冒充GPU训练等价。
2. 保存完整版本/衰减/跟踪名与形状dtype/累计更新数/全部shadow；恢复拒绝缺名、错配方、错误时钟、非有限或部分加载。再次更新必须接原时钟，不重复库时钟丢失问题。
3. 评估使用可恢复上下文：只临时装入EMA参数，无论正常或异常都还原在线权重、保持Parameter对象/optimizer引用；禁止嵌套或在已交换状态更新EMA。冻结参数和buffer仍用原值。
4. 真实四卡记录EMA内存、实际步数/影子数、原数据抽取；原固定参考/冻结/Adam及完整保存门继续保持。实验无效、在线优于EMA同样是正常结果，不强行组合已有负收益候选。

权重文件将同时保存原`model_state_dict`（online）及独立`trainable_ema_state_dict`（shadow＋时钟）；普通加载仍得到online。后续动作评估必须显式选择分支、装入全部已验EMA shadow并保留冻结部分，不能仅换成这个checkpoint路径就声称使用了EMA；当前没有提交这项动作检查或部署。

目前容量/独立日程的选择仍未完成，分模块裁剪helper未接入本配方；本轮不是EMA-PyTorch默认配方或SWA的复现，不能预承诺SR或更快收敛。

## 实现检查进度

17:40：首版helper a49d7e3在robo独立`git_worktrees/ema_cpu_20260914`通过46项CPU用例（1.78s）。包含FP32/FP64递推、非法状态原子拒绝、八步玩具online/Adam/梯度/RNG不变、临时平均异常还原，以及磁盘保存后**另一个CPU进程**完整恢复再更新；这不是实际策略的新进程续训。当前已补trainer独立接入和每次评估后逐位在线还原检查，新的集成源仍待CPU回归；真实policy训练/四卡显存门和方法收益均未发生。

17:42：ee970b0独立`git_worktrees/ema_screen_20260914`集成257 CPU passed（3.03s）。随后补加载模型前真实每rank余量检查（55% allocator上限＋1GiB额外余量）；七项新CPU边界待验，新源和实际launch另记。不用“设置了显存比例”替代“共享卡确实有这些空闲显存”。

17:46：在前一CPU进程退出后将该独立源安全ff至d28581a，264 CPU passed（2.90s）。17:45:33唯一提交1707751，随后PID/argv/status实测为等待220792/ticks334478266；launch SHA `f9fb4d947c1648b8726a0e8650caefb4225ba540e7bc905bdf3bfa12b31f78ef`。该源现在活跃，后续不得pull/热改。原marker→CoT不变，待完整前驱/四卡门后才训练，当前无策略/动作/SR收益。
