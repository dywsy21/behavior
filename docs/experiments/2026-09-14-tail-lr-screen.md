# M-03-S：避免重新升高学习率的独立短对照

2026-09-14 17:58北京时间预登记，owner Codex。**尚未实现、排队或训练；代码commit待固定。** 唯一候选`fm_tail_lr_v1`，不是泛化超参搜索，不重复已完成control/其他方法。

## 已核实的选择依据

CPU mmap只读A4的`formal/checkpoints/step_2500.pt`，六个Adam组的当前保存LR全为`1.0000000000000002e-6`，initial_lr全为1e-5；scheduler last_epoch=2500、_step_count=2501、base_lrs全1e-5、_last_lr全为上述末端值。该LR是保存后的当前值，不另声称最后一次实际update用了完全相同浮点值。原`.hydra/config.yaml`的SHA `4d45b4c2ae8872b8e4a88916c4143d922b8cf0e76eedaa6a4116d473d9ef2483`复核一致（100 warmup/2500 cosine/min ratio0.1）。这次检查不创建策略、不做前向/optimizer更新或读取新训练样本。

之前A4曲线在后期低LR段改善；已有500各臂重新建Adam并升至1e-5，纯FM固定80也略高于父权重，但部分动作误差改善。[原诊断](2026-09-13-fm-learning-efficiency-audit.md)、[同入口M-04](2026-09-14-joint-ki-screen.md)。这些事实足以选择一个保守日程对照，**不足以证明高LR是根因、低LR必然更好或某方法无效**。

## 固定单项与预算

- 唯一变动是整段学习率函数：AE与LoRA均恒定`1e-5 * 0.1`，0 warmup/不重新爬峰。原对照是50 warmup后cosine至0.1倍、总500更新。峰值、曲线形状、累计LR暴露一起属于本曲线变动，不将此对照拆称三个独立因果；它不保证train loss降得更快。
- 从同A4-2500（SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`）/新Adam重新开始。betas0.9/0.95、weight_decay0.03、原分组decay规则、全局clip1、rank8、数据/采样/seed41/批量/评估不改。**不恢复父Adam矩，不称精确继续训练。** 降LR也改变AdamW的有效衰减总量，不把LR积分当实际参数更新倍率。
- 对照复用`fm_action_control_v3`500（`efce4dfe232f85ac18f7fca66b562360ba23d839f3f74748f658b5911b5c33f9`），不重跑control。EMA的online曲线只作补充、不可预先假定其增广/RNG完全相同；本候选不开EMA、joint/KI、分组裁剪、marker或增容。
- 原五task950 train/50 eval、train-only stats `846bcbeac181df5555cb5d40d4183d61743a556df5cf8c8a8fb1bc17e2a40b19`、原80 manifest `a365370d81596cb720ba5df38e6935aa1e0a2bdcbf1fbc0c54dfd52bbf45b254`；4×micro2×accum2/global16，6帧/32动作/执行0:16/真实23维完整。无新标签或恢复数据，不接管队友工作。
- 拟唯一输出`robo:/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/fm_tail_lr_v1`。先CPU默认路径/500个真实scheduler LR/状态恢复/前驱门；之后等待现有EMA1707751（spec `d195fd580899e5d947df5daa8d768d6be44315ca61a7c441beab159e2d8fd2b7`）完整500及EMA保存验收，再5步四卡保存门＋从原A4独立最多500/8000 draw。训练前仍检查原实际GPU资源，不挤占旧服务；无新增正式墙钟限时，保留原smoke安全限制。
- 每100仍原未加权固定80、完整原数据抽取/梯度/冻结/保存回读；不改指标、不自动放松原初值参考门。身份/非有限/资源/保存/前驱失败即停，0自动重试/额外5000/部署。本登记没有新增动作生成或仿真预算，最终结果后再决定必要的小型动作检查。

本轮完成的是日程选型与实测入口；容量仍需要单独适用性/效果证据，不能把这个候选或已排队EMA当作整个M-03/goal完成。
