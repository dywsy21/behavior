# M-01：动作专家学习率×2的500步筛选结果

2026-09-13，owner Codex。**AE×2没有通过本轮原固定80窗口的离线筛选，暂不纳入“有效方法组合”。** 这不是成功率结论，也不代表所有学习率调整或SFT都无效。

## 实验身份与实际训练

两臂从相同A4-2500完整权重初始化，新Adam与50步warmup/cosine500；不是原A4优化状态的等价续训。均为原950 train/50 eval、seed41、四卡global16、六帧三相机、预测32/执行0:16、真实23控制，冻结高层和其他基础权重。唯一配方变化是动作专家峰值LR从1e-5增至2e-5；VLM LoRA都为1e-5。

两臂均完成500次optimizer调用、保存与实际回读：504 Adam状态/可训练张量、冻结不变、模型/Adam有限、四rank RNG。每臂8000次真实train抽取、950条轨迹全覆盖，各task1600次/190条轨迹。四rank完整1000条microbatch回执的SHA逐对一致，**已验证全程抽样来源及顺序，而不仅是前缀**；这不意味着所有解码像素也逐位一致。

## 相同未加权固定80窗口

| 指标 | control：AE 1e-5 | candidate：AE 2e-5 |
| --- | ---: | ---: |
| task0 | 0.14149776 | 0.14052702 |
| task1 | 0.16176960 | 0.16331595 |
| task2 | 0.27347011 | 0.27856336 |
| task3 | 0.16389004 | 0.16870400 |
| task4 | 0.25037853 | 0.25460990 |
| 五task均权总FM | **0.19820121** | **0.20114405** |

AE×2总FM比同预算control高**1.485%**，仅task0改善、其余四个退化。相对A4父参考0.1969439941，control高约0.638%，AE×2高约2.133%。这些是固定窗口/噪声的诊断值，不是全eval、统计显著性或完整任务SR；两臂均未在此轮跑新闭环。历史A4局部抓取/0/3完整评测不能当这两份新权重的结果。部分训练期间与工程探针共享GPU，因此不使用整轮墙钟声称收敛加速。

## 证据位置

以下run都在robo的`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`下，实际训练源码为fb40145；原A3 source SHA `356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281`。

- 父A4 SHA：`6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`。
- `fm_control_v1/formal/checkpoints/step_500.pt`：`def222a6674e6ac92e6ee982c22836b789240f1542c459d5de2111cd646a244e`。
- `fm_ae_lr2x_v1/formal/checkpoints/step_500.pt`：`7f1c9acdfe52d3ffd6e98038c46a6d743a07766b1188e20c7b45262048396753`；17:45:40（北京时间）完成全部验收。
- candidate `formal/fixed_diagnostic/step_500.json` SHA：`a58d046c2f2f9ca3a22bfe81584ebb49132652ba32b0e7c164321e96c6db4e90`。
- 每臂`formal_checkpoint_inspection.json`为保存回读验收；`formal/coordination_sample_receipts_rank0.jsonl`等为全程采样来源，四rank相同SHA分别为`208076c4…`/`e43213a5…`/`77e2ac92…`/`756cdd00…`。

## 下一步

不部署替换A4、不自动延长这个LR候选。A4-AR四卡5步保存门已通过，正式500作业已从原A4重新启动；原生AR、Beta分层、执行段加权、同入口FM/joint/KI分别按[执行计划](2026-09-13-dual-track-execution.md)排队。CoT、真实闭环、条件服从和有效组合仍须完成，整个goal未结束。
