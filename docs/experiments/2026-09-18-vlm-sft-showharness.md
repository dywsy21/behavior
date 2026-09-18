# H-09：Show-Harness 小模型语义动作 SFT

负责人：Astra/max 子代理。独立分支 `feat/vlm-sft-showharness-20260918`，基点 `6da8c80fb94e748c479ddd558290c6315f6e163d`。主代理负责 H-08 harness，本文只维护 H-09。当前是小规模方法验证，不是50任务大训练。

## 最新状态

- 2026-09-18 20:12（北京时间）：已 fetch 团队 `origin/main=33677bd`；基点包含该提交。主仓有主代理未提交改动，未 pull/切换/覆盖；新建独立 worktree `/home/wsy/behavior_worktrees/vlm-sft-20260918`。GPU1 独占核验为 A100 80GB、0MiB，旧 PID38931 已退出、8918空。`/mnt/sdc1`余94GiB，登记本块新增≤8GiB且始终保留≥80GiB；不下载大权重。尚未训练。
- 已核官方原论文、项目及训练配置；已确认专家数据含同帧 RGB、23D动作、61D本体、每臂实际末端位置/姿态与原始技能区间。准备检查连续16帧运动能否无歧义归入通用微动作；混合运动、边界和未知样本会保留排除原因，不能伪造教师标签。

## 原论文与复现边界

[原论文 §3.4、§5.1、附录§7.1–7.2](https://arxiv.org/pdf/2609.10522)：FT使用Qwen3.5-2B，通过原生文本预测语义动作；LM线性层rank64 LoRA，冻结vision/projector。附录设定7.9k单臂样本、40epochs、LR1e-4、cosine/warmup0.1、bf16、256²视图、effective batch32，报告**单H200少于2小时**；不是27B或任意机器人/数据规模的时间保证。采集输入和动作通过共同接口严格配对。机器人与BEHAVIOR R1Pro、移动底盘、长任务明显不同，不能直接外推完整任务成功率。

[官方训练配置](https://github.com/showlab/Show-Harness/blob/main/train/configs/qwen3_5_2b_lora.yaml)当前写30epochs、`freeze_multi_modal_projector: false`，与论文40epochs/冻结projector存在差异。本轮明确采用冻结vision/projector，不称字节级复现。代码启用fused CE以避开大词表全序列logits开销；本轮须先量真实吞吐。

[官方训练接口说明](https://github.com/showlab/Show-Harness/blob/main/train/README.md)特别指出训练与服务chat-template和历史字段要一致；本轮从同一函数构造训练prefix和推理prefix，并逐token核验，仅assistant动作被监督。遵循原生词表、不另加动作head。

## 首块预登记

主要假设、数据/训练/物理预算固定在 `configs/vlm_sft/h09_first_block.json`。基座复用已有Qwen3.5-2B，GPU1；最多600更新/3小时，effective batch16，rank64；200/400/600仅小adapter检查点，最多新增8GiB。真实内存与估时在小batch门后补入。

选择task0 radio、task1 trash、task3 plates三类；每task12训练、2验证、2测试来源episode，原每task末10个留出episode绝不进训练，原开发实例task0/138和task3/242排除训练。来源instance与episode分组冻结后才抽帧；最多3456训练样本，不重建大集。测试用于一次最终比较，验证用于检查训练与选择checkpoint。所有未来动作/物理字段只生成离线target，不进入actor；actor只接收当前RGB、本体、自然语言任务/技能和过去动作。

闭环初始预算：同协议2起点×未微调/微调，各≤40决策/1280新控制/1200秒，GPU1顺序运行。原前缀单列。两个模型使用完全相同的控制、安全门、提示与重置规则；成功只认官方物理判据，另报告局部行为、拒绝、动作和延迟。数据/映射审计失败就先纠正具体根因，不消耗此物理预算。

## 尚未完成

- [ ] 来源分组冻结、同状态微动作映射和机械检查。
- [ ] 分层人工图像/标签审核及可定位记录。
- [ ] 实际训练mask、梯度、保存/回载与吞吐门。
- [ ] 完成有界训练与adapter SHA。
- [ ] 留出未微调/微调比较及行为失败分类。
- [ ] 同协议物理闭环和视频审核。
- [ ] 主代理独立代码审查。
