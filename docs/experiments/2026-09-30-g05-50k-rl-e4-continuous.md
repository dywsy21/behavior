# E4：E3收尾后无总预算持续RL

负责人：Codex。分支`feat/rl-dense-aggressive-20260929`。用户2026-09-30明确授权；只延长已登记路线，不扩任务/机器，不改变学习率、奖励或模型架构。最新状态：14:23北京时间已服务器arm，唯一CPU接续PID199272等待E3原训练＋六对final，尚0 E4 GPU。

## 假设与接续规则

假设：维持E3相同on-policy配方持续采样更新，能继续改善更早接管，并最终改善无专家前缀的完整任务成功。当前局部曲线不能证明后者。

E3必须先按原计划结束训练，并用预先选定的最后完整权重完成固定六回合评测。CPU接续器只在原监管exit0、六回合身份/物理账一致、所有旧PID退出且GPU0/2/3空闲后准备并启动一次。旧run失败不会绕过评测或盲目重启。GPU1其他作业不动。

父模型仍为robo `/mnt/sdc1/robodojo/outputs/g05/r1pro/behavior5_nomem_bs8_4gpu_20260923T135319Z/checkpoints/step_50000.pt`，SHA `c465044b025a487c42fddce17d45059b314a547e1b672f27cbdbbb7cfe2f6d48`；其上加载E3最后完整AE/noise/critic delta，恢复Adam、Torch/CUDA及learner Python RNG、actor/critic累计步、课程前缀与成功历史。运行时确切delta路径/SHA写新manifest和resume_load回执，不预猜最后步数。

原TRAIN task0 `turning_on_radio`实例1/138、env seed0、原动作/Parquet/标注SHA和holdout隔离保留。新轨迹继续仅TRAIN；原public_test301/302 × seeds17/23/41、3224控制/无前缀是开发评测，不回灌PPO，也不称盲测。

## 不再有的上限与仍有的保护

- `max_controls/max_training_controls/max_batches/max_new_actor_updates/max_actor_updates/max_training_seconds/max_active_wall_seconds`全部显式null，不用大整数假装无限。批次使用无限iterator。
- 单回合自主控制1024、三连胜前缀提前96及原768 gap floor、真实23D控制/27D表示、FM32预测/16执行不变。AE1e-7、noise1e-6、critic1e-4、4 PPO epochs、4 critic steps、BC0.1、KL/backtracking及原signed dense potential不变。
- 三批无奖励/无accepted更新只记警告，不再以学习进展门结束。保留数据/数值/真实模拟器/物理对账失败停止；不盲目重启失败作业。
- 保留至少100GiB空闲＋8GiB单checkpoint写入预留，每批/保存前检查；保留全部权重，不自动删唯一文件。若磁盘或资源故障仍需处理，持续训练不代表无限物理存储。
- TRAIN累计控制继承E3，旧最终评测控制另列`prior_evaluation_controls`；`cumulative_physical_controls`包含两者。中断历史物理步不丢、不重复扣，评测不能伪装训练步。
- E4不再有预算终止final。后续固定checkpoint评测另按相同开发矩阵登记；本次效果图明确区分在线训练课程与已有固定完整评测。

## 文件位置及启动

robo新run：`/mnt/nvme_tmp/robodojo_g05_rl_20260928/e4_continuous_v1`。

CPU接续器：同父目录`e4_handoff_v1/{claim,launcher,status}.json`、`watch.stdout.log`、`prepare.stdout.log`。它的`waiting_for_e3`不等于E4已开始。

新私有runtime：`/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e4_continuous_v1`；只复制已验证退出E3的私有编译缓存，绝不复用活跃可写runtime。

已冻结源码`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/g05_50k_rl_continuous_883aec8`，准确commit `883aec894e0dec7ccf065ab798cc11c2e65494a4`。UTC06:22:15已唯一arm，PID199272、`status.json.phase=waiting_for_e3`，14:22:44复核存活，E4 OUT尚不存在。唯一入口：设置`BEHAVIOR_RL_OUT/BEHAVIOR_RL_RUNTIME/BEHAVIOR_RL_PREVIOUS_RUNTIME`后运行新冻结源`continuous_handoff.py --arm`；本次已执行，不要重复运行prepare/launch，不改当前E3源码/manifest。

## 当前曲线与可复算证据

原始小体积摘要：[曲线数据](2026-09-30-rl-learning-curve-data.json)。绘图：`scripts/rl/plot_learning_curve.py`，图位于本地`/home/wsy/behavior/artifacts/g05-50k-rl-20260928/rl-learning-curve-20260930.png`，不入Git。

14:12快照19批，已保存202。实例1的1076/980/884/788专家前缀各5/7、4/5、3/4、0/3；实例138前缀1096仅2/19。每条轨迹横轴为生成它的更新前actor，而非随后吃掉它的更新后actor。滚动窗最多5次，换前缀清零并断线。完整reset仅parent0/6、E2 0/6；E3未完整不填零。只能说部分课程有推进，尚无完整成功率提升证据。

14:19本地69 RL＋5 flow回归通过，覆盖真实toy actor跨旧2094、循环跨旧批/控制上限、三批无进展继续、完整六回合依赖拒绝缺项、在训不调用GPU/prepare、重复arm拒绝。14:22服务器69 RL中68过/1 CUDA项按CPU模式跳过、5 flow全过；真实method_dense/prepare导入及旧新recipe验证通过，无环境改动。另补IPC连接到达顺序可变时按具名worker关闭回执＋逐条物理日志验证的回归。等待未来真实prepare/SHA/恢复/新更新，不能把arm等同训练已续接。分支尚未经另一成员独审，不合main。
