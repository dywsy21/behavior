# 恢复数据 → A800 小规模 SFT：准备与验收

负责人：Codex；分支 `feat/recovery-sft-prep-20261009`。此页是10月9日批准的准备工作，不是新训练启动或数据放行通知。真实进度看 [plan](../plan.md)；方法/预算看 [RL方法计划](../RL_METHOD_PLAN.md)。

12:50 CST检查点：八卡硬件通信、L0真实父权重/FM梯度（ff584c1）及H0成员prefix/observer梯度（c25d06c）均通过，A80070＋8测试通过，0optimizer，所有诊断GPU已释放。242ZIP数据转移完整hash通过；原100task/3200个不同来源实例专家锚点索引已生成。三种恢复训练池依旧未放行；[回执](results/2026-10-09-recovery-sft-preparation.json)明确列出尚未完成的训练整合/数据/校准门。

13:04 CST补充：冻结874d39e八卡**微型合成网络**验证了通用运行时的非均匀尾批全局梯度、模型/Adam/RNG精确恢复与单rank错误共同拒绝更新；它不是实际G0.5八卡训练验收。独立[W&B工程探针](https://wandb.ai/hanhanyy-fudan-university-school-of-management/behavior2026-g05/runs/26d903a19327)已真实写入/读回并结束，仅`engineering/*`指标，无合成loss混入训练曲线。0新增G0.5更新，所有GPU已释放；完整derivative trainer接线、数据/校准门仍待。

## 现有事实

- `raw-v1`封存242个已关闭候选ZIP，347,415,662B，86个episode/43任务。结构、原index SHA、三相机图像、动作时钟通过。去重399个重叠控制行，无矛盾。
- 6194个视觉锚点；5734个有完整32步同意图真实动作。这是“目标可读取”，不是“这些动作正确”。
- 精确asset↔BDDL映射后有360个局部抓持成功候选锚点，合为19段连续证据/11个episode；相邻窗口可能仍为同一次事件。不能称360次成功或19次恢复。
- 主agent亲审75个窗口/19张sheet/675幅原RGB，覆盖所有提出成功候选的archive及任务/事件分层。每窗口只看三个时刻，**不是全视频验收**。逐条记录见 `configs/recovery_sft/owner_review_v1.json`。
- 所有训练审批仍为空。`admission-v1`三个池分别拒绝训练；缺FAILED/IN_PROGRESS语义验真、正确交接/恢复续段、完整动作质量检查和独立留出覆盖。
- `bringing_in_wood`实例9/111的模板与状态scope不匹配，2个episode/179锚点隔离。不能猜对象数字后缀、用另一模板强行补齐。

## 服务器与文件

共享根：`/data/workspace/wsy/behavior2026`。本轮选lc1，lc2已有队友GPU进程不动；不使用robo。独立环境沿用 `envs/g05-py310-cu128`，不安装/升级。代码仅Git传送到独立 `src/recovery-prep-<commit>`，不用scp/rsync覆盖源码。

| 内容 | 共享根下路径 |
| --- | --- |
| 原专家数据 | `datasets/memlite-stage1-20260930-v4` |
| 候选封存/审计/准入 | `datasets/recovery-candidates-20261009-v1/{raw-v1,audit-v4,admission-v1,source-metadata}` |
| 本轮CPU/GPU证据 | `runs/recovery_sft_preparation_20261009` |
| 高层SFT父权重 | `runs/memlite_stage1_high_100task_v1/checkpoints/step_00048045_save_0027.pt` |
| 低层SFT父权重 | `runs/memlite_stage1_low_100task_v1/checkpoints/step_00098414_save_0021.pt` |
| 低层原TRAIN归一化 | `manifests/memlite-stage1-v4-action-bounds/stats.json` |

父权重完整hash与10383父模型相同，详见 `configs/recovery_sft/a800_p2_v1.json`。不能不传显式state就调用旧stage1初始化：那会回到B1500/A42500，而不是正式SFT终态。

本地小证据/媒体位于 `/home/wsy/behavior/artifacts/recovery-sft-prep-20261009`，不入Git。提交的是代码、配方、人审记录与摘要，不是数据/权重/视频。共享数据的落盘成功须看transfer回执，不用“rsync已启动”代替校验。

## 准入与采样

1. `prepare_recovery_corpus.py`全量验结构/源TRAIN manifest/高低父SHA，按task-instance分组并排除原SFT留出和开发保护组；物理记录`row[k]`描述`s[k+1]`，RGB锚点`k`只能使用截至`row[k-1]`的结果证据。
2. `check_recovery_admission.py`分别签发outcome、planner、action池。审批必须绑定精确sample/hash、原媒体和物理语义审查、事件ID及可用时刻；outcome通过不等于动作通过，动作审核要覆盖32步及终点。结果头每条样本绑定具体并行技能成员。
3. 初始工程数据门：train至少8个独立已审事件、dev至少4个，二者各至少2个来源实例组；outcome还需真实IN_PROGRESS/SUCCEEDED/FAILED覆盖，planner需真实纠正续段。这只是最低试跑门，不是校准或统计充分性的证明。缺类不造标签。
4. `VerifiedRecoveryActionDataset`只接受SHA绑定且ready的action池。真实pre-state61维、实际已apply动作23维与同帧JPEG从原ZIP读取；经原processor/normalizer成为27维，补齐位仍`[7,8,17,18]`。特权状态、reward、物理标签和review ID不进入模型samples。
5. `finite_mixture_schedule`约70%原专家/30%合格新动作，**每事件每遍最多一个锚点**，最多5遍；原专家跨任务混合。64全局batch实际为45专家+19新事件，尾批只按现有事件配比缩小，不复制填满。两类来源分别计数/评估；分布不同不只看合并loss。

未签发数据不能靠改`training_ready`绕过，也不能把这次75窗口抽帧QA当逐样本审批。现有候选保存的是片段，不是完整初始状态+全程动作：不能仅凭RGB/proprio恢复世界或保证回放到失败现场。

离线记录的`history_is_partial=true`必须保留：窗口内可见的同意图时长只是下界，不能冒充完整在线ledger计数。H1后续需要从同episode的完整planner日志重建既发意图/记忆/持续时间，并与context ID、父权重、真实时钟交叉绑定；缺失时应掩码或拒绝，不能补写历史。

## 训练配置与梯度边界

- **H0：** 冻结整个高层，member-conditioned答案前context与归一化本体输入最多4个过去/当前检查点；detach后进小GRU适配器和四类结果头。loss只更新适配器/头，低层完全不参与。并行成员需要不同条件prefix，不能复制同一个bundle向量假装分别判断。当前小验为每成员单独一次VLM prefill，**没有实现与正常规划共享prefill，不能宣称零额外推理成本**；部署前须实测或优化合批/共享表示。当前是独立模块+图验收，不是已部署或已校准功能。
- **H1：** 高层正确交接/恢复CE，初始LR `1e-6`、micro1/global32候选；使用分来源实例OOF或独立冻结observer的预测反馈，不能拿oracle结果直接当部署输入。新高层hash会使旧特征/校准失效。真实数据、OOF生成器及完整训练/部署整合仍是后续门。
- **L0：** 从low98414精确恢复，包括全部192个已训LoRA张量。`configure_recovery_expert_only()`保留其值但冻结，只开放322个动作专家张量；FM loss不回传VLM或高层。初始LR `1e-5`，micro4/global64候选；32步预测/16步执行/起点0，4噪声样本、原stats不变。不能重构无LoRA的Stage-A替代“冻结LoRA”。
- 每线最多1000更新、合格新事件5遍或4小时先到为止；不为凑预算重复一个小数据集。H0小头未必值得占8卡，正式world/microbatch须按实际loader和数据量验收；不要把原bs256强加给很小的恢复数据。
- W&B计划沿已可写team `hanhanyy-fudan-university-school-of-management` / project `behavior2026-g05`，新group `memlite-recovery-p2-20261009`。仅用0600的共享秘密路径，准备小验0optimizer不新建训练run；正式run必须验证真实写入，不能把配置存在称已经接通。

### 有限数据的八卡运行时

`src/g05/utils/training/recovery_runtime.py`提供独立的通用更新原语，尚未取代完整derivative trainer：

- 对已经准入的全局事件顺序分片，microbatch尾部不复制真实监督样本；空rank只做一个零权重前向，真实样本数/有效分母/事件遍数都不增加。仅适用于G0.5这类行间独立、无跨样本BatchNorm的模型。
- 各rank先累计`loss × 有效分母`反向，DDP平均后乘`world/global_denominator`，再clip与Adam；不能平均各rank的mean loss。零全局监督拒绝weight decay/更新。
- 每个更新前先协调读取/前向错误；任一rank异常均拒绝继续，已发生的optimizer异常必须从上一个完整checkpoint恢复，不把部分rank更新当作已提交步。
- schedule指纹绑定代码、父权重、admission、配方、world/micro及精确样本顺序。调用方仅在更新成功返回后推进游标，沿用已验的原子checkpoint和全rank RNG保存。

双CPU及八NCCL rank的小网络真实Adam/恢复测试已通过，3新单测＋原6运行时回归通过；父模型未参与这部分更新。**正式入口仍须先做数据准入，并将真实G0.5 loader/目标、预算监管、验证及在线W&B整条接合验收**，不能直接把这些函数或合成测试当作完成的训练程序。

## 环境/图验收命令

从自己的**干净冻结worktree**运行，先看GPU是否空闲；绝不对活跃源码pull。

```bash
source scripts/infra/activate_a800_training.sh
export PYTHONPATH="$PWD/src"
# 包含实际验证过的NCCL P2P/IB关闭和bond0，以及torchcodec所需NPP/FFmpeg库路径。
# 未source时即使torch可import，也可能在读专家RGB视频时失败。

CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=30s 900s \
  python scripts/rl/memlite_online/tools/check_recovery_parent_graph.py \
  --recipe configs/recovery_sft/a800_p2_v1.json --component L0 \
  --output /data/workspace/wsy/behavior2026/runs/recovery_sft_preparation_20261009/NEW_l0_graph
# 完成并释放GPU后，H0用另一个新输出目录单独运行，不并发占同卡。
```

以上读取两条已验原专家TRAIN样本，只做真实前反向，**无optimizer**。H0采用明确的合成图测试目标，不把数值当物理标签/loss效果。八卡硬件通信小验已通过，但单卡模型图通过也不等于新SFT八卡全链验收。

`prepare_recovery_sft_ticket.py`只产CPU准备票并列出阻塞，不启动训练。下一步须完成逐样本质量门、H0特征/校准、H1预测反馈/续段、完整derivative trainer的DDP/断点/W&B小验和固定短TRAIN-dev合法起点验收。不能以本页或空审批文件充作“所有准备完成”。

## 留给下一冻结RL的修复

- 采集器从真实env object_scope保存label-only精确实体映射；同对象同一手连续确认抓持，覆盖任务工具，不只看最终goal对象；少量正常抓持窗口补充失败偏置。双臂之一在PLACE不再掩盖另一只手的意外掉落。
- 缓存critic冻结前缀特征，分别计算optimizer.step前后value loss、return/value统计与EV；目标方差近零时EV为null。actor KL门不改，旧日志不回写。
- 因果反馈ledger按session/task/instance/episode/高层及observer SHA隔离；同意图持续时间/刷新次数不因重复规划清零，明确RETRY才开始新尝试；最多保留64种近期意图的重试计数。预测结果要求校准、足够置信度及至少两个不同检查点一致，否则UNKNOWN。

这些改动**尚未进入10383在训源**。原共享RL按原24h截止运行，不因本轮准备自动延长；本轮没有恢复1000条公共评测。
