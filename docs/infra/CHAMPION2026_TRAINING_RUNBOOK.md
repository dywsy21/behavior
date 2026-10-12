# 冠军权重 → 2026：训练链与操作边界

当前是实现/验收手册，**不是新大训练授权**。默认
`configs/champion2026/shared100_v1.json` 的 `formal_launch_approved=false`，
正式入口会在访问 GPU、W&B 和创建输出前拒绝。用户确认配方后再记录授权、
commit/push，并创建新的冻结 worktree；不要在现有运行目录修改这个开关。

## 数学与数据

- 父权重：公开 shared50 intermediate，revision `114902410a6439ad1a0b58a112b84159d3ab3f65`，不是最终四专家之一的 cp2。
- 保留原 50 个 task / 596 个 stage embedding 行；扩为 100 task / 1087 stage，其他原参数完整继承。真实动作 23 维、模型补齐为 32 维，不能沿用 G0.5 的 27 维填充位置。
- 作者原生 FM + 0.05 FAST CE + 0.1 stage CE，4 个 flow 噪声采样，不冻结视觉/语言参数，不开启 KI。保持原归一化、FAST codec、相关噪声、30 步动作目标。
- 原 TRAIN 准入 18,895 演示、stride16 共 12,443,010 原生锚。固定每视频原 phase；105 TRAIN 与 6 eval 整源隔离继续有效。MEM 技能边界不截断冠军原生动作目标。
- 每全局 batch256 都有全部 100 task，各 2 或 3 个不同锚；8,000 批共 2,048,000 唯一样本，各任务 20,480。按任务无放回，预取不能推进优化器游标；恢复后下一批身份 SHA 必须相同。
- AdamW、梯度 clip1、EMA0.99。旧参数 peak LR2.5e-5，新追加行在 **Adam 更新之后**乘4；warmup300、cosine 到8000步时的10%。这不是把梯度乘4，也不是完整过一遍数据。

## lc1 单节点八卡运行

共用根 `/data/workspace/wsy/behavior2026`，独立环境
`envs/champion-openpi-py311-v1`；不要更改 shared env、队友 lc2 或重新登录 VPN。
每次核实八卡确实空闲，固定源码及登记补丁。激活后必须显式增加确定性 flag：

```bash
source scripts/infra/activate_champion_a800.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD/src"
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_exclude_nondeterministic_ops=true"
```

入口为 `scripts/train/train_champion2026.py`；必填 `--recipe`（同一冻结源码里的
上述 JSON）、`--output`（新 run）及 `--credential-file`（现有 0600 私密文件，
绝不把内容放入命令/日志/Git）。W&B 固定已有 team 的 `behavior2026-g05`，仅 online。
`--engineering` 是另外一个 3 步/30 分钟验收模式，产物明确不可部署；
不把它当作长训练开关或绕过批准的方法。

24 小时从首次启动起计，**包含断线/暂停间隔**；恢复不会重置该时间窗。
每1000步及正常退出前保存。入口在加载模型前自动检查整个剩余保存计划：
实测全状态54.49GB，保守按56GiB/份；8000步保留8份，加1份任意步紧急保存、
15GiB EMA和256GiB余量，新run至少需要775GiB空闲。恢复只计算余下保存，
已有文件继续占实际磁盘；每次保存还有独立实时空间检查。系统不自动删除旧 checkpoint。
SIGINT/SIGTERM 只请求停止，当前更新结束后完整保存；不要第二次强杀保存进程。

## 保存、恢复、评估、导出

- `checkpoints/step_XXXXXXXX`：参数、Adam、EMA、相关噪声等 Intermediate、随机键、step/weak_type、采样游标及冻结运行合同。逐叶 SHA、实际 HOST→各物理分片回读都检查。只有 manifest 完成并原子提交的目录可恢复；`.incomplete-*`保留现场但不可恢复。
- `latest-checkpoint.json`：当前 run 的已提交恢复点。`--resume-latest` 仅恢复同代码/数据/环境/配方、同 W&B run；不得从另一个实验冒充继续。非有限更新等失败需诊断，入口不自动 retry。
- `events.jsonl`、`sampling-journal.jsonl`：记录真实完成步数、耗时、loss、批身份和 task 计数。云端 append history step 与优化器 step 分开，避免恢复旧 checkpoint 时云端静默丢日志。
- 固定原 eval：100 task 各16、共1600，旧/新50各800。每1000步、初始和末尾做 EMA forward，时间/停止条件不足时不伪造结果。无 augmentation，独立固定噪声；stage CE 单独计算，因为作者 `train=False` 原接口漏返回该项。逐任务与两组 FM/FAST/stage/总损失分开报告；不把它当作 Q/SR。
- `exports/ema_XXXXXXXX`：作者 Orbax `params/` + `assets/`，附 task-layout 与 manifest；每个 EMA 数组匹配其完整 checkpoint，导出后用作者 `restore_params` 逐字节回读。推理仍会按作者路径转换 BF16，此格式验收不等于物理控制成功。

## 尚未通过就不能写成“已就绪”

2026-10-12 已证：全部8000批CPU采样、W&B同run在线恢复、八卡默认scatter
非确定性与确定性微核、整模型同进程和新进程/新engine恢复后下一步全部302叶精确一致、
真实原留出1600条EMA前向有限且全覆盖、全模型工程EMA作者loader逐数组回读，60项CPU测试。
证据见[工程验收回执](results/2026-10-12-champion-training-chain.json)。这些不是正式训练效果。

新推理入口支持`formal_ema_v2`：完整文件SHA账本、100任务映射、训练/推理模型配置一致，
拒绝工程checkpoint和未登记参数。旧公开父权重路径有兼容回归。
实际全模型回读的是EMA v1；后来补文件账本的v2只做了CPU小模型回归，不能称v2物理验收已过。
还需：新正式权重的2026物理/提交容器验收、成员独立审查、用户批准新大训。
长时间CLI＋在线W&B尚未整体运行；数学/保存/恢复/评估engine和W&B生命周期是分别实测的。
公开 cp2 的首个完整 DEV radio 回合仍是 Q0；不以工程通过掩盖这个效果问题。
