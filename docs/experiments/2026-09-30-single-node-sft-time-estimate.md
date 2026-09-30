# 单节点八卡 A800：全数据一遍 SFT 的时间预算

2026-09-30 11:39（北京时间），Codex / PLAN-MEM100-TIME。**这是旧四卡 A100 日志的条件外推，不是八卡 A800 训练实测。** 本轮仅本地元数据计算、归档配置核查及 robo 日志只读检查，未启动训练、GPU benchmark、模型迁移或修改共享环境。

## 1. “全部过一遍”先定口径

固定官方 `behavior-1k/2026-challenge-demos@4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`。本轮重新读取本地全部 100 份 episode metadata 的 `length`：20,000 演示、210,916,774 帧。按每 episode 从第 0 帧开始、步长 s、允许尾部 mask，候选锚点数为 `sum(ceil(episode.length / s))`：

| 候选起点间隔 | 全部 20,000 演示锚点数 | 95% train 近似样本数 | global batch 32 近似更新数 |
| --- | ---: | ---: | ---: |
| 每帧，stride 1 | 210,916,774 | 200,370,935 | 6,261,592 |
| 每 16 帧，stride 16 | 13,191,664 | 12,532,081 | 391,628 |
| 每 32 帧，stride 32 | 6,600,830 | 6,270,789 | 195,963 |

- 95% 是数量估计，不是已发布 train manifest：按实例留出 5% 不保证恰好留出 5% 帧，正式数据还须排除未标注/不对齐区间、处理技能边界。不能把这些原始锚点全部视为已经合格的 MEM-Lite 样本。
- stride 16/32 是显式稀疏取窗的比较情景，**本轮未配置或批准这个改动**。稀疏窗口遍历不等于每个观察帧各作一次训练输入；预测 32 动作、执行前 16 动作并不会自动减少采样起点数。
- 六帧历史是每个样本的输入，四次 FM 噪声抽样是其内部计算；都不算四/六个新的独立观察样本。
- A4 的 `episode_round_robin_v1` 优先覆盖 episode/技能，各 leaf 内做 rank 分片的置换循环，不是简单的全局不放回遍历；短 leaf 可以先重复。因此 `len(dataloader)`、抽够 N 次或每条演示至少抽过一次，均不能宣称全部 N 个窗口唯一覆盖。本文是等量抽样预算；真正一次全覆盖需冻结有限索引并审计覆盖。

本地证据：`artifacts/memlite-100task-design-20260927/episode-meta-000.parquet` 至 `episode-meta-099.parquet`；原版本及聚合 SHA 见 [9/27 分组证据](2026-09-27-memlite-100task-partition-draft.json)。

## 2. 两套真实旧配方，不能混算速度

### A4：计划沿用的六帧 MEM-Lite 低层

- robo 四张 A100 80GB PCIe；三相机×六帧历史，SkillFM、AE＋LoRA，microbatch 2/GPU、累积 2，全局 batch 16。
- 2,500 次更新、40,000 次样本抽取。首 forward 为 2026-09-12 15:48:57.161397 UTC，最终 checkpoint 保存完成为 23:39:05.435 UTC。
- 两者间 28,208.274 秒，即 7.836 小时，平均 **11.283 秒/更新、1.418 样本/秒（四卡合计）**。包括这段期间的周期诊断、保存及其他等待；不是独占 GPU 的纯计算微基准。
- 500 步对照的另一路纯 FM 记录也给出 8,000 draw / 5,725.812 秒 ≈1.397 样本/秒，说明 A4 并非仅启动阶段异常拉长。两者均不能代表优化后吞吐上限。

原始证据：

- `robo:/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/logs/model_debug_rank0.log`（首 forward）。
- 同 run 的 `train.log`、`coordination_run_receipt.json` 与上层 `formal_plan.json`（终点、四卡、microbatch/累积、完成状态）。
- [M-04 机器记录](results/2026-09-14-joint-ki-screen.json) 的 `training.fm_action_control_v3.last_row`。

### 普通 G0.5：单帧、无 MEM-Lite 的历史参照

- 本地 100k run 保存配置：四 rank、microbatch 16、accumulation 1，即 global batch 64；`obs_size=1`，离散动作 CE＋连续 FM，`fm.joint_training=false`，每观察四个 flow samples。
- 只取最后一次续训的连续区间 99,000→100,000：日志 2026-09-27 20:14:35.382→21:11:28.071，共 3,412.689 秒，平均 **3.413 秒/更新、18.754 样本/秒（四卡合计）**。
- 区间包括周期 eval/数据等待，不包括 100k 最终 checkpoint 保存；不要把早期重启/空档一起平均，也不要按 run 名里的 `bs8` 覆盖实际保存配置里的 16。
- 这不是 A4 的同配方性能对照。历史帧数、microbatch、可训练参数、梯度路径、实现和资源并发均不同；不能据两者相差约 13.2 倍就断言“MEM-Lite 理论上必然慢 13.2 倍”或“删掉记忆就会加速这么多”。

本地证据：`artifacts/g05-100k-eval-20260928/source/{train.log,training-config.yaml}`。

- train.log SHA256：`54492f395e8aa3b2947e44dc21ab4dda1eabef452ac3d2fa61adf2934a583808`。
- training-config.yaml SHA256：`c9ab6bf23f9aa3a69c068a92747fd42f4ee6eb42c428dd35dd19390e51444ec5`。

## 3. 八卡条件外推

先作明确的理想假设：保持每 GPU 配方与效率，八卡吞吐等于原四卡的两倍，且新版数据加载不新增瓶颈。于是 A4 约 2.836 样本/秒，普通 G0.5 约 37.507 样本/秒。

`天数 = 训练锚点数 / 八卡实际样本每秒 / 86400`。

| 95% train 口径 | 沿用 A4 六帧低层 | 普通单帧 G0.5 参照 |
| --- | ---: | ---: |
| 每帧分别作观察起点 | 817.7 天（约 2.24 年） | 61.8 天 |
| 显式 stride 16 | 51.1 天 | 3.87 天 |
| 显式 stride 32 | 25.6 天 | 1.94 天 |

若按字面使用全部 20,000 演示、完全不扣 holdout，分别再乘 `1/0.95`：A4 为 860.8 / 53.8 / 26.9 天。这只是说明统计口径，**不建议把留出集回灌训练**。

这是旧实现的算账锚点，不是 A800 性能上限，也不是承诺耗时或统计置信区间：

- A800 节点无 NVLink，已测 NCCL 需要 `NCCL_P2P_DISABLE=1`，八卡未必线性翻倍；但独占资源、更大 microbatch、合适后端与数据管线优化也可能明显好于旧实验，不能单向当作保守上界。
- lc1/2 读共享 NFS、lc3 读本地盘；百任务随机视频解码与旧五任务数据工作集不同。尚无正式模型的八卡端到端吞吐。
- 全局 batch 增至 64 会让更新数减半，但若仅靠累积倍增，每步计算也增加，不会让总时间自动减半。
- 按旧 A4 理想八卡/global32，30k 更新约 3.92 天、50k 约 6.53 天；这是 0.96m/1.6m draw，只约等于 stride16 候选 train 数的 7.66%/12.77%，不是完整 epoch，更不是已经收敛。

## 4. 高层要另算

官方 `skill_summary.csv` 实际 35 类、合计 406,341 条技能段。若**仅假设**每段产生一个有效高层决策，95% train 约 386,024 个样本；并行技能可能要合并，一段也可能产生多个检查/记忆样本，故这不是已构造的高层训练集大小。

旧 B-parent-format：四卡 micro4 / accumulation1，1500 更新即 24,000 draw；pretrain 日志 2026-09-10 09:27:48 至 10:42:32 UTC 共 4,484 秒（含初始化/保存），四卡约 5.35 样本/秒。照同样理想翻倍假设、每段恰一决策，95% 一遍约 **10 小时**。新高层 token 长度、决策密度、监督反馈样本未冻结，不报确定工期；不能把低层 2.109 亿帧直接套到高层。

证据：`robo:/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/formal_b_parent_format_1500_v1/logs/pretrain_log.txt`；本地 `artifacts/local-archive-20260912/root/memlite-resume.L6rqZ6/b_parent_format_config_preflight_v1.json` 中真实 launcher micro4/four-rank，`b/configs/model/g05.yaml` 默认 accumulation1；全量正式 high manifest 仍未生成。

## 5. 开训前的实际决策

不要直接把五任务诊断用的小 batch 配方照搬成百任务正式长训。先完成既有代码/数据准入，再登记约 100–200 更新的八卡端到端基准：分离数据等待、模型前反向、通信、保存/验证，报告实际观察样本/秒和峰值显存。正式数据需要按任务/技能/事件定义覆盖，而不是只看 epoch 字样。若选择 stride16，接触/闭爪/释放/技能切换短事件应单独核覆盖；其加密会改变上表样本数，不能仍冒充同一 epoch。

基准、抽样策略变更和正式训练均是下一步建议，**本轮未启动**。原数据/RL 分工、holdout 和训练授权边界保持。
