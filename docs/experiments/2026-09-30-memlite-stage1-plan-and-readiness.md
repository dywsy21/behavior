# 阶段1：高层一遍、低层120小时——方案与开训审查

> 20:53北京时间实施后更新：本文保留18:08的审查原始证据，不代表当前仍缺S1–S9实现。正式链已在独立feature补齐，v4数据发布、lc1高16步/lc2低32步及八rank保存恢复通过；源码d0528b4、[最新操作手册](../infra/MEMLITE_STAGE1_RUNBOOK.md)和[结果JSON](../infra/results/2026-09-30-memlite-stage1-acceptance.json)。没有启动正式长训，合main前团队独审仍待；节点和实际排期以新手册为准。

2026-09-30，Codex / REVIEW-MEM100-STAGE1。审查基线`ce77988`；相关源码与已测`4e59b5b0e34030907c7e6b87b7134f832f01ac14`一致（其后差异仅文档）。

**结论：八A800的模型计算图通过，但百任务正式训练链未通过准入，不能直接开长训。** 本轮是方案和最终检查，没有改变训练代码、重建百任务标签、启动长训或发布模型。以下明确区分实测、建议及未完成实现。

本方案覆盖[原百任务设计](2026-09-27-memlite-100task-training-design.md)中的旧“两遍/60小时/三组专家优先”等预算。阶段2/3方向不变：冻结高层做语义合法的低层适配，再基于跨任务真实失败轨迹做反馈/恢复SFT与通用RL。

## 1. 阶段1安排

主要假设：在完整100任务上，分别训练规划/记忆与技能条件化FM，比继续小任务集重复拟合更适合作为后续协同与恢复的共同起点。不是在本阶段证明自动恢复已经学会。

| 项目 | 高层H | 低层L |
| --- | --- | --- |
| 建议节点 | lc1，单节点8×A800 | lc3，单节点8×A800；数据本地盘，优先给更高吞吐的低层 |
| 初始化 | B-final model-only，950状态全部恢复 | A4 `step_2500.pt`，1138状态/192 LoRA状态全部恢复 |
| 更新范围 | planner VLM、原可训练投影/本体模块；冻结视觉、动作专家、结果头 | 动作专家＋原r8/alpha16/dropout0.05 VLM LoRA；其余冻结 |
| 监督 | 真实技能/规划/记忆CE；memory token权重0.25；无物理证据的outcome/终止字段mask | 真实执行技能作为条件的FM；保持全部23控制维，不做动作token CE |
| 有效batch | 256 = 8卡 × 每卡4 × 累积8 | 256 = 8卡 × 每卡32 × 累积1 |
| 数据预算 | **每个已发布高层TRAIN候选恰好一次** | 循环已发布低层TRAIN候选；**累计作业运行120h或200,000次更新，先到为止** |
| 完成标准 | 发布清单全部覆盖，含尾批；不因估算步数取整漏样本 | 截止前保存可恢复状态；记录实际遍数、唯一覆盖和当前遍游标；不强求四遍 |

节点只是建议分配，本轮仅核lc3八卡空闲，没有占用lc1/2或给它们排队。两节点各自DDP、没有跨节点梯度同步，也不是一个global512模型。lc2留给数据验收、有限评测/后续准备，不为填满算力另开搜索。高/低层训练实现与集成负责人Codex；数据/纠正轨迹和通用RL维持团队原分工，见[任务板](../TEAM_PLAN.md)。

### 输入、超参和日程

- 单帧三相机256²，三路严格同一逻辑时刻；未来32步连续动作、执行前16、动作起点0。stride16稀疏的是**观察起点**，不是把动作降采样成每16帧一个。高层输入包含有界因果记忆；不能在跨batch隐状态中延续另一条演示。
- 每条逻辑episode固定一次`r∈[0,15]`，候选为`r+16k`。采用已计数的seed17与来源身份哈希；三相机、高低层共用相位，低层下一遍仅改shuffle顺序、不改r。高低层各自按监督合法性过滤，不能假定其最终样本数完全相同。阶段1不偷偷插入额外边界样本改变用户指定网格，短技能覆盖不足先报告。
- 高层峰值LR `1e-5`，warmup建议1,000更新，按最终发布清单计算实际总更新数（目前约48,920），随后cosine至`1e-6`。高层实际样本显著少于估计时，warmup改为总更新约2%，不机械保留1,000。
- 低层AE与LoRA各`1e-5`，分别记录参数组；warmup 2,000更新、cosine按200,000更新至`1e-6`。120h通常先到，提前终止时记录实际LR，不伪称已经走完日程。新阶段fresh optimizer/scheduler；同run续训才恢复旧状态，不从A4旧step2500的LR日程续算。
- AdamW `(0.9,0.95)`，clip1，BF16 autocast/FP32参数和optimizer，gradient checkpointing，no EMA。高层wd0.03且norm/bias不decay，沿高层已测分组；低层若严格沿已测配方，其AE/LoRA可训练参数均wd0.03，**低层测速使用平铺参数，不把它说成已测过norm/bias免decay分组**。正式参数组应保存完整名字/数量回执。
- FM保持原时间约定、Beta(1.5,1.0)、每观察4组噪声、`fm_weight=1`；不同时叠加KI、扩LoRA rank或改执行段加权。4组噪声不是4份不同观察。
- 高层已测DDP `find_unused_parameters=false`，低层为true（单帧不执行历史相关LoRA支路）；128MiB bucket、gradient-as-bucket-view。低层载入192项LoRA而实际该单帧路径有182项梯度是已验合同，不将10项历史分支无梯度误判成恢复失败。高层326项可训练张量有梯度；冻结组无梯度。
- 按现有集群证据保留`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=bond0`；默认不强制算法、不改共享env。启动时检查实际import目录、8个不同LOCAL_RANK和GPU归属，独立冻结Git源码。

低层120h解释为**累计作业运行墙钟**：初始化、读取等待、验证、保存都算；恢复不重置预算，离线排队/停机不算。rank0统一广播截止状态，在更新边界一致停机，并在预算内预留实测保存耗时的余量；不能简单`timeout`杀进程后指望存在新checkpoint。此预算机制目前未实现。

## 2. 时间与开销

来源：[八卡实测](2026-09-30-memlite-oneframe-a800-benchmark.md)、[高层JSON](../infra/results/2026-09-30-memlite-high-batch256.json)、[低层JSON](../infra/results/2026-09-30-memlite-batch256-stride16.json)。

- 固定相位全候选13,182,390/遍；简单乘95%得到约12,523,271。这是**数量外推**，不是已发布TRAIN索引；按实例切分、标注合法性和边界过滤后须重算。
- 高层29.19–31.41观察/s：上述约95%候选**一遍110.76–119.16h纯计算**。可先预留约5–6天工程排期，但不是保证；100任务新文本若比旧五任务缓存更长，时间/显存还可能增加。
- 低层120.292观察/s：120h完全用于计算时约51.97M观察/202,993更新/4.15遍。方案另有200k更新上限；真实120h包含非计算开销，通常达不到这一理想值。
- 仅作开销敏感性示例：若有效计算占墙钟80%–90%，约162k–183k更新、41.6M–46.8M观察、3.32–3.73遍。**这不是实测loader效率，更不是至少三遍的保证。**
- 高层“一遍”与低层“120h”是不同停止条件。高层有数据开销可能超过120h；总阶段1结束以二者都完成为准，不为凑同时结束偷偷把高层截断或低层续期。
- 旧I/O探针热轮约160窗/s只在单作业、特殊读取顺序下测得。高低层目标合计约150窗/s，共享盘并发余量不宽；尤其高层在lc1走NFS，**必须在正式loader到位后测两节点并发数据等待**，不能把网络9.4Gbps等同训练吞吐。

### 监测和保留

- 固定分层验证：每task32窗，共3200窗；高层每1000更新、低层每2000更新做一次，记录真实验证耗时。每遍结束/阶段结束可扩大到每个留出来源的预登记窗口，不声称这是全部留出帧。
- 高层报告总CE、去memory/各字段CE、技能/结构错误、task宏平均；低层报告固定噪声FM、每task/技能/控制组及有效scalar数。eval用原未改目标，不因训练重采样或mask分母变化制造下降。
- 高层每2000更新、低层每5000更新保存完整状态；保存耗时纳入预算。run内预先约定保留最近两个可恢复点、最佳验证点和每遍终点，旧实验/唯一权重不自动删除，先核剩余盘与保留清单。
- 非有限loss/梯度、来源越界、mask/任务组成/参数恢复合同失败立即停；验证出现持续异常退化时保留现场并复核。best checkpoint和final checkpoint分开记录；CE/FM降低不代表完整任务成功率。
- W&B只读取安全配置的凭据，不进Git、配置回执或日志；每层单独run，绑定同一数据发布ID和源码commit。

## 3. “每个batch都混任务”的精确定义与当前答案

**当前不能确认，正式入口还没有实现该保证。**

`scripts/finetune.py:862`若未启用特殊sampler，会在`:896`使用`ResumableDistributedSampler`：它全局shuffle后按rank切片，不看task身份。shuffle可产生混合批，但不是“每批必须含多个task”的约束。启用旧factory也不能解决新合同：`src/g05/utils/common/task_event_sampler.py:63`只接受branch/task_event/motion_recovery/fm_motion，实际调用`strategy=coordination_v6`会抛ValueError。现有task_event的task等权循环还会重复小池，不能充当完整一遍。

CPU可复现反例（**合成数据，不是观察到真实100任务坏batch**）：64索引，前32属task0、后32属task1，world8/micro4/seed17/shuffle=true；rank1第二个microbatch为`[62,48,50,44]`，四个都task1。这证明普通shuffle本身没有每卡混合保证，不证明当前百任务global256经常单任务。

需要实现/验收的合同：

1. 先冻结完整合法候选清单、来源split和固定相位，再在全局生成**无放回、task-aware**顺序；按候选自然总量覆盖，不能为了每task等权把短任务重复几次，同时漏掉长任务后半段。
2. 每个有效global256批至少两个不同task；进一步约束每卡完整microbatch（高4、低32）也至少两个task。不能要求micro4含100任务；global批不硬塞每task相同数量。通过预先分配/保留不同task的尾部候选防止短任务先耗尽后长任务独占收尾，而不是末尾重复小任务凑数。
3. 任务内打乱episode/合法anchor，尽量避免同episode相邻窗口挤在一起；从同一全局计划切8rank/累积片，禁止各卡独立抽签形成重叠。高层八个micro轮次组成一次256观察更新。
4. 尾批不`drop_last`，也不把DistributedSampler填充的重复样本计为新监督。可重分配最后批次、用显式零权重填位保持collective一致；若最后有效数据不足以满足micro级混合，须重排前一批/明确失败，不能悄悄撤掉保证。记录尾批真实有效batch，而不是始终记256。
5. loader返回的实际(task,episode,frame)必须等于计划身份。坏文件/缺标签当场报告，不随机换样本。整个epoch集合等于发布清单，覆盖1.0、有效重复0、8rank交集0。
6. 全清单CPU审计每一个batch和rank微批任务数、来源/技能覆盖、重复及尾批；训练首批/每个周期记录可核对摘要，运行时继续断言。采样审计ID只进入外侧回执，不进入模型文本。
7. 恢复从**已完成optimizer update**对应的消费游标重建相同顺序；保存seed、epoch、发布清单SHA、phase规则、RNG与预算。预取出来但没训练的样本不算消费。

## 4. 最终代码审查：明确阻塞项

这些不是“A800性能差”，而是目前Git只迁入了测速需要的模型子集，没有迁完正式数据与训练编排。优先级P1表示会阻止正确长训；不能用通过的旧单元测试掩盖新入口缺失。

| ID | 级别 | 证据与影响 | 放行条件 |
| --- | --- | --- | --- |
| S1 | P1 | `configs/data/behavior2026_r1pro_rgb.yaml:30`仍指外层cache；lc3实际该根无`meta/info.json`，嵌套snapshot有 | 正式高/低配置固定真实snapshot、全部输入身份；不靠临时shell覆写隐藏错误 |
| S2 | P1 | `task_event_sampler.py:63`拒绝coordination_v6；默认shuffle不保证混任务，固定stride脚本只是计数器 | 新的无重复混任务固定相位sampler实际接DataLoader，完整epoch审计通过 |
| S3 | P1 | `scripts/finetune.py:600`与`checkpoint_utils.py:562`通用恢复未调用低层remap/LoRA post-load，也未调用两层`configure_coordination_trainability`；模型前向明确要求该gate。测速在各自restored函数手动完成这些步骤 | 正式入口严格恢复950/1138状态、192 LoRA，先冻结再DDP/optimizer，核全部参数组和实际更新 |
| S4 | P1 | `scripts/finetune.py:1402`只做`loss/accum`；变长高层token分母/低层边界有效action分母不同，会平均各局部均值，而非全局目标 | 正式8rank+累积梯度与单进程同一global batch逐项对照；CE按有效权重、FM按有效scalar归一，零监督批不偷偷decay |
| S5 | P1 | 当前`memlite_sidecar.py:55`为v2/v3/v4字段归一，`base_lerobot_dataset.py:1160`仍旧intent覆盖；独立`memlite_v6_projection.py`保护存在但不是完整loader接线。`SKILL_MAPPING`仅15项，当前官方summary35项；新20项会UNKNOWN且SkillFM拒绝 | 补齐正式v6单帧builder/overlay、35技能/双手关系/高层task父目标格式，保证高低标签同状态，不造物理反馈标签 |
| S6 | P1 | 未发布100任务MEM-Lite标签、训练索引和旧新来源holdout对照；普通task-tail切分不是旧五任务来源保护。已有metadata长度与标注duration非统一差值 | 原始RGB/23D动作/标注逐episode时钟对齐、保护split、train-only norm验收、本人分层人审后冻结发布 |
| S7 | P1 | `base_lerobot_dataset.py:1276/1293/1317`坏样本/无标签会随机换索引；sampler计划正确也不能证明实读正确 | 正式路径禁止替换；实际身份回执与计划逐项相等，坏例修复/隔离后重新发布清单 |
| S8 | P1 | `scripts/finetune.py:1039`按loader长度整除accum估epoch步数，`:1391`只整组更新；263候选在8rank默认sampler被补成264、每rank33，micro4/accum8只有首256参与这次“一遍”更新 | 尾批实权重归一且完成更新；不遗漏、不有效重复，每个合法候选一次 |
| S9 | P1 | 当前checkpoint保存optimizer/scheduler/游标，但未在入口保存/恢复各rank RNG、清单身份、累计预算；无120h一致停止；`:741`直接torch.save最终名 | 原子保存/发布、保存失败保旧点；8rank中断续跑与未中断对照后续样本及状态一致；累计预算恢复不清零 |

补充S4的CPU数值反例：8rank×8累积，共64个局部批，交替设置有效权重1/9、损失梯度10/1；旧局部均值平均梯度5.5，全局权重目标梯度1.9。不是浮点尾差。低层当前全32有效、相同micro的测速不会暴露此问题，但正式含技能边界/尾批时必须检查。**上一轮修好的Liger非均匀CE反向问题与此是两件不同的事**：前者是局部loss后端，本项是跨rank/累积的归一。

## 5. 本轮实际验证与后续最小路径

实做：

- Git fetch/pull成功（首次TLS失败后重试），main仍33677bd；独立审查分支从main纳入ce77988，没有改其他工作树。当前无active goal，不重开历史等待作业。
- 复用已有VPN，仅新建本任务SSH连接；没有重登ec。lc3八卡0MiB，4e59b5b源码clean，shared env不改，0GPU训练更新。
- 本地高层算术5/5过；本地模型测试缺OmegaConf所以未收集成功，转到原A800已配环境，未在本地/共享环境安装包。
- A800原冻结源：11个相关测试文件**105/105通过，5.75s**，含高/低模型合同、CE、采样、checkpoint、train-only统计及数据完整性工具测试。命令如下；不将旧行为单测通过等同本报告S1–S9已经修复。
- 实际factory拒绝coordination_v6、普通shuffle单task微批、非整除尾批与加权梯度反例都已在CPU复现。服务器确认真实data根和35技能表。机器可用、计算图可用，与正式链可开训分开报告。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q \
  tests/test_memlite_high_benchmark.py tests/test_memlite_high_runtime_contract.py \
  tests/test_memlite_oneframe_benchmark.py tests/test_memlite_training_route.py \
  tests/test_memlite_hl_end_template.py tests/test_memlite_conditioning.py \
  tests/test_memlite_branch_sampler.py tests/test_memlite_task_event_sampler.py \
  tests/test_training_checkpoint_finalization.py tests/test_train_only_stats_sampling.py \
  tests/test_infra_dataset.py
```

进入正式训练前的顺序（待实施，不是本轮已开始）：

1. Codex整合正式单帧trainer/严格恢复/归一/采样与保存停止机制；数据负责人提供合格100任务split/标签/归一清单，接口同步，不能用旧30行缓存凑齐。
2. 全候选纯CPU采样覆盖验证＋100任务真实getitem抽查；本人图像/标签分层审查与代码独立审查。没有通过的技能/任务明确阻塞，不删掉后仍称全100任务。
3. 两拟用节点各做有界正式入口工程试跑：建议每层至多64次临时更新、每节点GPU墙钟≤60分钟，覆盖新任务最长输入、真实loader、梯度/边界、保存回读与中断恢复；两节点同时读取测NFS等待。该预算是后续建议，本轮未启动。
4. 用上述真实吞吐/最终候选数更新排期，再冻结源码/配置/manifest/SHA启动阶段1。合main前仍需另一成员独审，不能把本次本人复核叫独立review。

没有这些证据，最终状态只能是**NOT READY**；不能向团队发“8×A800完全适配，直接开120小时”的确认。
