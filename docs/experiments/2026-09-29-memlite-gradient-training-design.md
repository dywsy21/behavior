# MEM-Lite高低层怎样训练：损失、梯度与协同顺序

2026-09-29，Codex / PLAN-MEM100-GRAD。补充[百任务训练提案](2026-09-27-memlite-100task-training-design.md)。这是代码核查后的设计说明，不是新训练/环境修改授权。只读本地源码及既有归档，没有连接robo/A800、重连VPN/ec cli、运行模型或修改训练代码。

## 1. 结论与三个不同的“一起训练”

建议：**高低层参数分开、SFT可以并行；低层内部AE与LoRA联合；高层内部规划与记忆联合，反馈头先单训再择机联合；最后通过真实闭环数据交替协同。** 不把FM穿过离散技能文字的端到端反传作为首轮方案。

- 同时开两个训练作业：资源安排，不会自动让梯度跨模型流动。高层先于低层训练完成也不是正确技能BC的前提。
- 同一模型内部多目标联合：例如高层规划/记忆CE，或高层CE＋outcome分类；共享参数上梯度相加。
- 高层生成技能→低层动作→物理结果的端到端训练：当前字符串/整数token/解析边界不支持普通FM反传。需要显式改成可微接口，或设计策略梯度等信用分配；不能只写总loss相加。

当前高层、低层是两个独立policy实例。从同一G0.5初始化不等于共享同一组可训练参数。`fm.joint_training=true`只控制**低层VLM的KV是否detach**，不是高低层联合开关。

## 2. 参数归属（按实际后期实现，而非旧v11残差模型）

| 符号/部件 | 参数范围 | 推荐首轮状态 |
| --- | --- | --- |
| `H` 高层上下文与生成模型 | 高层`model.vlm.*`；存在的`model.multi_modal_projector.*`、`model.proprio_embedder.*`也在`planner_vlm`组内 | 规划SFT更新；沿用B-final完整planner组，不误称只有LoRA |
| `O` 结果判断头 | `outcome_head.*`，LayerNorm＋四分类线性层 | 先冻结；有合格结果数据后独立训练 |
| `L` 低层语言视觉适配 | 低层`model.vlm.*lora_*`，A4为r8 | FM更新；原始VLM权重冻结 |
| `A` 低层动作专家 | `model.action_expert.*`，包括动作输入、时间编码、注意力/MLP和输出投影 | FM更新 |
| 其余低层模块 | 视觉塔、原VLM参数、projector/proprio编码 | 保持A4冻结合同 |
| 高层不使用的动作专家及视觉塔 | 高层policy内对应模块 | 冻结 |

`memory`、`intent`/当前技能、`memory_update`是数据/输出字段，不是同名的一组`nn.Parameter`。一个高层VLM和LM head完成多字段自回归预测。表中“更新某组”表示有可达梯度，不保证组内每个张量在每个batch都有非零梯度。

## 3. 高层AR：规划、记忆及有依据的状态文字

高层输入为当前及过去的图像/本体、任务、已有记忆、上一技能和可部署反馈。输出按固定schema包含父目标、当前技能bundle、EXECUTE/RETRY/REPLAN/STOP、记忆更新等。用teacher forcing，即每个待预测token的上下文含其之前的正确回答token；没有把未来图像喂给模型。

对监督位置j，`ell_j = -log p_H(y*_j | observable_context, y*_<j)`。把答案token分为规划、记忆、可验证结果/完成状态与结构字段，可以分别报告CE，但**这些通常是一次AR forward/一个CE目标的不同位置，不是独立的网络头**。

旧B-final实际目标为：

```text
L_AR = sum(valid_j * weight_j * ell_j) / sum(valid_j * weight_j)
weight_j = 0.25  对memory_update字段
weight_j = 1.0   对其余有效监督字段
```

因此它不是“各字段先分别平均，再L_plan + 0.25 L_memory”。假设有效规划/记忆token数分别Np/Nm，两个简化字段的实际系数是Np/(Np+0.25Nm)和0.25Nm/(Np+0.25Nm)。字段独立归一是另一种候选目标，需明确实现/对照，不能冒充已运行配方。本提案首轮优先保持旧token级目标并分字段报告。

梯度分别为：

```text
规划字段CE → 高层LM head → 高层VLM → 可训练的高层上下文编码器
记忆字段CE → 同一个高层LM head → 同一个高层VLM → 同一上下文编码器
有证据的结果/完成文字CE → 同上；不直接经过独立outcome head
```

它们不更新低层LoRA或动作专家。高层视觉塔冻结，不能把“梯度到达视觉相关表示”说成“视觉塔权重也训练了”。回答中的状态文本与独立分类头不是一回事；状态文字CE只教模型写出状态，不能代替分类头监督/校准，更不是官方任务成功判据。

旧planner-only把物理outcome、未经核验的task_complete等位置屏蔽，不给全UNKNOWN/false补伪标签；B-final不构造outcome head forward/loss，头也不在optimizer中。

### 3.1 记忆没有跨真实时间自动BPTT

本次`memory_update`可以学会整理已有历史；但生成文字后写入状态、下一次重新tokenize，这个历史边界没有连续计算图。后一次CE不会通过上一轮采样字符串直接回传到上一轮生成过程。需要长程记忆行为，就要构造因果、多时点、包含真实偏离/修正历史的训练样本，而不是假设“多次调用同一个VLM”已经等价于贯穿整条轨迹的反向传播。

同一条teacher-forcing序列内部，后面字段的loss仍可通过注意力更新前面token的表示及共享参数；按字段分loss不等于梯度隔离。输入token的`labels=-100`仅取消该位置的直接CE，并不会阻止它作为上下文参与其他位置的梯度。

## 4. 反馈/结果分类：先固定高层训头，再有条件联合

取答案开始之前的因果上下文隐藏向量`h_ctx = H(observable_context)`，分类头给出IN_PROGRESS/SUCCEEDED/FAILED/UNKNOWN。禁止取teacher-forcing答案之后的hidden来分类，否则会读到真实结果答案。仿真物理状态可以给训练标签，但不能进入这个可部署输入。

```text
L_out = sum_i valid_result_i * CE(O(h_ctx_i), result*_i)
        / max(1, sum_i valid_result_i)
```

没有物理/人工依据的样本`valid_result=0`；缺标不是一个“确定UNKNOWN”训练例。归档C1仅接收有证据的前三类，缺标保留UNKNOWN语法且mask=false；未来若监督真实不可判定的UNKNOWN，须另行定义可靠标签合同，不能把缺失类别自动补齐。

### 4.1 热身阶段：只训练O

冻结一版已经训练好的高层`H_k`，执行`O(stop_gradient(h_ctx))`。只有分类头更新；高层、低层、动作专家都不动。这在归档C1 `high_outcome_only`路径已有代码，但不代表百任务Git入口已经整合，也不代表已有可部署的校准权重。

先验证高层现有表示是否足够识别结果，减少随机初始化分类头一开始扰动规划模型的风险。若固定表示不够，再考虑下一阶段，不保证单训线性头必然足够。

全batch没有有效标签时，O-only应跳过optimizer step，而不只是反传0；Adam动量/AdamW衰减仍可能改变参数。多GPU/累积步的稀疏标签归一与step门要按全局有效数核验，不能各rank独立平均、独立决定跳step；现有单条C1接口不冒充已验8卡实现。

### 4.2 可选联合：L_AR + lambda_out L_out

在标签质量、分类校准和规划回归通过后，允许`L_out → O → H的上下文路径`，仍不通向低层。高层共享参数接收AR与结果分类两种梯度；O只接收结果分类梯度。分类loss不直接经过LM输出投影；如果LM head与输入embedding绑权，它们可能因共享embedding参数而一起变化，不能把这说成又计算了文本CE。

旧完整`planner_outcome`代码中`outcome_head(vlm_hidden[context_index])`没有detach，正是此路线；但它要求`memory_update_ce_weight=1.0`。上轮提案的“memory0.25＋outcome0.5”不是现有模式可以直接运行的配置。首轮可保持planner-only0.25、随后O-only；若要同时启用0.25和结果联合训练，必须另做配置合同/字段mask/梯度回归后才放行。`lambda_out=0.5`仍只是未经验证的候选，不是有效配方结论。

任何高层表示更新都会改变分类头输入分布；头的特征缓存和校准回执必须绑定高层checkpoint，不能在换H后继续沿用旧校准结论而不复验。初始不建议把H与O同时作为两个快速变化模块无约束长训。

## 5. 低层FM：AE与低层LoRA一起训练，高层不参与

一条低层样本是同一时刻的观察、当前真实技能`s*`、未来动作`a*`；不要求先运行高层生成技能。技能是mask掉文本CE的输入条件，低层不必先学“写出技能”。保持A4历史/动作合同：三相机六帧、预测32、执行0:16、23维真实控制补齐为27，仅补齐位`[7,8,17,18]`和无效动作时刻不计误差。

原pi时间约定：

```text
epsilon ~ N(0,I), tau按已固定Beta变换采样
x_tau = (1-tau) * a* + tau * epsilon
C = LowVLM(base_frozen, LoRA_trainable; observation, task, s*)
v = ActionExpert(x_tau, tau, C)
L_FM = sum(valid * (v - (epsilon - a*))^2) / sum(valid)
```

FMHelper还乘显式`fm_weight`；实际数值须和固定配置一起解释。每状态4组时间/噪声复用同一VLM条件，不是4个新机器人状态。训练使用真实动作构造噪声插值是FM定义的一部分，不等于允许把teacher-forcing的正确动作token放进VLM前缀。

```text
L_FM → 速度输出投影 → 动作专家各层/动作输入与时间编码
                        ↓ 通过cross-attention的K/V梯度
                     低层VLM计算图 → 低层LoRA
```

原始低层VLM参数不更新，但包含冻结权重的运算仍可参与对LoRA的链式求导。不能为省显存把整个低层VLM包进`no_grad()`，那样LoRA也会断梯度。`eval()`则只切换dropout等行为，并不等于冻结。[PyTorch官方梯度机制](https://docs.pytorch.org/docs/2.14/notes/autograd.html#locally-disabling-gradient-computation)

源码`FMHelper.train_step`在`not joint_training`时对KV执行detach。因此：

- A4式AE＋LoRA纯FM：`joint_training=true`，不detach KV。
- 冻结所有低层VLM、只训AE：可detach KV。
- 若detach KV但仍声称用纯FM训练LoRA：LoRA没有这条监督，属于配方错误。
- 本轮不默认重新加入低层离散动作CE/KI；高层规划CE与低层离散动作CE必须分开命名。此前joint/KI是低层方法实验，不是高低层联合实验。

## 6. 梯度总表与离散边界

表中的“高层”包括其实际可训练上下文路径；“是”指计算图有连接，非保证所有参数都有非零梯度。

| loss/模式 | 高层H | 结果头O | 低层LoRA L | 动作专家A |
| --- | --- | --- | --- | --- |
| 规划/技能/决策文字CE | 是 | 否 | 否 | 否 |
| 记忆更新文字CE | 是（与上一行共享） | 否 | 否 | 否 |
| 有证据的状态/完成文字CE（若启用） | 是 | 否 | 否 | 否 |
| 结果分类CE，固定H热身 | 否 | 是 | 否 | 否 |
| 结果分类CE，高层联合阶段 | 是（上下文路径） | 是 | 否 | 否 |
| 纯FM，KV不detach | 否 | 否 | 是 | 是 |
| 纯FM，KV detach的AE-only对照 | 否 | 否 | 否 | 是 |

上述所有路线均不更新约定冻结的视觉塔。高层与低层不共享可训练参数时，`L_AR+L_FM`的一次backward只是两张独立计算图的梯度相加，并不会建立模型间因果信用分配。当前文字链为：

```text
高层logits → 采样/argmax整数token → 解析技能字符串 → 低层重新编码
                     ↑ 普通反传在这里断开
```

低层可学会如何响应给定技能，但FM不能自动告诉高层“刚才不应选TRANSPORT而应重试GRASP”。冻结高层或不冻结高层都不会改变这条离散断点；更不要把冻结高层里的参数解冻就称端到端。

若未来让高低层共用可训练VLM，FM可以通过共享参数改变高层行为，但仍不等于穿过采样技能对该决策分配梯度。那会引入表示干扰，需一个明确管理共享参数的optimizer，不能两个Adam各自更新同一权重。可微软技能/latent接口、策略梯度等是独立架构/算法路线，本轮不实施。

## 7. 推荐的具体顺序

### P1：并行、独立监督训练

高层从B-final候选初始化，正确因果上下文→正确规划/记忆的AR CE；低层从A4候选初始化，正确技能＋同状态专家动作→FM。H与L/A采用独立optimizer、scheduler、梯度累积和裁剪，不强求一一配对batch或相同步数。规划在决策/检查时刻抽样；动作在控制时刻抽样，不能把长静止片段重复成大量同一高层指令。

高层首先更新planner组，结果头保持禁用；低层更新AE＋LoRA，KV不detach。沿用上轮建议的量级：高层LR候选1e-5、低层AE/LoRA复现对照各1e-5；AE提高至2e-5或LoRA改5e-6属于单独短对照，不因为这次解释而自动采用。预算、数据版本与正式训练授权另定。

### P2：固定高层版本，结果头热身

H达到格式/技能/记忆回归门后固定H_k，使用有依据的同状态结果数据训练O；头LR1e-4是起始候选，不是已验证最优值。先看各类召回、错误报成功、校准/弃权和实际技能切换。无合格标签就保持UNKNOWN_ONLY，不能从正常演示区间结束自动生成SUCCEEDED。

### P3：有条件的高层内部联合

必要时小LR更新H和O，`L_AR+lambda_out*L_out`，混入正常规划/记忆数据防遗忘，并重新校准。P3可跳过：固定H的O若已够用，不为了“一起训练”额外解冻。低层依旧只收自己的FM，不用物理特权标签作actor输入。

### P4：闭环收集，固定版本交替改进

1. 固定`(H_k,O_k,L_k,A_k)`在预登记train来源运行有限闭环，记录模型实际到达的偏离状态。
2. 同一状态上由合格教师/人工/已验证恢复轨迹给出正确技能、因果记忆/结果、真正执行的纠正动作；保留数据owner的发布/QA边界。
3. 高层错选技能：修正H的CE样本。结果误判：修正O训练/校准。技能正确而动作失败：给L/A增加该状态的正确动作FM样本。可分别更新或分开作业同时更新，阶段内版本固定。
4. 与正常跨任务数据混合，评估`H_{k+1}+L_k`、`H_k+L_{k+1}`的必要小规模诊断及新组合，识别收益来自哪层。预算有界，不自动追加多轮5000步。

这就是协同训练：共享语义合同、真实状态分布和纠正数据，而不是强行共用一张梯度图。原失败动作不直接做正向BC，错误技能也不能配上另一技能的专家动作；仅把模型生成技能随机替换进旧演示并不构成合格scheduled sampling。

以空抓为例：同一失败观察上的结果标签教O识别未抓住；高层目标教H选择RETRY/安全纠正并记住上次失败；纠正轨迹教L/A重开、对齐、闭合、验证。三种监督可以源于一条轨迹，但不是同一个loss，也不保证仅靠多看正常抓取就自然出现恢复。

## 8. 开训前的梯度验收（待实施，不冒充已测）

1. 每项loss分别清grad并反传，按H/O/L/A导出梯度存在性、范数和实际更新；记录无监督/全mask情况，不只看`requires_grad`。
2. 单独FM应更新A/低层LoRA，高层及冻结基座逐位不变；切KV detach后LoRA不再收到FM梯度。
3. 单独规划/记忆CE应更新高层且不更新低层；O-only反传只改头，完整高层联合分类允许回到上下文。
4. token/input mask与冻结参数是两回事；取context hidden的索引和attention causality做答案扰动反例，确保O看不到真实结果token。
5. 校验optimizer参数集合互斥且完整，清除阶段切换前残留grad，核保存/恢复后的集合。H和低层独立裁剪，避免无意对两者做同一次全局clip。高层内部多个loss的相对梯度也需监控；不把原始CE/FM数值调到相同当作平衡标准。
6. 无标签全局batch的跳步、有效token/scalar归一、DDP累积与8rank一致性须实测。不能据代码存在或这份图表声称已通过100任务训练准入。

## 9. 本轮证据与版本边界

本地归档根`artifacts/local-archive-20260912/root/`，以下是本轮实读文件；它们不是目前Git训练入口。没有声称本轮连接robo核对了活跃导入路径，也没有重新运行大模型梯度检查。

| 路径（归档根相对路径，FMHelper例外） | 核查位置 | SHA256 |
| --- | --- | --- |
| Git `src/g05/models/g05/helpers/fm_helper.py` | `train_step` KV detach、masked velocity MSE | `90b1f955610bda3184d823a24e2eeee718a895f3d23f4d8b66f10d6b4e2a4992` |
| `memlite-resume.L6rqZ6/a2_lora_history_candidate_v7_samplercoverage/src/g05/models/g05/g05_policy_memlite_skill_fm.py` | `configure_coordination_trainability`、`forward_train` | `1b152034d35eb0c99906b54bf25b6d684a89efb4d3fa8cf43a08f230f112014c` |
| `memlite-resume.L6rqZ6/b_parent_format/src/g05/models/g05/g05_policy_memlite_planner_outcome.py` | 0.25模式限制、参数组、字段mask、planner-only/完整loss分支 | `9f794c8b13f6b7e3eb764ff9be850a7aecbdba0054848fbee710c5bdb80b307d` |
| `memlite-resume.L6rqZ6/b/src/g05/models/g05/helpers/ar_helper.py` | `cal_ce_loss` token加权归一；非独立字段归一 | `53f20318356af93bb2762fe40ed142a2e0f0779bfda111dbdf12a762a45c8ec5` |
| `c1_payload/src/g05/models/g05/g05_policy_memlite_planner_outcome.py` | `forward_outcome_only`冻结上下文、结果mask与step门 | `376ab069a1749664024fc5821cd46e5c4b2fef640db896a3d472abfa46d7d5d9` |

旧实验的实际配置/权重和效果边界沿用[9月27日核查](2026-09-27-memlite-100task-training-design.md)。[joint/KI首筛](2026-09-14-joint-ki-screen.md)已记录全局clip造成的优化耦合，但它是低层实验，不能推导“高低层联合已经失败”或“本提案必定更好”。

## 10. 2026-10-01补充：阶段3的负例、恢复区间与纠正动作

Codex / PLAN-MEM100-STAGE3-DATA，研究与接口提案，**尚未实现、构造或启动阶段3**。沿用当前100任务训练计划，不改阶段1冻结源码/环境/预算。数据队友负责采集与QA、RL队友负责奖励/critic，本线程负责高低层训练合同及集成。本次本地核查基线8c6adb1，检索截至2026-10-01；以下方案是对原论文的项目适配判断，不是BEHAVIOR实测收益。

### 10.1 先区分三种监督，不把负例直接当FM专家动作

1. **结果/事实数据**：之前哪次技能尝试成功、失败、仍进行或不可观察；教O辨别状态，也可供独立reward/critic使用。整任务结局、局部技能结局、恢复是否成功分别存储。
2. **高层恢复决策数据**：在当时可获得的观察与历史下，应继续、重试、换计划还是停止，接下来执行什么技能、更新什么记忆；教H的CE。
3. **低层纠正动作数据**：从实际偏离状态真正执行并验证有用的动作；教L/A的FM。失败动作本身不做正向BC，也不简单取负FM loss。失败transition可交RL/critic，但需要相应算法，不是现有FM入口自带的功能。

相同物理状态可同时是“上一次GRASP的失败例”和“重开、重对齐、再抓的正确动作例”。成功恢复动作是偏离状态上的**正向示范**。高层不必新增笼统RECOVERY token；优先保留RETRY/REPLAN与真实技能，对齐已有schema-v6。

### 10.2 rollout标注：尝试、故障事件、恢复行为分开记录

建议采集sidecar记录以下信息；这些是审计字段，不直接扩展或混入现有actor输入白名单：

```text
source_instance / source_episode / split / policy_and_schema_version
attempt_id / parent_attempt_id / goal_and_target_binding / arm_or_bundle_member
observation_time / action_start / actual_executed_length
event_type / event_time_or_interval / evidence_kind / evidence_available_time
attempt_outcome / outcome_validity / observability / termination_reason
recovery_start / recovery_end / recovery_success / controller_source
restore_identity / branch_seed / perturbation_provenance / review_record
```

- **skill attempt**是一次具体目标/对象/手臂绑定的尝试，不是整段任务。双手bundle先标每个成员的结果，再按预先声明的AND/OR及依赖聚合，不用一个手的成功掩盖另一个手的失败。
- **failure event**需要物理或经过审核的视觉证据。例如闭合并尝试抬起后目标仍在原处、手是空的；已保持物体在非计划释放时脱落。技能时间到、夹爪闭合、专家段结束、暂时不动或完整任务没成功，均不单独构成物理FAILED真值。
- **stalled/blocked**先作为诊断/复查原因；必要时保留IN_PROGRESS/UNKNOWN，不把“缺前置条件”自动等同不可恢复终止。最终超时可记官方预算内task_success=false与time_limit，但不是物理不可恢复证明，也不把全段每帧改成FAILED。
- **recovery interval**从有依据的纠正行为/接管开始，直到进入经验证的可继续状态；不要求回到专家完全相同的关节姿态或轨迹。仅输出RETRY、重复动作或改变技能，不足以证明已进入有效恢复。行为起点与结果须分开，失败的恢复尝试也保留。
- 恢复成功可以是局部抓稳/重新满足前置条件，不等于整任务完成。第一次尝试FAILED在恢复后仍然FAILED；新attempt可以SUCCEEDED。若无法定位边界，保留时间区间与不确定性，不虚造精确帧。
- 记录错误发生时刻、最早合法可观察时刻、接管/纠正时刻、恢复验证时刻。若稳定性需要后续K步确认，监督available_time设在确认后；不能把未来帧/结果文字倒灌给早期outcome输入。未来结局预测属于另一个risk/value目标，不能与“现在已失败”混用。

以空抓为例：接近与闭合仍进行→抬起时确认目标未随动、第一次抓取失败→重开并调整/新attempt→目标随手稳定抬起、恢复局部成功→继续搬运。恢复区间不能粗暴从第一次异常一直延伸到整个任务结束。

### 10.3 真值和效率：物理事件筛候选，人工检查边界与例外

优先建立**按技能族复用**的事实验证器，而不是给100任务各写控制规则。OmniGibson已有统一对象状态接口及IsGrasping、Inside、OnTop、ToggledOn等关系，可作物理事实来源；具体版本、物体能力和抓取模式仍需实测，不假定每个技能都有完备oracle。[官方对象状态说明](https://behavior.stanford.edu/omnigibson/object_states.html)

- 抓取核指定对象/手臂、抓持信号、物体随动与保持；闭爪不是抓稳。运输承载物同时核“手持容器”和“内容物仍在容器上/内”，不能只看盘子没掉。
- 放置核目标容器/表面、释放、稳定支持关系；开门核实际开度及后续可通行条件；开关核目标设备实际状态。液体/颗粒/切分必须用支持的状态和官方任务语义，不套用刚体距离阈值。
- 将每控制步轻量事实日志与低频模型决策时钟对齐；保存真正执行的动作，不把未执行的未来32步全部当真实transition。事件附近保留密集RGB，稳定区稀疏抽样；不得用阶段1stride16硬切全部失败边界。
- VLM可提出疑似时间段、描述现象、归纳类型；物理验证器和人工审核裁定。不能只按注入的故障类型填真值：扰动可能没造成失败，或造成了另一种后果。
- 人审覆盖全部故障/技能族、双手、遮挡、成功近失误、自然恢复与恢复失败；全审冲突和边界模糊例，同时随机审自动通过/自动拒绝例，避免只审困难例而无法估计总体错误率。评价按事件/来源episode统计，不用海量相邻帧虚增独立样本数。

[FailureSpot，2026-09-03](https://arxiv.org/abs/2609.04277)明确研究整条失败轨迹标签污染正常前缀的问题，使用动作异常弱监督与不确定性驱动的时间标注。我们可借鉴候选筛选和主动人审，但动作停滞/突变只是提示；例如HOLD的正确动作本来就可以很小。[SAFE](https://arxiv.org/abs/2506.09937)支持轻量latent检测与校准的路线，但预警分数不等于物理事件真值；分布改变后不宣称校准保证仍自动成立。

### 10.4 成功演示能生成什么，不能生成什么

| 方法 | 可以得到的监督 | 必须保留的边界 |
| --- | --- | --- |
| 挖最终成功演示中的空抓/重试等片段 | 自然失败尝试＋实际纠正动作 | 最终成功不证明局部无错；原数据是否含这类片段尚未逐量统计 |
| 保持视频，换对象/目标/条件问题 | 目标满足/语义匹配的负例 | 核目标真实存在、确未满足；不伪装成原策略真的执行过新指令，更不沿用旧动作做新指令的正向BC |
| 成功视频截断到完成之前 | 尚未完成、部分进度、near-miss | 不自动标在线FAILED；这可能只是IN_PROGRESS或人为截断的删失样本 |
| 亮度/轻微遮挡等视觉增强 | 观察鲁棒性、证据不足时弃权 | 一般仍是正例；刚体旋转/相机变化须同步坐标与动作，不能任意改RGB却保留矛盾标签 |
| 演示起点＋实际执行的动作/状态扰动 | 真实偏离状态、故障与成功/失败恢复分支 | 需要仿真/真实交互并重新渲染；扰动后原动作后缀不自动仍正确 |
| 世界模型生成失败/恢复视频 | 辅助语义与恢复提议候选 | 图像逼真不等于接触动力学成立，不未经物理执行验证就当23维控制专家真值 |

[RoboReward](https://arxiv.org/abs/2601.00675)直接从成功视频通过反事实任务重标和截断生成负例/部分进度，适合补“是否满足目标”的训练。其目标是片段/episode结尾的reward评分，不是给在线四分类outcome凭空制造失败发生时刻，也不提供真实低层恢复动作。

[DART](https://proceedings.mlr.press/v78/laskey17a.html)是在**采集演示时实际注入动作噪声，并让专家在到达的新状态纠正**；不是给已有动作数组加Gaussian noise就完成恢复数据生成。借鉴时按末端/底盘/关节各自单位、相对物体尺度和真实模型误差设计有界、时间相关扰动；夹爪时序作为单独事件，不对23维均匀乱加噪声。不向27维表示的补齐位制造“动作”。

[AHA/FailGen](https://arxiv.org/abs/2410.00371)用仿真关键帧扰动生成抓空、滑落、位姿错位、顺序错误和对象错误等失败，是合成**失败解释/结果样本**的直接参考，但本身不是已移植的R1Pro恢复动作教师。[MimicGen](https://mimicgen.github.io/)的对象相对轨迹变换、衔接并执行筛选可作为恢复教师候选；双臂、移动底盘、接触/液体和新对象需要适配，不能只复制邻近演示后缀。

[ChauffeurNet](https://waymo.com/research/chauffeurnet-learning-to-drive-by-imitating-the-best-and-synthesizing-the-worst/)支持“合成偏离＋对不良后果的监督”这一方向，但它的驾驶中间表示与我们的三视角RGB/接触动力学不同；轨迹几何扰动不是无需新观察的通用机器人负例生成器。

世界模型路线已出现[Dream2Fix](https://arxiv.org/abs/2603.13528)，用生成的失败片段及结构化纠正建议训练VLM。我们已有目标仿真器，首轮更优先物理重执行；该工作不证明任意合成23维恢复轨迹都可以直接用于FM。

### 10.5 推荐采集核心：同一偏离状态的恢复分支

先固定阶段1/2候选版本，在TRAIN实例取得少量自然rollout和可恢复状态；另从合法演示状态产生有界故障，补稀缺类型。每个候选状态保留“不干预继续”的分支，以及少量有理由的重试/恢复分支。教师可由可验证的几何规划、经过执行筛选的演示重定向、人类短接管或已有策略候选提供；LLM只给候选不能充当动作正确性oracle。

同状态、相同任务约束/剩余预算及配对seed重执行；局部后置条件、后续稳定性和短链继续可行性均通过，才接收为纠正动作。若要声称某高层纠正优于原决策，须配对比较多个随机后续，不因一次成功就认定原决策必然错误。**从失败后状态开始的恢复**与**回退到错误前重做决策**分两种来源：后者能教预防错误，却不能自动填充前者缺失的纠正动作。

[MAGMA-GEN，2026-09-17](https://arxiv.org/abs/2609.20056)与高层MEM-Lite很贴近：教师诊断只是候选，回到同状态重执行才选择监督。它主要改高层离散决策，不能替我们补出低层本来不会的动作；本项目仍须独立验证FM纠正段。

“能否救回”不是只由一个状态决定：它依赖教师、可用技能、剩余预算及恢复协议。[Kintsugi-VLA，2026-09-25](https://arxiv.org/abs/2609.31048)用restore-and-branch估计这种相对可恢复性，适合启发困难起点筛选；目前仅单一仿真Franka任务的证据，不能外推为百任务保证。少数次没救回只能记录未证实可恢复，不标物理必然不可恢复；边界也未必单调。

[RaC](https://arxiv.org/abs/2509.07953)强调恢复段和随后正确完成子任务的纠正段都要保留。对我们，这意味着不能只教松开/撤退，否则可能学成反复退回而不完成任务；下一段再抓、保持与继续执行同样需要监督。

工程前置：目前发布的数据是RGB＋动作/标注/meta，**未验证包含可精确恢复的完整仿真快照**。须先验证从训练实例重放/保存/恢复后，物体、机器人、控制器、抓持/接触、必要粒子状态和观测时钟一致，再谈大规模分支复用；不因讨论自动下载raw或宣称有免费任意帧reset。

### 10.6 现有接口缺口、样本配比与验收

本次实读：`src/g05/utils/memlite_skill_protocol.py`已有evaluated bundle、outcome证据/可用时刻、监督mask字段；`src/g05/models/g05/helpers/outcome_head.py`四分类且无标签mask与真实UNKNOWN分开；`src/g05/utils/training/stage1_model.py`固定单帧、planner-only/outcome loss=0。旧`observable_outcome_template`仍要求三相机×六帧，旧`collect_memlite_recovery.py`是旋转/几何恢复采集，不能当通用100任务管线。

- 阶段3结果观察器需要因果时序：建议先比较缓存的尝试起点、最近关键帧、当前帧/特征与真实执行摘要，不默认将高低层所有推理改回六帧。纯单张当前图＋“曾发过GRASP”的文字无法可靠区分刚接近、空抓、已抓又掉落。旧六帧合同与新单帧服务适配、合法输入白名单、可观察性mask和时序回归是独立待办，不热改阶段1。
- 缺失真值用loss mask=0；确因遮挡等在合法输入下无法判断，才作为有审核依据的UNKNOWN。仿真知道的对象位置、故障注入参数、未来结果不得进入actor/memory事实输入；即便教师知道答案，也要审查当前观察是否足以支持该监督。
- O先固定H训头并校准，H使用正确恢复技能/记忆CE，L/A只使用实际合格纠正动作FM；跨技能边界的32步动作窗按新合同切分或mask，不混入另一技能的动作。FM不反传到生成离散技能的H。失败transition交RL/critic专用数据视图，区别termination与time-limit/bootstrap。
- 按技能族×故障×严重程度×来源抽样，保留正常成功及困难但未失败的例子。恢复FM可先提议正常80%/合格纠正20%，仅作有足量唯一来源后的起始候选；结果头按事件类别和来源均衡并在自然分布上校准，不照抄80/20。一个起点的众多分支不当作独立来源，不用重复几段补固定训练小时数。
- 先在6–8个异质任务、覆盖抓取/运输/放置/关节操作/双臂等技能族做协议及数据效率小验证，具体实例、seed、GPU/仿真墙钟与交互上限须预登记后批准。先做旧数据分析及少量物理验证，不立即百任务全量rollout、重建或新长训。
- split在源实例/episode层先固定，所有扰动兄弟、片段、重标问句同split；原5%/public_test绝不回灌。另留恢复类型相同但任务/对象不同的迁移检查：例如不提供task0空抓恢复训练，用其他对象的恢复数据测试task0的纠正能力，不称整任务zero-shot。
- 指标分别报告：标签错误率/边界误差；O误报成功、误报失败、漏检、检测延迟、弃权与校准；高层恢复选择/低层服从；配对起点局部恢复率、正常任务退化、完整自主官方SR。给计数和按来源独立性处理的区间，不把诊断前缀辅助成功报成完整任务SR。标注抽查通过不声称全量绝对零错误。

GAIL/IRL可以利用演示学习奖励/策略，但判别器的“像不像专家”不是物理FAILED真值，也不自动给出恢复段及正确动作；BEHAVIOR已有物理事实可利用，暂不把另起GAIL/IRL作为标注前置。[GAIL原论文](https://arxiv.org/abs/1606.03476)；用户摘录中SEARN/SMILe的特定实验比较不作为跨任务普遍排序结论。

### 10.7 阶段1未结束时可以并行准备什么（2026-10-01）

Codex / PLAN-MEM100-PREPDATA。以下为用户本轮询问对应的**待办与分工建议**，不是新数据构造/仿真/训练已执行，也不擅自占用空闲节点。正式阶段1继续使用冻结v4数据和d0528b4源码；已有split、全量内容校验、普通技能标签和采样器覆盖验收复用，不重做。

| 优先级/依赖 | 工作与交付物 | 主要负责与准入边界 |
| --- | --- | --- |
| P0，无需模型/GPU | 基于v4的episode/segment元数据，建立100任务×35技能的关键事件候选和视频定位索引；记录对象/手臂绑定可信度、双手并行、技能交接、源实例/split与完整帧时钟。扩充事件维度，不重复一般覆盖统计，不把标注区间终点当成功。只新增sidecar，不复制整套RGB。 | 数据负责人；本线程核读取/监督合同。先小metadata，再按需读取动作列和短RGB，避免共享盘全量解码竞争。 |
| P0，无需最终模型 | 固定6–8个异质任务的初始小样本，从成功演示中筛重复开闭夹爪、回撤重试、重新放置等候选；建议首轮约200个待审核短片段，人工区分真实纠正、正常动作和无法判断。交付clip索引、尝试/恢复边界、实际动作范围及证据，不凑失败类别数量。 | 数据负责人；候选不等于已批准训练数据。无物理状态时以合法视频证据为限，边界模糊或因果不可辨保持无标签mask。 |
| P0，无需最终模型 | 确认结果/恢复schema和审核规程；从上述片段及后续物理样本建立小型人审参考集，区分训练、校准和测试用途。涵盖可见成功、未完成、局部失败、恢复尝试、遮挡；对歧义例双人复核。交付原片段、标签、最早可确认时刻、审核/争议记录。 | 数据负责人＋本线程；沿用原实例split，留出样本及衍生物不回灌；反复诊断集明确为开发集，最终测试另留。没有可靠失败类就标明缺失。 |
| P1，无需最终模型，可CPU/人工先做 | 对已核实片段少量构造目标不满足和部分完成的反事实问答，绑定label_kind；相同片段保留正负对照、拒绝含糊目标或仅靠措辞可猜的伪负例。交付goal-satisfaction视图，与真实attempt-outcome和action-SFT视图分离。 | 数据负责人；不把截断IN_PROGRESS标FAILED，不把换指令后的旧动作作为正向FM，不自动开启大VLM标注作业。 |
| P0工程前置，无需最终模型，但真实验证需仿真资源/预算 | 实现事件日志及状态保存/恢复验收入口；先挑少量TRAIN场景检查恢复后观察、物体/控制器/抓持/必要粒子状态和后续动作一致性。交付restore回执、合法动作/观察时钟和可复放起点。CPU可先做日志协议测试；实际仿真预算须确认后执行。 | RL/仿真负责人；本线程对齐输入白名单和训练导出。不能将RGB/机器人61维状态当完整世界快照，不默认下载raw/depth或认定A800图形仿真已适配。 |
| P1小规模，需上述restore/教师/物理门 | 在少量成功演示起点做有界扰动，验证失败事件和恢复教师；正常、不干预及纠正分支分开，输出结果标签和实际合格动作。它是接口/数据质量样例，不代表最终策略真实故障分布。 | 数据与RL负责人协作，唯一run owner另登记；不等待阶段1也可做，但不立即开展百任务规模合成。 |
| 等待固定且有代表性的阶段1/2模型 | 大规模自主rollout、按真实频率分配失败类型、批量生成高层预测intent的阶段2适配数据，以及最终结果头特征缓存/校准。 | 只读完整已保存checkpoint可做有限接口探针；旧模型或早期点的失败只能作预演，不能冒称最终分布。特征缓存必须绑定高层版本。 |

先完成的目标是**一套能复核、能喂给正确训练分支的小数据包及可复放起点**，不是预先堆固定小时数的负例。建议顺序：来源/事件索引与schema→约200候选人工筛选及反事实小样→独立预算的restore与物理教师小验证→稳定模型上的针对性扩展。例子数量只是首轮建议，未实测产量/人工耗时，不构成开工授权。

资源约束：优先轻量metadata/动作读取及事件附近CPU解码；旁路输出、低并发、限制I/O和缓存，避免抢lc1/lc2共享盘吞吐。没有因本轮查询确认lc3或robo有可用GPU；若需要仿真或标注模型，先核资源与已有作业，不改共享env/当前数据release，不给阶段1热加新标签。

#### 2026-10-02 P107 calibration40：已封存候选的受控时序视觉审阅

`calibration40-v1` 是 `annotation_calibration` 侧的 sealed metadata queue，不是 student/train 视图。经独立 review-clear 的`0ccbe4e` 已在 lc3完成受限 CPU decode 并 sealed resume：40条、400 requested slot、392 distinct episode-local frame、8个边界 clamp、1,176 native PNG和40张 review-only contact sheet；完整候选 bundle/QA 的 manifest 为`ca9a3cd9735b4b9c4e145ae282cbc62d85362709c332e77432c4fae2fdac63f3`。根分支的 candidate pipeline（`943ab07`、`9c99147`、`a731045`）亦已独立复审；它仍不改变这些图片的 candidate-only、training=false 身份。

四个 shard 均持久化于`/home/wsy/behavior-annotations/p107/temporal-annotations/calibration40-v1/shard{0,1,2,3}`，root-model现已原分辨率审完40个anchor camera-triplet（38个page02，边界q36/q38为page00）及6个future-only page04 audit。可见校准的41 queries最终更正为2 YES（q18 hold/support、q24 OPEN_DOOR）、8 NO（q10/q11/q15/q16/q26/q31/q34/q35）、31 masked UNKNOWN：q0只有body contact、无法证实目标press point；q13的head camera不代表robot-base orientation；q1/q17原声称也均降为UNKNOWN。root仅批准这10个可见校准判断；它们不是attempt/outcome/recovery/action标签，future也不能进入actor anchor。中央合并`/home/wsy/behavior-annotations/p107/calibration40-merged-v1/canonical_reviewed_calibration.json` SHA `8e95c0e0365a7f0821255e98441cee2189d2286f81f8f085d08bbd9e2ab8a71f`、manifest SHA `419e0c8f5bfc8dbce9a1441edd550238b664e64666c4aa8b22af2a99debb4586`、root receipt SHA `31955044c0b4b37918bc76cb913a0984ebd917e43738308833d8adbfa7f4160a`已核：41个`event_id`+exact `goal_relation` composite join与55个causal native-PNG hash通过；q24 multi-query只见于临时summary，未污染source canonical。strict counts仍为10 known calibration auxiliary/31 masked/0 student-outcome-recovery-action-BC。s1 agent attestation已解identity；此前standard pack真实run因`_read_sealed_index`物化403,257行/1.018GiB而MemoryError。stream fix `c776a7a797b3049b11af93dbbc65e5ab7fe72701`已独审通过并以根commit `67b8b52`集成：review full scan15.15s/119,116KiB、all strict validation/40 retained通过、sealed-source注入second pass拒绝无output、旧fixtures byte-identical；根本地packer15/15、实际audit4/4、CLI help/pycompile通过。它仍只消除有界读取缺陷，standard pack实际rerun尚未发生，绝不称package/release通过。

此次provenance是`root_model`/`human=false`，但E2的操作定义是**主代理/root亲自**查看至少120个分层图像/短片及标签，并不是另一个外部human gate；当前40个anchor在数量和分层上仍未达E2，且未审完整1,176 native PNG/timeline。不得把该provenance改写为`human_reviewed=true`，也不能将E2或global goal标为完成。formal student、attempt/outcome、recovery/action BC、FM-positive、release与训练准入仍均为0。ownership只可按helper shard的`event_id` join；全40 metadata虽有37项位置不同，仍为有效`event_id`双射；generation `7f`→consumer `efdd`是批准lineage。规程固定在 [P107 时序视觉校准标注指南](../data/P107_VISUAL_ANNOTATION_GUIDE.md)：三路原生 RGB 无ground-truth footer，actor因果帧与future offline audit分开；以实际frame/PTS/camera locator、明确问题和target binding标`YES`/`NO`/`UNKNOWN`。source skill是尝试目标而非结果，metadata end/close/timeout不能生造失败或成功；image-only也不产生BC/control/正向FM。

`/home/wsy/behavior-annotations/p107/sampling-audit/cal40-phase-v1/cal40-phase-v1-stats.json`（SHA `939b7148d91480dd0a7b000ca3d7f2ab248502694350fe7e42e868bac0e0a772`，metadata-only、无新标签/图像）显示CAL40 40/40与full-v3 403,257/403,257事件均用`observation_frame=segment.start`。这是系统性选样问题：当前候选不能代表段内phase，也不支持attempt outcome/recovery发生率或学习效果结论。`deb000e`的`p107-phase-sampling`已建`/home/wsy/behavior-annotations/p107/phase-balanced-calibration40-v2`：mini-index SHA `a43bbec04a86d0b13f9ad06f4825fcd7c16ab469cf13c3a65a555a5ab0eca09b`、queue seal `400613905529c710fcba62ebe4b7167c6df974ada94256d05a0c7bdc94b46c8f`；40个distinct TRAIN calibration groups、31tasks/31skills（94/100/101/103缺），ENTRY/MID/TERMINAL_OR_TRANSITION/REPEATED_METADATA_QUERY各10，17.3s build/17s sealed resume/351KiB、selector11+renderer11测试过。terminal-adjacent transition不等于success，跨episode repeat仅是metadata query而非failure/retry。`/root/review_phase_sampler`已独审PASS、根以`b3c961b`只移植selector/test/design note；review核7 seals/350,878B、legacy7f resume、40 unique/non-parent IDs和source group/clock合同，根再跑selector2+renderer11 tests及两CLI help均过。实际同一受限batch的lc3 decoder已由timeout parent PID`1722446`在`2026-10-01T18:11:00Z`（10-02 02:11 CST）以CPU0/`nice 19`启动：400 requested slots、396 distinct（4 boundary clamp）、预期最多1,188 native RGB；7 sealed inputs、`b3c961b`/renderer `5bcb3451…81c9`及legacy7f固定，输出`/data/workspace/wsy/behavior2026/p107/runs/phase-balanced-calibration40-v2-rgb-attempt2`与同级`.receipts/`。预算仍1CPU/4GiB/30min/1.9GB/0GPU，8×A800为0MiB/0%；这是**运行中**，不是PNG/RGB/packet、annotation、release或训练完成。additive actor-query contract `5ea8912`只在隔离树就绪，`/root/review_actor_query`仍在独审；而query producer `e7187061`独审P1×3阻断：`pretty_entity`会截断5–8位noun tail（`box_of_oatmeal→boxof`、`boxing_gloves→boxing`、`electric_switch→electric`、`bottomcabinet→bottom`），geometry缺lid/drawer/left/right，effect query又未含实际entity；root表格QA同样检出，v2尚未用于标注，author必须另存修订artifact后再审。standard calibration pack仅获`67b8b52`后的binder真实rerun权限，尚无run receipt；formal student、recovery/action-BC、live DART和训练仍为0。live DART仍等用户RTX4080与lc许可、冻结教师/运行时和official TRAIN fresh-reset证据；不用robo，未启动仿真或训练。
