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
