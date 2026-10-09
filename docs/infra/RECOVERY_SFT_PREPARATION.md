# 恢复数据 → A800 小规模 SFT：准备与验收

## 2026-10-09 19:53 CST最新授权

用户要求已签核新数据先在A800开训，覆盖下方“待全100齐才训”的时间顺序。新分支`feat/recovery-pilot-a800-20261009`/配方`a800_pilot_20261009.json`：H0单卡小头→L0八卡global8/micro1→真实OOF反馈H1八卡，原SFT终态高48045/低98414不变。新联合release仅搬迁重新验真原逐点审批，未审候选不训练；H1增加正常原专家规划回放，避免全RETRY目标。5事件遍、各线4h异常保护，不自动凑1000更新；数据采集仍无新增额度。部署校准未通过则UNKNOWN；此SFT不等于异质短技能RL已启动。实际运行/结果以plan顶部为准，旧准备票和工程产物不覆盖。

负责人：Codex；分支 `feat/recovery-sft-prep-20261009`。此页是10月9日批准的准备工作，不是新正式训练启动通知。真实进度看 [plan](../plan.md)；方法看 [RL方法计划](../RL_METHOD_PLAN.md)。

## 17:34 CST范围覆盖：先扩展100任务，暂不训练结果头

**18:25 CST进度覆盖：** 活跃GRASP为e515109 / `grasp-collection-v3`，活跃开合为d7d00be / `articulation-collection-v5-gap`，接续v6。已封第二不可变快照`closed-grasp-corpus-v2`：168ZIP/16,183真实分支控制/12,264原JPEG/4,088观察全读通过，2,388完整干净动作窗/85实际RETRY因果请求通过，0optimizer；**不是168ZIP语义全部批准**。新左手vacuum15仅5项逐点审批（3结果＋1plan＋1动作），SHA18602b0f…f4941b；正常对照三组6结果另签，不增加独立事件。洗衣机169首次强门恢复有156面板人审记录，跨技能标准封包/准入待接，不能误称已进训练release。类别名与实例ID混用是新发现的源标注问题：新绑定只能依据原示范中唯一实际接触/关节变化，原全竞争对象记录和派生回执均保留、建库重算，不按物体名称后缀/最近距离猜。下述17:34计数/运行源保留作历史。

用户要求收集全100任务通用恢复数据后再一起训练H0。当前工作分支为`feat/recovery100-collection-20261009`；下方16:31的15来源release仍是不可变历史小试，不代表本轮已完成。

- 全20k演示元数据盘点完成，保护原holdout后19k来源。99任务出现GRASP文字，实际只有97有可绑定的单目标GRASP；另外三任务通过开合等机制补齐，不能用含糊目标造标签。
- 首轮预选325来源覆盖100任务：旧40、93新任务×(2TRAIN+1dev)=279、缺口三任务各一TRAIN/一dev开门。**预选不等于完成或通过。** 279的七卡独立冷采集已运行，另有117跨41任务OPEN_DOOR原来源预选，后续物理通过情况另报。
- GPU0–6跑冻结`3e11e95` GRASP；GPU7用`c340b42`继续单关节开合验真，4并发冷场景加载用于GRASP。只允许已证明属于同一采集的其它GPU辅助渲染小上下文共存；不动其它工作负载。
- 新GRASP前11闭合来源33ZIP/2,540分支控制/1,929RGB/643锚点已全量实际读取，349干净动作窗口、15真实RETRY因果链验证过。根亲审三来源117原RGB面板，签发9逐点结果、3实际恢复意图、3个干净32控制窗口：`closed-grasp-approvals-v1.json` SHA`44ac737aa50d3cd1d28414b972e4316d5f064639ec434010a18307b8b3d0c806`。其余自动候选仍未批准。
- 开门初试虽物理Open翻转，但仅3–9度开缝，根亲审75面板后明确拒绝其功能性成功标签。新门用单一精确关节方向与行程：开度>=35%且official Open为真才提成功候选，<=10%提已丢失证据，中间UNKNOWN；关门需<=2.5%且official Open为假。多关节/双向未绑定拒绝。此仍只是间隙代理，**不能自动证明后续动作可达或全任务完成**，须人审。
- 原参考重放失败（如bringing water:98完整1,531控制仍闭门）也保留，不将原标注结束、没进展、超时判为FAILED，不挑同源幸运重试。质量拒绝记录在`recovery100_quality_rejections_v1.json`，新已审记录在`recovery100_owner_review_v1.json`。

当前还缺：剩余来源真实完成、全100任务有效/无效覆盖表、导航/放置/开关/运输等非GRASP机制、全量结构校验与分层人工准入、逐类别独立dev事件及校准。H0/H1/L0/新RL皆未开训；不把本轮候选或局部教师恢复当演员模型SR/Q提升。具体路径见[服务器目录](../SERVER_LAYOUT.md)。

## 16:31 CST验收：首批GRASP恢复SFT技术准备完成

本节覆盖以下历史“等额度/无恢复数据/起点全未通过”状态。用户已取消准备工作的reset、控制步和时长额度，本轮以实际质量验收收尾。**完成的是第一轮局部恢复方法试验的技术准备，不是100任务全技能数据覆盖、已校准结果头或方法有效性证明。** 13:04/14:24原回执保留，最新机器可读证据为 [v3验收摘要](results/2026-10-09-recovery-sft-preparation-v3.json)。

| 交付 | 实际验收结果 |
| --- | --- |
| 旧RL | 已安全停在共享update180；权重、Adam、8rank RNG与旧轨迹均保留，不重开公共评测 |
| A800训练工程 | H1/L0各8卡真实2更新；step1保存→显式恢复step2、完整CPU重载和W&B读回通过。工程权重不替换正式SFT父权重 |
| 新采集记账 | 4任务40来源，35完成分支、4参考抓取未复现、1停滞隔离；33,007实际控制。45恢复候选分支只来自26来源，不冒充45独立恢复或actor SR |
| 完整候选实读 | 105ZIP、10,436分支控制、7,926RGB、2,642观察锚点全部解码/动作/因果校验；1,575干净32控制窗口、51真实RETRY。未审核部分仍不可训练 |
| 冻结准入数据 | 15独立来源，9TRAIN/6dev，45结果标签（每来源失败/推进/成功各一）、15实际恢复意图、15个正确32控制动作窗；我亲审585原RGB面板并结合精确目标/同手物理记录，仅签具体窗口 |
| 实际数据消费 | 所有准入H1/L0样本在lc1正式processor实读；60因果成员请求/105次高层prefill已缓存，16.97秒不含权重加载。重载/头前反向通过，梯度不回流骨干，0optimizer |
| 课程起点 | 全15来源一次冷进程验收，13通过（7TRAIN/6dev），2拒绝。13例都重放纠正后连续同手同目标抓持128控制，起点39原RGB面板亲审 |
| 最终回归 | 105项RL/数据测试＋8项因果反馈测试通过；准备票固定成功状态、父权重/数据SHA，错误或跨版本证据不能通过 |

`preparation-ticket-v3.json`明确记录`technical_preparation_complete=true`、`remaining_preparation=[]`，同时`execution_ready=false / formal_training_authorized=false`：它是验收清单，不是自动启动训练的许可。独立团队review尚待，feature没有合main；本轮不启动新正式SFT/RL。

### 现在用哪些文件

以下相对共享根`/data/workspace/wsy/behavior2026`：

| 用途 | 路径 |
| --- | --- |
| 已验训练数据，唯一当前准入 | `datasets/recovery-local-corrections-20261009-v1/local-admission-v2/` |
| 对应实际图像/动作/上下文 | 同release根`local-corpus-v3/{raw,audit,history,proposed-plans}` |
| 冻结高层特征 | `runs/recovery_sft_preparation_20261009/h0-accepted-feature-cache-v2/` |
| 课程证据与原图 | `datasets/recovery-curriculum-starts-20261009-v1/` |
| 最终准备票和各门证据 | `runs/recovery_sft_preparation_20261009/preparation-ticket-v3.json`、`accepted-preflight-v1/` |
| 全批候选，非训练release | `datasets/recovery-extra-candidates-20261009-v1/local-corpus-final-candidates-v2/` |
| 实际验收冻结源 | `src/recovery-prep-4424b35`，回归源`src/recovery-prep-841bae6`；工程训练源仍`8f47fd7` |

准入SHA `08684fb665a3113617463138db100c09cd12674dbe137bae3e0d0d3327db54ed`；特征SHA `b82fb5e6864593bfc7933b01dbf969bbe56f77fdc928956ff0db81884fe06ffd`。旧`admission-v3`只有19个自然rollout成功结果标签，与新`local-admission-v2`不是同一版本线，不按名字末尾数字混用。原105候选与15来源签发池也不能混用。

### 起点怎么用，哪些没有通过

合格路线为：新冷模拟器→加载原来源的完整seed世界/控制器/任务与指标时钟/RNG→重放原32控制fault→验证同目标当前确实未被抓持且位置误差≤5mm→交给待测actor。每个分支独立memory，只包含之前真实发出的意图；oracle只留在诊断字段。

`failure-start.pt`已检查同进程恢复，但**其直接跨冷进程加载没有另行认证**，不要跳过上述已验证路线。原seed与记录动作仍在10383 `runs/recovery_prepare_sim_20261009/local-coverage-v2`或`local-coverage-v4-cold`，具体来源/对象/branch SHA见13项清单。冷验只测试保存时的场景、官方post2和单env，不能当作任意场景/向量环境恢复保证。

`make_pizza:179`和`preparing lunch box:95`冷重放漂移分别6.070/5.772mm，拒绝用于该课程，不调门、不挑幸运重试；原采集实际执行过的SFT动作标签仍是真实数据。暖场景复用曾继承非TRO柜门状态，整条暖采集路线已隔离，runner默认拒绝。coffee142的静止非抓持死循环也隔离；新检测器用实际日志在控制128识别，不依赖资源倒计时。

### 后续训练依赖，不冒充已完成

先H0冻结骨干训练结果头/时序适配器，再产出真实分组OOF反馈供H1；L0只训练动作专家、冻结原192个LoRA，保留100任务专家锚点。正式父权重仍高48045/低98414。样本/梯度/续训入口已验，但**没有训练好的新H0/H1/L0，也没有新增SR/Q结果**。

这15来源只支持初始GRASP方法小试。当前6个dev来源不足每类别20独立事件/置信度门，因此即使H0拟合能跑也不能开runtime readiness，反馈仍UNKNOWN。完整通用RL还需其它技能的课程/奖励实测、更多独立校准事件、预测反馈服务接入和固定对照；不能靠重复窗口凑数量，也不能说三个训练池过门等于通用恢复已经解决。旧SFT见过这些原TRAIN实例，新dev仅对本轮恢复增量留出，不叫从未见过的泛化测试。

以下为历史检查点与接口说明，状态以本节为准。

12:50 CST检查点：八卡硬件通信、L0真实父权重/FM梯度（ff584c1）及H0成员prefix/observer梯度（c25d06c）均通过，A80070＋8测试通过，0optimizer，所有诊断GPU已释放。242ZIP数据转移完整hash通过；原100task/3200个不同来源实例专家锚点索引已生成。三种恢复训练池依旧未放行；[回执](results/2026-10-09-recovery-sft-preparation.json)明确列出尚未完成的训练整合/数据/校准门。

13:04 CST补充：冻结874d39e八卡**微型合成网络**验证了通用运行时的非均匀尾批全局梯度、模型/Adam/RNG精确恢复与单rank错误共同拒绝更新；它不是实际G0.5八卡训练验收。独立[W&B工程探针](https://wandb.ai/hanhanyy-fudan-university-school-of-management/behavior2026-g05/runs/26d903a19327)已真实写入/读回并结束，仅`engineering/*`指标，无合成loss混入训练曲线。0新增G0.5更新，所有GPU已释放；完整derivative trainer接线、数据/校准门仍待。

## 历史：14:15 CST验收结论（已被16:31检查点覆盖）

**训练工程链已验，恢复数据与仿真起点门未全过，因此没有启动正式恢复SFT/RL。** 全部自有GPU验收进程已结束，lc1八卡0MiB/volatile不可纠正ECC均0；10383旧RL已安全保存第180次共享更新后停止，未删除旧产物。[本轮补充回执](results/2026-10-09-recovery-sft-preparation-v2.json)记录真实证据，旧13:04回执保留作历史。

| 检查 | 实际做了什么 | 结论边界 |
| --- | --- | --- |
| L0低层 | 8f47fd7，lc1八卡/global16/原专家32行，step1保存退出后恢复到step2，共449.19秒 | 322个动作专家张量更新；192个原LoRA保留并冻结，其余冻结参数SHA不变。不是效果训练 |
| H1高层 | 同冻结/八卡/32行，保存后恢复共2更新/546.71秒 | 326个高层张量更新，其余冻结SHA不变；未使用未验真恢复目标 |
| 检查点/W&B | 两条线均完整SHA及独立CPU重载，Adam步2/八rank RNG齐全；线上history各有1、2步且finished | 证明原子保存与显式续训链，不把两步loss当方法有效 |
| H0特征缓存 | 31a46c7，精确高48045冻结，4个真实三相机观测/4次member prefill，cache SHA重载与实际头消费通过 | 本次每请求只有1个真实检查点；多时刻因果/填充由单测覆盖。0optimizer，diagnostic cache不得进正式训练 |
| H1新反馈输入 | 1个真实观测＋4种明确合成反馈，经正式processor通过；前置memory/上一意图正确，结果/低层loss掩码为false | 只验证格式，不生成训练标签或冒充已校准预测 |
| 原媒体精审 | 19个锚点/168幅原图＋每锚点前6控制的精确目标/手物理证据 | 只批准19个GRASP结果标签，保守13事件；0动作/0planner/0恢复成功批准 |
| 仿真准备 | 两个原TRAIN实例各352示范控制，pizza另有两段16控制重放，总736实际控制 | 两例抓到指定目标；旧pizza世界/控制器/RNG重放并非完整生产起点恢复，时钟等补丁仅CPU验过 |

86项A800 RL/数据/wire回归和8项因果反馈测试全部通过。后续新增H0显式事件预算负控已本地8项pipeline回归通过：OOF与final训练的**合计**事件曝光必须同时满足用户票和5遍总上限，不能用最大允许5遍覆盖一张更小预算的票。独立成员代码review仍待，不合main。

历史上曾等待追加reset额度；**用户14:44已明确取消本轮全部准备额度，此等待已作废**，后续实际采集/验收见本页顶部。不能据此继续要求重置次数批准。

## 现有事实

- `raw-v1`封存242个已关闭候选ZIP，347,415,662B，86个episode/43任务。结构、原index SHA、三相机图像、动作时钟通过。去重399个重叠控制行，无矛盾。
- 6194个视觉锚点；5734个有完整32步同意图真实动作。这是“目标可读取”，不是“这些动作正确”。
- 精确asset↔BDDL映射后有360个局部抓持成功候选锚点，合为19段连续证据/11个episode；相邻窗口可能仍为同一次事件。不能称360次成功或19次恢复。
- 主agent亲审75个窗口/19张sheet/675幅原RGB，覆盖所有提出成功候选的archive及任务/事件分层。每窗口只看三个时刻，**不是全视频验收**。逐条记录见 `configs/recovery_sft/owner_review_v1.json`。
- 初始`admission-v1`审批为空；最新`admission-v3`只签发19个逐成员GRASP结果（TRAIN17点/11事件/9来源组，dev2点/2事件/2来源组）。三个池仍拒绝训练；缺FAILED/IN_PROGRESS语义验真、正确交接/恢复续段、完整动作质量检查和dev事件覆盖。逐条判断见`configs/recovery_sft/exact_outcome_review_v2.json`，不能将其推广为整个clip批准。
- `bringing_in_wood`实例9/111的模板与状态scope不匹配，2个episode/179锚点隔离。不能猜对象数字后缀、用另一模板强行补齐。

## 服务器与文件

共享根：`/data/workspace/wsy/behavior2026`。本轮选lc1，lc2已有队友GPU进程不动；不使用robo。独立环境沿用 `envs/g05-py310-cu128`，不安装/升级。代码仅Git传送到独立 `src/recovery-prep-<commit>`，不用scp/rsync覆盖源码。

| 内容 | 共享根下路径 |
| --- | --- |
| 原专家数据 | `datasets/memlite-stage1-20260930-v4` |
| 候选封存/审计/最新准入 | `datasets/recovery-candidates-20261009-v1/{raw-v1,audit-v4,admission-v3,source-metadata}` |
| 完整因果上下文/人审图与物理证据 | 同候选根`causal-history-v3/`、`span-review-v1/`、`outcome-approvals-v3.json` |
| 本轮CPU/GPU证据 | `runs/recovery_sft_preparation_20261009` |
| 新工程run（非正式模型） | 同run根`l0-trainer-v1/`、`h1-trainer-v1/`，各有同级`.supervisor`、`*-ticket.json`、`*-audit.json` |
| H0诊断cache / H1 processor回执 | 同run根`h0-feature-cache-v1/`、`feedback-processor-v1.json` |
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

原窗口内的`history_is_partial=true`保留不回写。新`causal-history-v3`从同episode完整planner日志重建6194锚点/86episode的既发意图/记忆/时长，与实际低层context SHA全部交叉绑定；提供当前`observable`及规划前`predecision`两种视图，H1不能把本次答案放回自己的输入。只有这份完整join可以解除部分历史限制，缺失时仍拒绝，不能补写历史。历史JSON不包含物理标签。

## 训练配置与梯度边界

- **H0：** 冻结整个高层，member-conditioned答案前context与归一化本体输入最多4个过去/当前检查点；detach后进小GRU适配器和四类结果头。loss只更新适配器/头，低层完全不参与。并行成员需要不同条件prefix，不能复制同一个bundle向量假装分别判断。当前小验为每成员单独一次VLM prefill，**没有实现与正常规划共享prefill，不能宣称零额外推理成本**；部署前须实测或优化合批/共享表示。当前是独立模块+图验收，不是已部署或已校准功能。
- **H1：** 高层正确交接/恢复CE，初始LR `1e-6`、micro1/global32候选；使用分来源实例OOF或独立冻结observer的预测反馈，不能拿oracle结果直接当部署输入。新高层hash会使旧特征/校准失效。OOF生成器及真实trainer已实现；真实正确目标、实际OOF产物及校准后部署整合仍是后续门。
- **L0：** 从low98414精确恢复，包括全部192个已训LoRA张量。`configure_recovery_expert_only()`保留其值但冻结，只开放322个动作专家张量；FM loss不回传VLM或高层。初始LR `1e-5`，micro4/global64候选；32步预测/16步执行/起点0，4噪声样本、原stats不变。不能重构无LoRA的Stage-A替代“冻结LoRA”。
- 每线最多1000更新、合格新事件5遍或4小时先到为止；不为凑预算重复一个小数据集。H0小头未必值得占8卡，正式world/microbatch须按实际loader和数据量验收；不要把原bs256强加给很小的恢复数据。
- W&B沿已可写team `hanhanyy-fudan-university-school-of-management` / project `behavior2026-g05`；本次两条工程run在`memlite-recovery-p2-20261009-engineering`，均真实写入/读回/finish。将来正式run使用独立group `memlite-recovery-p2-20261009`，当前没有启动。仅用0600共享秘密路径，禁止把凭据、checkpoint或原视频上传日志服务。

### 有限数据的八卡运行时

`src/g05/utils/training/recovery_runtime.py`提供独立的通用更新原语，已由新`train_memlite_recovery.py`接入（旧Stage1源码不变）：

- 对已经准入的全局事件顺序分片，microbatch尾部不复制真实监督样本；空rank只做一个零权重前向，真实样本数/有效分母/事件遍数都不增加。仅适用于G0.5这类行间独立、无跨样本BatchNorm的模型。
- 各rank先累计`loss × 有效分母`反向，DDP平均后乘`world/global_denominator`，再clip与Adam；不能平均各rank的mean loss。零全局监督拒绝weight decay/更新。
- 每个更新前先协调读取/前向错误；任一rank异常均拒绝继续，已发生的optimizer异常必须从上一个完整checkpoint恢复，不把部分rank更新当作已提交步。
- schedule指纹绑定代码、父权重、admission、配方、world/micro及精确样本顺序。调用方仅在更新成功返回后推进游标，沿用已验的原子checkpoint和全rank RNG保存。

双CPU及八NCCL rank小网络的梯度与Adam/RNG逐位恢复测试已通过；新增实际高低模型各2更新/保存续训也已通过。同步CPU读样本时显式保护CUDA RNG，避免原数据worker里的`manual_seed`重置主进程FM噪声。当前通过的是原专家工程分支，**正式恢复数据分支仍须逐池准入**，不能用工程票绕过。

### 实际训练入口与先后顺序

1. `cache_recovery_features.py`读取已签发outcome成员/规划前观测，以冻结高层提特征；正式模式要求SHA绑定且数据池ready。`--diagnostic-limit`仅接受1–4请求，产物带不可训练标志。
2. `train_memlite_recovery_h0.py`实际训练三个分组OOF头＋最终结果头，训练/校准/目标来源组互斥，同一事件总曝光不超过票内预算及5遍。做温度校准并导出逐成员真实预测；支持完整Adam/游标/RNG保存恢复。**当前没有运行H0拟合或校准。**
3. 校准不是“拟合loss降了”即通过：每个已知类别至少20个独立真实事件和20个高置信预测，precision≥0.9、Wilson95下界≥0.8，错误成功率上界≤0.1。未通过则反馈UNKNOWN，不能为启动训练降门。H1 dev用只在TRAIN拟合/校准的fold头，不能拿在该dev上调过温度的最终头输入H1 dev。
4. `train_memlite_recovery.py --component H1|L0`分别消费验证目标/实际执行动作；H1只用预测反馈，L0约70/30原专家与验真新事件。三个池和三个产物独立，H0结果批准不自动批准其动作或规划。
5. 两类trainer必须经`scripts/infra/launch_memlite_recovery.py`的CPU票据/源码/父权重/文件SHA/GPU空闲/空间/累计预算门。工程票只能8卡、micro1/global16、累计2更新/1800秒；正式票另需明确授权与ready数据，当前不存在可开训的正式票。
6. H1完成后，在使用其新backbone的observer上重提特征并重新验证校准；旧高层SHA的校准不得直接套用。真实反馈服务接入、时延和短闭环效果都须验证，不因trainer产物存在就部署。

本次工程产物只可复现实验，不替换正式高48045/低98414父权重。L0保存16.54GB、H1保存26.59GB各保留两步，续训会校验完整内容；这类小数据每步保存的开销已计入监督器预算。

## 环境/图验收命令

从自己的**干净冻结worktree**运行，先看GPU是否空闲；绝不对活跃源码pull。

```bash
source scripts/infra/activate_a800_training.sh
export PYTHONPATH="$PWD/src"
# 包含实际验证过的NCCL P2P/IB关闭和bond0，以及torchcodec所需NPP/FFmpeg库路径。
# 未source时即使torch可import，也可能在读专家RGB视频时失败。

CUDA_VISIBLE_DEVICES=2 \
  python scripts/rl/memlite_online/tools/check_recovery_parent_graph.py \
  --recipe configs/recovery_sft/a800_p2_v1.json --component L0 \
  --output /data/workspace/wsy/behavior2026/runs/recovery_sft_preparation_20261009/NEW_l0_graph
# 完成并释放GPU后，H0用另一个新输出目录单独运行，不并发占同卡。
```

以上读取两条已验原专家TRAIN样本，只做真实前反向，**无optimizer**。H0采用明确的合成图测试目标，不把数值当物理标签/loss效果。八卡硬件通信小验已通过，但单卡模型图通过也不等于新SFT八卡全链验收。

`prepare_recovery_sft_ticket.py`只产CPU准备票，不启动训练。初始GRASP小试的实际类别/纠正续段及TRAIN-dev冷起点现已验收，见顶部；H0拟合校准与H1/L0效果验证属于之后的数据小训。既不能以“没有训练好的头”宣称trainer不可运行，也不能把本次技术准备完成推广到所有技能部署就绪。

## 留给下一冻结RL的修复

- 采集器从真实env object_scope保存label-only精确实体映射；同对象同一手连续确认抓持，覆盖任务工具，不只看最终goal对象；少量正常抓持窗口补充失败偏置。双臂之一在PLACE不再掩盖另一只手的意外掉落。
- 缓存critic冻结前缀特征，分别计算optimizer.step前后value loss、return/value统计与EV；目标方差近零时EV为null。actor KL门不改，旧日志不回写。
- 因果反馈ledger按session/task/instance/episode/高层及observer SHA隔离；同意图持续时间/刷新次数不因重复规划清零，明确RETRY才开始新尝试；最多保留64种近期意图的重试计数。预测结果要求校准、足够置信度及至少两个不同检查点一致，否则UNKNOWN。

这些改动**没有热改原10383训练源**。用户随后明确批准提前停止，旧共享RL已在13:12 CST安全退出并独立验证第180次完整checkpoint；下轮只用新冻结源、独立run和预算，不自动复活旧RL或1000条公共评测。
