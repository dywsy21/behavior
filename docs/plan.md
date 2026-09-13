# Goal执行计划：MEM-Lite协同训练与成功率提升

本文件是goal的执行总计划与实时进度入口，保留原有E0–E7路线、验收条件和执行证据。原路径为`docs/archive/MEMLITE_COORDINATION_EXECUTION.md`，2026-09-12按用户要求迁至此处；后续直接维护本文件，不再另建平行版本。

当前仍是为50任务大训练筛选通用方法的小规模准备阶段。三人分工、任务ID和实验预算见[团队任务板](TEAM_PLAN.md)，通用RL路线见[RL方法计划](RL_METHOD_PLAN.md)，文件位置见[服务器目录表](SERVER_LAYOUT.md)。这些文档与本计划应同步维护，历史记录不得覆盖用户最新要求及当前预算。

**2026-09-13最新职责：** 用户将更多训练数据、特别是错误恢复数据交给一位队友，将通用RL交给另一位队友；本线程/Codex集中研究高低层怎样训练更有效、条件服从与协同接口、相应方法评测和集成。原E0–E7目标/质量要求不缩减，但不重复承担队友的数据扩充或RL实现；需要新数据时提出明确接口与证据需求，不擅自平行重建。

**续接约定（用户最新明确要求）：** 超参/训练方法答疑已经完成，不再重复回答，候选依据见[方法文档](experiments/2026-09-13-fm-training-method-candidates.md)。compact后从下方最新真实执行记录续做FM/AR训练与闭环，核验既有进程后再行动；不能把历史问句当成当前问题。此约定已加入AGENTS.md。

## 实时进度（最新记录在前）

每完成一项实质工作或出现状态变化，立即更新本区及相关待办；规则见[AGENTS.md](../AGENTS.md)。记录时间、负责人/任务ID、做了什么、真实结果与证据、剩余问题和下一步；不等整轮工作结束才补写，不以聊天消息代替落盘。

### 2026-09-13 19:11（北京时间）：A4同历史诊断完成；未发现LoRA绕过或位置错位

- **Codex / AR-01，实际完成/0更新：** `ar_decode_consistency_a4_v1`的2604744已退出，result SHA `161649772d3a5b8a519a6f5916b6dd1201d26df1de3bbca2e341e23f4cbea43b`。两原train窗口各61个token，teacher完整前向与缓存强制同历史的MRoPE/类型mask全部一致，122个下一token的argmax有121个一致；96个LoRA模块在完整/缓存/自由路径均实际调用，没有发现整条解码绕过适配器。
- **数值与限制：** 两行full/cached eager CE分别15.08357/15.12535、13.05716/13.02461；最大logit差0.75/0.5，非逐位一致，不擅自声称所有数值路径完全等价。两次独立自由生成仍37 tokens且漏组。当前证据不支持用“位置错位或LoRA未执行”解释该原A4两样本，但尚未检查训练后的500权重、所有窗口或所有误差来源。
- **下一步：** 待当前500权重真实完成后复用诊断入口核验；现在继续实现显式原生Subtask-CoT视图与终止边界接续，输出监督只复用已审核同状态skills，不构造bbox/trace/FAILED/SUCCEEDED或新release。先真实输入/生成门再登记CoT训练，现有队列不变。

### 2026-09-13 19:09（北京时间）：AR同历史诊断46项CPU通过，原A4只读GPU检查已启动

- **Codex / AR-01，代码/真实CPU通过：** 新`probe_ar_decode_consistency.py`在同一权重/两条原train上，比较完整teacher-forcing与实际AR缓存循环的每token原始logits、目标rank/CE、MRoPE/类型mask及LoRA模块调用；另外分别自由生成，绝不把强制token历史当部署输出。robo独立`ar_decode_consistency_20260913`固定039e268，新增6项＋40项原回归共46 tests passed；GPU结果待验。
- **实际运行：** `ar_decode_consistency_a4_v1`于19:08:51启动，PID2604744，日志在同级`.launch.log`；原A4 SHA `61867047…`与原10行缓存中的前两microbatch各首行、seed17自由生成、GPU1/两CPU线程/最多40%单卡显存。两行各1完整teacher前向＋1真实缓存teacher诊断（最多96 tokens）＋1无GT自由生成（沿用原actor显式预算，不更改生成行为），0优化/保存策略/仿真，无自动重试。输入/权重/预算失败即停止；数值差异原样报告而不套未经验证的通过阈值。A4-AR及原生/五方法队列继续固定旧源码，不改训练配方。

### 2026-09-13 19:07（北京时间）：A4-AR固定80 CE下降，但自由生成仍全部漏组

- **Codex / AR-01，实际进展：** formal2316504已超过319/500，仍由1940901监管；原生及五方法后继未重复提交。原固定80的CE从step0的18.6797803降至100/200/300的14.0474166/8.2061711/7.1059599；原初始FM参考0.196943994070与A4逐值一致，step300参考0.1971009450（辅助指标，不是AR控制误差）。`formal/eval_step_300.json` SHA `83e11977c037f9b60dc4dfe742cd371cf88a6a275ea407414ed9d0e1d037d947`。
- **未通过部署门：** 四个评估时点每task固定一条自由生成均0/5完整动作组；step300均37 tokens、只有双臂残差块后终止，缺lower_body/双夹爪，未执行仿真、没有新SR。CE下降不能代替完整可用动作；不放宽23D合同或用GT/FM填空。
- **下一诊断优先级：** 在现有500预算内继续观察，不热改活跃源；先查同一token历史下teacher-forcing与缓存解码的实际数值一致性、LoRA推理路径和终止位置，再加原生Subtask-CoT对照及闭环。该检查只读原train和已有权重、0优化/仿真，具体有限GPU预算在运行前登记；CoT仍待实现，不把计划写成已完成。

### 2026-09-13 17:57（北京时间）：A4-AR四卡5步保存回读通过，正式500作业已启动

- **Codex / AR-01，实际通过：** `ar_a4_fulltrain_v2/smoke/checkpoint_inspection.json`passed，5次optimizer调用/192 Adam状态/80条实际train抽取、冻结不变、完整模型/Adam/四rank RNG回读通过；临时权重SHA `9f22d74f9886ee31a4bda9598a25de22e6680e90ce2b1176b5b02f4bf9f9a84c`，峰值rank0 reserved23,129,489,408 bytes。五步FM均0、只CE，真实LoRA更新在warmup第二次调用核验；第一步LR0被正确记录，不冒称零LR造成参数更新。
- **真实接续：** 原supervisor1940901已在17:55启动formal trainer2316504，仍固定171898c的`action_queue_20260913`。正式从原A4重新初始化/新Adam，不接5步临时权重；此时处于正式初始化阶段，尚未观察正式更新或宣称完成500。原生2222863及五方法队列保持等待；当前main文档8d27478已同步robo，活跃源码不热改。
- **下一次续接：** 先核验formal实际step/原初始80参考值、后续每100的CE/完整组自由生成；继续CoT与闭环准备及已排队FM方法，不重新答选型问题、不重复启动任何run。5步门是工程验收，不是AR方法收益或新进程断点恢复验证；goal仍active。

### 2026-09-13 17:50（北京时间）：AE×2正式500完成且未改善固定80；A4-AR已进入四卡保存门

- **Codex / M-01，实际完成：** candidate于17:45:40完成全部500 optimizer调用和保存回读；权重SHA `7f1c9acdfe52d3ffd6e98038c46a6d743a07766b1188e20c7b45262048396753`，504 Adam/504可训练状态变化、冻结不变、全有限、四rank RNG通过。原supervisor1902909/torchrun1912524均已退出，未重跑/追加。
- **效果与完整来源核验：** 原固定80总FM=0.2011440476，对control0.1982012083高1.485%；仅task0改善，其他四个退化。两个run四rank完整采样回执SHA逐对相等，各1000 microbatch，共8000真train抽取/950轨迹/每task1600次；不冒称像素逐位相等或新SR。暂不把AE×2纳入有效组合，详细证据/限制见[LR筛选结果](experiments/2026-09-13-fm-lr-screen.md)。候选step500诊断SHA `a58d046c…`，父A4不被替换。
- **AR真实接续：** 原A4-AR supervisor1940901已通过前驱500验收，四卡smoke trainer2307504运行，17:48仍在初始化/数据阶段，尚不能声称已完成5更新或正式500。native2222863及五方法后继保持等待。下一次继续核验AR实际更新/保存门；不重新答超参、不重复提交队列。CoT、闭环、M-03条件化候选与有效组合仍未完成。

### 2026-09-13 17:44（北京时间）：后续五臂全部已提交并核验真实等待，尚无新增训练更新

- **Codex / M-02、M-04：** 84110fa固定源码的五份supervisor于17:40–17:42实际提交；逐个PID/argv、status和spec SHA核验通过，均为verified_live_dependency、0更新。nvidia-smi进程清单中没有这些等待PID，未占GPU。依赖链为native2222863→Beta2289674→exec-weight2291489→新入口FM2294898→joint2296750→KI2297891；前一run完整500验收后下一run才开训，失败就停止，不是五轮并发抢卡。
- **唯一证据位置：** 全部在`dual_track_fm_ar_20260913`，对应spec SHA依次为`191a5393…`、`a6b18a6c…`、`e16d8670…`、`2d8595a6…`、`b9a7eec4…`；完整run/PID/SHA表见SERVER_LAYOUT。共用活跃worktree`method_screen_queue_20260913`固定84110fa，即使等待也不得pull或切换。
- **下一次续接：** 先核验FM AE×2的500完成/固定80/权重回读，再核验A4-AR进入smoke/formal；不要重新启动已排队任务或重复超参答疑。等待时可继续准备CoT和实际闭环接口，不能把上述排队/CPU检查当方法有效；两条路线效果、M-03条件化候选和有效组合仍未完成。当前部署权重及旧服务未变。

### 2026-09-13 17:39（北京时间）：串行筛选源码144项CPU检查通过，开始提交预登记五臂

- **Codex / M-02、M-04：** 独立worktree `method_screen_queue_20260913`固定84110fa，robo实际144 tests通过（1.58s），包括七个前驱配方/父权重、错误native身份/预算、500 checkpoint损坏、FM等待绑定和原模型回归。现按下述已登记顺序提交五个独立5+500 run，等待native2222863，0额外GPU直到前驱完成；提交后逐个核验真实PID/argv/状态/spec SHA，不用仅有文件判断已运行。17:38观测FM467/500，原A4-AR/native-AR仍等待，盘余665.82GB。

### 2026-09-13 17:37（北京时间）：省略语义审计完成；后续有限方法对照接续准备

- **Codex / AR-01，真实结果：** 861942c在robo 7项标准库检查通过，`ar_native_omission_audit_v1`完成10原train/20目标编码与20既有自由生成审计；result SHA `429d8fa2e9d09e63477dd94433b58ca166442c4ca8b07e29ea1f0b456590b09e`。20输出已有块均完整、无缺残差，但全部漏目标编码应保留的lower_body；task0还漏left_control，task2漏正在二值变化的left_gripper。9/10目标窗口允许省略双夹爪，另1只允许右夹爪；真实5/8步尾部窗口不改变本次noop判断。不是所有缺组都来自合法noop；也不将归一化0当物理保持、不据此宣布历史打转根因/AR不可行。0VLM/优化/仿真，无新release或部署合同放宽。
- **Codex / M-02、M-04，实质代码/待验：** 两现有trainer增加仅限已声明run的串行前驱类型/父权重/配方校验，FM等待也不占CUDA、绑定等待helper SHA；沿用已验PID+ticks+argv与500 checkpoint完整验收。8项FM标准库测试通过，新增10项真实action runtime CPU测试待验；活跃fb40145/171898c/6e2587b三个worktree未改。
- **后续五臂预登记：** 接在既有原生task-AR后依次`fm_beta_stratified_v1`→`fm_exec_weight2_v1`→`fm_action_control_v1`→`joint_a4_fulltrain_v1`→`ki_a4_fulltrain_v1`，每臂独立原A4父/新Adam、原950/50、seed41/四卡global16/4 worker/50warmup/cosine500，5步保存门后独立500；M-02另外各有原runner两临时更新GPU门。Beta/前16步2:1加权各只改一个因素，对照已完成原FM control；后面三臂共享新loader与LoRA/AE1e-5，分别FM、CE+FM不隔离、CE+FM隔离。原固定80口径不变，每100记录；后面三臂另有CE/自由动作。总计2500正式更新，不是五轮5000或全矩阵搜索；数值/输入/保存/前驱失败即停、不重试/不部署，不借用前驱训练权重，盘余至少120GiB，旧服务保留。当前只写好入口与预算，尚未提交这五臂；真实CPU通过后逐个登记准确commit/PID/spec。仍需效果比较、闭环、CoT、条件服从及有效组合，不把排队当完成。

### 2026-09-13 17:31（北京时间）：增加原生AR省略部件的只读语义审计

- **Codex / AR-01，实质代码：** 新`probe_native_ar_omissions.py`及7项标准库单元检查（已全部通过）。严格区分完整但省略组、缺残差/截断，以及codec认为noop的组；不填零、不修改部署合同。源码确认旧codec会省略恒定二值夹爪，而缺NN组解码成归一化0，这不能自动当作物理保持。
- **下一只读预算：** 拟`ar_native_omission_audit_v1`，新独立Git worktree/commit；只对原10条train做开关noop两种20目标编码，按task/episode/frame/requested_index/bundle精确连接两份已完成native GPU回执的20自由生成记录。2 CPU线程、0VLM/优化/仿真、不构造release；报告目标编码的合法省略与实际缺失，不用专家答案修补actor。首次身份/解析错误即停，真实结果尚待运行。FM和两份AR训练队列继续不变；main文档已同步99ee6c2，robo协作clone已ff pull，活跃源未改。

### 2026-09-13 17:26（北京时间）：原生task正式训练已提交，真实等待A4-AR

- **Codex / AR-01：** `ar_native_task_fulltrain_v1`于17:24:39提交，supervisor2222863；spec SHA `2f01682d6912b14cd7c8d4d6371694cd81ea1303756968d3722bda68811c42af`，源6e2587b。17:25真实PID/argv与status共同核验：等待1940901（start_ticks327195368），0GPU/0原生AR更新。前驱成功验收才执行下述原生5+500，不是已开始神经训练。当前三个活跃训练/等待worktree全部保持固定，不热改。原生训练及最终闭环结论仍待完成。

### 2026-09-13 17:24（北京时间）：续接规则落盘；原生task短门完成，准备正式串行训练

- **Codex / AR-01，真实新结果：** `ar_native_task_gpu_gate_v2`已退出0，result SHA `09d6244d99053c8295fad280267044fabc0b8df8709ee33739a7c8e8efb5577c`。原生权重/原生词表、192 LoRA的CE连通与FM零梯度、两真实临时Adam更新/冻结检查通过；同输入CE15.3403177→15.3013954。五task更新前后均缺完整动作组，未保存策略/下发模拟器；仅工程可训练性通过，不是AR效果收益或SR。
- **下一有限训练预登记：** 使用已119 CPU/真实task门验收的独立worktree `ar_native_task_20260913`、代码6e2587b，拟`ar_native_task_fulltrain_v1`。只等待既有A4-AR v2（supervisor1940901）真正500完成/权重验收，再进行四卡5步保存回读门→从原生G0.5 SHA `072211e5…`和新LoRA/Adam独立500；不从A4或前驱权重接训。原950/50、seed41、global16、每rank4 worker、LR1e-5/50warmup/cosine500、原80辅助参考＋CE及五task自由生成；无MEM-Lite planner依赖、无CoT监督、不自动部署/追加。输入/依赖/数值/保存错误即停止，保留120GiB盘余量。当前仅预登记，尚未提交队列。
- **真实前驱/边界：** 17:23核验FM候选387/500，1902909/1912524仍活跃；A4-AR1940901确在verified_live_dependency、0更新。不重启这些任务或热改其worktree。原生预训练允许省略不活跃部件与我们全23D合同之间的语义差异仍须分析；完整训练、CoT、闭环、KI/joint及其余FM候选/组合均未完成。
- **用户续接要求/同步：** 已在AGENTS与本计划记下不再重复超参答疑，下一轮按实际run续做。上一并发fetch/pull出现FETCH_HEAD多目标错误，已改为顺序、显式当前分支ff-only pull并确认up to date；无覆盖/重置用户文件。

### 2026-09-13 17:13（北京时间）：原生skills两更新门完成；task-only真实门开始

- **Codex / AR-01，真实验收：** `ar_native_skills_gpu_gate_v2/result.json`complete，SHA `4b527cbabe11f4770a32901f5a2146abe1d4666d83ad4a35a2384bf0fd514d6d`，2144305已退出0。原946权重逐项精确、192新LoRA/零B初始化通过；CE连通192 LoRA（初始96非零，符合零B）、FM0，两次真实临时更新/192 Adam/冻结状态检查通过，峰值reserved23,758,635,008 bytes。固定输入CE15.248372→15.296524，未改善，不包装为方法收益。
- **自由生成/限制：** 五task更新前后均不满足全动作组合同（task0为19 tokens，其余各37；并非用满生成预算），没有下发仿真动作或保存临时权重。这说明原生基座在当前skills条件下也不能直接满足完整23D输出要求，但两更新不是否定AR能力/已完成迁移训练。需同任务模板及原完整数据实训，再分清预训练noop省略语义、格式学习与控制质量。
- **下一实际运行：** 新`ar_native_task_20260913`固定6e2587b，119 CPU tests实过；提交`ar_native_task_gpu_gate_v2`，GPU1同两更新/五task前后生成预算、原生词表、无planner依赖，结果待验。FM step300=0.2009022778，而同step control0.1975404179，仍是未完成500步的候选点；现有训练/排队预算和源码不变。

### 2026-09-13 17:10（北京时间）：补原生task-AR无planner输入接口，准备task臂真实检查

- **Codex / AR-01：** native-skills v2实际进程2144305运行，已写restoration、真实梯度以及五task更新前生成回执；最终两更新result仍须核验。native CPU输入result SHA `099858a2d15c41241d3806833cb444db17e277cc05d71e928804f45fad4ee150`。原10行本体实测全部6×27，不将图像历史误当动作历史。
- **实质接口修正/待验证：** 原生`native_task` actor不再要求一个并不输入网络的MEM-Lite技能投影；只接收精确官方动作模板、任务/本体标识、18图像槽与六帧本体状态/真实padding mask，并剥离memory/skills/outcome等sidecar。普通skills actor的语义检查保持，训练侧同状态审计保持；新增无planner客户端/禁止teacher输入的CPU测试。此修正尚未跑真实task臂，不能宣称闭环已通。
- **下一步：** 新独立源码通过CPU后，用同原生权重、原数据/seed和两临时更新预算执行native-task v2，串行等待skills进程实际结束。原生完整5+500仍须两门及输入验收后才排队；FM当前已实查306/500，A4-AR仍等待，不改它们的源码或预算。

### 2026-09-13 17:06（北京时间）：原生词表117项CPU＋20真实输入视图通过，进入GPU v2

- **Codex / AR-01，实际完成：** 独立`ar_native_vocab_20260913`固定82ccb19，robo 117项CPU passed；`ar_native_input_gate_v1`真实10 train行×skills/native_task共20视图complete，原生动作区间[248077,252187)、EOV252187/state252188，无HL_END，全部60-token/8码块、前缀一致/无截断/23D完整。不是自由生成效果。
- **运行中：** 仅提交原生skills `ar_native_skills_gpu_gate_v2`，新源已明确保留原生词表；仍GPU1两临时更新＋五task前后生成上限、0发布权重/仿真。task臂等前者实际结束后才运行；旧失败v1保留，活跃FM/A4-AR源码未动。完整native训练入口的CPU编排检查已包含在117项内，但尚未正式排队或神经更新。

### 2026-09-13 17:03（北京时间）：原生加载首门失败定位到HL_END挪动state，保留原生词表重验

- **Codex / AR-01，实际失败/0更新：** dc0d085的native-skills进程2081801已退出1，v1日志和manifest保留；严格946项检查在`model.vlm.input_proj.weight`处拒绝252189→252190行的默认部分加载，尚未执行优化/自由生成，不重复声称native已通过。源码定位：MEM-Lite无条件在`<EOV>`后、`<state>`前注册`<HL_END>`；官方原实现没有HL_END，因此原生state252188被挪到252189。动作词表范围未因此挪动；尚不能把此兼容性缺陷宣称为历史打转的已证实根因。
- **修正/待验：** 原input_preprocessor与Git副本逐字一致，现仅加显式`register_memlite_hl_end=false`原生模式，默认true完整保留A4/MEM-Lite词表；原生模式不加新token，继续要求946基础权重逐项原样恢复，不放宽部分加载/补零门。进程内声明第四个Git扩展及基础policy的processor绑定，不热改原源码/环境。CPU真实native tokenizer门拟`ar_native_input_gate_v1`（10行×skills/native_task两视图，0 VLM/优化）；再新native-skills/native-task **v2** GPU门，不启动尚未运行的task-v1。
- **原生正式训练草稿：** 完整action trainer已增加显式native父权重/条件选择，以及只等待已声明`ar_a4_fulltrain_v2`真正完成、验500权重后才启动的串行依赖；原生不会从A4/前驱权重接训。仍原950/50、四卡global16、5步保存门→独立500、原80＋自由生成；A4参考值只约束A4初始化，native另记其原始参考。新增CPU用例待验；当前未提交原生正式训练队列，不把草稿算实训。
- **FM现状：** 原AE×2继续，step200固定80=0.1988764574，同步数control=0.1972260463，候选暂差约0.837%；不作最终500/SR结论，不改现有预算。完整双路线及组合闭环继续待办。

### 2026-09-13 16:51（北京时间）：原生AR入口92项CPU通过，开始真实native-skills短门

- **Codex / AR-01：** 新独立worktree `ar_native_20260913`固定dc0d085，在robo实际92项CPU检查通过（包括原FM/AR/KI回归及新增native加载/模板/五task取样）。首个`ar_native_skills_gpu_gate_v1`已提交GPU1真实入口，预算同上：原生权重、两临时更新、五task前后自由生成，0发布权重/仿真；GPU阶段结果未出。启动不算验收，须检查真实进程、restoration/gradient/result回执。
- **后续：** 同源native-task臂等前一进程结束才运行，不挤占第二张卡或热改当前源。现有FM500与AR等待不变；完整原生AR训练/CoT/闭环及FM候选组合仍待完成。

### 2026-09-13 16:49（北京时间）：确认原生G0.5身份，增加明确的原生AR初始化与模板

- **Codex / AR-01，实查完成：** 上轮新增首个AE×2诊断证据，属于progress；本轮clean pull/fetch后实查1902909/1912524仍活跃（最近日志153/500），1940901仍verified_live_dependency。原生权重实际为`/mnt/sdc1/robodojo/checkpoints/G05/g05-base/checkpoints/model_state_dict.pt`，11,440,519,964 bytes，SHA `072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327`，与本地HF下载metadata一致（revision `e312be81e90c56a55bcb26b57429bd39a335b449`）；946基础状态、0 LoRA，仅权重无Adam。配套config SHA `c98af37352c0f600341d2ecbdb49fcdfa87812198654448991615065fa1a4461`，27D接口；没有重新下载权重。
- **实现/待验证：** 新`native_action_initialization.py`逐项验证原946状态与新192 LoRA的零B初始化，拒绝A4 adapter混入或随机/部分加载。AR数据视图增加`native_task`，与原`BaseSamplesBuilder`动作模板精确对应（包含终止竖线），不改已有skills/task视图。现有GPU入口支持显式native/A4起点；native每task选一条真实train窗口，在两临时更新前后各自由生成一次，保留不完整动作失败而不补零/FM。新CPU用例及语法检查已写，真实CPU/GPU尚待验收；不热改两个活跃worktree。
- **下一有界执行：** 拟新`ar_native_skills_gpu_gate_v1`与`ar_native_task_gpu_gate_v1`，GPU1串行、原10行train缓存中的原前两行作两临时更新、seed41/LR1e-5，五task×前后共10自由生成/臂、0保存权重/仿真；前者分离native与A4初始化差异，后者检验官方动作模板/纯任务条件。实际代码commit/输入与权重SHA由manifest固定；当前GPU1约45GiB可用，门槛40GiB，不停别的任务，共卡时间不作公平吞吐。过真实学习门后才接原950/50的有界原生AR训练，不重复十行拟合冒充正式训练。
- **CoT边界：** 发布权重配置predict_cot=true，而官方R1Pro微调recipe默认false；原五任务parquet无原生CoT字段。先验证原生权重的动作-only微调入口，不捏造grounding/action-hint标签，不冒称复现论文BEHAVIOR权重或完整CoT配方。原生推理CoT与监督可用性仍待进一步核实，完整双路线/组合/闭环未完成。

### 2026-09-13 16:35（北京时间）：AE×2首个100步评估尚未改善，超参答疑不改活跃训练

- **Codex / M-01，只读实查：** 16:33:35核验FM supervisor1902909/torchrun1912524仍活跃，`fm_ae_lr2x_v1`处于formal；首个原固定80结果`formal/fixed_diagnostic/step_100.json`为0.1986116943，SHA `dca01b74cc0e9f488274b64b3d39bd02c546cce7cc33668bd14b9843146f1ad6`。同更新数control100为0.1971548254，候选高约0.739%；相对A4父高约0.847%。这是中途一个固定窗口/噪声诊断点，不是500步结论或SR，不能据此终止或追加预算。各task依次0.14445460/0.16000390/0.27245573/0.16426686/0.25187739。
- **解释/方法顺序：** A4留出均值改善0.92%不等于训练拟合速度；既有10行100步能降83.3%，不能把问题直接归为LR太小或clip过强。低成本优先分组LR/合理日程、保持原Beta期望的时间分层；实质方法候选是离散动作CE训练低层VLM＋FM训练动作专家的KI-inspired隔离，并以无隔离joint作对照。执行段/动作组加权须保留原未加权评估，容量和梯度冲突处理按诊断决定。既有工程门不算方法收益。
- **未完成/安全边界：** AR supervisor1940901仍verified_live_dependency、0更新；不把排队或KI辅助CE称为纯AR效果。本轮只复核配置/进程/结果并同步文档，未热改源码、改LR/停止进程、新启训练或仿真。保持现有500步上限，候选最终验收后才按既定依赖接AR；完整双路线训练、闭环和有效组合仍未完成。

### 2026-09-13 16:21（北京时间）：AR v2实际处于已验证等待，FM候选42/500

- **Codex / AR-01，已提交但尚未更新：** 171898c在robo实际80项CPU回归通过；`ar_a4_fulltrain_v2`于16:18:15提交，supervisor1940901，method spec SHA `27d22490bb4ccaa9319cc37699a7754471dd91ba60d97f6f5995b0881c7cc930`。16:20实际`status.json`为verified_live_dependency，确认前驱1902909的start_ticks327057321；无CUDA、0 AR更新，不是仅有启动计划。固定worktree `action_queue_20260913`现被等待进程使用，不能pull/switch/热改。
- **预算/接续条件：** AE×2须真正500完成并验收权重SHA，才依次启动AR四卡5步保存门、从原A4重新初始化的500步；microbatch2/累积2/global16、每rank4 worker/prefetch2、seed41/LoRA1e-5/50warmup/cosine500、全部原950 train来源及原50 eval隔离、每100原80诊断与每task一个自由生成。AR不使用FM监督/动作补齐，不更新B、不自动部署。v1等待API失败记录完整保留，无已训练步数被重跑。
- **FM现状/未完成：** 1902909/1912524原候选进程保持，最新日志42/500，没有新的固定80候选点或SR。control500及其0.1982012结果已完成；AR真实训练、原生AR/CoT、KI/joint同入口对照、其他FM候选和有效组合、真实闭环仍未完成。继续按具体活进程检查，不因SSH观察超时重启任务，整个goal保持active。路径与协作进度同步到main文档，模型实现仍只在feature待独立审查。

### 2026-09-13 16:17（北京时间）：AR排队首版在Python等待API处停止，0更新；补兼容等待

- **Codex / AR-01，实际失败：** bcfec9f再次74 CPU通过后，`ar_a4_fulltrain_v1`于16:12:54提交，supervisor1919165、spec SHA `3494e1d4d3c6882c827633e4738c13e2b36b9011c49fda78396dc8666b5ab8a3`；16:12:57因服务器该Python没有`os.pidfd_open`退出，进程已实查不存在。未启动AR trainer、0优化/0GPU/0仿真。失败run保留，不把排队提交算作AR训练。
- **修正/待验：** 不升级共享环境，改为每10秒核验`/proc` PID＋启动ticks＋准确argv的只读等待；识别消失/zombie/PID复用，始终不发送信号。仍须前驱真正complete并验500步权重SHA才放行；新增6项含真实当前进程的CPU回归，拟新`ar_a4_fulltrain_v2`，不覆盖v1、不重试FM或重复任何已训练步数。
- **FM实证/参考门：** AE×2真实1902909/1912524仍在，日志15/500；此刻四rank分别已记录32/31/31/31个microbatch，与control对应前缀完整采样回执逐项相等。只证明已观察前缀，不冒称全500已配对。AR参考门result SHA `7530aa9facead5b5d7b9ac4a7c4fb2f5f6f9e46eae4bdbb8faf72bea94d0d568`，两前向完全相等证据保留。整个goal继续，当前没有新的方法收益/SR结论。

### 2026-09-13 16:11（北京时间）：AR真实参考数值逐位相等，准备提交串行AR500

- **Codex / AR-01，真实完成：** 6454980的`ar_reference_metric_gate_v1`退出0，两次真实无梯度前向均0.0640164241194725，与原control同父/同两行/同seed771结果完全相等；AR/连续标志与CPU/CUDA RNG恢复，0 optimizer/仿真。这证明该输入上的参考口径，正式初始80窗口仍须独立复现父A4，不能用两行替代全80。
- **实施调整/理由：** 将尚未启动的AR/joint/KI/同入口FM配方改用原A3已有的每rank4 worker/prefetch2（不再把初稿workers0同步读视频用于正式训练），保留新入口私有loader RNG与全部来源校验。目的为避免GPU等待CPU视频读取；未实测吞吐增益，也不宣称与旧入口每个随机增强逐位相同。所有新入口对照共享此设置；保存RNG/游标不是已验证stochastic worker的精确新进程恢复。对应CPU用例更新，待重验后提交`ar_a4_fulltrain_v1`串行等待AE×2。
- **当前边界：** AE×2正式1912524运行中；AR尚未提交，原生AR+CoT、KI/joint实训、方法闭环/组合尚未完成。本轮未新增/修改训练数据或部署actor，未合入未经独立审查的模型代码到main。

### 2026-09-13 16:08（北京时间）：AE×2正式500进程启动，AR真实参考门运行中

- **Codex / M-01：** AE×2四卡5步保存回读passed，checkpoint SHA `6da58b5f9e3e86f5f95623fb065be5573b1570c70e882e6d432e1a3d0083ba6a`，504 Adam均5、冻结不变、四rank RNG与各20次真实抽取。正式torchrun1912524已启动（16:07实查），仍从原A4初始化，尚未得到500更新结果。
- **Codex / AR-01：** 最新6454980在新`action_ready_20260913`再次通过74 CPU测试。现启动已预登记`ar_reference_metric_gate_v1`（GPU1、2真实无梯度前向、0更新/仿真），基于原两行/seed771核验prefix-only FM参考；只在候选已进入formal后启动，显存充足、不停其他服务。此CPU/GPU共享区间及整轮墙钟不作为公平加速对照。AR排队仍等该门结果，未声称AR训练已开始。

### 2026-09-13 16:05（北京时间）：完整AR编排74项CPU测试通过，补真实参考评估门

- **Codex / AR-01：** 7a7941a在独立`action_queue_20260913`实际74项CPU测试passed，包含checkpoint模型/Adam时钟/动量/RNG/下一sampler游标破坏检测；这是编排门，不是新进程恢复或实际AR500。串行等待使用真实pidfd、核验具体argv与前驱权重，不因观察超时重启任务；supervisor新增隐藏CUDA，等待时不占GPU。
- **拟参考门/预算：** 新`probe_action_reference_metric.py`，`ar_reference_metric_gate_v1`预定GPU1、完整父A4、原2行train缓存/seed771、仅2次无梯度前向、0优化/仿真。比较新AR模型入口的prefix-only FM和已完成control GPU门原值0.0640164241194725，并验训练路由及RNG恢复；当前仅语法检查。等AE×2进入正式阶段并确认显存余量才运行，避免阻塞其阶段资源门。共享区间不计公平吞吐；队友服务保留。
- **下一步：** 参考门通过后提交`ar_a4_fulltrain_v1`等待当前AE×2真实完成，再独立5+500；若参考不符则修正而不提前长训。当前AR尚未排队/更新，AE×2四卡5步仍运行，完整双路线结论未完成。

### 2026-09-13 15:59（北京时间）：AE×2已启动并通过GPU门，AR正式配方65项CPU检查通过

- **Codex / M-01，运行中：** `fm_ae_lr2x_v1`于15:55:15启动，supervisor1902909、固定fb40145，method spec SHA `d661502417afe2dce58f8e3b26a5f7579e94dd4f0f3e08ec798b08eb6e76b160`。15:58核验真实进程与单GPU门：A4完整恢复/原评估口径相同、4 microbatch/2更新通过；四卡5步torchrun1903487运行中。正式500尚未开始，候选效果未出；未改变旧服务或已完成control。
- **AR-01实现验收：** 新独立`action_trainer_20260913`固定2f095df，新增11项加已有54项共65 CPU测试通过；包括真实Adam warmup前两更新、参数覆盖、原FM入口保留与评估不进入训练。尚未真实运行完整AR trainer，不能拿单元门声称训练完成。
- **下一有界执行预登记：** 为持续利用四卡且不让两轮四卡训练争抢显存，新增仅依赖已声明`fm_ae_lr2x_v1`的串行启动门：验证具体supervisor argv并以pidfd等待其实际退出，再验500步checkpoint与SHA，成功后才放行AR的5步保存门→独立500。依赖失败就停止，不重试、不从该FM候选权重接训；AR仍从原A4初始化。拟新run `ar_a4_fulltrain_v1`，首版无新进程resume宣称、无自动部署；新等待和checkpoint篡改测试待验。用户goal内的原生AR/CoT、KI/joint实训、后续闭环和有效组合没有缩减。

### 2026-09-13 15:54（北京时间）：FM control500完成验收，准备唯一AE×2候选；完整AR训练编排已写

- **Codex / M-01，前轮为progress：** 上轮完成54项回归、40真实AR视图和Git同步；本轮clean pull/fetch后重新核验真实进程。control于15:50:53完成，supervisor1499025/torchrun1508221均已退出；`formal_checkpoint_inspection.json`passed，step500 SHA `def222a6674e6ac92e6ee982c22836b789240f1542c459d5de2111cd646a244e`，504 Adam均500、全部模型/Adam有限、冻结状态未变、四rank RNG与各2000次train抽取验收。不是仅凭PID退出认定成功。
- **实际效果/限制：** 原固定80 step500=0.1982012083，相比A4父0.1969439941高约0.638%；五task依次0.14149776/0.16176960/0.27347011/0.16389004/0.25037853。此配方是同A4新Adam/50步warmup/cosine500，不是原日程的等价续训。没有新的SR，不能因该短对照未改善就否定SFT。旧AR短GPU门曾共卡，因此整轮墙钟不作公平速度指标。
- **M-01下一臂预登记：** `fm_ae_lr2x_v1`，仍Git fb40145/同A4 SHA/原950-50/seed41/四卡global16/500更新、每100原80；唯一变化AE1e-5→2e-5，LoRA维持1e-5。先同源GPU两更新与四卡5步保存回读，再从父A4独立正式500；不部署/自动追加/清理旧文件。15:52各卡空闲约50/79/49/49GiB、盘余637GiB，保留120GiB；六个旧服务保留。拟启动，非已运行。
- **AR-01/M-04实质实现：** 新`train_action_method_probe.py`接入已验原完整loader、四卡global16、AR/joint/KI与同入口纯FM对照；全A4初始化、先5更新保存回读再独立500，CE/FM严格分开、原固定80 prefix-only FM及每task一条目标自由生成、实际draw/冻结/Adam/RNG证据。首版workers0使数据与模型RNG可定位，故不和旧四worker FM入口声称同墙钟/完全相同随机过程；初始完整80须实测复现父A4参考值。当前语法门通过、11项新CPU测试待执行，实际AR训练尚未启动；新进程断点续训仍须另外验证，不能把保存回读冒称已验证恢复。该AR臂为FM训练后A4的技能条件微调，不是原生G0.5+CoT复现；后者及闭环/组合仍在goal内。

### 2026-09-13 15:38（北京时间）：54项回归与AR全部真实码块校验通过

- **Codex / AR-01：** 新独立worktree `ar_blocks_20260913`固定7edf644，robo实际54项CPU回归通过；`ar_input_gate_v2/result.json`complete，SHA `f4fe54428154821af39a4c53062d959edd8d618c13094cb1fbc6bd3324f361ea`。10条原train×4视图均完整60动作tokens/8码块；每个残差级与两个夹爪码块齐全，prefix/无截断/真实23D约定保持。0神经前向、0更新、0仿真；不将正确teacher target等同自由生成已过。
- **状态/下一步：** 原control最新438/500，未重启/追加/热改；已完成的是AR格式与数据入口、AR/KI/joint的两更新工程门，实际AR和CE+FM方法训练、FM AE×2候选及闭环均未完成。严格保持单变量/同父与原未加权指标，接下来完成control500验收后启动已定AE×2，并将真实完整loader接入有界AR训练；不是重复十行拟合。新代码只在feature，计划同步main，等待独立代码审查后才合并实现。

### 2026-09-13 15:35（北京时间）：joint与原完整数据管线门完成，AR补齐完整码块校验

- **Codex / M-04、AR-01：** 本轮已fetch，保留上一阶段五份未提交修改，未强pull/热改活跃FM源。`joint_gpu_gate_v1/result.json`complete，SHA `fb48b231c918a9bd7412bff65713837414afd14f5250329c6afb4c5debfe16ad`；不隔离时FM连通322 AE和182 LoRA，CE连通192 LoRA而不连通AE。两次临时更新完成，固定输入CE降至14.83285、FM升至0.0676576；与KI一致只通过工程门，不是效果结论、没有发布权重。
- **完整数据入口：** c220e73的`ar_loader_gate_v1/result.json`complete，SHA `1bb36c7aa62d911bc1e6901846dcd85f5908791ee613194232bfd0c26428869c`。真实原train loader取出五task共10行并验来源；五个eval窗口只核验身份、不参与训练。dataset长度8,898,502/451,241是train/eval帧窗口数，不是轨迹数；原950/50切分与归一化、固定80身份保持。新loader使用私有worker RNG，不能在未配对核验前声称与原FM trainer所有随机抽样完全相同。
- **实质修正/待测试：** 原codec解析会对缺失残差级/短码块补零，单看absent keys不足以识别。纯AR新增仅依赖静态codec元数据和生成IDs的完整码块校验，拒绝缺级、截断、重复或越界；不读取专家答案、不强填动作。八项CPU用例与真实40视图目标探针已补，当前仅语法/diff检查通过。下一独立CPU `ar_input_gate_v2`验证，不覆盖v1，不改神经训练loss或活跃FM服务。
- **对照与答疑：** 15:34核验原supervisor/torchrun仍在、417/500；step400原固定80为0.1983362647，仍略差于A4父0.1969439941，不据此提前定性最终效果。继续完成分组LR对照及实际AR/joint/KI训练、自由生成和闭环；超参值得试，但小样本可拟合与完整任务0/3不支持盲目把全局LR、clip或batch一起调大。整个goal未完成。

### 2026-09-13 15:19（北京时间）：KI真实梯度与无答案泄漏通过，准备接原完整train loader

- **Codex / M-04，实际工程门：** 9045cf7的`ki_gpu_gate_v1/result.json`complete，SHA `eb5c6366051d736df45983c815818b3a7d09ac63da975cd4b2cbe11fc47efec6`。CE连通192 LoRA、不连通AE；FM连通322 AE、不连通LoRA。保持观察/连续目标/噪声不变，实际改变60个teacher-action suffix tokens，CE15.07449→26.21026而FM逐位保持0.06588463485，Qwen的prefix recurrent门通过。两次临时Adam更新、514份状态step2、冻结参数未变；部署检查只走FM，23D形状/有限值有效，无辅助AR调用。峰值reserved约28.4GiB。
- **严格效果边界：** 两更新后固定输入CE下降但FM从0.0658846升至0.0675914；不是方法收益，临时权重不保存/部署。旧control回执322 AE+182 LoRA有梯度，而新CE覆盖192 LoRA，因此Adam数量504/514不同来自梯度覆盖，不是本轮额外解冻AE（两者均322张量）。接着同预算独立`joint_gpu_gate_v1`检验不隔离对照，仍不能替代正式三组训练。
- **AR真实训练准备：** 新增`action_training_data.py`，复用原A3完整五任务dataset、train-only归一化、episode轮转采样与collator；显式核验train/eval不同resolver与原固定80身份。拟CPU`ar_loader_gate_v1`：原train真实五microbatch/10行及每task一个eval窗口仅做输入身份检查、2 CPU线程、0模型/优化/仿真，不构造release、不回灌eval。当前只做语法检查；通过后才能接实际950条train来源的有界AR训练，不重复拟合十行缓存冒充正式效果。

### 2026-09-13 15:13（北京时间）：纯AR真实梯度/两更新通过，自由生成确认缺少下半身和夹爪组

- **Codex / AR-01，工程实证：** 9045cf7在robo通过46项CPU检查；`ar_gpu_gate_v2/result.json`complete（SHA `50215961993d57088fdfa2460a30308d47f6b03e9161ca70eb58df13558fa5ae`）。完整A4恢复后，真实CE连通192个LoRA张量、当前输入187个梯度非零，FM恒0/不连通；两次临时Adam更新、192份Adam状态均step2、冻结参数未变。固定同输入CE 15.0744896→14.8460464；两次更新的clip前范数18.16/24.49。峰值reserved约22.13GiB。没有保存临时权重或仿真，不能当正式AR学习/方法收益。
- **实际自由生成失败及定位：** 更新前后各实际生成37 tokens（含末尾`|`），仅`left_control_0/right_control_0/left_control_1/right_control_1`，而正确目标60动作tokens含`lower_body_0/1`与两个gripper组。本人用真实CPU tokenizer/codec重解`before/after_raw_generation.json`，明确missing=`left_gripper,lower_body,right_gripper`；不是本次标签mask遗漏、不是解码器临时删组，也不是用满生成预算。新入口拒绝下发此不完整动作，未填零/FM补齐。该父权重是FM训练后的A4，不是原生上游AR复现；短短两更新不构成AR不可行的结论。
- **下一步/并行状态：** 继续准备原950条train的有界AR训练，目标是学会完整格式与真实控制，而非只做teacher-forcing；还须自由生成/闭环。先同9045cf7、A4/原train/GPU1、单行/两临时更新预算运行独立`ki_gpu_gate_v1`，核验CE→LoRA、FM→AE且不入LoRA，以及真实teacher-suffix不泄漏；joint另门。15:10 control真实进程仍在、288/500，预算和活跃源未改；所有方法最终效果仍未完成。

### 2026-09-13 15:05（北京时间）：AR GPU首门在独立codec设备迁移处停止，定位并补回归

- **Codex / AR-01，真实失败：** `ar_gpu_gate_v1`在9570e1c完成A4全部1138状态/192 LoRA逐值恢复并写`restoration.json`，但首个编码前向因动作张量在CUDA、独立ActionCodec仍在CPU而退出1；0 optimizer更新、未保存临时权重。该tokenizer不是policy的nn.Module子模块，`model.to()`不负责迁移；原finetune在模型迁移后另有`model.action_tokenizer.to(device)`，本次新GPU探针漏了这一步，不将其误报为训练数据或历史AR根因。
- **修正/待验证：** 新增`prepare_action_updates`复用该显式生命周期，补五项CPU配方/真实行切片测试；下一新run为`ar_gpu_gate_v2`，仍两次临时更新上限、同A4/原train/GPU1，不覆盖v1日志或放宽动作完整性检查。当前代码只做语法检查，v2尚未运行；FM control仍正常更新，15:03日志252/500，没有重启。

### 2026-09-13 15:01（北京时间）：41项CPU回归及40视图真实输入门通过，准备AR真实GPU门

- **Codex / AR-00，实际完成：** 新独立worktree`ar_inputs_20260913`固定3807531，41项AR/策略/FM CPU测试通过。`ar_input_gate_v1/result.json`complete，SHA `5a6ca932949dabc4198e546045e8deb52d9c0977eed81f01fd96a7665a70ddbe`；十条原train、五task、四种视图共40行，全部prefix一致/无GT泄漏到prefix/无截断/真实控制组齐全/token往返相等。每行60个动作tokens，完整序列1317–1395 tokens，实际动作词表区间[248077,252187)，当前Qwen hook无需偏移修正。0 VLM、0优化、0物理，不能当自由AR已通过。
- **Codex / AR-01、M-04，拟运行：** 新增`probe_action_training_gpu.py`，从完整A4-2500权重单次恢复，单行原train microbatch、两次临时Adam1e-5更新、实际CE/FM各组梯度、无teacher-suffix泄漏与更新前后目标自由生成；不保存/部署临时权重。先唯一纯AR门`ar_gpu_gate_v1`；joint/KI在独立进程分别验收，不能混成已验证KI。新脚本仅语法门通过，尚未真实运行。
- **资源/边界：** GPU1只作短工程门，启动时要求至少40GiB空闲、两CPU线程、0仿真，不停其他服务/训练、不热改其源码。可能与当前FM control共卡，故该重叠区间及整轮墙钟不能用于公平加速比较；训练更新数/样本/学习率/固定评估口径均不变。无效自由AR预测单独记录，不能用零动作或FM补齐伪装有效；非预期运行错误即停。后续仍须真实有界AR训练与闭环、FM候选/组合效果，goal保持active。

### 2026-09-13 14:51（北京时间）：AR/KI入口补全CPU回归和真实tokenizer门，待运行

- **Codex / AR-00、AR-01、M-04；前轮分类为verified wait：** 上轮核验真实FM进程并得到首个固定80，不把答疑/计划当新方法收益。本轮再次fetch，保留未提交草稿未强pull；14:43原control真实进程仍在、日志141/500，不重启或热改fb40145。
- **实质实现：** 补全独立`G05PolicyMEMLiteAction`/`ar_training_methods.py`，明确纯AR仅LoRA、joint/KI为AE+LoRA以及FM梯度路由；保持原v6输入校验。新CPU回归覆盖参数范围、Qwen prefix recurrent边界、目标拒绝及单一推理来源。源码发现旧decoder对空串返回零动作但absent为空，故新入口额外拒绝无有效动作token；同时处理AR解码CPU张量与GPU mask的设备差异。没有据此归因旧打转或宣称真实策略已通过。
- **下一工程门/预算：** 新`action_training_runtime.py`只挂载三份声明Git扩展，其余神经/数据仍固定A3 source SHA；`probe_action_training_inputs.py`拟用既有十条五任务原train缓存，检查四种表示视图共40行的训练/推理prefix相等、GT反事实不影响prefix、标签位置/无截断/全部23维与token往返。2 CPU线程、0 VLM/0 optimizer/0仿真，无新数据release，首次错误即停并保留run。当前六份文件仅py_compile与diff检查通过，torch检查和真实输入门尚未运行；之后仍需纯AR训练、自由生成、闭环及KI真实梯度验收。

### 2026-09-13 14:40（北京时间）：超参/训练方法答疑，control首个固定80结果已出

- **Codex / M-01，只读复核：** 用户追问改变超参或具体训练方法是否值得。本轮读实际A4审计、冻结SkillFM/FMHelper与配方，并复查PI KI及MolmoAct2一手说明；未改活跃源码/超参、未重启或新增训练。开工已fetch；本地有三份未提交AR/KI草稿，保留原状、未强pull，草稿尚未完成实际模型验收。
- **真实进度：** 14:39核验supervisor1499025及正式torchrun1508221启动身份仍一致，日志显示119/500正式更新。`fm_control_v1/formal/fixed_diagnostic/step_100.json`已写出：80窗口、五task均权FM=0.1971548254，较父A4的0.1969439941略高约0.107%；该单点不是最终结果或方法有效性证据。结果SHA `ce9a6b8f8c6256c78f0f5d92a2f43c48645df3acb7854e88488ac21fb144830c`。rank0真实梯度回执显示AE与LoRA均更新、冻结参数未变。
- **判断/下一步：** 优先完成既定AE分组LR单变量对照；原Beta内时间分层、执行前16步适度加权与KI-inspired双监督保持候选，不能把改loss标尺、十条样本记忆能力或更平滑的曲线当泛化/成功率提升。当前只有control在训，AE×2、KI、纯AR策略训练/闭环与有效组合均未完成；高层B、队友数据/RL职责及五任务预算不变，goal仍active。

### 2026-09-13 14:14（北京时间）：四卡保存回读通过，control正式500步进程启动

- **Codex / M-01：** `fm_control_v1/smoke_checkpoint_inspection.json`passed；真实回读step5（SHA `56cc992a81463180344c871a345368c368a15623bc01db17b6411947f4243c03`），504份Adam计数均5、504项可训练状态变化、冻结状态未变、所有模型/Adam有限、四rank RNG保存、每rank20条真实train抽取、归一化一致。只通过工程门，不发布临时smoke权重。
- **当前运行：** supervisor1499025已启动正式torchrun1508221，`status.json`为formal/running/max_steps500；重新从A4-2500权重开始而非smoke，seed41/新Adam、四卡global16、AE与LoRA1e-5、warmup50/cosine500。正式模型/数据初始化中，尚无完成500更新或新固定80结论。源worktree仍fb40145，不热pull。
- **下一步/边界：** 确认正式真实更新，完成并验收control后跑唯一AE×2单因素候选，再比较原固定80/动作和闭环，不能拿不同权重训练日志loss直接比较。其他FM方法、有效组合、纯AR策略训练和闭环均未完成；整个双路线goal保持active，无额外训练/成功率承诺。

### 2026-09-13 14:07（北京时间）：control真实GPU门通过，四卡保存回读短测运行中

- **Codex / M-01：** `fm_control_v1/gate/result.json`passed，A4全1138状态/192 LoRA完整恢复后，3次真实前向证明本control入口的原评估loss与CPU/CUDA RNG逐位一致；4个原train microbatch完成2次临时优化、504份Adam计数均2、动作专家与LoRA更新、冻结参数逐值未变。峰值reserved 35,475,423,232字节；无诊断权重保存或混入正式初始化。
- **实际阶段：** 原trainer同参数配置/资源门已通过，四卡5步短测torchrun1499487运行中；须取得`smoke_checkpoint_inspection.json`并验证保存回读才放行formal500。不把control的GPU门当其他非默认时间/权重选项都已验证，更不当loss/SR收益。
- **协作：** 14:00启动及AR编码结果/边界已docs-only同步main `d26b57e`，robo协作clone已ff pull；活跃训练worktree仍固定fb40145、未热改。AE×2候选、其他FM方法组合和AR策略训练/闭环未完成。

### 2026-09-13 14:00（北京时间）：M-01 control编排已启动，真实GPU门进行中

- **Codex / M-01，运行中：** robo独立worktree固定`fb40145d62b387ead5e9b25ea8a45c4a2fef57cc`，22项CPU检查通过后启动supervisor1499025；run `dual_track_fm_ar_20260913/fm_control_v1`，`method_spec.json` SHA `569455fb9ca4c0417cae9a998c767e82cdf846c9703a7f64adb62fb7718a9d03`。启动前再次确认无其他训练/仿真，六个旧服务保留，不热pull此worktree。
- **当前阶段/边界：** 正在GPU1加载A4并检查真实未改评估口径、两次临时优化；随后须四卡5步checkpoint回读通过，才自动进入独立500步正式control。编排启动不等于已训练500步或方法有效；AE×2及其他候选、AR策略训练/闭环均未完成。
- **证据/下一步：** 进度`status.json`，详细`gate.log`/`smoke.log`/`formal.log`，每一门失败即停且保留原证据，无自动重试。继续核验实际更新和正式阶段，结果及时同步团队main文档，实验代码只在feature。

### 2026-09-13 13:58（北京时间）：AR十行编码门完成，FM真实训练配方待GPU门

- **Codex / AR-00：** Git `2dd2cac`，robo真实CPU 15项测试通过；`dual_track_fm_ar_20260913/ar_codec_gate_v3/result.json`complete，10条原train/五任务，8个完整16步窗口＋5/8步末尾窗口，合计141个有效执行目标。全组原32/holdpad32均完整解码、原有效前缀不变、候选不受未执行后缀变化影响。直接16步10/10不支持，不能只改配置horizon上线。
- **结果边界：** 逐行归一化有效维RMSE均值原32为0.0219491，holdpad为0.0215759，7/10行改善、3/10变差；约1.70%均值下降仅是小缓存工程诊断，不是AR学习/泛化或成功率证据。v1/v2失败保留。AR后续仍需明确codec/任务条件/自由生成的真实训练与闭环。
- **Codex / M-01，拟开训：** 新增`train_fm_method_probe.py`，显式hash绑定原A3 trainer、A4安全编排与新训练forward扩展；7项本地标准库配方测试通过。先启动唯一control：A4-2500完整权重初始化、新Adam/scheduler、seed41、原950/50数据/采样、6帧/32预测/0:16执行、四卡global16、动作专家与LoRA均1e-5、50步warmup/500步cosine至0.1。真实GPU恢复/评估口径/2次更新门→四卡5步保存回读→独立正式500更新，每100步原固定80、500步checkpoint；不自动再训/部署、不占队友数据或RL职责。
- **对照/资源/停止：** 后续ae_lr2x仅动作专家2e-5，LoRA保持1e-5，其余含初始化和采样顺序相同；本轮不同时组合方法。robo四卡余约50/81/50/50GiB、sdc1余669GiB；GPU1做单卡门，正式四卡保留现有服务，磁盘保留120GiB、数值/身份/恢复失败即停、无自动重试。尚未启动该编排，大模型门/正式更新/双方效果均未完成。

### 2026-09-13 13:53（北京时间）：AR探针补齐真实轨迹末尾情况，训练forward隔离入口已写

- **Codex / M-00、AR-00：** v2已完成首条原32步与holdpad对照，直接16步codec不支持；随后因第二条仅5步真实动作、探针错误假定所有行均16步有效而停止。只读检查全部缓存：10行中8行32步有效，另2行仅5/8步有效；这是原轨迹末尾的真实padding，不是损坏数据，不能编造补齐为专家监督。
- **修正/待验收：** 编码候选支持真实连续有效前缀，内部尾部复制最后一个有效动作；仅对真实有效步评分，完整16步与部分末尾窗口分开计数，缺组/掩码断洞仍拒绝。新增CPU边界测试及只对grad-enabled train forward生效的policy入口，保留原`forward_train`/FM评估实现身份；当前未跑新测试或大模型更新。
- **下一步：** 新版本Git固定后运行CPU门和codec v3，继而实际GPU更新门及M-01同A4父权重的分组LR对照。v1/v2失败保留，不用首条codec误差下降作为AR方法结论。

### 2026-09-13 13:46（北京时间）：FM的13项CPU检查通过，AR首轮编码门因继承noop配置停止

- **Codex / M-00、AR-00：** 开工已clean pull/fetch，robo独立worktree固定`7d7a8cf`。现有Python环境CPU执行`test_fm_training_methods.py`及`test_fm_velocity_adapter.py`，实际13项通过；覆盖原helper loss/gradient/RNG逐值不变、Beta分层、23D梯度和异常恢复。尚无真实大模型更新或方法收益结论。
- **AR真实失败/根因界定：** `dual_track_fm_ar_20260913/ar_codec_gate_v1`加载实际codec后，在首条样本发现两个gripper组缺失并按门槛退出。回查配置`dropout_noop_parts=true`和源码：恒定二值夹爪被当noop省略；这是本次探针继承了FM不使用的codec配置，不是新发现FM丢失夹爪，也不能据此解释历史AR全部失败。旧AR v9已显式关闭该开关。
- **下一步：** 保留v1失败manifest；新探针显式记录关闭noop dropout的“全部动作组”诊断配置，并保持缺组即失败，真实重跑新v2。FM继续真实GPU梯度/固定评估口径门，继而有界训练；两条路线仍未完成。

### 2026-09-13 13:32（北京时间）：双路线goal开始实施，FM可控扩展已写，AR编码门准备中

- **Codex / M-01、AR-01；上一goal轮分类为进展：** 前轮源码证据确认了KI缺失和监督等价关系，但尚无新方法训练。本轮重新检查Git及robo真实进程：无新训练/仿真，GPU1空闲，其他卡各约30GiB旧服务保持；sdc1余669GiB。新分支`feat/dual-track-fm-ar-20260913`从最新main `dbc89c8`建立，不热改原快照。
- **用户完整目标：** “所有有效的方法都要用上；重新研究纯AR是否可行，两条线都推行实验，结束后给结论；仍为最后大训练做验证准备”。旧仅FM规则按新要求放开独立AR实验，未缩成只做离线loss。执行和验收见[双路线实验计划](experiments/2026-09-13-dual-track-execution.md)。
- **已写代码/待验证：** `src/g05/utils/training/fm_training_methods.py`实现有作用域、可恢复的原Beta等概率时间分层和执行段加权，以及AR执行前缀内部padding候选；对应CPU测试已写、尚未执行。本地无torch，将从Git独立服务器worktree用现有环境测试，不安装/改动共享环境。
- **AR证据/下一步：** 旧v10报告证明部分32步codec后半段影响前16步刹车，不等于AR路线本身不行。本轮已回读原train处理缓存：5个真实microbatch/10条样本、五task、动作[2,32,27]、SHA `237acf01b29bd0d6806ed1a11d9033a747640b3ea9e0bf7246b7a62b4e92be81`。先真实codec前缀往返门，继而训练和闭环；缓存仅用于工程门，不能作为整条路线效果样本。新训练、AR rollout、方法组合与最终结论均未完成。

### 2026-09-13 13:15（北京时间）：训练超参与方法候选完成，只分析未开训

- **Codex / A-02：** 核对SkillFM、FMHelper与Qwen3.5 prefix缓存实现，并查PI Knowledge Insulation、MolmoAct2当前微调配方及PCGrad一手资料。明确`joint_training=true`只是FM→VLM梯度连通，当前SkillFM禁止离散动作目标；不是已实现CE＋FM/KI。
- **候选/新结论：** 保留单变量分组LR初筛；新增原Beta分布内4次时间分层、前16步适度加权作为低成本候选，KI-inspired双监督作为更实质训练机制。推导同次未裁剪动作重建MSE仅为t²加权FM，不冒称独立监督。所有收益均待实测，不能照搬PI倍数或用task3退化证明梯度冲突。
- **边界/下一步：** 方法池与梯度/teacher-forcing泄漏、codec/23D等验证要求见[训练方法候选](experiments/2026-09-13-fm-training-method-candidates.md)。本轮未改模型/超参、构造数据或启动训练/仿真；不自动跑全候选，先确定一项有界对照。队友数据/RL职责不变，A-02与整体goal未完成。

### 2026-09-13 12:56（北京时间）：FM下降慢的只读审计，未启动新训练

- **Codex / A-02：** 回读A4实际配置、全部25份固定80诊断、离线W&B的250条训练指标和既有10样本拟合结果。重要更正：固定80并非持续下降，600步升到0.201226，后半程LR降低才降至0.196944；不能仅凭两端点归因“LR太小”。250个已记录clip前梯度范数均<1，但未检查未记录的每一步。
- **学习能力证据：** 既有10条原train样本100步实验FM 0.107716→0.017985、生成动作归一化RMSE 0.433966→0.124399，说明局部能快速拟合，不代表泛化。A4训练日志末microbatch均值与固定80口径不同，不能拿二者直接认定欠拟合/过拟合；自主完整结果仍0/3。
- **建议/边界：** 先固定train/eval双曲线、关键动作组诊断，再一次最多两臂各500步的单变量动作专家LR筛选；分组LR、关键阶段采样/加权、LoRA容量及吞吐优化均未执行/验证。不改数据、不开展队友恢复/RL、不追加训练。证据SHA、口径和候选预算见[学习效率审计](experiments/2026-09-13-fm-learning-efficiency-audit.md)；A-02及整体goal未完成。

### 2026-09-13 12:24（北京时间）：A4完整收音机三回合结束，0/3

- **Codex / A-02、B-03，本轮评测完成：** 固定301/302/303全部正常退出、各3224控制，官方目标均未满足，合计9672控制/78次高层调用，成功0/3。三份完整step trace分别严格连续1–3224，每一帧官方`done.success`均false；不是短预算或服务故障。303 result SHA `fcd94065b99e8650454b33afd17ccc344fb94e677784ed84204c26db5f9afff2`，`summary.json` SHA `6ba4d1ed5ade55a6abcf72ff8d430554fd60a766e95752e307c1da10e950c578`。
- **实际失败阶段：** 三回合分别在1152/512/1024切到GRASP，之后持续到预算结束，无TOGGLE；三份诊断各覆盖3224实际控制，均0错误/0确认持有。不是仅缺少抓取指令，但当前证据不够把因果进一步断言为某个单一损失/架构问题；局部有演示前缀的抓取成功没有转为本轮自主完整成功。303视频仍在复制/待本人视觉复核。
- **退出/边界：** supervisor1479480已退出，专用高1479622/8784、低1479621/8783由自身编排SIGTERM关闭，三个仿真均0退出；GPU1空闲、显存恢复原状态，其他服务保留。无追加训练/新回合/评测回灌。三初态小样本不宣称总体真实成功率必为0；全goal、主仓模型整合和协同训练改进仍未完成。
- **12:28最终复核：** 303视频已到本地、SHA与服务器相同，本人查看其25张全程抽帧＋末帧，确认从桌子另一侧接近后仍未夹住。三段合计75抽帧＋3末帧和全部9672控制的官方结果/物理记录均核对；本次完整评测及视频交付完成。报告、团队任务板和服务器目录已更新，不把这轮测量完成写成整体方法有效或总goal完成。

### 2026-09-13 12:09（北京时间）：A4收音机302完整结束，暂计0/2，303已启动

- **Codex / A-02、B-03：** 302正常跑满3224控制，26次高层调用，耗时818.32秒，官方未成功，result SHA `ec049bf012b08c95e51d274bd2feede5ccc1e0e91a333429638426d0b0373643`。运行中同样已自主进入GRASP但未观察到持有对象，完整物理/视频正在复核。
- **进度/预算：** 两个已完成回合0/2，第三个预定303已启动；没有替换失败回合、增加seed、改权重/高层或缩短预算。302视频正复制本地，待303完成后统一报告原始计数与失败阶段；整轮仍未结束。
- **12:12视频/物理复核：** 302视频复制与SHA核验完成，本人查看25张全程抽帧＋末帧；高层512时进入GRASP，剩余2712控制仍未持有，0诊断错误/0确认持有，后段收音机倒在桌面。不是只缺少正确抓取指令；结论/视频身份追加在完整评测报告，303继续。

### 2026-09-13 11:55（北京时间）：A4收音机301完整结束，0/1，继续固定302/303

- **Codex / A-02、B-03，实际部分结果：** 301正常跑满3224控制，耗时808.92秒，26次自动高层调用，官方未成功；result SHA `5aa127330248caa4c19eefa6bbd7df7f59aea6d64af573ba007fa51e003ae4a9`。不是短预算/通信故障；运行中看到高层从NAVIGATE切到GRASP，但尚须完整物理记录/本人视频核验失败阶段。
- **边界/进度：** 当前只有1个已完成回合，暂计0/1，不冒称三回合或总体SR。302已按原清单启动，303仍排队，权重/策略/seed/3224预算均未变。视频正在传本地`artifacts/a4_radio_full_20260913/instance_301/`，不用于训练；继续完成预定3回合，保留全部结果。
- **12:01本人复核：** 301视频已到本地且SHA一致；本人查看25张全程抽帧＋末帧，结合3426行物理诊断（3224真实控制/0错误/0确认持有），确认卡在抓取执行。高层1152时从NAVIGATE切GRASP，之后未开机；不是高层从未发出抓取，也不是抓住后忘切换。详细条件/暂时结果见[完整收音机评测](experiments/2026-09-13-a4-radio-full-eval.md)，302仍运行中。

### 2026-09-13 11:39（北京时间）：A4完整收音机三回合编排已启动

- **Codex / A-02、B-03，运行中：** 本地/服务器7项CPU检查通过，Git独立worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_radio_full_20260913`固定`be23b06b045fcc06e6fa8ab6dfb724aeebe95f31`，supervisor PID1479480。run `/mnt/sdc1/robodojo/behavior_dev/a4_radio_full_20260913_v1`的`manifest.json`、`launch.json`已写出，正在加载前身份核验；不把编排启动当作已完成回合。
- **执行与判据：** 高B-final/低A4专用服务、真实7次wire门后跑301→302→303。每回合从零动作/空命令记忆重置，最多3224控制；官方任务目标是指定收音机`toggled_on`，抓住或模型自报都不算成功。基础设施失败单列并停下检查，不静默加入失败分母/换实例重试；正常跑满失败则继续固定下一实例。完整权重/物理循环不变的回归通过，代码仅在feature发布，未合入main模型实现。
- **11:42实际通信门：** 两个专用服务ready；A4-2500完整base/adapter逐值恢复为true、B仍UNKNOWN_ONLY，7次真实低层调用全部通过六帧/23D/mask/0:16检查。`wire_probe.json`已passed，首个完整回合进程1480089已启动进入初始化；尚无回合成功率，不修改正在运行的be23b06源码。

### 2026-09-13 11:37（北京时间）：按用户要求准备A4完整收音机评测

- **Codex / A-02、B-03：** 用户看到局部抓取后要求“跑全程看看成功率”。本轮转为A4-2500＋不变B-final自动高层，从官方初始状态开始，0演示前缀、0固定oracle技能、0特权反馈输入；不追加训练，不开展队友恢复数据/RL。
- **预先冻结：** `turning_on_radio`，public_test实例301/302/303，环境seed0、policy seed17，每回合完整3224控制、充分墙钟5024秒；合计最多9672控制，3回合后停止，不根据结果改权重/策略或自动重试。301是反复用过的开发对照，302/303的官方初态文件已确认存在；小样本不冒称50任务总体SR或官方榜单。
- **代码/资源：** 新增显式checkpoint步号完整低层服务、独立三实例manifest/runner/supervisor，复用已审核六帧/mask/0:16神经实现与官方物理判据。新增服务GPU0/8783低层、GPU2/8784高层，仿真GPU1；其他服务保持。工作分支`feat/low-fm-a4-effectiveness-20260913`，拟run `/mnt/sdc1/robodojo/behavior_dev/a4_radio_full_20260913_v1`。当前为代码门，尚未启动或取得完整SR；通过后Git固定版本、7次真实通信检查，再顺序完整评测。

### 2026-09-13 11:16（北京时间）：A4局部成功证据与本人视觉复核完成

- **Codex / A-02，实质进展：** `actual_analysis.json`通过全部31事件/29条历史请求/464条实际模型控制的回读核对；动作与原物理回执逐值相同，模型六帧输入hash与独立捕获一致。frame912右手持有指定`radio_89`连续10个不同物理帧，超过同一6帧门槛；旧A3同起点1280模型控制未成功。这是一次L1局部改善，不是高层或完整SR验收。
- **本人检查：** 已查看A4全程24抽帧、末尾12帧、最终原分辨率帧，并对照A3的26张抽帧。可见右手接触并夹住收音机、对象姿态变化；没有把短抓取扩大为离桌/运输/整任务成功。视频已传本地`/home/wsy/behavior/artifacts/a4_effectiveness_20260913/A4-radio-e121-L1.mp4`，源/本地SHA `a3dd6da39dc2417f8b591aac09a025c521a32063e4f700e4405984cbaaca7669`；完整小结见[本轮效果报告](experiments/2026-09-13-a4-training-effectiveness.md)。没有构造/发布新训练数据。
- **比较限制/资源：** 两次起点本体六帧逐值相同，但三相机RGB有微小平均差异（0.848/0.508/0.563，uint8），不冒称完全相同像素。A4临时服务1474269已正常收到本supervisor的SIGTERM并退出，supervisor也退出，GPU1空闲、其他服务保留。训练及本轮两项诊断均已结束，无后台追加训练。
- **当前决策与待办：** 冻结A4候选低层，先少量跨起点/seed复现，再做自动高层＋FM的完整小型对照；暂不启动新损失/架构或再2500更新。继续研究高层意图、低层服从和可信完成反馈的分阶段/交替训练，但当前高层仍UNKNOWN_ONLY，不能将oracle反馈直接当部署输入；相关数据扩充/RL不重复队友工作。P0-02主仓整合、独立完整SR及高低层协同目标未完成，goal保持active。

### 2026-09-13 11:05（北京时间）：A4局部GRASP出现因果成功，待完整证据/视频复核

- **Codex / A-02，实际已完成：** A4 L1于11:01:48正常结束，448原前缀后真实执行29×16=464模型控制，31事件/913次物理观察。frame896为`target_not_held_by_any_arm / IN_PROGRESS`，frame912为指定`radio_89`的`any_arm_grasp / SUCCEEDED`，`initial_satisfied=false`、`policy_causal_success=true`。本次固定GRASP的停止规则与A3相同，达到物理因果成功即停止，不要求成功后继续跑满1280。
- **区别/限制：** 同起点A3在1280模型控制内无稳定抓取；当前是一次train实例、oracle-skill局部改善候选，不是自主高层/完整任务SR，也不证明运输或后续开关任务成功。正回执还须逐项核对实际消费动作、真实历史、精确目标/稳定计数和本人视频，不能只见SUCCEEDED字段即放行。
- **证据/下一步：** `completion.json`已complete，完整结果SHA `b81afbabb720e3015997375da4a48e00d00076f3463c965a125ff641692e4f35`；正在下载视频到本地忽略目录并复核。暂停原先“若失败则改训练损失”的分支，先确认这个真实局部收益，再做有限独立/协同验证；不回灌评测轨迹或追加训练。

### 2026-09-13 10:59（北京时间）：A4单次L1真实服务通过，仿真启动

- **Codex / A-02，运行中：** Git固定`c7fb287`，独立worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_prefix_20260913`；服务器4项CPU测试通过后启动supervisor1474264。run `/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1`，模型GPU0/8782、仿真GPU1，`launch.json`/`service.launch.json`/`rollout.launch.json`分别记录实际进程。
- **实际验证：** 新服务已完整载入A4-2500且7次真实wire全部通过，逐次核对六帧时钟、4个padding位、23真实控制、动作起点0及实际checkpoint身份；模型调用约0.61–0.62秒/chunk。仿真进程已启动进入环境初始化，尚无新局部成功/完整SR结论；预算仍448原前缀＋最多1280模型控制，临时服务由本次supervisor在结束时回收。

### 2026-09-13 10:50（北京时间）：170次配对完成，A4仅有限动作改善；批准单次L1复测

- **Codex / A-02：** `a4_paired_actions_20260913_v1/result.json`已complete，实际170次/0优化/0physics；两权重各1138状态/192 LoRA全量恢复、42条训练/推理prefix和重复seed通过；A3额外与原84份预测逐字节一致，旧结果没有因评测实现变化而漂移。
- **配对结果：** seed17右臂原始动作误差均值0.01648380→0.01516740rad（约降7.99%），seed29为0.01497355→0.01448188rad（约降3.28%）；但42窗分别仅18/20窗改善，两seed误差差值中位数均略正。16个原观察明显运动窗的误差下降约10.08%/4.80%，比保持当前姿态更好均仍10/16；夹爪误差没有一致改善。不把少数大运动带来的均值改善泛化成所有动作改善，不宣称SR或条件服从已提高。
- **下一步唯一假设/预算：** 同原radio121/138、同448真实前缀、同GRASP、policy seed17，A4是否将有限离线运动收益转为稳定抓取。仅一次L1，最多80×16=1280模型控制＋448原前缀，7次实际wire检查先行；GPU0模型/GPU1仿真，0高层调用、0训练、不改物理判据、不释放训练数据。复用已经修正的六帧/mask/起点0运行库，新建step-explicit服务入口和新run；没有先扩五任务或再训2500。若仍同机制失败，停止重复该闭环，转向条件学习/损失机制的小对照。
- **10:53代码门：** 新增`serve_low_fm_prefix.py`、`a4_prefix_pilot.py`、`a4_prefix_wire_probe.py`，只将完成step和旧只读runtime位置显式化；真正构造/重置/推理的`FormalALowService`类AST与旧修正版完全相同。4项CPU窗口/step/隔离/AST测试及原9项配对测试通过，当前尚未启动仿真。拟输出`/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1`；结束自动关闭本次专用服务，不关闭既有服务。
- **10:55跨版本测试修正：** 首次服务器CPU门因Python3.10与本地新版的`ast.dump`字段不同而失败，训练/服务/仿真均未启动、run未创建。测试改为AST定位后核验整个服务类的原始源码字节SHA `7d5e0f17daf3c16a3557d8b2c36f7d794c3da49797f75634bc99549b39f4551f`（旧/新逐字节相同）；不放宽实际类/模型身份，不改正在使用的旧runtime。

### 2026-09-13 10:44（北京时间）：A3/A4配对动作诊断已启动

- **Codex / A-02，运行中：** Git commit `f9d8937c6048333e5613097ee20b4c6d16c8f760`，服务器独立worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_effectiveness_20260913`已显式fetch/pull并核对干净；9项测试在服务器再次通过。supervisor PID1472661，run `/mnt/sdc1/robodojo/behavior_dev/a4_paired_actions_20260913_v1`；`launch.json`绑定版本/预算，`process.json`记录进程，A3/A4日志分开。
- **实际阶段/边界：** A3 worker已开始加载与依赖校验，尚未得到生成结果。每个权重单独进程顺序加载、释放显存；170次上限，0优化/0physics/不部署，不停止其他服务。继续核对真实生成和旧A3逐字节复现，再读取配对差异。

### 2026-09-13 10:40（北京时间）：A4身份/固定窗口比较核实，动作配对脚本待GPU执行

- **Codex / A-02：** 已独立重算A4最终文件SHA，与`6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`一致。A3最终及A4全部25份固定诊断具有相同manifest `a365370d81596cb720ba5df38e6935aa1e0a2bdcbf1fbc0c54dfd52bbf45b254`、window-set `7ade41ed2407bf29f032cc5bd0de51b240b292e7c3ab1b560859ec1159ece527`、noise/time sampler `2a77e3006b1531bd0f5b7c655578698002eea47ac2ace7f4cc64046013b7d1e5`，均为80窗口，不是full eval。
- **实际指标：** FM 0.198776845081→0.196943994070（约降0.922%）。task0/1/2/3/4依次0.142679→0.142559、0.163655→0.158998、0.274372→0.269659、0.161039→0.164092、0.252139→0.249411；task3变差，task4的GRASP子组也变差，不能用总体微降掩盖差异，少窗口不作统计显著性声明。
- **已写配方/验证：** `scripts/experiments/paired_a3_a4_actions.py`与9项标准库测试全部通过。复用已审核原train radio121/138的42观察×seed17/29，每权重加1次重复，共170次真实FM生成/1卡/0优化/0physics；采用原始绝对右臂目标、真实四维mask和0:16，不把clamp后的训练target逆变换当原始控制。A3须与旧84份同输入输出逐字节复现；本轮没有错意图干预，不将同正确条件的动作精度称为意图服从证据。
- **下一步/同步：** 服务器GPU1核对空闲，脚本尚未启动；从Git独立worktree运行，失败即记录、不覆盖/自动重试。feature尚未合入main模型实现。依据配对动作收益选择一次训练机制试验，队友数据扩充和RL不重复开展。

### 2026-09-13 10:23（北京时间）：A4完成，转向配对动作与训练机制验证

- **Codex / A-02；上一goal轮分类为实质进展：** 已启动并验收真实四卡训练，本轮重新核对外部状态。A4于07:39结束，supervisor/torchrun均已退出；`formal_checkpoint_inspection.json`通过2500次优化、504份Adam状态、全模型/Adam有限、冻结参数不变、四rank RNG及每rank10000条真实样本回读。最终checkpoint SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`；5个每500步完整checkpoint保留、last.pt指向step2500。仍将独立重算当前文件SHA，不把回执替代文件身份核验。
- **初步学习结果：** A4固定80留出窗口FM=0.1969439941，原A3约0.198777，约下降0.92%；不是完整eval或成功率。正式比较还须逐项核对manifest、window-set及noise/time sampler身份，再报告逐task/skill变化；不自动再加训练步数。
- **旧A3闭环已完成：** `native_a3_aligned_development_pilot_v3`五项均正常消耗完整3224/7901/20682/20544/17770动作，全部官方失败，修正版结果0/5；这是相同A3/B-final及反复使用的开发实例，不是A4的SR。旧campaign已退出，不重复启动该五项。
- **本轮假设/预算：** 现有正确监督的追加学习是否改善关键运动，不能只看FM均值。保持相同输入/normalizer/mask/动作起点，先做A3/A4配对离线动作诊断：复用原train radio121/138的42个已核验锚点、seed17/29，最多约200次生成/1卡/无优化/无新physics；检查关键运动误差、相对保持姿态的进展、条件响应。根据证据选一个训练机制验证，不开展队友的恢复采集或RL；训练方法、主仓整合及独立完整任务成功率目标仍未完成，goal保持active。
- **协作：** 开工main已pull/fetch，独立分支`feat/low-fm-a4-effectiveness-20260913`；不热改仍在服务的旧源码，结果写新run，代码配方经Git同步。

### 2026-09-12 23:55（北京时间）：A4真实训练已推进，补足GPU0显存余量

- **Codex / A-02，运行中：** 正式任务已于23:48:44进入训练循环，四个`formal/coordination_grad_receipt_rankN.json`均证明第2次优化中动作专家和VLM LoRA实际更新、冻结参数逐字节不变；每rank动作专家322份非零梯度、LoRA182份非零梯度（192份参数中的10份允许未使用，与既有图结构一致）。截至23:55样本回执进入第30步，不将预取/采样步号冒称完成步数。
- **配置不再变动：** 原五任务950训练/50留出轨迹，A3-5000权重初始化的新阶段，四卡global batch16，seed29/lr1e-5，新增2500更新；每100步固定80留出窗口诊断、每500步保存。无正式训练墙钟截止，无自动重试/追加/部署；高层B不训练，RL与物理反馈头未启动。
- **显存干预与此前承诺更正：** 实际四卡初始化后GPU0仅余1143MiB，因此23:54:39仅关闭我们旧版A3前缀诊断服务`3276541 / 8778`，并非保持所有旧服务永不停止。操作前核验UID1003、完整argv与`a3_prefix_radio_e121_l1_v1/service.launch.json`逐项一致、进程起点1789195491.89与启动回执相差不足1秒、旧rollout确已complete、端口无活动连接，且当前campaign只用8773/8781。以pidfd发SIGTERM，未删除任何文件或停止训练/当前评测；23:55进程已退出，GPU0余16257MiB，当前高层8773/低层8781监听保持。
- **证据/后续：** 所有run仍在`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912`；`launch.json`中的未停止其他进程是启动时事实，上述后续显存调整以本条为准。尚无正式500步checkpoint或新固定80结果，更无SR改善结论。之后读取最终checkpoint验收、配对离线动作与留出指标，再决定是否进行少量局部闭环；不得把训练正常当作方法有效。

### 2026-09-12 23:43（北京时间）：四卡保存回读通过，A4正式进程已启动

- **Codex / A-02：** 四卡5步正常结束，实际回读`smoke/checkpoints/step_5.pt`（SHA `dc31736599d3562ae7a1a8e82129248cd292dd71ac5fe7ecf90dee3d84345194`）：504份Adam计数全5，504项训练状态变化、冻结部分逐字节不变，模型/Adam均有限，四rank RNG完整，四卡各20条实际train样本、覆盖五任务，normalizer与A3一致。`smoke_checkpoint_inspection.json`通过；这是工程验收，不是模型效果。
- **正式运行：** supervisor `3719946`已自动启动torchrun `3729064`，`status.json`为`phase=formal/max_steps=2500/wall_seconds=null`。从原A3-5000重新初始化而非smoke权重；进入正式模型/数据初始化，须继续核对早期真实优化与四rank梯度回执。正式输出`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal`。
- **协作同步：** 启动配方仍固定在Git分支的`e463932`；计划/目录/任务板已单独同步main（`13b03b1`），未把未经其他成员review的运行脚本或旧实验模型整体合入main。当前源码不热pull。

### 2026-09-12 23:34（北京时间）：A3父权重真实GPU门通过，四卡短测启动

- **Codex / A-02：** `overnight_a4_20260912/gate/result.json`已实际通过：1138模型状态/192 LoRA逐字节恢复检查后，四个原train microbatch完成两次临时优化，动作专家与VLM LoRA真实更新、冻结参数不变，loss和梯度有限，峰值reserved 32,631,685,120字节。未保存或混入诊断权重。
- **阶段：** 四卡短测的原trainer同argv配置/资源预检已通过，`smoke/`开始构建模型；须等5步checkpoint保存及实际回读通过，才能把`formal/`称为正式训练。后台编排会自动继续，无训练墙钟硬截止。

### 2026-09-12 23:31（北京时间）：A4后台验收/启动编排已运行

- **负责人/任务：** Codex / A-02。服务器通过Git同步后在独立detached worktree固定`e463932740cbb3977a2b975be824c21d9dc96f45`；运行脚本SHA `3e21038efc97e68ccfb79b87f584025690ee035f43fd246b1c34b2e9a0ff5342`。11项CPU安全测试通过，正式训练墙钟限制为`null`，此前两次Git分支追踪失败均未产生训练。
- **已提交运行：** supervisor PID `3719946`，运行根`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912`；源码配方在`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_overnight_20260912`。`launch.json`固定配方commit、原A3源码SHA、父checkpoint SHA及预算；`status.json`记录实际阶段；`supervisor.log`/`gate.log`/`smoke.log`/`formal.log`分开保留。
- **当前边界：** 后台编排已启动，不等于正式训练已执行。先真实两次更新/父状态核验，再四卡5步保存回读，均通过后自动启动独立`formal/`的2500更新；固定80诊断在`formal/fixed_diagnostic/`，checkpoint在`formal/checkpoints/`。仍未改变高层B、数据、既有仿真和服务；下一步确认真实门及正式早期更新。

### 2026-09-12晚：用户取消正式训练墙钟上限，继续启动

- **负责人/任务：** Codex / A-02。用户最新要求“不用限时，跑起来就行”：取消正式训练8小时硬截止，仍按2500新增更新完成，每500步保存，保留磁盘/数值错误保护；不因时间到而终止训练。以下8小时记录是此前方案，已被本条替代。
- **实际状态：** 首次Git取代码只更新了`FETCH_HEAD`，服务器clone的窄refspec未产生对应`origin/feature`引用，因此独立worktree尚未创建、训练尚未启动。没有重复任务、没有模型更新；改用明确远端分支ref同步后继续。

### 2026-09-12 23:27（北京时间）：过夜运行配方与安全测试完成，真实GPU验收待执行

- **负责人/任务：** Codex / A-02。`scripts/experiments/overnight_low_fm.py`只编排原封不动的A3训练快照：新父权重实际两次更新验证→四卡5步训练/完整checkpoint回读→2500步正式阶段。任一门失败即停止，不自动重试，不停止其他服务，不修改现有训练源码/数据/环境。
- **安全/验证：** 新增10项CPU测试全部通过，覆盖训练预算、父权重、路径隔离、GPU/磁盘余量、限时停止只作用于自己创建的进程组。正式阶段8小时封顶，每500步保留完整checkpoint；异常中止只保证已完整保存的checkpoint，尚未保存的更新可能丢失，不冒称信号触发即时保存。
- **同步/证据：** 运行配方commit `ff20200`，位于`feat/low-fm-overnight-20260912`，未合入main模型实现；服务器从Git取得独立固定版本。拟运行根`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912`，仍须核对真实GPU门和正式优化步数后报告训练已开始。

### 2026-09-12 23:20（北京时间）：A-02单次过夜低层训练准备

- **负责人/授权：** Codex，分支`feat/low-fm-overnight-20260912`。用户明确要求按计划启动一晚训练；本次取代此前“仅整理、不新开训练”的工作范围，但不启动50任务大训练、反馈头或未实现的RL。
- **主要假设：** 在修正采样覆盖后的A3上，再给一个有界的低层学习阶段，检验动作能力不足是否仍能靠现有正确监督继续改善。只训练动作专家＋VLM LoRA，MEM-Lite高层B-final不动，不引入评测回流/未验收纠正数据，不宣称这是新架构或严格单变量消融。
- **拟定预算：** A3-5000全量权重初始化，原五任务950/50切分，六帧/未来32动作/执行前16；四卡microbatch2×累计2，新增最多2500更新（40000次样本抽取），新阶段optimizer/scheduler与seed29，非等价断点续训。沿用已验证lr1e-5与原采样，固定80窗口每100步，完整可恢复checkpoint每500步；训练墙钟上限8小时，预计新输出低于100GiB。正式配置/真实GPU门尚未通过，**当前未启动训练**。
- **依据/资源：** 原A3的1000更新约2.9–3.1小时，因此2500更新约7.5小时；源盘剩余约763GiB。既有修正版评测task0/1/2均已完成失败（0/3），task3在跑、task4待执行；不打断该序列及其高低层服务。四卡存在服务占用，启动前再次核对峰值余量。
- **代码边界/下一步：** 先核对不可变A3训练快照与父权重、真实首步/梯度/保存，再启动独立run。Git跟踪新运行配方和证据索引，现有模型实现仍以固定源码SHA绑定；不以此次启动冒称P0-02主仓整合完成。早晨首先比配对离线动作误差与留出FM，再决定是否值得做少量局部闭环，loss下降不算SR提升。

### 2026-09-12：计划迁回docs并启用实时更新规则

- **负责人/范围：** Codex；仅文档与协作流程，不启动训练或评测。
- **已完成：** 原goal计划迁为`docs/plan.md`，完整保留原E0–E7内容与历史记录；AGENTS新增实质操作后实时更新、长任务状态区分和交接同步要求；同步调整文档导航。
- **已有最新阶段记录：** 仓库同步、安全归档及三人分工已完成；模型进度仍以[团队任务板](TEAM_PLAN.md)中带核验时间的记录为准。本次只迁移文档，没有重新核验或改变正在运行的实验状态。
- **剩余/下一步：** P0-02最新实验代码整合、低层关键动作学习、高层可信完成反馈和通用RL验证仍按团队任务板推进；文档整理不代表goal成功率目标已经完成。
- **证据/验收：** 原计划正文已与迁移前Git版本逐字节比较一致；旧路径已移除，新入口/导航存在，AGENTS及plan均未被忽略，差异格式检查通过。只改文档，未运行模型测试；Git同步以包含本条的提交及远端HEAD为准。

## 历史执行记录与原E0–E7计划

以下保留旧检查点原文；其中“当前”“下一步”“goal保持active”等描述均属于当时记录，不是实时状态。旧本地资料迁移后的路径映射见[服务器目录表](SERVER_LAYOUT.md)。新增工作写入上方实时进度区；不删除失败记录，发现旧结论错误时明确补充更正。

**版本：2026-09-12 v31；当前状态：在实际同输入的504次对照中确认两处推理偏差：纯FM缺失4个补齐维度的mask，以及把6帧图像历史误当5步动作历史、执行预测的5:21而非0:16。84/84旧输出逐值复现了错误偏移；同输入的训练/服务神经路径84/84逐值一致。两个修正使配对右臂误差均值下降约3.8%/17.6%，但关键运动仍弱，尚不能宣称抓取或SR改善。独立修正版服务已启动加载，下一步是真实通信及同权重、同448步前缀/GRASP的闭环复测。旧A3五任务0/5结果保留，不能将含推理bug的旧结果全部归因于架构。原装罐11787步已结束，本人复核发现原演示亦有一次罐外放置。反馈头/校准、成功纠正、协同训练、主仓代码集成及独立成功率改善仍未完成。全部本人执行，无subagent，goal保持active。**

本文最上方的“单人继续执行”记录是当前状态；下方带日期的早期评审、分工和 pending 描述是历史过程，不覆盖当前实际结果。

本计划服务于唯一的产品目标：在未见实例上提高完整 BEHAVIOR 任务成功率。训练步数、训练/离线 loss、单元测试通过和模型规模都不是完成条件。每个版本必须留下可复现的权重、数据、配置、代码、随机种子和评估证据，才能讨论成功率是否上升。

## 2026-09-10 单人继续执行

### 最新：两处实际训练—推理偏差与修正后的因果复测（v31）

- **动作mask实际缺失：** 原训练的27维合并动作包含23个真实控制和4个补齐维度，补齐索引为`[7,8,17,18]`。训练FM输入将这4维清零；target-free原生推理跳过动作处理，未生成`action_dim_is_pad`，实际服务将None传入FM，导致这4维初始和积分中的无监督噪声进入动作专家。实际未遮蔽输出的补齐维度最大绝对值约2.66–2.73，新mask下全部精确为0。底盘/躯干7维均保留，未以mask修复之名删底盘。
- **动作时间起点实际错误：** 该权重训练的是从当前时刻起的未来动作，42个原始锚点前16步目标已逐值验证；但通用processor在未显式设置时把`num_obs_steps-1=5`用作动作执行起点。对84条旧实际网络输出穷举起点，84/84精确等于新复现原预测的`5:21`，而非`0:16`；最大差异0.285671rad。图像历史数量不能用来推断历史动作数量。
- `a3_radio_e121_matched_original_inputs_v2`完成42原train锚点×2seed及首例重复，共85次生成；实际Dataset索引、技能条件、归一化目标核对通过。末锚点含18个有效动作和14个padding，比较器按真实有效位处理；另有1锚点经历真实clamp边界，比较器按实际正反归一化重新对齐原绝对动作。旧诊断失败保留，不冒称原数据损坏。
- `a3_input_route_factorial_v2`完成504次生成：42锚点×seed17/29×6种输入/路径组合。同输入的直接prefill与原生入口84/84逐位一致，原训练源与原服务源84/84逐位一致。原图像/原本体与重放图像/本体交叉只作诊断，不能当真实物理样本准入。早期“原输入比回放输入更好”的比较同时改变了mask和动作时间起点，**已确认存在混杂，不再将其单独归因于图像域差异**。
- `execution_alignment_analysis_v1.json`核验全部504份预测hash，并对同一批回放输入做mask/起点的2×2分析。seed17的右臂误差均值：旧0.0180216、仅mask0.0175277、仅起点0.0174256、两者0.0173315rad；seed29依次0.0215536、0.0201253、0.0187187、0.0177681rad。两者同时修正时36/42及37/42锚点误差下降；但13个明显运动锚点中优于保持姿态的仅6/13及4/13，没有证明关键动作学习已充分。
- 独立`native_a3_aligned_prefix_runtime_v2`和`native_a3_aligned_full_runtime_v2`只叠加静态动作元数据mask与显式动作起点0，不改旧服务、1138项模型状态、192 LoRA、归一化、历史图像、控制器或物理判据。16项mask/起点测试、6项分析测试、19项实际历史/传输回归通过。实际新服务PID3599746，GPU3/8780，`a3_aligned_radio_e121_l1_v2`；启动不等于模型已就绪、通信通过或仿真成功。7次真实网络推理通过后才启动GPU1固定GRASP复测。
- 原`demo_canmeat_e807_feedback_trace_v5`完成11787条未改控制，0模型/优化，原轨迹末尾没有官方完整任务成功终止。7069UNKNOWN/3479IP/1239SUC，27段中11段出现稳定目标谓词，不是11次任务成功。本人查看30全程帧和15关键帧（44不同帧）：bratwurst233在`[8888,9100)`的PLACE_IN未成立，9027右手释放后视频可见香肠在罐外；后续另一根香肠才成功入罐。报告`ORIGINAL_CANMEAT_E807_PERSONAL_REVIEW.md`，视频已本地；不将原专家身份当每段成功证明，不擅自造FAILED标签或训练准入。
- 下一步按因果顺序：完成修正版真实通信→同权重固定GRASP闭环及本人视频核验→相同B-final/开发实例的完整任务对照→根据剩余失败决定低层关键动作学习与高层条件/反馈协同训练。最终需未用于诊断的独立实例完整任务成功率改善及主仓实际集成；旧0/5、绿色测试、离线误差下降都不满足goal完成条件。

### 历史：A3完整0/5、真实低层错配与原训练小样本可学习性（v30）

- 上一goal轮及本轮均为实质进展：完成A3完整五任务、原GRASP闭环核验、实际80000训练抽样运动审计、100次真实临时优化、原动作反馈扩充及本人视觉审核。没有把运行文件、绿色测试或训练步数当作goal完成。下面结果均来自实际退出状态／完成凭据，不是计划推测。
- `native_a3_final_development_pilot_v1` 五任务分别完整消费3224/7901/20682/20544/17770控制，全部官方失败，总计0/5；耗时825.438/1923.977/5446.270/4972.557/4714.288秒。所有动作与逐步物理trace核对一致，无teacher前缀。沿用B-final UNKNOWN_ONLY和已反复开发的public_test301/seed0/policy17；这不是独立盲测或总体SR，评测数据永久不回灌训练。后三任务result SHA依次为 `9c5510066a9967f1751dc381b032e3c1b1a3a1728e96ca76050a3afd0019f90f`、`dde48a2700b63b731729222bc061c3385af5709b88681670b289c621803f23b9`、`5015a1015b07d2b4d142807a47da624d48b9342796902197cd1b9fa6fd27158a`。
- **实际条件服从未解决：** 万圣节 `[384,16256)` 要求OPEN_DRAWER电视柜left，15405却确认持有candle90；18736才出现匹配GRASP candle91的稳定成功，18816切换，随后PLACE_IN未成功。19712以后要求抓candle89，20349实际持有candle91。收盘子 `[256,1408)` 要求OPEN_DOOR冰箱right_door，1227却实际持有plate93；4608–18944长期NAVIGATE bowl91，最终无任务进展。装罐5NAVIGATE/134OPEN_DOOR，无后续GRASP。这些不能用格式合法、低CE或单纯增加速度解决。
- 五段完整视频都已传本地。本人审阅收音机23准确帧、其他每任务30准确帧，并核对对应物理持有／语义；不是逐帧看完。报告 `A3_EVAL_FINAL_PERSONAL_REVIEW.md`。装罐真实机体曾达到88.025°，但17760采样已恢复约0.013°；收盘子虽画面歪，真实机体最大仅1.272°。没有仅凭画面给两者都贴“倒地”，也没有把描述性倾角直接制作FAILED监督。
- A3原起点正确技能 `a3_prefix_radio_e121_l1_v1` 已完成80×16=1280模型动作，80个chunk后均IN_PROGRESS，无高层调用，1729实际观察捕获，所有80次真实六帧输入hash通过。本人看26准确视频帧；原动作448前缀与模型贡献分开，不能排除chunk中间短暂持有。本例不是完整任务SR，仍不准入纠正数据。
- **实际原动作审计而非静止猜测：** 80000抽样逐一映射到1952920条原数据行，episode/frame/task/global-index与950条train身份一致。radio GRASP4016样本中1953条右臂未来16步RMS≥0.005rad；plates GRASP3440中1929条达到该阈值。数据并非没有明显运动，阈值也不被当作idle/成功标签。结果 `a3_actual_raw_motion_coverage_v2.json`。首次把原state宽度误假设为57而拒绝，v2按实际61维重查，旧失败保留；不是原训练被修改。
- **100步临时可学习性测试完成：** `a3_original_train_capacity_100_v1` 从实际A3最终完整1138状态／192 LoRA载入，在10条已有原train样本上做100更新（200微批、400draw、world1、micro2×accum2、fresh Adam、常数lr1e-5）。训练与target-free实际prefill的token及modality mask逐项一致。AE和LoRA真实更新、其余逐字节不变；同样本／同seed的生成RMSE均值0.433966→0.124399，右臂误差0.013153→0.004482rad，优于hold-current的17/20→20/20，固定噪声FM0.107716→0.017985。8项测试通过，实际峰值reserved32.864GB。无权重保存／部署／正式训练准入，无eval或physics数据；不能用同训练样本拟合宣称泛化或意图服从。结果SHA `a03250d1e5a9164d1da853f263470a6d91c85729a7fb47848a76168cf638be40`。
- 为避免拿不同样本作域差异结论，新增 `probe_a3_matched_original_inputs_v1.py`：固定原radio121/138的448..1104共42锚点，直接走实际released train Dataset/processor；先验证准确分割索引、原动作减当前q的归一化目标、同任务全文／父目标／技能，再用原A3同seed生成。13项组合反例检查通过；实际CPU准备正在执行，未提前声明42次真实对照完成。额外检查确认C1 context虽然字段名叫task_name，实际保存的是完整自然语言指令，不是短任务名；未把这个猜测当根因。
- 原feedback v5 e91/97实际1611未修改控制后官方success=true；467UNKNOWN/1083IP/61SUC，0模型／optimizer。本人看26准确视频帧，全轨迹检查通过；它是原演示成功，不是模型SR或失败状态纠正。can-meat e807/10原11787动作准备通过，GPU3已启动并实际推进。trash e286/e308有初始及中间语义缺口，被prepare拒绝；其中一次误launch也因缺manifest拒绝，0模型／物理动作。未补造标签，未删除失败。
- 精确plates v4追加本人29时点×3相机=87面板审阅。OPEN/CLOSE交接隔离仍为77条，原谓词事实不变，715新候选全部training_admissible=false；旧v3的125条个人审阅不冒充新v4全量认证。报告 `ORIGINAL_FEEDBACK_V4_V5_PERSONAL_REVIEW.md`。跨实例校准、真实FAILED/恢复与成功同状态纠正仍缺，未虚构门槛或教师。
- 接下来：完成同帧原训练／回放输入配对，依据真实动作误差和条件响应决定低层训练改动；补足多任务可验证完成反馈及其交接含义，训练并校准observable-only结果头；取得真实失败后的成功纠正，才进入交替协同和独立完整SR验证。任何临时探针都不能替代主仓最终集成和独立成功率提升。

### 历史：A3真实闭环、相同状态复测和反馈标签语义修正（v29）

- A3独立serving组合 `a3_serving_composition_v2` 恢复全部1138项模型状态（192 LoRA），逐字节相符；23项runtime检查与3项控制器实际socket检查通过。真实BEHPY→服务7次模型调用验证三相机六帧，末次真实历史16/32/48/64/80/96，返回有限16×23控制。初次客户端误送dataclass导致msgpack失败，未产生模型/物理动作；修正仅wire映射，旧失败保留。配置/运行源不热改。
- A3五任务manifest SHA `0f63e3d619ce838d7bf77d7a110a28f931af04638c41ce563a9a50714d7bbeda`；GPU1完整pilot `native_a3_final_development_pilot_v1`，高层权重、public_test301、环境seed0、policy17、全部官方预算均同A2。收音机3224步、825.438秒，失败；6 NAVIGATE/20 GRASP，无PRESS、无实际稳定抓取。本人看23个全程关键帧，视频已本地展示，报告 `A3_RADIO_FINAL_PERSONAL_REVIEW.md`。捡垃圾7901步、1923.977秒，失败；右手3292确认持有trash_can116、3297稳定，3584才切换导航can113，延迟287控制；4224至7901抓can113未成功，右手仍持垃圾桶。后者不等于违反某个指定手臂命令，也不自动制造FAILED。尚未完成的后三任务不进入结果分母。
- A3训练实际40000微批次凭据、80000不同原始train样本，五任务各16000、各190/190轨迹。技能段覆盖760/760、2280/2280、4000/5045、3532/3532、4000/5153，仍未全覆盖任务2/4所有技能段。第一次离线审计误把每微批2样本当每更新4样本，立即拒绝；依据实际训练器两次累积记录修正到独立v2并完整核验，不是训练错误或丢失样本。`a3_actual_5000_sample_coverage_v2.json`。
- 85次A3原成功回放状态预测完成，42个状态及seed17/29/首状态重复与A2的实际输入、条件和目标逐一相同。需要明显右臂运动的13状态，A3进展投影增益中位数0.01667/0.01641，A2为0.01838/0.01778；优于保持当前姿态的A3为6/13、5/13。A3首次close仍960/944，原控制952。不能把loss下降4.16%当进展学习已解决，也不能拿投影增益作动作放大倍数。GPU3 `a3_prefix_radio_e121_l1_v1` 使用原448步前缀、固定原GRASP、同seed17和最多1280模型控制；独立GPU0/8778服务真实载入。新前缀runtime只换A3源码身份，19项检查、18次真实历史通信证明通过，实际闭环结果待完成核验。
- 精确部件原plates train762/242回放 `demo_plates_e762_verified_parts_v4` 实际10709原控制完整通过轨迹审计，0模型/训练；6296 UNKNOWN、2828 IN_PROGRESS、1585原谓词SUCCEEDED、9段出现因果稳定谓词成立。**这不是9次完整任务成功，也不是全部技能已满足交接要求。** 右冰箱门首次原Open稳定成立在561（约8.41度），原动作直到1275仍继续开到约90.33度；默认阈值只有6.75度（5%行程）。本人已看334/561/1275三相机原视频/回放对照。新v2构造器明确屏蔽所有未校准OPEN/CLOSE（含混合bundle），不篡改原谓词、不任设统一90度成功标准。13项正反例通过；715新候选中77条被隔离，其中原SUC47、IP30；保留可监督候选IP174/SUC59，全部仍training_admissible=false，尚非反馈头训练准入。
- 下一步直接服务于成功率：完成A3真实闭环与本人视频检查；核对实际80000训练抽样对应的原始动作运动分布，判断如何加强关键接近/闭合动作学习；扩充不同train实例的真实物理反馈与本人复核、建立可信交接语义和独立校准；约束高层真实父目标—对象—技能一致性而不偷喂场景oracle；获得真正成功纠正后再协同训练、主仓集成、独立实例完整SR验证。不是继续盲目5000步或用模型格式合法替代服从。

### 历史：A3训练完成，A2完整0/5，低层进展与高低目标错配证据（v28）

- 上一goal轮属于实质进展：新增85次真实A2原成功回放状态动作预测、9项精确部件映射检查及两个真实非扰动物理资产探针，形成新的因果证据和候选代码。2026-09-12重新检查发现原训练与五任务runner PID均已退出；训练receipt为complete/returncode0，A3实际step5000权重SHA `865193f1c8a257d72ea159bd7a46cebf945b0fd24905110b831f0e8a3438e940`，不是仅凭文件存在推断完成。
- A3正式阶段为原A2最终权重基础上新增5000更新，保留六帧三相机、FM及23D实际控制；采样轮转修正和新optimizer/scheduler阶段不能伪称单因素消融。固定80样本最终FM `0.19877684508101084`，A2最终 `0.20739499653573149`；逐task为0.142679443/0.163655016/0.274372173/0.161038742/0.252138852。约4.16%的固定集loss下降不是完整eval、对象服从或任务成功证明。下一步先装配同神经数学、同历史/normalizer/动作桥的独立A3推理快照，验证实际权重恢复与真实通信后做闭环；不覆盖A2及既有结果。
- A2 plates task3实际20544/20544控制，5493.567秒，官方失败；161次高层请求为22NAVIGATE、3OPEN_DOOR、136GRASP，无PLACE。路径26.382米、累计转动1019.351度。step1247–1280明确请求GRASP bowl_92，但右手物理确认持有plate_93；1537在NAVIGATE期间失去持有确认，未自动标成意外掉落/FAILED。2048–17664共15616控制、122次高层请求持续GRASP plate_92，实际独立observer报 `semantic_object_scope_binding_not_unique`。不把该UNKNOWN伪装成oracle确定失败，也不偷喂场景object_scope给策略。原始高层raw_text确认`Task goal: cleaning up plates and food`由模型生成，**不是服务端遗漏Parent后的静默fallback**。完整视频已本地展示，本人看30张全程面板；result SHA `677b48d3ea9f2e767f67daa20897b9d0416d5511a4459944556bf667f95d7d91`、physics SHA `2985b60f50733a6fb9cba1f284f53ffe6c2907d99f0aec2652275ad7f0ae6725`。
- A2 can_meat task4实际17770/17770控制、4544.509秒，官方失败；139次高层请求为6NAVIGATE、133OPEN_DOOR，无后续GRASP/PLACE。路径15.928米、转动1610.818度，逐物理步17770均UNKNOWN，无诊断缺帧。完整五个回合现为0/5，每任务仍只有复用public_test301/seed0/policy17这一例，绝不是盲测或总体SR。task4 result SHA `f8ecc3567eba106d106b031117dcc9e406d9fc12b5b0449cf219994434e2fb48`、physics SHA `4249d9791007dbf12944248352fa1c3dce0eeb4c68683889d86c709e5517fbbd`；视频已传本地，个人视觉检查待完成。
- A2 oracle-low radio train e121/instance138在原448控制前缀、正确原GRASP、真实6帧stride16条件下，完成80×16=1280模型控制，80次chunk后观察均IN_PROGRESS，无稳定抓住、无高层调用。所有80模型请求与独立捕获的真实过去观测hash一致；不能排除chunk中短暂持有。本人看28张全程面板、5组六帧历史90面板、12张原视频/回放比较。报告 `A2_ORACLE_RADIO_E121_PERSONAL_REVIEW.md`，实际result SHA `01a52a70d5fb7b5b7284a29cc73b0f3bf7401cdca6987f700ac22abe9dee360d`。
- **控制器不是这一例主要证据指向：** 核对真实机器人配置为绝对位置关节控制，A2右臂80个chunk末端实际q与末条设定值的RMS误差中位数0.000235 rad、最大0.000876 rad。夹爪收到闭合后也实际闭合。A2首次close为源帧1518，原控制为952；不能用修改电机增益或统一提前566帧当根治。`execution_tracking_v1.json` SHA `90d438956d86351cc4d14e476f1cacff6aeb012aed44afd6a0163b5148e67f4a`。
- **原成功状态预测进一步缩小原因范围：** `a2_original_radio_replay_state_predictions_v1`在42个实际原回放状态上分别seed17/29，首状态重复seed17，共85次真实模型调用，15.08GB峰值分配，0新物理动作。全部输入是过去6帧；未来原动作仅作比较目标。同状态同seed重复逐字节一致。seed17首close960、seed29首close944，接近原952；但在原右臂确实需运动的13个chunk（未来16步相对当前q RMS≥0.005rad）中，预测沿原进展方向的投影增益中位数仅0.01838/0.01778，优于保持当前q的仅6/13、4/13。原动作只是一个有效续行，并非唯一正确答案，不能把这两个增益当动作放大系数。证据表明**专家状态上的动作进展学习不足也存在**，不是“teacher-forcing完全正确、纯闭环漂移”这么简单。
- 原radio+plates成功动作回放产生715条past-only outcome候选：482UNKNOWN不监督、174IN_PROGRESS、59SUCCEEDED，0FAILED；只有两个train实例，不冒充C1 learner rollouts或C2纠正。本人查看全部59成功及按各段/结果分层的共125条三相机当帧、18组六帧三相机历史，总699个显示面板，明确检查抓稳/释放后6帧门。真实冻结B的processor125条均通过三组[1,6,3,256,256]和7字段observable入口检查，无模型/optimizer。报告 `ORIGINAL_RADIO_PLATES_OUTCOME_PERSONAL_REVIEW.md`；候选SHA `70df77a3b075969edc59a97e174feabe86352cf8fceef017bdb947c6c2322448`、实际processor结果SHA `e32c2e694320acc50e44b8f9ea0f15b632e30c0fbe314001d0113d87f7dd2581`。仍training_admissible=false，缺跨实例校准与真实失败/恢复，未伪造准入。
- 为补齐OPEN/CLOSE真实反馈，两个train场景reset后实际读取冰箱和柜体Open相关关节、body1移动link、metadata、实际加密资产SHA及loader MD5；前后完整physics dump逐字节相同、时钟41不变、0控制/模型。已生成 `actual_live_exact_part_registry_v1.json` SHA `aedc0609dec8cd522b9e5342f0a184e6e5a08b81d9195b87ab4593431634ed95`：精确支持fridge/petcxr的left_door/right_door及bottom_cabinet/rhdbzv的left_drawer/right_drawer。**left/right别名仍未验证，不猜测映射。** 新v4采集只向新observer实例注入已核验resolver，不改模拟器/旧v3/已完成评测/策略输入；20项CPU回归通过，第一次6项失败为未设置选定g05源的测试导入路径，失败XML保留。新plates回放已准备10709原控制，实际新物理结果仍待验证。
- 下一阶段按可解释瓶颈推进：A3实际闭环与原状态进展复测；精确部件绑定后的原控制反馈覆盖及本人审核；高层真实目标绑定/父目标生成和低层条件服从联合诊断，避免仅靠语法合法或低CE放行；补齐C1实例隔离的反馈训练/校准、真实失败后成功纠正及协同训练。主仓源码不热改；最终必须在未回灌的独立实例上验证完整自动成功率提升才完成goal。

### 历史：第三个完整失败、原始动作对齐证据和真实历史低层隔离诊断（v27）

- A3正式训练已超过200次实际更新。四rank首个更新均记录冻结部分逐字节不变、动作专家和VLM LoRA有效更新；原A2父权重没有被诊断smoke覆盖。固定80样本FM：A2最终0.207395，A3第100步0.207970、第200步0.207871。当前基本持平略高，不能宣布改善、未欠拟合或SR上升。第200步task0–4为0.140496/0.177236/0.286874/0.158917/0.275832；它仍不是全eval。训练期尚无完成时才写出的actual_draw_summary是正常行为。
- A2万圣节task2完整执行20682/20682控制，5164.299秒，官方未成功；162次高层调用中16次导航、146次OPEN_DRAWER，2048–20681持续指向bottom_cabinet_rhdbzv_0的left部件，没有进入抓取/摆放。累计路径47.070米、转动4082.646度、净位移0.751米；低层每16步平均618.717毫秒。物理observer均UNKNOWN且没有持有对象变化，UNKNOWN不改写为FAILED。视频已传本地，本人查看30个准确视频帧，反复绕电视柜/壁炉/餐桌/沙发，未见有效抽屉操作。报告 `A2_HALLOWEEN_FINAL_PERSONAL_REVIEW.md`，完整汇总 `a2_final_halloween_actual_episode_analysis_v1.json`；result SHA `f8193762a4944abb6e3706f3290617b00848232c0eed6c7261201a2a1391098e`，physics SHA `f9e5da644b7300ef3eaa5d7a944b2b38bf1764f9bd35d336e0ba5df5c2630936`。只报告已完成的三例失败，后两例不提前填结果。
- 原始plates train episode762/instance242全段10709个原始23D控制实际完成，未到official terminal，0模型/optimizer，不是模型SR或C2成功纠正。9项轨迹审计反例通过，真实逐条控制/物理时钟/原半开技能区间核验通过；7个技能段有稳定因果成功，807帧SUCCEEDED、2344 IN_PROGRESS、7558 UNKNOWN、0 FAILED。本人查看所有19个技能边界和首次稳定成功附近，共27时点×3相机=81面板。可见开冰箱、右手放盘子、双手持碗和依次放水槽，仍不代替official完整任务判定。报告 `DEMO_PLATES_E762_FULL_SKILL_PERSONAL_REVIEW.md`；physics SHA `a2f67832cf666ff1bd5a7197f5ee61531f7a7c9336b11875d967ea603e73e9b5`。
- **明确的意图—动作边界问题：** 第一只盘plate93的GRASP标注在2317结束，本次回放首次确认抓住在2321，此时标签已经NAVIGATE；持有持续至3050释放。不能把2317伪造为成功，也不能说整段从未抓住。它证明该例原始标注边界不等于本次实际物理完成时刻，可能包含回放动力学差异，不据一例宣称全数据存在统一帧偏移。抓取/放置的释放是事件事实，未确认真实意外掉落，不新增FAILED标签。
- 开门/关门当前返回 `open_parent_identity_unverified`：现有oracle没有实际资产SHA及“指定部件→该对象自身关节/移动link/方向”的绑定。资产元数据可供构造，但必须核对实时Open相关关节；不使用整个对象Open状态冒充指定右门，也不猜left=left_drawer。该缺口影响反馈取证覆盖，物理oracle从未进入策略输入，因此不是直接驱动转圈的信号。
- 新隔离目录 `native_a2_prefix_runtime_v1` 保留运行中的v4原样。新的客户端逐物理步接收真实观察，在演示前缀N后送N−80..N的六帧、stride16，完整三相机/本体，且新鲜读回必须与最后实际捕获字节相同；非16倍数前缀467/4221也保留真实时钟，不假装0、不重复当前图。低层接收只放宽独立诊断的初始时钟，默认整任务入口仍要求0；window SHA只作身份校验，不入模型。旧16项检查通过，但实际C1记录器→客户端→原有localhost transport发现新客户端仍认旧mode名称；在新入口使用前修正，保留失败复现XML，19项完整检查通过。该新入口缺陷不能被拿来解释此前模型失败。
- `a2_prefix_actual_observation_probe_v2.json`用六个既有train窗口、覆盖五任务的18次真实保存观察，验证原图/本体/时钟、实际msgpack和服务端tensor/hash一致，0模型/physics/更新，无数据准入。它是旧C1观察的接口检验，不能伪称新A2物理评测。`run_a2_prefix_radio_l1_v1.py`已冻结 `a2_prefix_radio_e121_l1_v1/manifest.json`：原train e121/instance138、前缀448、原正确GRASP、80×16预算、原A2-final5000及policy seed17。模型拟用独立GPU0/8776，模拟器GPU3私有cache和排他锁；共享四卡训练时检查实际显存余量，不动GPU1完整评测及GPU2既有服务。模型和实际物理结果仍须另外验收。
- 曾怀疑C1 `Parent command`与训练条件不一致；直接查到真实A2训练e121 frame609的GRASP条件与该C1完全一致，故不把它改成`Task goal`，也不将此项未经证实的假设列为根因。实际低层服从、反馈头及校准、成功纠正准入、后续协同训练、主仓安全集成、独立完整任务SR提高仍待完成。
- 本轮网络传输失败留下4个自有rsync接收进程；仅经uid、精确argv、WORK目录、父子关系和启动时刻核验后用pidfd逐个停止，未停止训练/仿真/服务，未删除模型或历史。新候选逐文件校验后才继续；主仓代码及已运行源码均未热改。

### 历史：A3真实四卡验收及启动、C2完成和首帧结论更正（v26）

- 新阶段独立source `a2_lora_history_candidate_v7_samplercoverage` SHA `356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281`，明确打开episode轮转，父权重仍是原A2-5000（SHA `b57eed704179a1517f6582b7572c975f732d3bc3c49d80ce119eaa2f643ae640`）。每卡microbatch2、累计2、四卡global16、lr1e-5、5000更新；保留原950/50切分、统计、三相机六帧、完整动作和FM32/执行16。`resume_ckpt=null`，新optimizer/scheduler/RNG/采样阶段，不伪称等价续训或单变量采样消融。不加入C1/C2/public_test数据。每100步原fixed80、每1000保存，启动前保留足够新输出空间且不删除历史权重。
- 两版真实processor共20条batch4和10条batch2样本，包含所有五任务及此前漏覆盖的任务2/3/4轨迹，18路camera/time映射和完整动作通过。第一GPU预检误读训练模型不存在的serving诊断属性，在optimizer创建前失败；独立v2直接逐字节验证1138条完整父状态（含192 LoRA）、实际4个microbatch和2次累积更新，冻结部分不变，动作专家/LoRA均更新、Adam计数2，峰值reserved32,631,685,120字节，无保存诊断模型。结果SHA `cd0acb097feacf7477c7a4e85f0d36436a07c5a4dc4a5de2074e6670f428fa13`。另17项CPU反例通过，旧环境误配失败保留。
- 原launcher准确解析smoke/formal两份配置通过。`a3_episodecoverage_actual_four_gpu_fresh5_v1`真实四卡5次更新正常结束，11:39:52 UTC保存step5，checkpoint SHA `dfae4b5c4fe593449f6b9b7c0552df7014e72d68e42251a68fb7a91761736335`。`a3_actual_saved_smoke5_inspection_v1.json`本人实际回读通过：1138状态中624冻结条目逐字节不变、504训练条目变化，504份Adam计数全5、scheduler5、下一microbatch10，全部四rank RNG保存，80实际训练样本覆盖五任务且梯度累计计数一致；四rank首次梯度审计均证实AE和LoRA更新。诊断不是SR，也不混入正式父权重。
- 以上门通过后，`start_a3_phase_v1.py --phase formal`提交独立run `formal_a3_episodecoverage_5000_v1`，launcher PID3589152；正式从原A2-5000重新载入，而不是step5。仍须核验真实训练早期更新与显存，不把提交启动当完成。A2/B服务及GPU1完整评测保持原样。只在C2全部窗口完成且无活动连接后，用身份/进程起点/pidfd核验精确停止自有闲置旧A和G0.5两个服务，保留所有结果，见 `c2_completed_idle_service_stop_v1.json`。
- C2最后can-meat train episode807/instance10完成：原前缀1816、旧A1280，再同一未完成状态G0.5接管1280（3097–4376），全IN_PROGRESS，未抓稳目标hinged_jar_236，official terminal=false。4085–4090右手确认持有的是柜门top_cabinet_lkxmne_2而非目标，不凭此改成FAILED。接管期间无reset/load；第一query前后snapshot一致，0教师控制/更新、training_admissible=false。本人看19个全程和6个接管细节面板，视频传本地，报告 `C2_CAN_MEAT_E807_PERSONAL_REVIEW.md`。五窗口中四次接管全部未成功，另一个Halloween旧A已成功而跳过；没有成功纠正数据或物理复放认证。
- **首帧结论更正：** `initial_radio_rgb_render_only_v1`零控制/模型/额外physics，三次render及每次读取后的7次检查物理状态逐字节相同、clock41→41、本体不变。但是本人查看15个三相机对照面板时发现本次缓存首帧本来正常；没有复现旧“沙发”视角，不能声称刷新修好问题。口头“已复现”已向用户更正。候选未接入任何A2评测，报告 `INITIAL_RGB_REFRESH_PERSONAL_REVIEW.md`。新原始回放v3在step前复制观察只是因果记录防护，不是已证明旧缓冲区别名错误。
- 新 `demo_plates_e762_feedback_trace_v3`正在用原始train episode762/instance242全段实际控制采集，覆盖GRASP之外的原技能，首次局部成功不停止。原始动作不改、0模型/更新，training_admissible=false；不是C2同失败状态纠正。后续需实际物理结果、本人动作/图像/语义复核及独立准入，不能用原始标注边界虚构完成/失败。首批记录已出现抓取和放置的实际稳定成功，完整运行及审查尚待结束。
- 仍未完成真正可部署的物理反馈头训练/校准、失败与恢复覆盖、成功教师纠正及普通技能服从、主仓串行集成与独立完整任务SR提高。修正覆盖也不能单独解释已经190/190覆盖的收音机/捡垃圾失败，因此继续实际根因分析，goal保持active。

### 历史：真实采样根因、第二个完整回合与原始演示物理核验（v25）

- A2捡垃圾 `native_a2_final_development_pilot_v2_wirefix/task_1` 完成7901/7901实际动作、2089.831秒，官方未成功。7901条动作与保存数组逐位相等、物理记录无缺口。frame1029确认右手稳定抓住trash_can_116，1280高层切换为导航can_of_soda_114，完成反馈滞后251动作；3072至7901的4829步均未抓稳汽水罐，没有PLACE_IN。不能再将这次描述为高层始终抓垃圾桶。旋转1516.56°→483.63°，属于局部进步，不是整任务SR提升。本人查看27个全程面板及6个细节面板；视频已传本地，报告 `A2_TRASH_FINAL_PERSONAL_REVIEW.md`，汇总 `a2_final_trash_actual_episode_analysis_v1.json`。第7281–7318帧持有确认短暂丢失，不凭此自动制造FAILED标签。
- 已将有效A2 step1–3000和正确续训3001–5000的四rank共80,000样本，逐条与原真实Dataset/旧sampler重放核对，一条不差。旧调度只打乱任务、不打乱任务内源轨迹顺序，以平均叶子数在4192批结束epoch，4193又从头开始；较长任务尾部持续漏训。task0–4实际见过190/190/126/181/125条合格episode，合计漏138条；这是本次A2增量的覆盖事实，不声称此前祖先模型也从未见过。收音机和捡垃圾没有这一漏轨迹现象，因此不能用它解释所有失败。
- 独立 `a2_lora_history_candidate_v7_samplercoverage` 仅修改生产sampler，新增显式 `leaf_ordering=episode_round_robin_v1`：任务内按episode轮转，再遍历其技能；epoch按最大任务容量；帧索引用跨rank相同的leaf-local随机流再分片。默认source_order_v1逐位保留旧行为。8项新测试和6项旧回归通过，`a2_samplercoverage_regressions_v1.xml`。真实 `a2_samplercoverage_actual_source_probe_v1.json`（SHA `30b9c714cdfaea22c261c36ce921de8c810107b19607a0382e39196093407a7b`）验证新调度前250步已覆盖950个episode，5000步仍全部覆盖、六对rank索引交集为0；完整6442批epoch覆盖每个合格技能bundle。sampler SHA `7b084fdf00575b0325fa6e12fa2d4facbece07f8307e4ae9081985ec69f32af4`。本阶段只有CPU索引检查，无新模型/梯度/训练；改变调度不允许伪称旧checkpoint的等价采样续训。下一训练阶段须显式选项、真实processor batch验证、独立谱系与新run。
- 原始train episode121/instance138语义为NAVIGATE[0,448)、GRASP[448,1122)、PRESS[1122,1226)、PLACE_ON[1226,1562)。两次独立原始23D动作回放均在971达到稳定抓持；有效v2在1193触发官方success=true。`demo_radio_e121_alignment_v2_persist/actual_replay`1193动作逐位原始，物理trace SHA `2b0b0a44017c3094f566552becab307acfcb29352fd762f97b243ede48b316f3`。没有模型查询/更新，不是独立失败状态纠正，training_admissible=false。PRESS在官方终止帧刚满足，不能虚构六帧稳定oracle确认，也不能因该诊断oracle仍IN_PROGRESS否认官方成功。
- 原始回放v1受Kit shutdown进程退出行为影响，外层finally没有落result和actions，视频/完整物理trace仍保留；v2仅在evaluator上下文内持久化，7项回归通过。不得给旧v1补造正常完成回执。本人已检查原始/回放5个关键帧三相机对照、额外1/16/32帧对照、两个回放全程抽样。首帧head明显朝向不同，第1帧恢复；后续几何/抓持语义相符，物理状态不是逐位一致。
- 首帧问题的实际源路径：官方Evaluator.reset读取env.reset()[0]后，非灯光任务的_sync_lights_and_get_obs直接返回传入obs，未render；adapter.observation只投影缓存。独立 `initial_observation_refresh_v1.py` 在每次render和读取观察后检查完整序列化物理状态及physics clock逐位/严格不变，不执行控制或settling；真实初态probe尚未完成，不能提前部署或热改正在运行A2序列。首帧问题也不自动解释数千步后的抓取失败。
- C2 plates train episode762/instance242：原前缀4221、旧A1280、同状态G0.5接管1280；接管5502–6781全部IN_PROGRESS且双手未确认持有，未成功，无数据准入。本人看23个全程面板和6个接管细节，视频本地 `C2-plates-train-e762`，记录 `C2_PLATES_E762_PERSONAL_REVIEW.md`。原前缀中的搬盘动作不能归给A/G0.5。当前index13 can-meat episode807/instance10继续GPU3；评测task2继续GPU1，无热改、未停止外部作业。
- 后续仍以真实反馈覆盖、低层普通技能服从与完整任务SR为门：完成当前五任务和C2；验证并修正首帧读取；把新采样接入明确的新训练阶段；独立扩展成功即停/GRASP-only的C1采集协议，取得可确认的失败/恢复事件，补做实际结果头训练与校准。未得到成功纠正的记录不作为BC教师；没有独立SR提升不标记goal完成。

### 历史：首个A2完整回合、最终意图干预、三条C2结果及采集缺口（v24）

- A2完整收音机回合 `native_a2_final_development_pilot_v2_wirefix/task_0` 正常退出：3224/3224个官方动作、879.517秒、task_success=false。新manifest SHA `3c3ecea5612ec0c6c936633a0a5c60eb440ce256d972f93f445b642147972f0a`，runtime v4未热改。序列PID3544699已自动进入task1；这是新的有效完整结果，不混入此前0动作通信错误；其余未完成，不得报告本轮五任务0/5。
- `a2_final_radio_actual_episode_analysis_v1.json`重算保存动作与官方trace逐位相等，全部3224个动作都有独立物理诊断，无缺口。高层0–639为5次NAVIGATE（当前oracle导航判据UNKNOWN，不能称导航失败），640–3223为21次GRASP；后2584帧全IN_PROGRESS、目标两手均未持有，无因果抓取成功、无成功后掉落、无PRESS。因此该回合瓶颈是低层未完成抓取且高层反复同意图，不是抓好后没切换。
- 同一development实例旧A绝对旋转537.26°、新A2为129.11°，路径2.816→2.060米；单回合显示反复旋转缓解，但任务未改善。新低层FM平均611.94ms/16步；整段耗时还包含初始化/高层/仿真/新增诊断，不能按wall time单独归因模型。本人看过22个全程非空面板与6个近距面板，详见 `A2_RADIO_FINAL_PERSONAL_REVIEW.md`；本地视频 `/home/wsy/behavior/memlite-results-20260911/A2-5000-Bfinal-radio/rollout.mp4`。
- 最终A2的 `a2_step5000_actual_action_interventions_v1` 在GPU0独立进程完成15个已保存状态/69次真实推理，不占正式A2/B会话、不做物理或optimizer更新。1138条目/192个adapter全量逐位回载，15/15同seed重复相等；原始result SHA `49605fc8e5e03a549eb145aa2d330f84b6f41bc39ab81c730bcb93acf8cac752`。三组旧A/A2-2500/A2-5000的全部动作文件SHA核验并重新计算差异，保存 `a2_final_condition_comparison_v1.json`。
- 排除W12后12状态的“换技能RMS÷换seedRMS”中位百分比，旧A→A2-5000：底盘1.165→1.296、躯干1.258→1.931、左臂2.916→2.618、左夹爪0.366→0.544、右臂3.354→2.267、右夹爪0.563→0.950。目标替换主要6状态约0.385%–1.128%。仍无明显条件响应增强；这不是服从/成功率、不是单独LoRA消融，替换条件未证实在接收状态可行，绝不与原动作拼成伪BC。详见 `A2_FINAL_CONDITIONING_PERSONAL_REVIEW.md`。
- C2第二收音机train窗口episode91/instance97完成：前缀467、A1280、独立G0.5同状态1280，纠正1280条物理记录全IN_PROGRESS、无目标持有或official terminal，接管无reset。与首例一样不准入成功纠正、不伪标FAILED。本人20+6面板复核和全trace统计见 `C2_RADIO_E91_PERSONAL_REVIEW.md`；视频已在本地 `C2-radio-train-e91`。
- C2第三窗口task2 episode479/instance97完成：前缀1972后旧A执行448步，frame2420右手持有pillar_candle_91（candle.n.01_1），连续18个不同physics frame成功，超过6帧门槛；程序正确跳过G0.5。本人查看17个全程非空面板及最终6时刻×3相机18面板，右腕可见夹爪闭合于蜡烛周围；见 `C2_HALLOWEEN_E479_PERSONAL_REVIEW.md`。这不是A2表现、不是整任务SR、不是独立纠正，也没有数据集级准入。
- W6与以前旧A未成功记录的实际比较 `c2_w6_restart_input_comparison_v1.json`：前缀端点frame1972的六个本体通道逐位相等，RGB像素RMS差约1.15–1.46，首16×23动作RMS差0.000541、最大0.002719，语义相同。未隔离图像/后端/隐藏物理状态，不把成败变化归于新模型或单一渲染因素；需重复与输入一致性评测。旧/新记录均保留。
- 第四个C2训练窗口index9（task3 episode762、prefix4221）由未改既有入口启动，PID3549028、GPU3；旧A/G0.5服务仍在GPU0，A2/B在GPU2、完整评测在GPU1。没有停止外部工作、没有热改任何运行source、没有重训新模型。
- **新增结构性发现与下一步依赖：** 当前C1选择器只挑单GRASP；driver的`failure_reason_provider`默认空且两个工厂都未提供，GRASP未持有只返回IN_PROGRESS，首次真实SUCCEEDED又立即停止。多数失败/恢复事件因而在采集设计上不可见，不是继续多采同类窗口就能补齐的随机缺样。须新增独立采集版本、扩展真实技能/成功后轨迹和有严格反例的物理失败事件；主动释放/未知/超时不得误标FAILED。详见 `C1_FEEDBACK_COVERAGE_FINDING.md`。本轮只查清并写入依赖，**尚未实施该新采集器或训练四类反馈头**。
- 接下来：完成正在运行的五任务序列及C2；校验演示GRASP标签与实际物理判据的对应、训练边界/动作启动采样；实施上述反馈采集改版、取得真正独立成功纠正并复放/亲审，再做协同训练和独立提升验证。仅loss/旋转改善不满足goal，主仓尚未做最终实现集成。

### 历史：5000训练完成、真实通信故障修复与首条同状态接管（v23）

- A2 run `formal_a2_lora_history_5000_v5_arrowfilter_adamresume3000` 于04:18:52 UTC原子保存最终step5000并正常退出，run receipt为complete/returncode=0。最终checkpoint SHA `b57eed704179a1517f6582b7572c975f732d3bc3c49d80ce119eaa2f643ae640`，run receipt SHA `96c2f1a1e3ff6a8512b88f5cf6ee2ebbdf42f8310a1bc1905922279cf2a8da3e`。`a2_v6_actual_saved5000_inspection_v1.json`真实读回1138模型条目及504份Adam一/二阶矩均有限、六组Adam计数全为5000、scheduler=5000、六组LR=3e-6，四rank RNG/sampler及冻结source正确；无新optimizer更新。
- 最终固定80平均FM `0.2073949965`；task0..4分别为 `0.1385297216 / 0.1724527660 / 0.2832714375 / 0.1652814474 / 0.2774396101`。这只是原冻结80窗口，不是全eval、不证明服从或成功率改善；训练完成不等于goal完成。
- 首次完整A2尝试 `native_a2_final_development_pilot_v1/task_0` 保存了0动作和完整traceback：`history_admission.anchor_hashes`直接携带整数帧号作为msgpack map键，严格客户端在收到第一块动作后解码失败。现场真实服务复现 `a2_actual_wire_legacy_reproduction_v1`证明模型确实输出有限16×23动作，但没有任何动作进入仿真。该回合是基础设施失败，不计0/1；原日志/manifest/视频均保留。
- 修复仅在新runtime `native_a2_history_runtime_v4_wirefix`：内部历史索引仍为整数，出站诊断单独转换为字符串键，不改神经source/权重/输入/动作/严格解码器。旧v3保持冻结；只精确停止已空闲的自有A2低层服务PID3537189，未停止B、旧A、独立G0.5或采集器。新服务 `a2_final_eval_a2_low_service_v2`再次全量逐位加载1138条目/192个adapter，使用相同最终checkpoint和配置，并显式记录history runtime hash。
- 6项真实历史结构回归、22项新runtime回归、训练Python3.10和仿真Python3.11各3项实际localhost通信测试通过。`a2_actual_wire_fixed_mainpy_v1`与`a2_actual_wire_fixed_behpy_v1`均用失败回合保存的真实初态、原B生成的相同意图和seed17查询实际A2服务；两者严格解码通过，16×23动作与旧服务逐位相同，动作文件SHA都为 `af561669de7552edc102841677605f3b075a6bd8a79f664ba63716cba577940e`。这些测试无物理控制、无权重更新、非成功率。
- 重新冻结 `native_a2_final_five_task_development_v2_wirefix.json`，输出独立的 `native_a2_final_development_pilot_v2_wirefix`；仍是旧五任务public_test instance301/seed0、policy17、3224/7901/20682/20544/17770完整动作预算与原wall budget。只改变低层A→A2；B仍UNKNOWN_ONLY，未训练物理反馈。五个实例已因反复使用成为development pilot，永久eval-only，不称盲测/总体SR、不回灌训练。
- C2第一例 `c2_same_live_state_pilot_v1/c1v2-matched-t0-train-e121-f448-grasp` 已真实完成：train instance138，演示前缀448，旧A实际1280，原始G0.5在同一未完成状态接管实际1280，期间无reset/load/演示动作。第一查询前后snapshot完全相同；1280条纠正物理记录全IN_PROGRESS、两手持有列表全空、无official terminal，稳定抓取未达成，**不准入**成功纠正数据、不伪标FAILED、不认证教师、不声称物理复放通过。
- 第一例视频已传到本地 `/home/wsy/behavior/memlite-results-20260911/C2-radio-train-e121/rollout.mp4`。本人亲看20面板全程抽样及6面板接管细节，并核对全部1280条实际动作物理时钟/持有记录；结论与限制见 `C2_RADIO_E121_PERSONAL_REVIEW.md`。旧A长时间停在夹爪接触附近，G0.5接管有姿态调整但未确认抓稳；此窗口不支持把失败都归因于原地打转。
- 第二train窗口index1（episode91/instance97、前缀467）由既有未改C2入口启动，继续真实同状态接管；不因首例不成功就改标签或删除失败样本。任何后续成功都仍需单独物理复放及本人逐段意图—动作审查，不因最后GRASP成功将全任务G0.5之前的导航/开门动作贴成GRASP。
- 尚未完成：A2完整五任务结果及实际普通技能服从；多样且物理可验证的反馈类别；成功纠正及复放准入；基于这些数据的协同训练与独立提升验证；主仓串行集成。因此goal保持active。

### 历史：3500真实检查点与C2复放准备（v22）

- 新run于03:10:52 UTC完成step3500原子保存，checkpoint SHA `393ca7cf5a2862ceaaafd7da770879071d094a40077dd738cfc985a80bc9dfc9`，16,581,364,894 bytes。`a2_v6_actual_saved3500_inspection_v1.json` 实际CPU读回检查通过：1138模型state条目及Adam一/二阶矩均有限；6组、504份Adam状态的标量FP32计数全为3500，scheduler=3500，六组LR=`8.776425088350713e-06`，四rank RNG及采样续接一致，source仍为审定v6。检查自身无optimizer/GPU模型更新，正式run继续，不能标complete。
- 原3000固定80 FM=0.2142255375；新3100/3200/3300/3500为0.2131682757/0.2124158705/0.2095831787/0.2094292760。3500逐task为0.1349297131/0.1776061761/0.2818155112/0.1694403613/0.2833546181。只是固定80诊断，不是全eval或成功率；不因这点下降宣布低层服从或完整任务改善。
- `c2_replay_audit.py` 对新同状态采集输出逐条交叉核验：保存动作/physics trace/chunk中实际消费动作必须完全相同，接管前后无动作的状态相同、帧时钟连续、六帧stride16历史及文件hash正确、初态已知未满足、最终成功有6个不同的同目标真实物理帧依据。eval、自教师、提前admission和未来帧拒绝。`c2_physical_replay.py` 在调用方一次恢复后的独立环境中只复放原实际控制，无额外settling、reset或policy查询；起点误差阈值绝对1e-5且记录是否逐位一致，结果只覆盖局部GRASP，不宣称全episode计数/RNG/轨迹逐位等价。
- 新结构审计5项、加物理复放loop共9项真实parser/oracle＋显式假物理fixture检查通过，最终记录 `c2_physical_replay_tests_v2.xml`；v1测试记录仍保留。`replay_c2_trial.py` 的主训练环境和BEHAVIOR解释器CLI均通过，实际7个physics/controller/robot-config源路径可读取并记录SHA。**没有运行新Kit或真实复放**，这些不是实际专家认证。
- 同状态采集factory新增 `physical_runtime_sources`，记录当前simulator、robot、controller、scene、env、oracle和robot-config源；不改动作/观察/成功规则。当前factory SHA `6dc1642769269cc68f8a55f4e17710a527b87186f9d9c0b16517fb3e81e4ba13`；原C1冻结factory及原采集结果未改。新纠正原始轨迹仍必须按实际技能片段由本人审查；独立G0.5输入为全任务，不能因为最后GRASP成功就将此前导航/开门动作全部贴上GRASP做BC。
- `launch_c2_trial.py --index 0` 实际预检通过；将使用GPU3和已有且可写的独立 `kit_c1_gpu0_appdata_v1`（约7.3GiB，旧screenshots子目录不可读保持原样，不chmod、不复制）。与GPU1完整五任务评测使用的 `kit_c1_gpu1_appdata_v1` 分离；模型各用独立端口/会话。C2正式启动需要A2正常结束、旧A/独立候选实际服务身份匹配和GPU3空闲；当前未启动。成功候选的复放输出必须另建目录，原采集与结果不修改。
- 官方starter primitives仍显式声明WIP、仅支持特定绝对位置控制器，OPEN/CLOSE直接`NotImplementedError`，初始化还改grasp window。因此未拿它们代替实际纠正教师或改官方物理设置；若独立G0.5不能实际纠正，保留失败证据后再决定受验证的下一条物理控制路线。

### 历史：实际四卡正确续训与最终评测准备（v21）

- 新run `formal_a2_lora_history_5000_v5_arrowfilter_adamresume3000` 于02:41 UTC由launcher PID1870445启动，torchrun PID1871155、ranks1871166–1871169；02:47:28记录恢复step3000和原rank RNG，02:47:34开始训练。第二迁移副本SHA `c84094a57e3d46256bb1175d926e8cb800092da743b4da558776b627cc6a98b2`，仅来自原始合法3000，坏v5续训完全排除；原文件未改。
- `a2_v6_actual_resumed_updates_v1.json` 已通过：四rank实际3001/3002/3003的采样及条件逐值等于原run继续段；首步3001六组LR均为`1.2658877580481063e-05`；action_expert和vlm_lora均有效更新，冻结参数逐位不变；实际日志包含正确resume/RNG恢复且没有任何optimizer reset。此为真实更新证据，非CPU加载或配置测试；普通worker重建后的未来augmentation轨迹不声称逐位等价。观察到已超过3030步，训练未完成。
- v6对应serving源 `a2_history_serving_candidate_v3_adamresume` SHA `2c4525691cb799e43768d570898c8ed33eda9a852db1828d59d12a6c015b91c8`；`a2_serving_composition_v3_adamresume/source_receipt.json` 逐文件/AST绑定真实训练神经实现与已审原生FM接入。配置SHA `2ef4c1f9932d83e843a04eb45aa5e419862e8da1732d50d110ff019fc9b869c3`，snapshot SHA `f4ad901c6f3159878ec8dbce7e46d70df7b09f4bd9607b2ba592fa74d32eca59`。40项serving回归、20项runtime回归及3项真实localhost通信检查通过；五任务实际六帧训练/推理前缀、mask、最新本体反归一化锚全部相同。
- 在尚未启动的v3评测runtime中修补诊断漏采：此前`held_objects`仅chunk起点记录；现在每次实际physics action后都记录两手全部持有/未知对象，能保留块内瞬时抓错和掉落。加入对应3动作夹取/掉落fixture断言，`a2_runtime_v3_regressions_v2.xml` 20项重新通过。只读observer不进入模型、不改控制/成功判据；旧v1记录保留，原训练源未改。
- 最终服务与五任务入口已准备，尚未执行最终模型或新Kit：GPU2的A2/B使用独立8772/8773端口，GPU1依次完成五任务，防止多回合共享服务内部历史/RNG。GPU0的旧A与独立G0.5候选供GPU3同状态纠正；两个模拟器互不共享模型会话。入口要求训练完整正常结束、实际5000全量回载证据、来源/切分/相同预算验证；未见最终checkpoint前不冻结评估manifest。两项服务证据门测试及真实oracle/cache路径检查已通过。
- 下一步仍是完成并冻结A2 → 真实完整五任务development pilot及低层服从 → train-only同状态独立纠正与复放/本人视觉审核 → 补足真实反馈与协同训练 → 独立评估及主仓整合。没有新success-rate、反馈头更新、纠正教师或数据准入；不能用这些工程检查替代目标结果。

### 历史：纠正真实Adam续训状态恢复缺陷（v20）

- 首次v5 reader续训 `formal_a2_lora_history_5000_v4_arrowfilter_resume3000` 实际日志于02:20:21记录“Reset optimizer state for 504 params”。虽然迁移副本完整，通用 `fix_optimizer_state_after_resume` 随后将Adam的标量`step`同参数矩阵比较形状，错误清空一阶/二阶矩及计数器。这违反保留optimizer状态的承诺；不能只凭副本相等或scheduler步数宣称成功续训。
- 已通过精确argv/UID/PPID/start_ticks和pidfd停止仅此自有torchrun PID1788560。`a2_v5_bad_optimizer_resume_stop_v1.json` 保存原因与进程身份；原进程已退出，run为`failed`。最后记录batch step3095，3000之后这段尝试完全排除于正式谱系；未删任何日志、权重或旧迁移副本。此bug只影响本次实际resume，不解释此前fresh训练的0/5。
- 原始合法checkpoint仍为 `formal_a2_lora_history_5000_v2/checkpoints/step_3000.pt`，SHA `1795deb7986095ae3ea0eecbe31eb5c799d1adb1caebdbaf8575ee9fffcb97c0`。不采用坏续训的参数，不回退到初始A5000；有效A2训练进度仍为3000。
- 独立v6源 `a2_lora_history_candidate_v6_adamresume` SHA `f0ded5bef64485210f97eb8dc4ffbc79ed72d11c5014b38a450736f0d9e26592`。生产代码仅改变两处：Adam标量step不参与参数形状/dtype修复，畸形非标量step拒绝；协同训练resume若仍发生任何state reset，在首个optimizer更新之前立即失败。普通梯度、采样、数据、模型结构和reader均未改变，不热补丁原源。
- `a2_v6_optimizer_regressions_v2.xml`：51项通过，包含矩阵/标量/单元素参数、AdamW/AMSGrad、bf16参数下FP32计数器保真、真实moment形状错误、下一更新与不中断AdamW逐位相同、trainer拒绝reset、LoRA和checkpoint回归。v1因测试入口缺scripts路径而未完成的记录保留。
- `a2_v6_actual_optimizer_resume_preflight_v2/result.json`：实际原始3000模型1138 state条目逐位恢复；按真实trainability和真实参数组构建6组AdamW，回载504份状态。原函数确实复现504次误清空；新函数0次清空，全部状态数值/dtype/shape与原checkpoint逐值相同，模型未改。每份step=3000，scheduler=3000，六组LR均为`1.2658877580481063e-05`。CPU回载无optimizer更新、无simulator；不是实际四卡续训完成。v1预检遗漏正式trainability配置产生的参数组不匹配已保留。
- 正在准备新的 `formal_a2_lora_history_5000_v5_arrowfilter_adamresume3000`，从原始v4 checkpoint独立迁移到v6，不使用被排除的尝试。下一道门必须同时看到实际3001起采样与原继续段一致、四rank有效AE/LoRA更新且冻结不变、正确首步LR、日志无optimizer reset。未通过之前不称续训正确完成。
- 后续serving必须重新绑定v6训练身份；v19的v5-serving组合不能直接用于v6最终模型的来源认证。独立G0.5纠正候选本身不执行optimizer恢复，不受本次bug影响。真实C1/C2与完整成功率改善仍未完成。

### 历史：A2中途条件响应、独立G0.5候选与同状态接管（v19）

- 原四卡run于02:09:53 UTC完整保存step3000，SHA `1795deb7986095ae3ea0eecbe31eb5c799d1adb1caebdbaf8575ee9fffcb97c0`。迁移副本 `a2_equivalent_reader_resume3000_v1/step_3000.pt` SHA `f694b864dcb54ff36ba6ab1405f30dbb1a1cdbb0967ae7bc1e6954918d3a8ad6`；实际模型/optimizer/scheduler/采样位置/四rank RNG及其余原字段逐值读回一致，仅新副本更新明确的source身份并附父run谱系。`resume_identity_receipt.json` 和实际四rank配置预检均通过；原权重未改。
- 02:13 UTC，核验精确argv/UID/父子关系/start_ticks并持有pidfd后，仅SIGTERM原torchrun PID3628926。原160个自有进程均退出，最后观察到batch step3007；这部分3000之后的未存盘日志保留，但不计入新谱系。`a2_v4_step3000_resource_handoff_v1.stopped.json` 记录主动交接导致旧run `failed/returncode=1`，不是数值发散。未删权重/日志，也未停止他人任务。
- 新run `formal_a2_lora_history_5000_v4_arrowfilter_resume3000` 于02:14 UTC由launcher PID1788267启动，source v5 SHA `8efd3dbbaf2f49415a091c5c780f5c19efdebd923e43065a8e1c0ebbfb6598e8`；四卡启动时均空闲，实际命令带迁移checkpoint的`resume_ckpt`。目标累计5000、剩余2000，不重置优化器、scheduler或采样，不从A5000重新开始。待实际首批比对；普通resume会重建data workers，不声称后续augmentation/模型参数轨迹逐位等价。
- `a2_step2500_actual_action_interventions_v1/result.json` 已完成15状态、69次真实六帧FM推理，1138 state条目含192 adapter逐位回载，15次同条件同seed重复逐位相同。换真实同task条件造成的动作RMS相对换seed的RMS，各关节组中位比例为0.482%–3.300%，与旧A5000的0.365%–3.487%处于类似量级；排除已知初态不匹配W12后仍无一致大幅改善。详细口径和限制见本人 `A2_STEP2500_CONDITIONING_PERSONAL_REVIEW.md`。不是物理服从或成功率，也不是只隔离LoRA的消融。
- 独立纠正候选使用MEM-Lite之前的真实G0.5 step5000，SHA `1327c18f6da7697b1c3b0be15b3e9f78eb244fe21bc50c2f15b5c4b3cebf5642`，不是当前A模型，也不输入它没训练过的MEM-Lite条件。独立legacy serving源恢复原词表252189/state token252188，保留原FM路径；仅跳过原本未被选用的AR动作分支。实际946 state条目逐位完整回载。
- `ordinary_g05_correction_actual_probe_v7/result.json`（SHA `efc3905bef360c5f4a51c8a5c0d03fd6a2b280dc88335b7437ec00ec7581e9c5`）覆盖五任务五个保存状态：训练/推理实际前缀token和mask相同，占位动作扰动不改变前缀，32步FM输出经原接口生成完整23维控制，5/5同seed重复逐位一致。此处夹爪观测2维、控制1维；此前接入探针错误地复用了观测维度，旧失败目录全部保留。峰值分配约14.93 GiB，接口验证无模拟器动作、无optimizer更新，不是成功率或专家认证。
- 新 `c2_same_live_state.py` / `c2_official_factory.py` / `collect_c2_same_live_state.py` 在同一个官方环境内先执行原A，再直接让独立候选接管实际未完成状态。没有接管后reset、load_state、teleport或演示动作替代。第一条候选动作之前保存真实序列化状态，核对候选推理期间状态不变；逐动作记录完整23维实际执行命令、目标物理谓词、两手实际持有对象、六帧stride16历史和终止信号。保存的状态尚未通过重放，不冒称已经完成可复现纠正。
- 接管阶段从新的物理初态重新算因果信用；初态已满足或未知则跳过，成功需要6个不同动作后物理帧稳定成立。超时仍可为IN_PROGRESS，抓错对象保留真实held身份但不自动制造FAILED。候选始终`training_admissible=false / expert_certification=false`，必须之后真实成功、重放、切分检查及本人图像/动作审核才可能准入。
- 6项实际localhost通信检查和8项实际semantic parser/物理谓词＋假环境采集检查通过，明确不是物理试验。train W0的完整配置预检通过；所有eval/public-test实例与已知setup偏差的W12被拒绝。正式采集门要求A2训练结束、模拟器GPU空闲及独立Kit目录；未启动新Kit或候选服务，未改正在运行的训练源。
- A2读取迁移后的独立serving源 `a2_history_serving_candidate_v2_arrowfilter` SHA `0966f475479ed88e6bcb5280a67593ebbe57a622cf9c39bca9961a25aa4cb473`，与实际v5训练神经实现及已审原生接口逐文件/方法核对通过，绑定实际续训配置。28项serving回归、20项runtime检查和3项真实localhost控制器测试通过；五任务实际六帧前缀/掩码/最新本体反归一化锚再次通过。早先误用pytest加载带独立CLI的测试入口导致的零测试失败记录保留。旧source/runtime不热补丁；最终权重回载与真实闭环仍未完成。

### 历史：A2实际训练过半、等价读取优化和中途条件干预（v18）

- `formal_a2_lora_history_5000_v2` 于2026-09-10 14:58 UTC启动，源 `a2_lora_history_candidate_v4` SHA `174ab4e88533f7d348e50a92b17780bef8f7c875bac487477334cf3c00e8c51e`。2026-09-11 01:00前后仍可核实原launcher/torchrun/四rank存活，已到2700；checkpoint500/1000/1500/2000/2500均已保存。四rank第2步均证明AE和LoRA有效更新、冻结参数不变。不是训练完成。
- `a2_v4_full_checkpoint_reload_v2/result.json` 已完成真实四卡step5完整回载：1138个state条目（含192个LoRA条目）逐位一致、有限，优化器/调度器及四rank RNG存在。26项回归通过。该结果已经取代v17的“回载核对中”。
- 当前数据读取瓶颈不是显存不足：原sidecar每次只要一条episode，却先把整个row-group其他episode全部变成Python记录。10条原始轨迹逐字段/顺序相同，平均读取12.0568秒降到1.6073秒；这只是该环节约7.5倍，不冒称全训练7.5倍。优化源 `a2_lora_history_candidate_v5_arrowfilter` SHA `8efd3dbbaf2f49415a091c5c780f5c19efdebd923e43065a8e1c0ebbfb6598e8`，仅提前Arrow整数episode筛选及补一个测试fixture初始化，不改数据、模型、采样或优化器。
- 优化版56项测试全通过；`a2_v5_loader_parity_v3/result.json` 又用每任务一条真实完整轨迹逐字段/重复行/顺序核对，并检查缓存、字符串episode旧格式、空值、多row-group，全部通过。原组合release的reader identity不变。v1失败测试和v2错误输入路径产生的空目录原样保留，不改写成成功记录。
- 原训练已过半，因此放弃“从原A5000重新开始5000步”的早期切换方案。正在准备显式的`step3000 -> 总step5000`等价reader迁移：源文件和实际trainer argv除源/输出路径外必须完全对应；迁移只在新副本更新source身份并记录父checkpoint和完整谱系，所有模型、优化器、调度器、采样进度、各rank RNG逐值读回核对。原checkpoint/源码不改、不热补丁。10项身份/argv/优化器/RNG差异拒绝测试通过，实际旧新计划等价检查通过；尚无3000迁移权重、更未停止旧run。普通resume会新建数据worker，不声称未来所有augmentation或参数轨迹逐位等价。
- 固定80验证step2500平均FM为0.2264843605，逐task为0.1543062064/0.1671312761/0.3048764751/0.1855128645/0.3205949804。它不是全量eval或成功率。单帧A5000与六帧A2输入改变，不能单用跨架构loss差宣称变好；实际step2500条件干预正在跑，复用原15状态、69个条件/seed组合，没有optimizer或新物理动作。
- 六帧候选推理源 `a2_history_serving_candidate_v1` 与v4训练神经实现一致；五任务真实观察的训练/推理前缀token和mask逐位相同，反归一化使用最新而非最老本体状态。原生控制/历史一致性/只读逐physics帧物理trace共23项CPU检查通过。它们尚未完成A2真实模拟器闭环，旧B仍UNKNOWN_ONLY；不把只读物理observer冒充模型反馈头。
- 下一步：保留现有训练进度完成A2并冻结 → 相同协议的完整五任务开发pilot与低层技能服从诊断 → 根据真实失败补足C1与同状态外部纠正C2 → 经隔离评测再主仓集成。旧pilot公共实例已经用于诊断，重复评测须明确标为development pilot，不能包装成未见盲测。

### 历史：A2真实四卡小跑通过、完整checkpoint回载检查（v17）

- 第一轮 `formal_a2_lora_history_5000_v1` 在首个零学习率warmup更新的审计阶段自动失败，未保存正学习率更新checkpoint。不是训练完成、不是loss发散。旧v2源码、日志和失败记录全部保留；不能继续沿用v16的“正式训练运行中”状态。
- 根因：审计在DDP构造之前抓取随机LoRA初值；DDP首次广播把非主rank的LoRA A同步成主rank初值，被误判为零学习率optimizer更新。独立源v3只将可训练组基准重取移到DDP广播之后、optimizer之前，冻结组仍检查广播前后逐位不变。真实双进程CPU DDP复现旧错误，并验证修正后零LR延后、下一次正LR验证；凭据 `a2_ddp_audit_reproduction_v1/result.json`。
- 真实四卡 `a2_v3_actual_four_gpu_fresh5_v1` 于14:47 UTC保存step5并正常结束：run receipt `complete / returncode=0`，四rank均在第2步确认action_expert和vlm_lora有效更新，冻结参数逐位不变，第一步零LR延后记录完整。仍从原A5000和原真实训练池启动，没有将eval用作optimizer数据。这五步是诊断，不作为5000步正式模型或成功率证据。
- v3 source SHA `1330db096864fd7decad922cbb1f1828f735c69fd9303642e375b7fe8990cf35`。另在推理接入中发现旧LoRA checkpoint归一化仅去掉PEFT外层前缀，遗漏 `.base_layer.`，会导致重新加载时基座线性层未命中；v4仅修正该键名归一化，不改变训练数学、数据或A5000初始化映射。真实PEFT完整基座＋adapter加载、别名碰撞拒绝、冻结保护及formal-A共26项测试通过；进一步用实际step5做全模型逐位回载核对。该bug不适用于无LoRA的旧A5000，不能用来解释上一轮0/5。
- 当前新增队友yrm的四个GPU仿真进程，各占约4GiB；不停止、不修改。A2已测每卡batch4峰值约35.4GiB，下一次启动按每卡至少55GiB空闲检查并记录同机进程，保留资源余量。若不足则等待，不擅自挤掉队友任务。
- A2六帧低层服务、真实已消费动作计数、历史一致性和只读物理trace的集成仍在候选目录。反馈头仍UNKNOWN_ONLY；C2同实际状态的外部纠正专家、C1四类校准、完整任务成功率改善与主仓集成尚未完成。

### 历史：完整五回合结束、A2四卡启动（v16）

- 五回合均 `status=complete`、`task_success=false`、`termination_reason=full_official_max_steps`：动作数3224/7901/20682/20544/17770。只报告本固定pilot的0/5，不推断总体成功率，不混入先前0动作基础设施失败。五份已保存23D动作逐值等于官方实际step trace。完整个人复核见本地/WORK的 `A5000_BFINAL_FIVE_TASK_PERSONAL_REVIEW.md`。
- 最后task3耗时4471.325秒，48 NAVIGATE、10 OPEN_DOOR、103 GRASP，没有PLACE；task4耗时4237.889秒，3 NAVIGATE、136 OPEN_DOOR。本人查看task3全回合21张每1000动作关键帧、task4全回合23张每800动作关键帧，并核对全部技能切换。餐盘回合后段有地面餐盘/食物，装罐回合后段视野大量偏向天花板；缺少逐帧目标持有/门关节trace，不能把视觉接触当完成证明，不能将所有失败归咎于高层不切换。
- 五份视频、结果、动作、controller事件及官方step trace均在 `/home/wsy/behavior/memlite-results-20260910/A5000-Bfinal-{radio,trash,halloween,plates,can_meat}/`。最后两份视频本地/远端SHA相同：plates `4f270e9b7759026abc066771b34aef46492145c7d0955fd3bcda441e3cdf1de5`；can_meat `0da6a1e6b93bbab4f4b68a5116413cbf3e46baf4d5a90d66114b05e819fdeba1`。
- 所有模拟器退出、五份result完成且7个模型端口无活动连接后，以pidfd核验精确argv/端口/输出目录/进程启动身份，仅SIGTERM这7个已空闲的自有推理服务；`owned_pilot_services_release_v1.before.json`、`.signalled.json`保留。未删除权重、日志或视频，未触碰其他作业。
- A2真实正式loader预检 `a2_formal_real_preflight_v3/result.json` 完成：原train/eval 8,898,502/451,241帧、4个真实batch覆盖五任务、每条3相机×6历史帧，action为[4,32,27]。从真实A5000加载后945个保留参数逐位一致，固定80初始FM=0.2810064124。随后只用train样本执行3次真实DDP更新，loss为0.1119777113/0.0555782951/0.1539084613（不同batch，不当作收敛曲线）；冻结参数始终逐位不变，无checkpoint保存。峰值38,001,382,912字节；第二/三次纯更新约1.13/1.12秒，不含持续数据读取和四卡通信，不用来承诺正式吞吐。
- 该预检还验证正式梯度审计不把明确声明的LoRA A/B误当被冻结的VLM基座；只有精确LoRA组可豁免，基座意外解冻/变值负例仍被拒绝。真实PEFT、梯度审计、formal-A回归21项通过（先前独立接口/数据回归40项）。cross-attn-only只使用前缀K/V，最后层post-K/V的q/o/MLP共10个LoRA张量无梯度，连续3次DDP确认一致，正式开启find_unused_parameters；不把这些结构上不可达参数冒充有效更新。
- 正式 `WORK/formal_a2_lora_history_5000_v1` 已于14:15 UTC由自有launcher PID3597497启动，torchrun PID3597589、四卡、每卡batch4、全局batch16、5000步、每500存ckpt、每100固定80。训练源仍为独立 `WORK/a2_lora_history_candidate_v2`，source SHA `f7f1f895cf524d0e6e2ebbda08b36f405d8d4f2c10e3a5fd031bc0b7afecd979`；保持原950/50切分、原组合release、原train-only归一化、实际A5000初始化。启动前已有目录只是Hydra配置验证留下的3个配置文件与0字节train.log，经过精确白名单并记录原SHA后保留使用，不覆盖旧训练。当前仅确认run已启动，后续必须确认四rank有效更新和完整结束。
- 下一轮只读物理观察候选 `autonomous_physical_observer.py` 已完成6项真实语义parser＋物理oracle/假环境CPU测试；不伪造演示annotation ID，明确 `policy_input=false`、`training_admissible=false`。尚未集成真实runner、更未产生新的物理结果。A2六帧serving接入与训练/推理前缀一致性仍在做；C1四类反馈、同实际状态的C2纠正专家、完整任务成功率改善和主仓正式集成都未完成。

以下各节为历史状态，不覆盖本节。

### 历史：三个完整回合、全部C1审查与A2实际梯度（v15）

- task2 `putting_away_Halloween_decorations` 已执行20682个动作，`task_success=false`、`termination_reason=full_official_max_steps`，耗时4507.570秒。162次高层调用：112 NAVIGATE、24 OPEN_DRAWER、15 PLACE_IN、11 GRASP；低层每16动作推理中位566.44毫秒。全部保存动作逐值等于官方实际控制trace。三个完成任务只能报告本pilot的0/3；两个未完成任务不进入分母。
- 完整视频及动作/日志/分析已传本地 `memlite-results-20260910/A5000-Bfinal-halloween/`。本人已看每1000动作抽帧的21张关键帧：前段来回移动，手臂在地面容器和南瓜附近动作；后段到电视柜旁，长时间举臂和转向。约动作8576起高层切换放置/抓蜡烛等阶段，13184后保持NAVIGATE至结束。没有独立逐帧持有谓词，不能从图像接触宣布任何抓取成功，也不能把稀疏关键帧当完整逐帧复核。
- 资源交接只终止旧串行调度器PID3523520，原task2模拟器子进程3539926被保留并自然完成；pidfd身份/交接凭据为 `ab_parallel_handoff_v1.json` 和 `.confirmation.json`。task3 GPU0 / A8772 / B8775，task4 GPU2 / A8773 / B8776 已开始；两个B副本的完整load_receipt与旧8771完全一致。模型、冻结runner、种子、官方预算不变。旧串行sequence汇总不会生成，应根据各任务实际result汇总，保留调度交接记录。
- C1全部15窗980事件完成：691 train（2 SUCCEEDED）、289 eval（2 SUCCEEDED），余下976是有物理依据的IN_PROGRESS。四个新实现的抓取成功窗口为2/3/4/5，不能叫4/15完整任务成功率。没有FAILED/UNKNOWN类，独立成功实例仍不足，重复同窗帧不补足类/实例覆盖。三个候选均 `training_admissible=false`；最终B特征缓存295＋340＋345全部完成，无head训练更新。
- 个人审查共137条：window0 12条、window2 9条、windows1/3/6 26条、windows4/7/8/9/12 45条、windows5/10/11/13/14 45条。五份 `C1_*PERSONAL_REVIEW.md` 已在本地/远端签写；原渲染清单pending保留，未自动代签。最后一批window5事件15稳定持有5帧仍IN_PROGRESS，事件16稳定21帧才SUCCEEDED。所有结论来自本人查看实际图像和相应物理凭据，不是脚本批量通过137条。
- `C1_WINDOW12_REPLAY_ALIGNMENT_FINDING.md`：本人另外核对7组原始演示/回放同源帧。window12在7738–7818，演示柜门已开、瓶罐可见，真实动作回放柜门仍关；7818是A零动作接管点。因此该窗不能作正确接管初态下的低层服从成绩，也不能配原演示动作作C2教师；保留实际未持有标签与所有原始证据。window13/14柜门已开仍失败，不能将全部低层失败归因于这个setup偏差。官方DataPlaybackWrapper逐帧load_state而本次C1仅回放动作，二者不是等价状态恢复；进一步排查可用录制状态和物理配置，不热改正在运行的评测。
- `a5000_actual_action_interventions_v1/result.json` SHA `60cae5f31fc2188ac02f06f7674c345bae09fcc3f40a16f93b104efef7262bd3`：15个实际状态、69次原生FM调用，同条件同seed重复15/15逐位一致。更换真实同task技能条件带来的动作RMS相对换seed的RMS，各关节组中位比例约0.37%–3.49%；只换GRASP目标的9状态约0.31%–0.70%。这是条件响应弱的直接证据，不是物理服从/SR，也不是给错误条件配正确动作的训练授权；donor在该状态的可行性未验证，window12须另列setup限制。
- A2候选独立源 `a2_lora_history_candidate_v2` 保留已训A5000的动作专家，新增VLM LoRA并输入六帧三相机真实历史；FM梯度不再detach VLM cache，视觉/原VLM/本体感觉基座继续冻结。首次真实GPU检查发现PEFT FEATURE_EXTRACTION包装强塞HF `input_ids`，与MixtureQwen35的embedding/cache接口不兼容，在任何optimizer更新前失败；旧失败保留。改为通用PEFT wrapper，接口、checkpoint覆盖、负例及既有回归40项通过。
- 修正后的 `a2_history_gpu_preflight_v2.json` 是实际A5000权重、真实18图/1311 token的单样本GPU一步预检：连续FM loss0.099720478，AE322张量有非零梯度，LoRA91张量有非零梯度；实际AE322/LoRA182张量更新，冻结权重逐位不变，峰值分配21751148544字节。LoRA A初始步可因weight decay变化，不把182个更新冒充182个非零梯度。无checkpoint、无正式训练；下一步使用原950/50按实例切分的完整已审核组合数据，实际loader/多样batch/固定80/history容量检查通过后才启动正式A2。

以下带时间的小节均为历史状态，不能覆盖本节。

### 最新：两个完整回合、十个C1窗口与真实特征（12:08 UTC）

- 捡垃圾 public-test instance301/seed0 已执行7901个动作，`task_success=false`、`termination_reason=full_official_max_steps`，耗时1674.932秒。62次高层调用先22次NAVIGATE、后40次GRASP，目标始终为`trash_can_116`：动作0–2816导航，2816–7901抓取，没有进入放置阶段。低层每16动作推理中位587.95毫秒。稀疏底盘轨迹路程17.078米、净位移1.622米、累计绝对转角1516.56度，不能描述为全程原地不动。
- 本人已查看该视频每400动作的20张关键帧，后段右手附近可见蓝色罐状物，而高层仍指向垃圾桶；这是需验证的目标服从疑点，不是已被独立物理trace确认的持有对象结论。演示的另一个真实C1实例确实先把棕色垃圾桶带到罐旁，因此不能仅凭高层先抓垃圾桶就判策略错误。完整视频、动作、观察、控制日志和分析已在本地`memlite-results-20260910/A5000-Bfinal-trash/`；收音机对应`A5000-Bfinal-radio/`。两个完成回合均失败，只能报告本pilot的0/2，不能提前把其余任务计为失败或推断总体成功率。
- C1已完成indices 0/1/2/3/4/6/7/8/9/12；只有2（task0原eval）、3/4（task1 train）满足目标稳定持有且由A动作新实现。其余窗口耗尽1280个A动作仍是IN_PROGRESS，不把超时制造成FAILED。index5/11正在采集，10/13/14已排队。所有窗口保留原train/eval实例切分、真实演示前缀、相同A5000/seed17和独立物理凭据；不是从官方初态的整任务评估，更不是纠正专家C2。
- 本人新增查看window2的9条，以及windows1/3/6的26条三相机六时刻图像，逐条核对物理凭据，加原window0的12条，累计47个唯一实际C1事件。window2末两条右手稳定2/18个physics frame、window3末两条左手稳定2/18帧，前一条均保留IN_PROGRESS，后一条才为SUCCEEDED。三个个人记录为`C1_WINDOW0_PERSONAL_REVIEW.md`、`C1_WINDOW2_PERSONAL_REVIEW.md`、`C1_WINDOWS136_PERSONAL_REVIEW.md`；本批选中样本零未处理关键标签/时序问题，不等于全数据认证。所有渲染清单pending原样保留，没有脚本或子代理代签。
- `c1_actual_windows01236_candidate_v1.json` SHA `ef9a921e392145d780c44b069e3304a8ac8e79eb14dabf16455af49d72816306`：295条真实事件、0机械错误，269 train（1正例）/26 eval（1正例）。`c1_finalB_feature_interface_v1`完成3条真实GPU接口验证；`c1_finalB_features01236_v1/result.json`完成295/295实际最终B特征缓存，无优化器、无梯度更新，标签与模型输入严格分离，`training_admissible=false`。
- 真实特征使用七个可观察输入字段、实际六帧历史及最终B prefill。独立helper `native_c1_features.py` SHA `0b87f510a3cad86cb905e5b0381f3920c753a26808b0bf4d6856129bf7087d0e`修正了历史实现把含模态代码的mask求和当有效长度的错误：真实最后有效位置为1425/1889，旧算法会得2235/4091。新特征缓存不热改现有B服务；未来反馈推理必须采用同一可核验pooling/动作摘要协议。
- 正在补检新完成窗口并继续至少120条的个人审核。当前真实数据仍缺FAILED/UNKNOWN类别；正例只有少数实例、同窗事件强相关，不能凭数百事件声称四类反馈头已充分训练或校准。已训练B也仍是UNKNOWN_ONLY，不能未经对应训练就强塞成功/失败反馈。下一低层步骤是独立快照上的实际动作条件干预，以及LoRA/六帧路径的真实梯度与容量预检；不改当前冻结A/B、已启动评测或采集源，也不盲目再开5000步长训。

以下各小节为此前阶段记录，当前状态以上方最新小节为准。

### 最新：第一次完整物理评测与真实 C1（11:32 UTC）

- 原 `native_ab_final_pilot_v1/task_0` 在0动作退出，属于真实运行接口失败：主线程的 `asyncio.run` 包住同步 Kit 场景加载，触发事件循环重入并使模型连接心跳超时。全部失败结果/日志保留；它自然退出，随后试发的 SIGINT 发现进程已不存在，没有因此丢失动作。不能将其混作一次完成预算的模型评估，也不能在已启动尝试清单中隐藏。
- 独立 `native_ab_threaded_v3` 只将网络事件循环移到专用线程，Kit仍在主线程同步执行，模型权重、数据、种子、接口内容和动作预算不变。7项runner、2项线程/真实本机WebSocket心跳、3项原控制器回归通过。新 manifest `native_ab_threaded_five_task_eval_v3.json` SHA `6ab4eb729db7943e38c9b69a2a220879b0c7bb937a44e8daa5474df8c317858e`，新完整序列 `native_ab_final_pilot_v2` 已真实运行；旧冻结源和运行记录未热改。
- 收音机 public-test instance301/seed0 已正常执行3224动作，`status=complete`、`task_success=false`、`termination_reason=full_official_max_steps`，实际耗时745.631秒。26次高层调用中4次NAVIGATE、22次GRASP，未出现PRESS；动作数组逐值等于实际传给仿真器的23D控制记录。低层推理中位556毫秒/16动作，第一次冷启动约9.27秒。底盘按16动作采样的路程约2.816米、净位移1.784米、累计绝对转角537.26度；这是稀疏采样量，不等于全程原地旋转或目标接近证据。
- 本人已查看前段/中段三视角拼图与全视频每200动作的关键帧：早段确实走到目标桌前，后段有伸手、接触和继续转向的行为，最终未开机。本次 autonomous runner 没记录独立逐帧抓取谓词，不能仅凭“爪子贴着物体”判定抓住后不切换；还要区分低层没稳定抓住和高层完成判断失败。本地完整视频及动作/观察/结果为 `/home/wsy/behavior/memlite-results-20260910/A5000-Bfinal-radio/`，视频已在对话中显示；分析 `analysis.json`。
- 最终 B GPU自由生成 `b_parent_format_cuda_heldout_final_v1/result.json` 完成：真实最终权重SHA `d4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef`，正常配置BF16/CUDA，14/14正常解析、10/14完整技能严格一致、13/14父目标一致，约186秒。没有参考注入或优化器；不是14个任务成功率，也不是与CPU同精度数值一致性的证明。
- 真实 C1 index0（episode121/instance138）和index1（episode91/instance97）均完成448/467个真实演示前缀后1280个A动作，每窗82事件；`c1_actual_windows01_candidate_v1.json` 通过真实reader及processor，164条全为有物理依据的IN_PROGRESS。它们仍为 `training_admissible=false`，没有把超时、接触或演示结束改成成功/失败。index2（原heldout episode192/instance282）已完成394前缀＋384个A动作、26事件，完整物理结果正在汇总。
- 本人逐张打开首窗12张三视角六时刻图，并逐一读取对应独立物理凭据；图像/动作帧/held=false/稳定帧0相容，接触不能当抓住，原演示该段1122帧结束不能替代本次完成依据。人工记录为本地及远端 `C1_WINDOW0_PERSONAL_REVIEW.md`；原渲染清单 pending 保留，没有脚本代签。12条不等于120条；只覆盖一个train实例且只有IN_PROGRESS，不足以训练并校准四类反馈头。
- 图像记录采用真实原始分辨率：head720×720、双腕480×480，真实processor再按原配置变成256×256。两个审查拼图工具已显式生成显示缩略图并记录原尺寸，不再错误要求原始NPZ本身就是256。C1采集器、模型输入、相机/时间绑定与像素原始文件未改。
- 为利用本机显存余量，已增加独立A会话：8773/PID3532053、8774/PID3532115在GPU1，原A评测8770/PID3519880与C1用8772/PID3525221在GPU0；B8771/PID3519939在GPU1。每个C1使用自己的服务连接、policy seed17和Kit缓存。GPU0/1在新增Kit前分别仍有约55/42 GiB显存；现有Kit峰值约13 GiB，核实进程均属于本任务后，保留30 GiB左右余量启动共享卡仿真，没有驱逐其他作业。
- C1 index3（task1 train）在GPU0、index6（task2 train）在GPU1开始采集，原C1槽位在GPU2，自动五任务序列在GPU3。共享卡两窗先完成原封存campaign的validate，再用相同冻结collector的显式collect入口及相同参数运行；未修改物理判定/采集schema/源数据。这只是进程与资源安排变化，不是数据或模型放行。index3前缀4536步、index6前缀1972步，不擅自缩短。

以下各小节为此前阶段的记录；当前状态以上方最新小节为准。

### 最新：最终权重完成，转入真实 GPU 与仿真（10:51 UTC）

- `formal_b_parent_format_1500_v1/coordination_run_receipt.json` 已为 `complete / returncode=0`，四张训练卡全部释放。最终 `checkpoints/step_1500.pt` SHA-256 为 `d4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef`。`b_parent_phase_step1500_tensor_audit_v1.json` 实际读取完整父/子权重，950 个 state entries 全部有限，326 个独立训练张量均改变、623 个冻结项逐位相同；另一个 output projection 项与 input embedding 在父/子两份权重中均为同一 storage/view 的合法绑权重别名。不是用参数名或配置推断训练完成。
- 最终 fixed-80 新口径 weighted CE 为 `0.015796400`，task0–4 为 `0.005513448 / 0.006483420 / 0.007520075 / 0.055382853 / 0.004082204`。排除新增格式 token 的旧口径为 `0.016956503`；父目标格式 CE `0.006628338`。500/1000/1500 的新口径为 `0.015009386 / 0.015433353 / 0.015796400`，没有单调下降，不以低 CE 宣称阶段切换或任务成功已解决。
- 修正版 500 步的全部 14 个真实留出窗口已无参考注入生成完毕：格式合法 14/14、完整技能严格一致 10/14、父目标严格一致 12/14。旧 B3500 同样本同 CPU 路线为 14/14、6/14。逐例解释见本地 `memlite-resume.L6rqZ6/B_PARENT_FORMAT_GENERATION_REVIEW.md`；5/10/11/13 四例仍不匹配，其中样本 13 重复旧放置阶段。修正同时包含新的 500 次更新和 optimizer 重置，缺少“旧目标再训 500”的等量对照，不能把全部差异唯一归因于格式监督。不是最终权重结果或 SR。
- 推理严格使用独立 `b_parent_format_serving_source_v2`：实际训练源不改，只允许三个已核准输入/解析文件与其不同，其余 147 个 Python 文件逐字一致。前两次 CPU 复测暴露的缺少 inference builder、parser helper 已作为接口错误保留记录，不算模型语义错误；第三版正常完成全组。
- 真实 GPU 服务：GPU0 的 A5000/8770 为 PID 3519880，GPU1 的最终 B/8771 为 PID 3519939；均已报告 ready，独立载入凭据保存在 `ab_A5000_service_v1`、`ab_Bfinal_service_v1`。GPU2 的 `b_parent_format_cuda_heldout_final_v1` 使用最终真实 B、正常 BF16/configured backend，对原 14 个留出样本自由生成；该组尚未全部结束。
- 五任务评估 manifest 为 `native_ab_parent_format_five_task_eval_v2.json`，SHA `4c45b2f270efa741cb147f84b9b04c9409e5fe1ddb03a55c9cd7d2e9d8de981c`。固定 A5000/B3500＋1500、八个运行文件及官方配置、public-test instance301/seed0、policy seed17、零前缀、完整动作预算；反馈仍明确为 UNKNOWN_ONLY。`native_ab_final_pilot_v1` 在 GPU3 依次运行。每任务一例只能作描述性 pilot，不是精确总体成功率估计；模型/协议失败也必须在已启动尝试中如实报告。
- 仿真额外保存每 128 动作的真实三视角/本体观察，以及逐 chunk 机器人位置与朝向，全部仅用于事后诊断、不输入模型，便于区分真正旋转、受阻和错误阶段。它们在 manifest 冻结前已通过 7 项 runner CPU 回归；不是预先宣称物理运行正确。第一段实际视频结束后传到本地并亲自检查。
- C1 候选构造/人工复核拼图工具已准备，仍未生成真实 C1 标签或放行训练；不把此前 14 个 B 留出图像检查抵算 C1 至少 120 条个人审核。后续独立 A 会话与第二个 Kit 仅在 GPU 容量确认后并行采集。

以下单人执行条目保留各时间点的历史状态，以以上最新记录为准。

- A 完成证据：`formal_a_5000_v3_cache8_callbackfix_20260910T003600Z/coordination_run_receipt.json` 为 `complete / returncode=0`；最终 `checkpoints/step_5000.pt` SHA-256 为 `bffcb960c7169e3a6f41f96420878975a54d32399cd98bdd8c5e682ae602ea00`。step=5000、epoch=2，optimizer/scheduler/RNG/sampler 状态均已读回。
- 50 次固定诊断均为 80 个留出窗口、每任务 16 个，数值有限。最终五任务 FM loss 依次为 `0.150761 / 0.199297 / 0.284048 / 0.179708 / 0.262722`。
- 本次直接执行目录：远程 `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910`；本地 `/home/wsy/behavior/memlite-resume.L6rqZ6`。冻结的 A 训练 source 和权重不作修改。
- 配对诊断 `paired_step5000_v5/result.json` 已完成：正确条件平均 FM loss `0.215296807`，同任务非等价错误条件 `0.214517063`（错误条件反而低约 0.36%）；80 个窗口中正确条件更低 45 个。已逐样本保持 observation/action/mask、time/noise 相同。这不能证明意图服从；还要做实际生成动作干预与更多物理技能窗口。反事实技能来自真实同任务标注，但不保证在接收状态可行，因此不能单凭该 loss 判定模型完全不理解语言。
- 配对检查、容量测试及局部仿真均已退出；原低层服务 PID 3762572 已在视频和结果保存后正常终止，四张卡已释放给 B。当前正式 B 启动进程为 3882936、torchrun 为 3883019；run 目录 `direct_execution_L6rqZ6_20260910/formal_b_5000_v6_arrowfilter`，冻结 source 为同级 `b_formal_source_v5_arrowfilter`。4 rank × batch 4、BF16 autocast、5000 步、每 500 步 fixed-80/checkpoint；不使用容量测试的临时权重。最新已观察到 1464/5000，稳定约 2.6 秒/步，500/1000 步 checkpoint 已写出。四卡 step-2 梯度审计均通过：326 个高层张量获得有限非零梯度并更新，623 个冻结张量 bitwise 不变。训练中不占用额外 GPU 做推理。
- L1 使用官方 train instance 1、seed 0、真实消费的 265 步演示前缀，再由最终 A 权重执行最多 `80×16=1280` 步抓取动作。它明确是训练实例上的局部物理诊断，不能标成 heldout 或完整任务 SR。记录视频、实际动作、观察与独立物理状态；成功需连续 6 个不同 physics frame 满足条件，同帧重复查询不增加稳定计数。
- L1 实际结果 `native_l1_radio_v6/result.json`：演示前缀 265 步后，A 执行 78 个 chunk / 1248 个动作；初态及前缀后均未抓取，结束时右手持有 radio，连续 21 个不同 physics frame 为正（门槛 6），`newly_achieved_subgoal=true`。亲自检查的视频关键帧显示接近、伸手及较长时间的调整；没有持续原地打转，但没有证明抬起、打开收音机或完整任务完成。本地视频 `/home/wsy/behavior/memlite-results-20260910/radio-grasp-A5000/rollout.mp4`。
- B 容量检查 `b_capacity_v6/result.json`：实际 V10 完整载入，326 个 trainable tensor / 1,894,008,640 参数，batch 1/2/4 均有有限非零梯度和真实更新；冻结参数前后 SHA 完全一致。另一次训练集较长上下文 batch-4 更新的峰值 allocated 44,929,944,576 bytes。临时更新均丢弃，正式 B 仍从原 V10 父权重和新 optimizer 开始。
- 本轮修复：Qwen3.5 内置 vision merger 的完整载入检查；高层 raw exact-19 与格式化 AR 槽位相互覆盖；六帧视觉末层被丢回一帧以及随机 history drop；多样本文本左 padding 下的 context 位置；高层 eval 必须只访问真实 high anchors。每项均在独立 source 修复，A 和主仓未热改。真实 CPU fixed-eval 回归和 80 窗口 GPU forward 通过，冻结评估初始 weighted CE `7.201351732`。
- B 已另发仅限 planner 训练的只读数据 release：44,405 train / 2,304 eval，实例片段互斥；原 86 条人工审查中的 85 条通过、1 条排除及原始凭据均保留。sidecar SHA 为 `67a65b45…985e3`，release manifest SHA 为 `b23f1d6f…e915fd`。outcome 仍为 UNKNOWN + mask=false、反馈头冻结；不表示物理反馈数据获准。
- B 的 fixed-80 每 task 16 个，按真实 eval eligible pool 冻结，每 500 步重复同一集合，记录均匀 task 平均 weighted CE、每 task CE 和按 token 计权的 decision/parent/bundle/memory 字段 CE。它不是全量 eval、动作 loss 或成功率。日志仅本机离线保存，不向外部 tracker 上传。
- B 先前启动记录全部保留：v1 因发现继承配置关闭 BF16，在更新前主动终止；v2 因把单点 high anchors 按低层区间切给四卡而退出；v3 的旧 VQ 首批诊断仍要求已退役的 intent/status 字段，在更新前退出；v4 首轮为零 LR，新增冻结 outcome head 经 DDP 初始化同步后被同步前的审计基准误报。v5 仅把审计基准移到 DDP 同步之后，仍在 optimizer 之前，不豁免任何冻结参数。
- v5 正常更新至少 89 步后，为修复已实测的数据读取瓶颈主动终止；尚未生成 checkpoint，其临时更新全部丢弃。v6 仍从同一原始 V10 初始化重新训练 5000 步：仅把 sidecar 的 episode 筛选提前到 PyArrow 层，避免每次将整 row-group 的其他 episode 转成 Python 对象。全部 46,709 行、1,000 个 episode 的字段和顺序逐项相同；旧字符串 episode 格式亦通过回归。10 个真实 episode 的标签读取均值由 1.359 秒降至 0.0216 秒；实际训练稳定步耗约 2.7–3.5 秒，此前约 8.5 秒。该优化不改变标签、采样、损失或优化器。
- 四卡 high 采样器已通过完整 44,405-anchor CPU 回归：任务池先共同打乱再按 rank stride 分片，同 epoch 各卡实际取到的 anchor 集合互斥；resume 后缀精确复现，换 epoch 顺序变化。按任务均衡，不冒称独立技能均衡。旧 low 分片逻辑未改。
- 首批兼容修复已用五任务真实 processor 样本通过；双进程真实 Gloo/DDP 回归复现了初始化广播，并证明同步后冻结审计通过、故意篡改冻结权重仍被拦截。v4 失败不表示 outcome head 被训练过。
- B serving 的五任务前缀 token 等价检查通过；新增六帧 history/两阶段提案提交模块，并在真实已录制 frame 281/297/313/329/345/361 上通过 CPU processor 测试。该输入复放只是运行接口检查，不是新增演示或训练标签。高层只接相机/本体状态/已经下发的命令；记忆保持 K=3、verified_world_facts=[]，被拒提案不改记忆。
- 修复了 serving 解析器把合法并行 parent 中的竖线误当六字段分隔符的问题，覆盖 JSON 字符串中的竖线、转义引号、重排/缺字段。随后发现训练模板末尾 `|<HL_END>` 的单个竖线也须按协议处理，不能把布尔值误读为 `false|`。A 神经实现和已冻结 B 训练 source 不作热改。
- 新增独立原生 A+B 控制器、显式高/低双服务和 official-init runner。3 项 CPU 协议测试通过，覆盖真实本机 WebSocket、每 16 个已消费动作记录历史、每 128 个动作高层提案、低层接纳后才提交记忆及拒绝回滚。测试使用明确标识的模型替身，不是权重推理或仿真成功证据。尚待最终 B 权重自由生成和五任务官方初态、零演示前缀的 UNKNOWN_ONLY 自动闭环。
- fixed-80 B CE：初始 `7.201351732`、step 500 `0.020939882`、step 1000 `0.017639306`；1000 步逐 task 为 `0.004490911 / 0.009711375 / 0.013044239 / 0.047100078 / 0.013849928`。真实五任务单样本和混合 batch 的 token/causal-mask CPU 审计通过：CE 是 h[j−1]→label[j]，标签所在及未来位置被因果 mask 屏蔽；这是索引/可见性证明，不是完整模型数值不泄漏证明。
- **真实权重自由生成故障及修复：** `b_cpu_generation_step500_v2/result.json` 使用实际 500 步权重、真实 demo-observable 前缀、CPU FP32/eager/原仓 torch fallback。旧推理跳过 Previous outcome/Decision/Parent，反复输出技能和记忆，384-token 截断；并未执行这些拒绝提案。训练明确屏蔽 `Previous outcome: UNKNOWN` 和 `Task complete: false`，但旧推理要求 LM 自己发出它们，构成训练—推理协议不一致。
- 新独立 `b_serving_source_v2_format_constants` 只确定性输出上述两个无物理监督的常量字段及尾分隔/结束 token，仍让模型生成 Decision、Parent、完整技能和记忆，再严格解析及检查 K=3 recurrence；不拿示范标签修复语义。5 个纯协议测试和 5 个真实样本 token 逐项核验通过：11 个强制格式 token 与训练完全相同，采样器在异常后恢复。greedy 路径的 repetition/ngram 参数实际被原代码跳过，本次不把重复误归因于这些参数。
- 相同权重 SHA `9e61a0bfed936909678a2d7702a2cb1bf7aedef4b4308f7eb46e6996b317dd2f`、相同 sample 0/backend 复测 `b_cpu_generation_step500_v3/result.json`：168 tokens 正常结束，六字段和因果记忆合法，NAVIGATE radio 语义与该实际缓存样本一致。sample 1 复测同样在 177 tokens 通过，NAVIGATE trash_can 与样本一致。二者都是训练演示历史上的离线检查，不能称为任务成功或 GPU 推理验收；需继续覆盖边界切换、并行技能、K=3 和官方初态。
- C1 v2 独立候选已修复 6 帧 stride=16（旧版错误地连续取 6 帧）、官方 train mode 与逻辑 train/eval split 区分、完整自然语言 task 输入、scene 对象名到唯一 official scope 的 evaluator-only 绑定，以及连续 6 个不同 physics frame 的稳定门。zero-action receipt 与动作数组形状检查已修正，初态/前缀已满足不记新 policy 功劳；没有物理证据的失败仍为 UNKNOWN。
- C1 仿真采集新增 reset-after-instance-load、官方 terminal 的 chunk 中途停止、实际 consumed 与 served 分别记录、视频及 result 在 Kit 退出前保存、视频关闭异常不丢结果。8 项 CPU 替身回归通过（非真实物理评测）。采集连接改为单轨迹固定 A checkpoint/seed 的 native FM 会话，不在每个 chunk 重置 RNG；新增连接模块的 3 项真实本机 websocket + 假 A 响应回归通过，覆盖连续三块只初始化一次、错误初始权重和中途错误权重拒绝。C1 仿真使用 A 的 `native_source_eoc_v2`，不错误指向缺少低层安装协议的 B source；尚未收集或放行 C1 标签。
- C1 特征入口 `native_c1_features.py` 使用固定的真实七字段 observable 投影，不读取 outcome target。实测原始 16×23 浮点 JSON 导致 8583 tokens，新固定动作摘要为 1895 tokens；原来的 modality-code sum−1 池化索引为 4111、超出序列，现按最后非零 mask 的位置 1894。已录制真实六帧/early-padding CPU processor 检查通过，数值池化只用明确标识的假 prefill 测试，尚未做冻结 B 的真实神经特征提取或 C1 训练。
- 完整自动 runner 的另 5 项 CPU 替身检查通过：拒绝未完成训练/改变文件/缩短预算，官方成功与普通终止分开，中途只消费 3/16 步，失败先保存后退出。正式 manifest 尚未生成，因为 B 未完成；不以这些测试填报 success rate。
- C1 首轮 15 个真实源窗口已冻结于 `c1_windows_v1/selection_manifest.json`：每任务随机选 2 个 train、1 个原 5% held-out 实例，在该 episode 首个合格的单 GRASP 边界开始 A 诊断。原 episode→official instance 从真实元数据映射，不能用 episode+1 猜；950/50 episode 对应每 task 190/10 实例互斥。每个真实 23D 演示前缀逐帧完整，附原文件、sidecar、window/context 哈希。它们只是新采集的入口，不含新的 outcome 标签、不是 SR 或回灌数据；只有抓取覆盖，OPEN/CLOSE 的缺失部件映射及无证据 FAILED 类不能虚构填补。
- B step 1500 的 fixed-80 CE `0.016924639`，task-0003 升到 `0.060900262`，其余四 task 下降；应继续观察波动，不据单点判断过拟合。第三个训练历史 CPU 自由生成样本也在 213 tokens 通过；下一批转向真实 held-out 的初态、GRASP 切换、并行和长记忆，不把简单 NAV 复现当充分验收。

### 随后的真实权重根因复查（主代理单人执行）

- B 的五个简单训练历史样本最终均正常结束并与参考 NAV bundle 一致；这组简单检查现已被更难的留出诊断补充。最新已观察到 2609/5000，500/1000/1500/2000/2500 权重已保存，四卡继续运行，未热改其冻结 source。fixed-80 CE：step 2000 `0.016701076`、2500 `0.016660811`；2500 逐 task 为 `0.004833050 / 0.005356594 / 0.010750842 / 0.057249787 / 0.005113784`。
- A 实际动作敏感性：`a_cpu_action_sensitivity_v2/result.json` 使用最终 A5000，在录制的三个真实时刻 281/905/1465 固定相机、本体和同一 FM 噪声，分别测试真实同任务标注中的 GRASP/NAVIGATE/PRESS/PLACE_ON，并重复 GRASP 作确定性对照。重复条件全部 bitwise 相同。不同指令的机械臂输出差异很小；frame 905 的 PRESS/PLACE_ON 对底盘有较明显影响，其余两个时刻全维最大差约 `3.6e-4–9.8e-4`。这是 CPU FP32、单步实际生成检查；替代指令在接收状态的可行性未独立证明，不能从小差异直接推断完全不理解语言。
- A 只读链路审计 `a_cpu_conditioning_path_v1/result.json` 对上述全部 15 次生成逐一证明插桩前后动作 bitwise 相同。真实输入包含完整 parent/skill，未被截断或 padding 掩蔽；六个交叉注意力层均能读取这些 token。frame 281 的技能 Value 均值在换指令后相对 RMS 有约 3.7%–23.0% 变化，但实际动作影响弱，排除了“字段根本没进模型”的简单解释。注意力权重本身不是因果服从证据；后续以实际技能闭环判断。
- B1500 较难留出 cohort：`b_cpu_heldout_step1500_v1/result.json` 已完成 14 例，13 例通过结构和 K=3 记忆校验，5/14 完整技能严格一致，五例都是初态 NAV。另九例包括阶段不同、并行分支遗漏和一例结构重复/1024-token 截断。该权重 SHA `bc3d51ec63f9da88e5ad05d2de3ea43fd122dae535c91afcdb5a240da5c04c54`，不是最终 5000；不是自主记忆、GPU 验收或 SR。
- 主代理亲自看完这 14 个真实三视角六时刻拼图并记录逐例判断：[B1500_HELDOUT_PERSONAL_REVIEW.md](/home/wsy/behavior/memlite-resume.L6rqZ6/B1500_HELDOUT_PERSONAL_REVIEW.md)。收音机和放蜡烛等画面接近真实阶段边界，因此不把全部文本不一致硬算成物理失败；留出数据及其图像仍永不回灌。
- B 数值缓存检查 `b_cpu_numeric_cache_parity_v1/result.json`：样本 0 的 126 个 token、样本 6 的前 160 个 token，固定同一完整 teacher token 历史，整段前向和逐步缓存推理的 argmax 为 286/286 一致，位置编码全部相同，最大 logit 差约 `9.73e-5`、`6.48e-5`。排除了本次 CPU 路线的明显缓存错位，未证明 CUDA parity。该诊断 JSON 的 `rows.supervised` 仅是 processor 初始 label 存在标记，未应用 planner 字段 mask，不能当正式 CE 的最终监督范围。
- **第二个生成监督缺口：占位父目标。** 对实际 processor 及全 46,709 条 B 高层标签核查：31,906 条（约 68.3%）的父目标为确定的 `Task goal: ...`，但 parent semantic mask=false，整个父目标字段不受生成监督。逐 task 屏蔽数为 `0 / 3978 / 11183 / 8374 / 8371`。留出 cohort 的 6–13 八例正属于此类。teacher forcing 给了模型这个字段，推理却要求自行发出，构成又一处训练—推理缺口；不能将被屏蔽字段的低计权 CE 当生成能力。
- 正在做 `b_cpu_parent_intervention_v1` 因果诊断：只补参考父目标字段，其余技能和记忆仍由相同 B1500 权重生成。样本 6 从原先损坏/截断恢复为 259 tokens 正常结束，完整 GRASP soda bundle 与参考一致。该干预显式含参考信息，只供根因诊断，绝不部署、报正常分数或作为训练数据。准备的新训练修复必须把“已知 task-level 占位文本的格式生成监督”与“未知细粒度父目标的语义监督”分开，不能解除物理 outcome/terminal 的屏蔽，不能改正在运行的 B source；其余案例和回归尚待完成。
- C1 采集入口已在新 `c1_windows_v2_matched` 修正：保留原先五个 held-out 实例/起始帧/技能不变，重新从原 train 组为每个 task 选两个相同 canonical bundle 的实例，解决 v1 的训练/校准对象不匹配。原 950/50 episode 分组不变。15 个新窗口已逐个同原始低层 sidecar 核验条件、完整任务文本和真实 23D 前缀，全部通过；v1 记录保留。它们仍只是采集入口，不含 C1 outcome 标签，尚未进行 C1 训练或放行。

后续按实际结果继续：修复出现的运行问题 → 验证 L1 物理服从 → 独立 B 训练 → 收集/审核真实 C1 反馈、训练并校准 → 需要时收集真实 C2 纠正 → 官方初态五任务完整闭环评估与失败归因 → 主仓整合。不会用假标签、伪造成功率或重复审批文件代替这些结果。

主仓库为 `/mnt/sdc1/robodojo/GalaxeaVLA`。本文件与仓库内 `docs/memlite_coordination_execution.md` 内容同步。诊断依据保存在：

- `/home/wsy/behavior/memlite-coordination-audit.ygMSCU/诊断结论.md`
- `/home/wsy/behavior/memlite-coordination-audit.ygMSCU/协同训练策略.md`

### 2026-09-10 父目标监督修复与有界第二训练阶段

- 参考父目标干预已完整结束，结果 `b_cpu_parent_intervention_v1/result.json`（SHA `eaf22769cb1ac6a5e51baa92de0dca0547def38c3afa30cc369952fe2b740169`）。同一 B1500 权重、同一观察、同一 CPU 后端下，仅注入已有参考父目标及分隔符，八例均正常解析；6/7/8/9/12 五例恢复完整参考技能，10/11/13 仍有阶段差异。原先这八例自由生成均不匹配参考。这是含参考字段的因果诊断，不是 5/8 任务成功或训练修复效果。
- 独立修复源 `b_parent_format_train_source_v2`：只对已经存在、逐字匹配公开任务名称映射的 `Task goal: ...` 增加格式 CE；原细粒度 parent evidence mask=false 保持不变，详细不确定父目标仍屏蔽，UNKNOWN 和 task-complete 仍屏蔽，不生成新标签、不吸收 eval。全量 31,906 个占位值均匹配公开映射；14 个真实 processor 样本逐 token 回归通过，8 例只增加 8–11 个父目标 token，其余 6 例不变。
- 新增独立 `parent_format` 字段指标，以及排除新增格式 token、保留原 CE/语言/记忆权重的旧口径指标。真实 token 权重/掩码检查、默认 unit-weight 分支和 callback 合成汇总回归均通过。不是新的模型评估成绩。
- 第二训练阶段已实际启动：原 run 保存好 B3500 后，从该真实 checkpoint 完整载入全部张量（含冻结结果头），新 optimizer、1500 新步骤、4×batch4、LR 2e-5、100 步 warmup、相同原 train/eval 数据。run 为 `formal_b_parent_format_1500_v1`；最终文件应是 `step_1500.pt`，祖先链 B3500＋1500＝5000，不重命名冒充单 run 5000。每 500 步保存 checkpoint 和 fixed-80；这次没有正式 step-0 fixed-80 文件，不能把预检单样本称作正式初始 fixed-80。
- 新增显式 `memlite_v10_planner_descendant` 载入类型和权重来源门：保留原 V10 直接初始化的严格门；修正版必须提供实际 checkpoint 身份、stage/profile、完整 tensor 覆盖、CPU 真实前向及新旧 loss 口径等价凭据，禁止旧 optimizer/dataloader resume。来源验证负例和原 runtime 的 4 项配置/资产/参数组/loader 测试通过，原 toy 完整覆盖测试未列入这次 4 项。
- 后续仿真独立版本 `native_ab_parent_format_v2` 已适配真实分阶段权重身份，保持官方 public-test instance301、seed0、零演示前缀、原完整 task action budget。新增 ancestry/phase SHA 握手，未完成或旧权重不能混用。7 项 runner/身份/视频关闭异常回归和原 3 项控制器 CPU 回归通过；尚未运行该仿真，不产生 SR 结论。
- 原 B3000 fixed-80 CE 为 `0.016189480`；task0–4 为 `0.005996521 / 0.003910269 / 0.008599370 / 0.058709885 / 0.003731355`。此旧指标仍遗漏格式父目标，不能以其低数值声称已解决自由生成。

### 2026-09-10 09:47 UTC 修正版实训与评测准备进展

- 旧 B 的最终保留父权重 `formal_b_5000_v6_arrowfilter/checkpoints/step_3500.pt` SHA 为 `324df1153113342b6ab83fa27a34c91c8c67d7bbf2d2901128b35afa283ebd92`。在新修复实际预检通过后，主代理于 09:24:12 UTC 对已核实的旧 torchrun PID 3883019 发出 SIGTERM；最后日志为 3722/5000，3500 之后约 222 个日志更新未进入新模型。旧 run 的 failed/returncode=1 来自主动作业终止，并非发现数值崩溃；旧权重和全部日志保留。
- 新 launcher PID 2418204，torchrun 初始 PID 2418308，09:26:46 UTC 启动。冻结源 `b_parent_format_train_source_v2` 的源码 SHA 为 `fbd44156fe587eea2431e0b865b8e6c6e7268ce17a4bc895fac2eff6b1d0ffdb`，训练期间不修改该源。09:47 UTC 已观察到 350/1500，约 2.6 秒/步，四卡各约 68,500 MiB（66.9 GiB）显存且处于计算中；后续进度以实时日志为准。
- 实际配置预检 `b_parent_format_config_preflight_v1.json` 通过；实际 trainer argv SHA 为 `c6218bc5bb81ae5dd84dea3051c9c5b3d72d8a5e8d23f4ca06c7088de1d7b33c`。实际完整载入的 missing/unexpected/mismatched/partial 均为空，不能用配置预检代替此权重证明。
- `b_parent_format_phase_preflight_v3/lineage.json` SHA 为 `9a3ce2ed09cf76f8e798b80aad04963db844c89cb15f66344ae4e7883458f760`。同一真实 B3500 权重、同一留出样本 6，CPU 实际完整前向：旧口径 loss 为 `0.000002218050894953194`，新增 9 个父目标格式 token 的字段 CE 为 `7.163291931152344`，新总 loss 为 `0.5692702531814575`；排除新增格式 token 后，旧目标数值差恰为 0。原语义 mask=false、outcome/terminal 无监督保持不变。该检查没有优化器或训练更新，不能冒充 heldout 训练。预检 v1 因诊断脚本关闭训练参数标志、v2 因 CPU 不支持 CUDA-only CE 而退出；v3 只使用已有 CPU eager CE，正式 GPU 训练 CE 未改。
- 新 run 的四个 `coordination_grad_receipt_rank*.json` 均在 optimizer step 2 通过：326 个高层张量有有限非零梯度并实际更新，623 个冻结张量 bitwise 不变。低层 FM、视觉和 outcome head 继续冻结。
- **旧 B3500 无参考注入对照已完成**：`b_cpu_heldout_step3500_v1/result.json` 的原 14 例、相同 CPU 路线和观察全部正常结束，6/14 技能严格一致（初态 NAV 五例＋样本 6 的 GRASP）；B1500 为 13/14 格式合法、5/14 技能一致。父目标严格一致仍只有四例；样本 6 虽技能恢复，生成父目标仍错误，样本 7–13 的阶段/并行问题仍在。主代理已逐条读完这 14 份输出；图像取自前次本人已审阅的同一 14 例，不冒称再次新采样。此对照隔离“旧目标多训 2000 步”的部分影响，修正版是否额外改善尚待同组正常自由生成。
- C1 实际神经特征检查 `c1_real_features_cpu_v1/result.json` 通过：冻结的真实 B3500、已经录制的真实六帧三视角和本体/已执行动作，正常 history 与明确标注为人工构造的缺失历史压力测试均产生有限的 `[1,2048]` 特征，并与真实 prefill 最后非 padding hidden 逐值相同。带额外物理标签字段的输入被拒绝。这只是无更新的接口/数值检查，不是 C1 新数据或结果分类准确率。
- C1 的 `c1_actual_A_campaign_v1.json` 已冻结 15 个匹配窗口（10 train、5 原 heldout）及真实 A5000 来源，SHA 为 `9a4318e080e0e7997b15f8b395a83c7b5f24a45592a48b26bc14b131e39eeeb8`；第 0 个窗口已在实际 behavior 环境完成 validate。该文件明确不是数据训练 release，尚无新物理 outcome、尚未采集/审核 120 个 C1 样本；训练中不抢占 GPU 启动 Kit。

### 2026-09-10 10:21 UTC 修正版中途结果与独立推理完整性

- 修正版 fixed-80：step500 新口径总 CE `0.015009385979`、旧口径 `0.016279491632`、新增父目标格式字段 CE `0.002646262379`；step1000 分别为 `0.015433353009 / 0.016416606718 / 0.009138345189`。不能把新旧口径总数直接作改进幅度，也没有单调继续下降；最终效果仍由自由生成与物理任务评估决定。
- 真实生成检查暴露了独立推理源装配遗漏：直接使用冻结训练源时缺 `PlannerOutcomeBuilder.build_for_inference`；补输入接口后，实际生成 126 tokens 到了解析步骤，又暴露缺 `split_planner_event_fields`。前者在生成前退出，后者是程序依赖缺失，均不算模型语义失败。两个原结果目录分别保留为 `b_parent_format_cpu_heldout_step500_v1/v2`；v2 的已核实 CPU PID 2906878 在依赖故障确认后主动停止。四卡训练和其源码均未改。
- 当前独立推理源为 `b_parent_format_serving_source_v2`：从该修正版训练源复制，仅叠加此前实际 B-serving 已验证的两个输入 processor 文件和一个输出 parser 文件。`verify_high_serving_input_adapter` 检查全部 150 个 Python 源文件：这三个文件逐字匹配显式固定 SHA，其余 147 个（含全部神经网络、损失、tokenizer、normalizer）与实际训练源完全相同。运行中的 source 不热改，早期 v1 副本也保留。
- `b_parent_format_cpu_heldout_step500_v3` 是当前有效复测。与旧 B1500/B3500 使用相同 14 个原 heldout 窗口、CPU backend 和 demo-observable 历史，没有参考字段注入或训练更新。10:21 UTC 完成样本0–7：全部格式合法，0–4 的初态 NAV 与参考一致，5 仍生成 PRESS 而参考为 GRASP；6 的汽水罐 GRASP 与父目标均正确（旧 B3500 技能对但父目标错）；7 的南瓜 GRASP 恢复正确（旧 B3500 为 PLACE_ON）。后六例仍在运行，不给未完成分母或 SR。
- 原生仿真 runner 新增只读的真实三相机快照（每128个已执行动作）、每chunk前的机器人世界位姿记录；这些诊断不进入高/低模型请求，也不改变动作预算、策略切换或官方成功回调。7 个 runner CPU 回归重新通过，覆盖故障保存、三步中途停止、位姿不可用仍保存、诊断不混入模型输入。原3个控制器/WebSocket回归也已复测通过，新增源适配器身份负例通过。这些仍不是仿真或权重成绩。
- C1 后处理准备：`build_c1_real_candidate.py` 通过真实 reader 和真实 processor 审核完整物理采集事件，额外核对实际 A 身份、原实例/切分/前缀和动作计数；输出始终 `training_admissible=false`。身份正常/13种篡改负例通过，尚无真实 C1 数据运行。`render_c1_review_candidates.py` 准备分层抽取140个真实候选的三视角六时刻图，所有条目保持人工审核 pending；不会自动签署120条人工审核。
- 为训练后并行跑自动评测与 C1 采集，已复制约7.3 GiB只读历史缓存到独立 `kit_c1_gpu2_appdata_v1`，避免两套 Kit 共享可写数据目录。尚未开启第二套 simulator；须先确认实际 A 服务双副本的显存余量。原缓存未覆盖或删除。

## 2026-09-10 实质里程碑

- Formal A：冻结 source/config/reader/asset 的 run `formal_a_5000_v3_cache8_callbackfix_20260910T003600Z` 已真实运行。首轮是零 LR warmup；第 2 轮是首个经审计的正 LR 更新。step 500 的 fixed-80 返回后，已原子发布可续训 checkpoint `checkpoints/step_500.pt`（SHA-256 `b331f100…246e5`，`last.pt` 指向该文件）。CPU 映射读取确认它含模型、6 组 optimizer（322 state entries）、scheduler（last epoch 500）、4-rank RNG、sampler replay 及梯度审计证据；没有 SR、物理服从或完整 held-out 结论。
- fixed-80：相同 80 个 held-out 窗口（每 task 16，`full_heldout=false`）的 uniform-task FM loss 是 step 100/200/300/400/500 的 `0.356203/0.324865/0.300011/0.288323/0.298416`。step 500 的 per-task 值为 task0 `0.162936`、task1 `0.323140`、task2 `0.336231`、task3 `0.301770`、task4 `0.368006`。这些只是离线 FM 诊断；尤其不能把 loss 趋势改写成子目标服从、物理反馈或任务 SR。
- B：独立候选 `b_high_planner_only_source_candidate_20260909T031000Z` 已把 `high_planner_only`/6 obs、v2 46,709 条 high-only labels（44,405 train、2,304 eval）、B memory strict protocol、`.25` memory-update CE 和 runtime glue 组合到同一候选。Hydra 解析、真实 factory→Mixture→reader→PyAV→processor→PlannerOutcomeBuilder→official collate、reader unit、模型 profile/weighted-CE 与 runtime gate 均有 CPU 通过证据；独立 CPU 组合审查记录为 `d35262b8…`（原始组合 manifest `bb9a8619…` 保留）。它仍是 PENDING：高侧车 `training_admissible=false`，未做 V10 GPU graph-load、capacity 或 optimizer training。
- C1/C2：当前 B 原始 outcome 均为 `UNKNOWN+mask=false`，不会训练物理反馈。C1 需在真实 A train-instance 闭环中记录旧 bundle 的 pre/post 稳态真值、实际 consumed action，并严格分开 observable 输入与 privileged 物理标签；C1 只训练冻结主干上的 outcome head，受 task+canonical bundle scope 与 frozen split 约束。C2 仅可使用有真实物理证据和同状态纠正动作的样本；timeout/end 保持 UNKNOWN，禁止 policy/demo 伪教师。二者均未训练或部署。
- SR 入口：独立 native A+B autonomous candidate 正在实现显式双 checkpoint/source、native low/high 注入、observable-only 输入、official 五任务 public-test instance 301 / seed 0 manifest 与 trace/video 关联；它仍未运行 pilot、未产出 SR，也未改变 A/B 当前 gate。首版维持 `UNKNOWN_ONLY`，不等待 C1 校准，更不得重用 legacy AR、teacher prefix 或 client memory。
- 后续顺序：A 完成并冻结 checkpoint 后，先做 paired semantic-obedience 与 oracle-low 实际执行（短 48-action 只可作 smoke，之后才有足预算子目标测试）；B 再做已验证 loader/capacity、单独训练与诊断；随后才根据 C1 物理 outcome 和 C2 真纠正数据交替训练。完整 five-task SR 只在自动高层+自动 outcome+FM 的冻结闭环、官方 full-maxsteps 和明确 wall budget 下报告。人工观察到的 NAV→NAV+GRASP→GRASP 中间 bundle 仅约 6 帧，而当前 execute 16 action / 高层 128 frame 更新，故离线边界命中不能声称 runtime 切换及时；后续必须按实际 consumed action 与切换延迟单列验证。

### 尚未完成的完整验收项

1. **A 训练已完成**：最终 5,000-step checkpoint、固定诊断和恢复状态均已验证；不代表低层服从和任务 SR 已验收。
2. **低层服从尚未验收**：paired-80 与一次足预算 radio GRASP 已完成；需补实际动作对照及导航、开门、运输、放置等技能窗口，局部 grasp 不能覆盖其他技能。
3. **B**：真实 V10 graph-load、容量/loader 和固定诊断检查已完成；独立正式 5000-step 训练正在推进，之后仍须验证自由生成、切换和服从。EXECUTE-only、`UNKNOWN` outcome 数据不等于反馈学习或部署成功。
4. **C1/C2**：C1 仍需数据 reader、实际 only-head 训练/校准与 serving 接线；C2 只能收集真实物理失败后的同状态纠正，不能用 policy 或 demo 伪造 teacher。
5. **闭环与 SR**：经验证的 high、low 和 outcome 才能从 official init 做完整任务；以独立物理 predicate/官方终止回调报告五任务 SR、失败类型与切换延迟，而非 loss、文本或标注时长。
6. **主仓最终整合**：仅在上述独立证据齐备后，原子合并经过审查的 B/C/serving 字节；正在运行的 A source 绝不热改。

## 仓库约束与既有文档复核

- 已检查主仓库及其父目录的适用 `AGENTS.md`：没有发现适用于 `GalaxeaVLA` 的额外仓库指令。本计划仍遵守现有 dirty/untracked 工作区不得覆盖、不得 reset/checkout 的约束。
- 已复核 `docs/README.md`、`docs/data/schema.md`、`docs/deployment/serve_policy_mem.md`。其中 `shape_meta`/action-state 约束、观测历史时序和 `action_steps` 对齐要求在 E3/E4 中保持为回归门，不因 MEM-Lite 改造静默改变。
- 既有未跟踪 `TODO.md` 的唯一待办是 `joint_training=true` 的自动 loss balancing，已登记为 R2；它不是开启 joint training 的授权，也不替代 E3 的真实梯度审计。
- 本文写入时主仓库有 31 个已修改 tracked 文件和 96 个未跟踪源码/config/test 文件，均属于已有工作状态；本计划不会把它们当作已合并或已验证的实现。

## 0. 不可改变的训练/评估约束

1. 保留高低层分离和连续 FM 控制；协同是共享事件语义、互相匹配的数据和闭环状态，不是假称 FM 梯度穿过高层生成的离散文本。
2. 每个样本的 `parent_goal`、`active_skill`、目标/目的地/手臂、`previous_outcome`、`next_decision`、`memory_update` 和 `task_complete` 使用同一份版本化协议。`SUCCEEDED`（上一技能成功）与 `task_complete`（全任务结束）绝不可混用。
3. 低层 BC/FM 监督只能使用“从该状态出发实际执行该 `active_skill`”的专家动作。高层预测的错误技能不能和原专家的另一技能动作配对；此类记录只训练高层/结果判断，或者收集真正对应的专家分支动作。
4. 训练环境特权状态只可产生标签、做审核或诊断；不得作为部署模型输入，也不得通过对象 ID 到世界坐标的查询泄漏到测试时。
5. 原每任务 5% 留出集、五段 public_test 视频，以及任何其派生近重复样本永久隔离，只可评估/诊断，绝不回灌训练。新采集的同源纠正分支须以原始实例为组切分。
6. 没有真实纠正动作的失败记录不是正向低层动作标签；它可以训练结果头、失败识别或高层决策。
7. 任何完整任务评估必须报告实例、种子、最大时长、重置规则、成功判定、失败恢复次数、版本哈希。68.3 秒短程 rollout 不能叫正式成功率。
8. 不为“再跑一遍旧模型”启动大规模基线 campaign。已有 v11 与历史轨迹只作为锚点；只在固定、同预算、同初始状态的**改动版本**之间做必要配对比较。

## 1. 已知证据、其含义与限制

| 现有证据 | 可以据此做的决定 | 不能据此声称的结论 |
|---|---|---|
| 全量 sidecar：同帧可配对高低层 74,002/74,002 intent 相同；无未来目标帧 | 不优先修“高低层字符串/时间连接普遍错位” | 所有原始人工语义标注都正确 |
| 低层标签中 63.72% 为复合 `AND` 父目标；低层只拿单帧，未拿高层记忆/反馈 | 必须将可执行技能与父目标分开，并为低层提供已训练的条件/必要短历史 | 细粒度一定天然优于长子目标 |
| 冻结 v11 干预：普通子目标换条件时动作变化很小；恢复条件作用明显 | 下一轮先验收普通技能服从，而不是只看 recovery loss | FM 路线无效，或这些数字就是任务成功率 |
| v11 仅训练约 2.97M intent adapter；旧 FM/VLM、高层均冻结；训练路径有 `no_grad`/detach/优化器覆盖限制 | 真训低层 FM 专家须先做可达梯度审计，不能只翻一个开关 | 仅训练 5000 步已经比较过本计划 |
| 原语料没有 FAILED/REPLAN；恢复集只有 58 条、主要是脚本诱发旋转后几何纠正 | 需要真实策略失败及专家纠正、独立 outcome 监督 | 现有旋转恢复可覆盖抓空、掉落、开门/放置失败 |
| 高层冻结回放：标注历史下 13/20 intent 逐字匹配，陈旧初始历史下 2/20；切换后仍会滞后 | 高层、事实记忆和结果判断必须接受训练和陈旧记忆压力测试 | 高层“完全不会规划”，或这个小探针是正式成功率 |
| 定向测试 18 passed | 现有 sidecar/因果数据/adapter 合约当前可运行 | 新架构、数据或任务成功率已经通过验证 |
| schema-v6/48d6 候选曾通过独立 protocol/builder probe、data 34 项、model 16 项和局部解码；但人工抽查小样本证明 `parent_goal`/`target_parent_goal` 是含 `relation`、`memory_prefix`、`skill_idxes` 的原始 teacher repr，48d6 的 `_clean_text` 会将它原样投进 low condition / high target | 原始 relation/audit 必须和可部署的紧凑 parent 语义彻底分开；改正可作为带来源 hash 的 metadata enrichment，不必重扫视频/raw 9M | 48d6 不是最终无泄漏协议；任何 48d6 全量产物、CPU consumer 测试、人工单条发现都不构成 E1/E2/E3 放行 |
| 训练 loader 生命周期候选：checkpoint remap → base safe load → post-load LoRA receipt → stage gate 在 DDP/optimizer 之前的串联，旧冻结组在 source 中机械通过；runtime 作者正因净化 data protocol 更新 identity/receipt 回归 | 可保留加载顺序作为新协议下复查的对象 | Stage-A/Stage-B 已真实更新、optimizer 覆盖正确、GPU 训练稳定或任何成功率改善 |

当前 v11 低层权重 SHA-256：`e659db15463d8b809608ca29181f4761612c302e49de9d736ce20cd5c67fbb08`。它是诊断锚点，不是已证明有效的协同模型。

### 新数据与仿真评测的现实风险（2026-09-08 更新）

原始 demo 并没有提供可直接部署的、物理验证的子技能 outcome。任何无法由真实物理状态/观察依据确认的 `SUCCEEDED`、`FAILED` 或 `UNKNOWN`，必须保留为 `UNKNOWN`，并令该样本的 outcome 监督 `mask=false`；不得把标注区间结束、文本记忆、夹爪闭合或人工想象变成真假标签。

目前也没有可用的“当前策略失败状态 → 专家物理纠正动作”实现。symbolic teleport 禁用；仍在开发的 primitives 不是教师，不能拿来伪造失败后的低层 BC 轨迹。E6 只能在训练实例中获取可复放、真实 physics 可验证的失败与纠正，覆盖抓取、开门、运输和放置等问题，而不是把最终目标缩窄为旋转恢复。

完整五任务仿真必须使用官方 full-maxsteps，而不是遗留的 `3600s` wall budget：`task-0000=3224`、`task-0001=7901`、`task-0002=20682`、`task-0003=20544`、`task-0004=17770`。每次 E7 manifest 还须记录对应的 wall-clock 限制；长任务不能从短任务沿用同一个 wall budget。

### 正在发生的受限执行（不等于阶段放行）

- E1 全量 sidecar 曾按 48d6 构造；此前构造在 episode 600 左右被错误中止，唯一约 32 MiB 的**派生 staging**随之删除；原始数据和 v5 产物未受影响。经一次授权重启后，人工审核发现 parent raw-teacher 泄漏，故该运行及其后续产物均不能作为放行证据。raw 全量 builder 后来已经结束；clean enrichment 以来源 hash 做 metadata-only 处理，不重扫视频/raw 9M。全量 clean labels 已输出为 `ef2a64b08793bddbe46dc67dbb9302a4e25178cd4da1a219275cdb0627519368`，manifest 为 `7e7d9409d5a2bd8bb8836fc90f7ace54dbc8a9b559f56ab066728e67b454e5d1`；但按 task/skill/outcome 的全量统计、group split 审计和 E2 所需的主代理人工视觉审核仍未提供，故不能用于放行或训练。
- 全量 clean labels 的关系安全扫描曾命中 `previous_intent`/`active_skills` 的 `unbound_relation`。抽样包为 `data/review_receipts/memlite_v6_unbound_relation_semantic_safety_samples_20260909.json`（SHA-256 `779f328c6058a3b4d988641247d56d10882cc9078fde3ab07642595eac591664`）：候选仅在 task-0002 的 199 个 episode 与 task-0004 的 9 个 episode 出现。原 `1a458a85…` 的递归删 key 已替换为 strict whitelist；r2 metadata-only 输出 labels SHA-256 为 `666f8fc0b097ad963de7c3e484d4017c59e8d0dfc74eaaa6124d6c551ac31cc6`。最终 reader/projection `c340…` 与 r2 的生成协议 `6584…` 已明确区分：前者只增加 exact-19 projection/resolver 接口，不改变 r2 语义字节。完整兼容 receipt [`memlite_v6_clean_r2_full_mechanical_validation_20260909.json`](/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data/review_receipts/memlite_v6_clean_r2_full_mechanical_validation_20260909.json) SHA-256 `477b799233c42fdda380fdcfc76255b2572992904d25560126b26b9d20c1db8d`：9,171,004/9,171,004 immutable rows、strict projection、causal/mask/split 均 PASS（0 error），parallel rows 102,780、true pre-boundary gaps 2,651。它没有重扫 raw/video；根的视觉结论另以 `PASS_WITH_EXCLUSIONS` 绑定到 overlay/composite record，并不构成 physical proof、serving 或成功率结论。
- 48d6 的 source 组合曾得到 80 passed、1 expected contract fail；随后把净化 protocol 单独放入 source 时，旧 builder/recovery producer 仍产生旧 parent 文本，暴露出一次 torn-integration P1（6 项失败）。数据侧交付了兼容的 builder `2d429ade…` 和 recovery dataset `30c0e493…`；两者与 protocol `1a458a85…`、clean Base/sidecar/processor 在 source 原子组合后，data 39 + model 18 + runtime 35 项，合计 **92 passed**。这只证明该 TEST_FIXTURE_ONLY 组合的 CPU 机械接口；没有全量 clean 数据、训练或成功率结论。
- 后续 source 组合以外部 cwd、清空 `PYTHONPATH` 跑当前模型 CPU 合同，得到 **2 failed / 26 passed**，不是可忽略的环境差异：data processor `b832…` 在 builder 已做完语义投影之后，又把 raw `active_skills_json` 及成员/边界审计字段写入 nested `samples`；低层模型的 no-audit guard 因而拒绝。修复合同已冻结：raw bundle、member ID、边界只保留在 sidecar 及以 `(episode_index, frame_index, actual_branch)` 唯一查询的 `MemLiteAuditResolver`，不得经过 getitem、processor、collate、builder 或 tokenizer；训练 receipt 仅能经 resolver 查询审计，再同 batch 的语义投影和 mask 比对。实际 selected `behavior5` 还通过 `MixtureLerobotDataset` 构造顶层训练集；已核实当前配置是包含五任务的**单一** Base 子数据集，因此顶层只能在 `len(datasets)==1` 时转发该 Base 的 resolver/spec 和 source tuple 映射，任何多 source mixture 必须 fail-closed，不能只给 inner Base 提供 resolver/spec 或暗中泛化拼接身份。数据、模型、训练三方须以该单一 API 完成真实 low 与 Stage-B 回归后再原子合并；不能删除 guard、把 raw 换名、由 inner Base 临时绕过，或凭旧 92 项回归启动 GPU。
- 数据侧已交付**小样本**净化候选：protocol `1a458a85…`、1780-row clean fixture（labels `0c589484…`、manifest `6b00ff3b…`）与 enrichment `3e037f…`。全 1780 行都通过 `validate_v6_label`，三 parent/memory raw-marker 扫描为零命中（high 4、low 1776）；真实 low/high builder/prefix 验证了 audit 仅在 `annotated_parent_command_audit_json`，low condition 无 audit，high 的 old parent 在 EOC 前、current target 在 EOC 后，旧 raw-parent 反例被拒绝。它关闭的是该 fixture 上的 R15，不等于 full sidecar enrichment、人工审核或训练放行。
- 真实 selected Stage-A processor 的 clean low receipt 已确认 action `(32,27)`、三路 image 各 `(1,3,256,256)`、proprio `(1,27)`，并固定了 v11-compatible processor config `2e0614e6…` 与 train-only stats `846bcbea…`。虽 raw `base_qvel`/`trunk_qpos` dummy registrations 存在，实际 transform 会先替换为 `lower_body`，未调用 dummy normalizer；raw 23D→norm 27D→官方 23D 的实际 round-trip 误差为 `2.61e-8`。但 approved stats 的 `lower_body[3]` 方差为零，离支撑的合成非零输入会坍缩；这可能是有意的常量 DOF，尚不能自动判成训练 bug。data 必须比较五任务 raw pre-converter 与 converter 后的真实变异，sim 必须只读审实际 23D wire/joint-controller 语义；它是 E7 physical/data gate，不阻挡当前 Stage-A structural fresh-5。任何正式训练 receipt 仍须固定 config/stats/source hash。Stage-B 仍需用 18 个唯一 camera/time ID 对齐 train/serve，不能把 `num_obs_steps=1` 误读为所有模态都只有一帧。
- model 的 clean CPU 合同已为 19 passed，source 的 92 项组合回归包含 runtime 当前 handoff（runtime `e365491…`、旧 launcher `bad341d…`）。该组合测试的 `source_tree_sha256` 是 `cb0acd47890d3cfd2834d4ff84e21f32bffea9d6bfa77bcff415b922849a97fe`；它只供复现本次组合测试，不能写入未来 launch receipt。之后仅替换了 metadata-only enrichment 为 `7fc93096…`：data 39 项已重跑，clean fixture 与 66,028-row raw subset 的 labels 与参考输出逐字节一致；当前时点的 source 先后变更，不能把旧 92 项机械回归移植成“新快照全绿”。
- launcher 的相对 `oc.load` P1 已在 source 修复为 `a7cdcac0…`：只接受 candidate `configs/task` 内 YAML，identity 覆盖整棵 config tree，子进程 `PYTHONPATH` 只保留 candidate `src:scripts:root`，并在导入前验证 `g05`/配置资源属于同一树。source runtime focused suite 已重跑 **36 passed**。随后在 source hash `f23278a8b62456de0a6239251acab64c05abe17179c47c97a296f4fda7a01c72` 上完成真实无 GPU Stage-A launch preflight，receipt 为 `/mnt/sdc1/robodojo/behavior_dev/memlite_coordination_preflight_20260908/source_stagea_cpu_preflight.json`（SHA `a12fd230…`）：clean fixture、train-only stats、task config、exact `(0, low)→low` 路线、`(32,27)` action、`(1,27)` proprio、三路 `(1,3,256,256)` image 皆已记录，CUDA 前后均为 false、未建模型/optimizer、未保存 checkpoint。该 receipt 仍缺实际 `g05.__file__`、解释器、严格 `PYTHONPATH`、`parts_meta` 路径/hash 和 preflight 自身 hash，故 R16 仅部分关闭；这些字段与模型 YAML literal 修复会进入**下一** source snapshot 后重跑。GPU Stage-A 结构检查尚未启动，且只可在全局锁、源码/配置哈希冻结、官方只读资产路径和独立 receipt 下执行；它不是长训授权。
- 仿真抓取 P1 已独立复核：OG `IsGraspingState` 的 `TRUE/UNKNOWN/FALSE=1/0/-1` 已显式映射。新的 physical replay runner `9c35d1f8…` 将 source action 前的 1156–1161 observation 明确标作执行证据、非标签；outcome label 只使用真实 physics action 后同步得到的 1159–1164 history，anchor 是第三次 right-close hold 后的 1164，因此不混入未来 evidence。receipt 持久化 raw enum type/name/value 与 canonical tri-state，UNKNOWN 不变成 release；context 内先原子写 physical evidence，再允许 simulator `__exit__`。独立的时序/enum/receipt 测试 3 passed（候选总套件报告 35 passed）。
- 已完成一次**真实** calibration-only physics replay：before-exit physical receipt [`physical_evidence_receipt.json`](/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/artifacts/calibration_only/task0_radio_grasp_calibration_only_20260908T155548Z_3649569_186f0152/physical_evidence_receipt.json) 的 SHA-256 为 `68fb784e3df6adbe2113f1d8bd5f467e70469ce4d8ad581ab0f2fbaf0c1ce839`。它记录 pre 时左右手均为 raw `FALSE=-1`，三次真实 right-close hold（1162–1164）后右手稳定 raw `TRUE=+1`、左手 raw `FALSE=-1`；共消费 1165 个有限 23-D action，anchor/history 仍为 action 1164/1159–1164，物理 transition 为 true、outcome 为 `SUCCEEDED`。这是单一训练实例的物理校准证据，不是 policy SR；字段仍严格为 `reinject=false`、`teacher_eligible=false`、`calibration_only_not_teacher`，不得回灌为低层标签，也不能放行高层 outcome provider、多技能 feedback 或 E6 纠正闭环。
- E2 的不可委托人工视觉审核：主代理已实际看完 **140 条**新样本，结论为 `PASS_WITH_EXCLUSIONS`，不是全库物理或成功率放行。task-0004、episode 821 的 `jar_235` 原始 `HANDOVER` 方向在 `[7997,8267)` 有 270 帧重叠；原 r2 字节不变的 boundary/quarantine overlay 已完成独立代码审、真实 Hydra tuple、全量 9,171,004 行重算和根审阅证据，固定为 271 low + 2 high 隔离、312 metadata shortened horizon（其中 62 个实际 32-step mask 变化）。正式 A composite 仍必须把 overlay 三个 artifact、reader 代码及 root 决定绑进不可变 release record；仅 sampler digest 不够。v3 candidate 已显式接线 overlay，实际 Hydra/Mixture/Base receipt 为 `2f217578c5abb078602c53780ced5c99d638f8a7a89935e5541fafd8df0e56c1`：required 模式正确拒绝 PENDING，非训练模式真实读出 `ep821/f7996/low` 的 1/32 mask、7997 effective horizon、exact-19 projection 与零 raw leak。它仍是 PENDING；只能新建不可变 `RELEASED` record，不能原地改候选。task-0000/0001/0003 中末段各至少 18 条补样、task-0004 近末段至少 2 条补样，以及 r2 审阅迁移/覆盖 receipt 仍按数据 release 追踪。所有已审条目保留原始来源和投影未变的冻结 receipt，不能为了凑数重抽/重签；发现关键错误仍须修生成逻辑并扩大同类审查。
- **独立 B 线（不改 A 或 r2）**：诊断证明 first-clean 将每行 `memory` 与 `memory_update` 都写成同一 `f(previous_parent_goal)`，多步历史退化，父目标改变时 online 历史错一事件。后续只在独立 B source 建 causal overlay：保留现有 19 字段，`K=3`，输入 `memory_i=U_{i-1}`，目标 `U_i=append_K(memory_i, previous_intent_i)`；只存 issued/unverified 的完整 canonical semantic bundle/对象语义，不存 ID、audit、帧、边界或伪物理成功。model 负责人单独实现 `high_planner_only`（planner CE；outcome head bitwise frozen、readiness=false、v10 high init）；sim 负责人单独实现 UNKNOWN_ONLY 的限频 replan/历史保留/官方 terminal 权威/observable allowlist。B 需自身三事件链、K eviction、parallel bundle、parent fallback、UNKNOWN replan、serve prefix 一致性和独立复审，之后才由集成负责人在独立 B source 串行合并；它不得阻塞 A 的单步、5-step smoke 或人工 r2 放行后的 A-main 5000。
- **Stage-A R16、单 GPU 更新与 bounded smoke 已完成（仍非正式训练）。** R16 receipt 为 [`receipt.json`](/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/integration_receipts/stagea_source_cpu_e2e_r16locator_20260909T025400/receipt.json)，SHA-256 `64046be06870e2cea87065a4159df18d6f21f8fe7eb1b5124a4affe4fb2246e8`，CUDA 前后为 false。随后 PID `3807430` 对 task-0000/episode-0 TEST_FIXTURE_ONLY 完成一次固定 seed/noise 的真实 FM forward/backward/AdamW update，receipt 为 [`receipt.json`](/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/integration_receipts/stagea_gpu_one_step_f233_20260909T030300/receipt.json)，SHA-256 `50c7a3598057d7aecc1121ce11f854a7d499c634e67e335ea2238e2a09a9472a`：continuous FM-only loss `0.7394396067`、无 CE 项；仅 `action_expert` 的 322 个参数有有限非零梯度且更新，623 个冻结参数 bitwise 不变。zero-LR audit/checkpoint 补丁后，source identity 为 `74ce206e236bb5daa2ea37f04fd758830c21417d37c5f264234dad426f64b99e`；其 canonical config-only 与唯一 normal-async fresh-5 位于 [`memlite_stagea_smoke_only_ep0_zero_lr_audit_20260909T0830Z`](/mnt/sdc1/robodojo/behavior_dev/memlite_stagea_smoke_only_ep0_zero_lr_audit_20260909T0830Z)。该 run exit 0、执行 5 次 optimizer iteration：首轮 6 个 effective LR 均为 0，仅 defer；第 2 步以 `1e-6` 首次产生 action-expert 已验证写入，322 个梯度有限非零、623 冻结参数 bitwise 不变。step-5 checkpoint SHA-256 为 `975dc6aeba58b78745fb7d950b37e48e780c36bdc4dfec7449d9ac03d87860e6`，gradient receipt SHA-256 为 `d04e18e1987cd304aae224515d7f54649d3d9839636f06fdc7fe68b6a24f8cc9`；同身份 resume exit 0，未出现第 6 次 update。checkpoint 记录 model/optimizer（6 groups、322 state entries）/scheduler（last_epoch=5）/sampler replay/audit identity，但**没有 RNG state**；因此这个 bounded resume 只验证 no-sixth，不可夸大为可继续训练的精确随机恢复。RNG capture/restore 是考虑 A-main 5000 前的独立 resume P1。诊断时的 `CUDA_LAUNCH_BLOCKING`/分布式 debug 只用于先前 trace，不能用其耗时估计正式吞吐；本次 smoke 为普通异步环境。它仍不覆盖 E1/E2、全量 r2、人工审核或成功率门。

### 当前三条链路的边界

- **A：低层结构验证。** 仅用不受 R15/task-0004 冲突影响的 task-0000 episode-0 fixture，已完成 R16、一次真 FM 更新、canonical config-only、5 次 bounded smoke iteration 与 no-sixth resume；普通异步 smoke 的实际步耗不应与先前 blocking diagnostic 混用。下一 formal-A gate 是 reviewed composite release（R2+overlay+reader identity）和 resume 的 RNG capture/restore；两者通过后才可能考虑 A-main 5000。
- **数据 release：** r2 已完成全量机械核验，但仍受人工语义、补样、task-0004 双 branch 整 bundle 隔离和迁移覆盖 receipt 约束；不得把 120 条初审或机械 PASS 误报为全量数据放行。
- **B：高层记忆/serving。** `high_planner_only`、causal K=3 memory overlay 与 UNKNOWN_ONLY serving 是独立 B source 的后续工作。overlay 还必须构造“中途观测后继续同一完整 bundle”的幂等自环样本：结果是 `UNKNOWN/未验证`，不得伪造完成；否则周期性 UNKNOWN_ONLY 调用只见过切换意图的数据分布。它须验证同 bundle refresh 不重复事件语义、K=3 history 去重，并在少量可人工审查 fixture 与真实 prefix 上通过后才可串行合并。绝不改当前 A frozen source 或阻塞 A。

### E7 launcher 现状

模型只读确认当前唯一完整 official launcher `eval_memlite_v9.sh` 硬绑定 v5 AR、旧 sidecar 和 5000-step 假设，不能作为新 A/FM 的成功率 runner。`load_separated` 与新 A 训练 config/类接口还需最小 **eval-only** server/launcher 适配和 CPU 合同；它可以与训练并行，但当前未获启动仿真授权。固定 oracle bundle 只可用于从真实 demo 前缀经 `env.step` 到子目标初态的低层 probe，并以物理 predicate 评 subgoal success；完整自动 SR 必须由真实 high+low 从 official init 开始，不能固定首 bundle 跑整任务，也不能按标注时长假切换。

## 2. 统一事件协议（所有实现阶段的硬依赖）

训练和运行中每个决策点都记录以下紧凑的结构化字段或确定性文本模板：

| 字段 | 语义 | 可接受值/例子 |
|---|---|---|
| `parent_goal` | 较长的任务目的 | `place plate in refrigerator` |
| `active_skill` | 此刻唯一可执行的技能 | `NAVIGATE` / `OPEN` / `GRASP` / `TRANSPORT` / `PLACE` / 显式并行分支 |
| `target`, `destination`, `arm` | 目标指代、目的地和执行手 | plate 93 / shelf 4 / right |
| `previous_outcome` | 已发生的上一技能观察结果 | `IN_PROGRESS` / `SUCCEEDED` / `FAILED` / `UNKNOWN` |
| `next_decision` | 下一调度行为 | `EXECUTE` / `RETRY` / `REPLAN` / `STOP` |
| `memory_update` | 可从过去观察确认的事实 | 右手持盘子；右门已打开 |
| `task_complete` | 仅表示全任务完成 | boolean |

结果标签必须有可检查的观察依据。抓取要求物体被正确手稳定持有；放置要求释放后处于目标区域并稳定；开门要求门状态达到要求。证据不足标 `UNKNOWN`，不把区间结尾、夹爪闭合或时间超时伪造为成功/失败。

## 3. 阶段、负责人、依赖与验收证据

负责人按**角色**列出；由集成负责人在开始前指派具体人。除“主仓库串行集成负责人”外，其他工作可在隔离工作树并行；未经审查不得合并到主仓库。

| 阶段 ID | 目标与工作范围 | 负责人 | 前置依赖 | 必须留下的验收证据 | 进入下一阶段的门槛 |
|---|---|---|---|---|---|
| E0：基线冻结与可恢复开发环境 | 记录当前 dirty 源状态、权重/数据指纹、评估预算；建立隔离工作树与恢复说明。只做文档/环境，不训模型。 | 主仓库串行集成负责人 | 无 | source manifest、工作树路径、`git diff`/untracked 哈希、容量检查；无训练进程证明 | 可从固定源状态复建隔离工作树，且主仓库未被覆盖 |
| E1：事件协议与候选数据 | 将 `skill_annotation`/`primitive_annotation` 转为版本化事件单元：过去观测、旧技能/记忆、outcome 的物理证据或 `UNKNOWN+mask=false`、下一技能/记忆、同状态连续动作和有效 mask。离线标签对每个并行组以原 annotation 时间覆盖导出的独立 expected-members 核验完整性；运行时 parser 只需验证当前 bundle 合法、唯一、语义稳定。按实例分组切分。 | 数据与标注负责人 | E0 | schema/版本号、生成日志、总量与按 task/skill/outcome/边界的计数、来源索引、UNKNOWN/mask 计数、每个并行组 expected/emitted 成员与差异、train/eval/public_test 隔离检查 | 无未来泄漏、无组泄漏；每个低层样本可追溯到同状态同技能动作；不存在无物理依据的 outcome 监督；离线并行标签不以自身 emitted list 证明完整性 |
| E2：数据语义审核与采样 | 对成功/失败/UNKNOWN、原子/复合/双臂技能及技能边界做分层审查；补齐 sampler，使 task×skill×boundary×outcome 均衡，恢复中的 brake/align/approach/settle 也单独均衡。 | 数据与标注负责人；**主代理亲自执行的人工审核负责人** | E1 | 至少 120 个分层样本的图像/短片和标签审阅记录、逐项通过/拒绝理由、修正记录、混淆矩阵、采样权重和实际 batch 组成 | 零未处理关键错误；UNKNOWN 和真失败均有明确路由；全量机械因果/mask/split 检查通过 |
| E3：低层 FM 可训练路径 | 从匹配的完整 G0.5/FM 初始化，令 `active_skill` 与对象条件进入上下文化图文条件，再到 FM。实现 FM 专家/条件模块训练路径；先冻结视觉/VLM，后续预留低层 LoRA/后段和短历史。保持动作坐标、归一化、32 预测/16 执行的基准约束。 | 低层控制负责人；主仓库串行集成负责人 | E1、E2 的小样本可用 | 参数冻结表、optimizer 参数清单、每组梯度范数、一步前后参数 SHA/差异、无梯度路径单测、动作编解码/边界 mask/服务时序回归测试 | 预期训练参数确实更新，冻结参数不更新；条件不再只在末端词嵌入残差中出现 |
| E4：低层技能服从训练与隔离评测 | 先训 FM 专家与条件模块；只有在 E4-A 不足时才打开 LoRA/短历史。评估“正确技能+学习低层”的导航、开门、抓取、运输、放置短技能，并做同状态、不同普通技能的行为级对照。 | 低层控制负责人；评测负责人 | E3 | run receipt（配置/数据/权重/代码/种子哈希）、每个 loss 的有效 token/维度、对象正确率、技能成功率、边界失败率、对照视频/轨迹 | 普通技能条件产生可观察且正确的对象/动作差异；不以 5000 步、单一 loss 或 recovery 下降放行 |
| E5：高层与结果判断 | 以 E1 的真实 outcome、下一技能、记忆更新训练高层语义分支和轻量结果头；结果头仅看当前及过去观测/动作。保留旋转守卫为专项兜底，不能代替 outcome。 | 高层与结果判断负责人 | E1、E2；E4 固定候选版本 | 成败/UNKNOWN 分类及校准、错误切换/提前成功率、陈旧 memory 与 previous intent 分离消融、结果触发日志 | 能区分进行中与真正成功/失败；对陈旧上下文不锁死；不使用未来帧或 oracle 作为部署输入 |
| E6：真实失败—专家纠正闭环与交替协同 | 在**训练实例**运行固定版本，收集当前策略实际失败；只能由可复放的真实 physics 专家/人工接管从失败状态纠正，产出失败证据、纠正动作和后果。symbolic teleport 和 WIP primitives 禁止作为教师。将高层、结果头与低层分开优化，轮流冻结固定版本采样。按需开放 LoRA/历史。 | 闭环数据负责人；低层负责人；高层负责人 | E4、E5 | 失败类别分布、每条纠正的前后状态和物理验证、专家动作、组切分证明、每轮固定版本/数据谱系、错误配对屏蔽统计 | 新数据包含真实抓取/开门/放置等失败，而不只是旋转；没有把错误高层技能与不匹配专家动作用于低层 BC |
| E7：完整自动评测、选择与再迭代 | 冻结明确版本，在独立 5% eval 和 public_test 上进行足时完整闭环：自动高层 + 自动 outcome + FM。官方 full-maxsteps 固定为 task-0000..0004 的 3224/7901/20682/20544/17770；按任务声明合理的 wall-clock budget，禁止沿用统一 3600s。先将 E4 的 oracle-skill 与 E5 的诊断-oracle-outcome 数字作为定位指标，严格同部署指标分开报告。根据失败归因回到 E1–E6。 | 评测负责人；主仓库串行集成负责人 | E6 | manifest、成功判定原始日志、每 task/seed 成功率与置信区间、q_score、完整 maxsteps/wall budget、时长、错误目标、提前成功、恢复率、视频和失败分类 | 在预先声明的预算中，完整自动成功率相对指定锚点有可复查提升；否则按失败证据迭代，不发布“成功”声明 |

### E2 的不可委托人工视觉验收

新数据完成构造后，主代理必须亲自查看图像/短片及其对应标签，不能由子代理、脚本或汇总数字代签。最少审阅 **120 个分层样本**，并覆盖全部五个任务；每个任务的 train/eval 类别；原子技能、复合技能、双臂技能与技能边界；以及 `SUCCEEDED`、`FAILED`、`UNKNOWN` 三种结果。它不是要求每个交叉组合都人为凑足样本：任何缺失或不足的类别都必须如实报告为“无数据/数量不足”，随后补采或调整采集方案，绝不能虚构标签来填格子。

每一条审阅记录至少保留：数据版本和 split、任务/episode/原始来源索引、对象 ID、`t` 前后时间戳、审阅的图像或短片路径、父目标/当前技能/outcome/记忆标签、审阅结论和理由。抽样须同时包含随机项与高风险边界项；抽样报告要能重新定位源数据，而不是只保留拼图截图。

如果发现某一类关键错误（例如因果方向、对象指代、结果依据或动作区间错误），处理方式不是只改这条被抽中的记录：先修正生成逻辑，扩大到同类分层重新审核，并重新跑全量机械因果、mask 与 group-split 检查。E2 的放行条件是**零未处理关键错误**和上述全量机械检查通过；这仍不等于宣称已对整个语料库做完视觉认证。

## 4. 训练与集成规则

### 4.1 低层的渐进开放顺序

1. 完整、相互匹配的旧 G0.5/FM 权重初始化；不把来自不同训练的 VLM 与 FM 专家直接拼接。
2. 先训 FM 动作专家和技能条件模块，冻结视觉与 VLM 主干。
3. 只有 E4 证明普通技能服从仍不足时，开放低层 VLM 的小规模 LoRA/后段；短历史通道也必须训练后才可依赖。
4. 每一次开放都重新执行梯度、optimizer 覆盖、一步参数差异、显存/吞吐和旧行为保真审计。

初始学习率仅作为探索起点：FM 专家约 `1e-5`、低层 VLM LoRA 约 `5e-6`、新模块约 `1e-4`、高层可训练参数约 `1e-5`。它们不是已验证最优值，亦不可复用 adapter-only 的显存/吞吐估计。

### 4.1-A Stage-A 与 Stage-B 的独立门

Stage-A 是单帧、三相机、低层 FM action-expert 的独立训练门。它需要自己的 clean 数据/E2 人工审核、candidate-root CPU 路线、真实一 GPU step 的梯度/更新/冻结审计和 train-only sampling receipt；不能因为 Stage-B 的六帧/18-image padding 或高层结果头尚未完成而无限期阻塞。

Stage-B（LoRA/短历史）和高层/结果头则必须在其自身启动前完成六帧的真实、非 padding 18 camera/time ID 训练—推理 trace，以及 observable-only outcome provider 的端到端验证。两类门不可互相借用：Stage-A 的三图通过不证明 Stage-B 时序正确；Stage-B 未通过也不能被拿来宣称 Stage-A 或完整闭环已经失败。

高层先按单独的 `high_planner_only` profile/run 训练 planner VLM：outcome head 必须冻结，参数覆盖为零、`outcome_validated=false`，且其 run receipt 单独列出。只有收集到真实 physics outcome 样本后，才允许另开 `high_planner_outcome` 训练结果头及其校准。这个未来 profile 不得混入 Stage-A 的参数组、loader receipt、结构预检或训练结论，也不以尚未完成的 outcome 训练阻塞 Stage-A。

### 4.2 高层、结果头与低层如何协同

- 高层学习下一技能、决策和事实记忆；结果头学习 `IN_PROGRESS/SUCCEEDED/FAILED/UNKNOWN`；FM 低层把当前**正确**技能变为动作。三个损失分别归一化、分别记录，不比较 CE 与 FM 的绝对数值。
- 高层离散文本/结构化技能由其自身监督优化；不声称连续 FM loss 会穿过 argmax/生成 token 更新高层。
- 只有预测技能与专家技能被验证语义等价时，才允许替换低层条件。其余行对高层纠正监督，低层 BC mask 为零；若需要另一技能动作，必须从同一状态实际采集分支。
- 结果头比长文本规划更频繁检查已发生的动作与观察；只有新的尝试与观察证据出现时，才更新重试/失败计数。

### 4.3 评估不能混淆的三层指标

| 指标层 | 输入 | 回答的问题 | 不是 |
|---|---|---|---|
| L1 | 正确技能 + 学习低层 | 控制器是否服从该技能、做对对象 | 完整自主任务 SR |
| L2 | 学习高层 + 诊断用真实 outcome | 规划/记忆是否是主要瓶颈 | 可部署系统 SR |
| L3 | 自动高层 + 自动 outcome + FM | 部署闭环是否真的完成任务 | 仅训练 loss 或短程视频 |

L3 是最终选择指标；L1/L2 只用于定位。所有版本都应同时给 L1、L2、L3，以免把某一模块的 oracle 优势误传为端到端提升。

### 4.4 最终证据只回答三件事

1. **A：低层条件是否真的改变控制？** 每 100 步的诊断必须固定官方 FM sampler 的 episode/frame tuple、动作 mask、batch shape、代码版本与 seed；用同一观测把正确技能条件和一个非等价技能条件作配对，记录每 task 的 FM loss。FMHelper 必须记录两种条件实际使用的 flow time/noise 并证明二者相同。这个反事实只回答“条件是否影响低层 FM loss”，**不是**任务成功率。
2. **B：高层是否有因果且稳定的记忆？** 只接受 K=3 的过去事件链、同 bundle 周期刷新不重复占位、真实边界切换和 UNKNOWN 保持/不宣称成功的证据。没有物理 outcome 标签时，`high_planner_only` 仍是 `UNKNOWN`、outcome mask=false；它不是自动完成或 SR。
3. **C：闭环是否真正完成任务？** 只由 simulator 的独立物理 predicate 和官方终止回调裁定。模型自报文本、served action 数、标注时长、固定首 bundle 或 loss 都不能代替成功。固定 demo-prefix probe 只报告子目标物理 probe；从 official init 的自动 high+low 才能报告任务 SR。

### 4.5 Formal-A 5000 的唯一执行计划

在 immutable `RELEASED` composite record、最终 source/config-only 和 RNG/fixed-diagnostic 回调都通过之前，不存在可执行的 5000-step 命令。通过后，唯一允许的入口是现有 `coordination_launcher.py`：由该**同一份**通过的 config-resource receipt 所保存的 `trainer_command` 程序化生成，并只由 launcher 写入新的 run/output 与 sampling-index 路径；禁止手写 dotted Hydra override 或第二份人工 argv。

该计划固定为 Stage `A/low/low_ae`、schema v6、`num_obs_steps=1`、三路 256×256 图像、连续 FM、4 个 DDP rank 与 max_steps=5000。最终快照放行后先复用已验证的 `preflight_memlite_skillfm_gpu.py` 做**一次** batch=8 容量预检：同最终模型/processor/tuple shape/assets，在临时进程真实 forward/backward 后允许一次 optimizer step，以计入延迟创建的 optimizer state；不保存训练 checkpoint、不改父权重、不计为 5000 进度。只有 OOM 时才可清理本次自身进程并以 batch=4 再试一次；80 GB 卡的目标峰值不高于 60 GB，之后四卡只采用通过的 8 或 4。容量 receipt 必须如实列出它与正式 loader 的差异；正式启动早期仍采样每卡显存、step time 与锁/PID，越过预写阈值即停止。checkpoint、fixed diagnostic 输出和最终 exhaustive heldout 结果都写入该唯一 run_dir。旧 one-step 的 21.616 GB 只作历史参考，不能被冒充为 fresh-5 或正式 run 的峰值。

## 5. 当前待解决项与明确依赖

| 编号 | 待解决项 | 首先由谁处理 | 阻塞的阶段 | 解决证据 |
|---|---|---|---|---|
| R1 | 事件协议如何从细粒度标注和物理状态构造，尤其是抓取/放置/开门的 outcome 依据 | 数据与标注负责人 | E1–E7 | 可版本化 schema、实例级来源和人工审核 |
| R2 | `joint_training=true` 的自动 loss balancing（现有 `TODO.md`）以及真正可达的训练梯度 | 低层控制负责人 | E3–E6 | 各 loss 缩放、梯度归因、参数更新审计 |
| R3 | `prefill()` 的 `no_grad`、KV detach 和 adapter-only optimizer 如何替换为可控训练通路 | 低层控制负责人 | E3–E6 | trainable/frozen 参数验证与回归测试 |
| R4 | 如何从观察而非部署 oracle 判断技能 outcome，并校准 UNKNOWN | 高层与结果判断负责人 | E5–E7 | 严格过去信息评测、分类/校准及误报日志 |
| R5 | 当前策略在训练实例上的真实失败和专家接管采集方式 | 闭环数据负责人 | E6 | 每条纠正可复放、可审计、组切分 |
| R6 | 评测环境的成功 API、足时时限、种子和视频/日志收集是否统一 | 评测负责人 | E4、E7 | 预先版本化 manifest 与可复放评测 receipt |
| R7 | 主仓库现有 31 个已修改、96 个未跟踪源文件的串行集成及冲突处理 | 主仓库串行集成负责人 | 全部 | 隔离工作树审查、逐次 cherry-pick/patch 审查；不覆盖队友改动 |
| R8 | 原 demo 缺少物理可验证的技能 outcome，UNKNOWN 如何保持为无监督而非伪标签 | 数据与标注负责人 | E1–E5 | `UNKNOWN`/`mask=false` 的全量统计、因果审计与人工审阅记录 |
| R9 | 真实 physics 失败后的专家纠正采集器尚不存在；teleport/WIP primitives 不可作教师 | 闭环数据负责人 | E6 | 可复放的物理失败—纠正轨迹、状态证据与专家动作 |
| R10 | 五任务官方 maxsteps 与任务级 wall-clock 预算绑定 | 仿真/serving 与评测负责人 | E7 | 每 task 的 maxsteps、wall budget、完成/超时原始日志 |
| R11 | Stage-A/B 的 loader 生命周期尚未把 checkpoint key remap、LoRA injection/restore、参数 stage gate 串到 DDP/optimizer 之前；Stage-B resume 还须拒绝缺失 adapter | 训练/评测入口负责人；低层控制负责人 | E3–E4 | 明确调用次序单测、真实参数组/optimizer receipt、resume 缺 adapter 失败测试 |
| R12 | 最终 v6 protocol 已同步到 data/model 候选；runtime/serving/collate 仍须以同一 hash 同步并证明 `task_name` 进入高层 prefix，`target_parent_goal`/outcome/audit 不进入输入，且 audit expected-member/source ID 不进 token | runtime/serving 负责人；数据/模型负责人复核 | E1、E3、E5–E7 | 每个 consumer 的 hash lock、白名单 projection 与 tokenizer/collate 反例测试 |
| R13 | 物理纠正与 outcome audit 的因果门：KNOWN evidence 必须严格早于标签帧；evidence-backed UNKNOWN 合法但不监督；物理纠正必须先证实未满足、实际消费至少一个有限 23-D physics action、再证实因果成功；OmniGibson `IsGraspingState` 必须显式三态映射（TRUE/UNKNOWN/FALSE = 1/0/-1），绝不使用 Python 真值；不可将部分 `IN_PROGRESS` 支持误扩展为“永久排除可恢复 FAILED” | 仿真/serving 与闭环数据负责人 | E1、E5–E7 | 对 shared protocol 的反例一致性、真实/同构三值 enum 下的抓取与放置 release 测试、pre/post state/action trace、无 teacher 的失败记录不回灌低层 BC、真实 simulator 复放证据 |
| R14 | 结果头和六帧视觉的部署接线：serve 入口必须传入实际 observable-only context provider，递归拒绝 oracle/audit；用 18 个唯一 camera/time ID 走真实 `encode_train` 与 `encode_inference`，证明训练和 serving 都得到 `head_t0..t5,left_t0..t5,right_t0..t5` 的同一最终 placeholder/pixel 映射 | 模型负责人；仿真/serving 负责人；独立复核 | E3、E5、E7 | 非 mock 的 high-policy prefix/hidden/head E2E 测试、旧 bundle outcome 触发日志、18-ID 严格顺序回归；禁止用“18 个图”或各自内部单测替代 |
| R15 | `parent_goal`、`target_parent_goal`、`previous_parent_goal` 及 `unbound_relation` 的语义投影：raw primitive relation、`skill_idx`、source/audit ID、memory prefix、v5 `intent/status` 不得以字符串 repr 进入任何 high input/target 或 low condition；raw teacher 只能作 audit。当前 `unbound_relation` 的递归删 key 不是白名单，须修正后重扫 full-clean label metadata。 | 数据/标注负责人；模型/训练 consumer 复核；主代理人工审核 | E1–E5 | 新协议/fixture hash、raw-marker 拒绝测试、关系字段白名单/反例、三个投影的精确 prompt 反例、真实 selected processor receipt、版本化 metadata enrichment 来源/目的 hash；随后重跑全部 data/runtime consumer |
| R16 | coordination launcher 的子进程工作目录与 Hydra 相对资源路径：不得因把 `cwd` 设为 source root 而使 `oc.load` 找不到官方 `parts_meta`；解释器、实际 `g05.__file__`、严格子进程 `PYTHONPATH`、source/config/preflight 自身 hash、stats/parts-meta 绝对解析结果均须进入 receipt | 训练/评测入口负责人；独立复核 | E3–E7 | 无环境继承的 launch-isolation 回归、真实 selected config 的 resolve 结果、失败时 fail-closed、可独立核验模块/路径/hash 的 run receipt | 先前 receipt 已证实真实 CPU 路线但缺可独立核验的模块/环境字段；未补齐并重跑前禁止经该 launcher 启动任何长训、恢复或正式评测 |

## 6. 隔离开发树与源状态恢复

当前主仓库是脏工作区，不能直接让多位实现者并发写入。E0 使用一个且仅一个预留根目录：

`/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908`

恢复设计：从记录的 `HEAD` 创建 detached `git worktree`，应用当前已修改 tracked 文件的二进制 patch，然后恢复未跟踪的源码/config/test 文件；每一步都以 source manifest 的 SHA-256 复核。`.venv`、`venv`、`checkpoints`、`datasets`、`outputs` 和任何模型权重均不复制、不链接、不删除。这样独立源码工作树预计约 9.9 MiB，而不是把主目录的约 11 GiB 环境/产物复制一遍。

该根目录的 `SOURCE_MANIFEST.md` 是具体源版本、补丁哈希、未跟踪树哈希、容量与重建命令的权威记录。新实现者只在该隔离树/其后继隔离树中工作；主仓库只由串行集成负责人合并经过测试和审查的变更。任何人都不得 reset、checkout、覆盖或提交主工作区的既有 dirty/untracked 内容。

## 7. 本计划的完成定义

计划不会因为文档写完、一个 checkpoint 到 5000 steps 或一组测试变绿而完成。只有同时满足以下条件才可宣布本轮路线达成：

1. E1–E6 的数据谱系、梯度审计、纠正数据和模型训练证据全部通过对应门槛；
2. E7 在独立、隔离的完整自动闭环评测中显示可复查的任务成功率提升，并报告失败模式；
3. 评测未污染 train/5% eval/public_test 边界，且主仓库集成可回滚/可重建；
4. 若某阶段未达标，保留失败证据，按表中依赖回到最早的可解释瓶颈继续迭代，而不是用更长训练替代诊断。
