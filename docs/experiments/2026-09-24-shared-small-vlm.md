# H45：训练期间的小VLM共享显存可行性

## 问题与边界

最新active goal允许在现有训练运行时尝试小VLM。当前四张A100各有约7489MiB可用显存，训练为xhz的3294346–3294349，不能停止或改配置。历史2B小输入峰值约4570MiB，不能直接代表当前6–9图/长上下文harness，更不能代表模拟器也放得下。

唯一主要假设：冻结Qwen3.5-2B能在保留训练显存余量的条件下，处理当前harness的真实公开语义请求。负责人Codex-parent。本实验只筛查资源与输出，不测新成功率、不训练，不取代官方完整回合。

## 冻结条件

- 配置：`configs/semantic_robot/h45_shared_small_vlm_probe.json`；代码：`scripts/semantic_robot/probe_shared_vlm.py`，必须干净Git独立worktree；实际commit记入结果。
- 权重：Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，完整10文件SHA与根目录清单固定。可选`generation_config.json`当前不存在，新增它也会拒绝，不能只记录版本字符串。
- 输入：原H38 task0 TRAIN-development138、seed0保存态，4个请求为规划、d0观察、d91观察、d91动作；30图顺序/RGB SHA逐一匹配原记录。只传原actor公开字段，不读取特权诊断标签来修动作。
- 所有原视角和文本保留，图像明确从最大640降到320，不声称这是与27B的模型单变量对照。
- 固定greedy/BF16/SDPA、Transformers5.7.0、原结构解码commit817f944；规划/观察/动作最大输出1024/320/64，原输入上限12000token。
- GPU2精确UUID/训练PID；PyTorch allocator≤4864MiB、另计512MiB非Torch开销、至少保留2048MiB。allocator限制不是整个进程的硬显存隔离，2s采样不能保证瞬时零风险，也不承诺共享计算无性能影响。
- 只运行一次，4调用；模型阶段600s、独立监管最多900s（含hash/CPU准备），无重试/新reset/训练。未知进程、低余量、OOM或到时，只结束新建worker。独占launch回执避免SSH断连后重复提交。

## 当前结果

2026-09-24 17:37北京时间：19 CPU测试过；4个真实保存态请求/30图CPU恢复及原图hash核验过。独立审查中，未调用GPU、未生成新输出、未启动模拟器。原H44两工程门已通过，不为新模型复跑那些门。

17:41：完整模型manifest及资源TOCTOU问题经独审指出后修复；首次CUDA和CPU模型加载后分配前两次全预算核验，已占自有context扣账，进程总显存上限5376MiB；采用UUID可见性。21 CPU与修后独立复审通过，无剩余实质发现。仍0 GPU调用，下一固定commit/远端单次探针。

## 后续判断

实际记录输入token、延迟、allocator/进程显存、语法和人工语义检查。格式合法不等于空间判断正确；保存态输出正确不等于闭环成功。只有探针资源/语义可用，且另有模拟器资源依据，才登记一个有限无专家/旧策略前缀完整回合。训练资源不足时不偷降保护余量或把局部抓取算成官方成功。

## H45实际失败与H45b修复

H45由99caf682在17:46提交，supervisor3438643/worker3438651。worker18.823s、监管27.538s后exit1，0次生成：在停止符一致性门拒绝。采样自有峰值4810MiB、最低空闲2674MiB，退出后7489MiB；四训练保持原PID。仅权重装载通过余量门，推理显存/语义仍未知。

完整本地证据`artifacts/agentic-vlm-goal-20260918/h45_shared_bundle_v1`。result SHA `02775d9c61a5df9ac92fd38cac2431e71ea5b4a3f74a53bd275093a0525f89b5`，supervisor SHA `88325174c6cecf5aa75838b8712fd8e1428f4e03e02d06bfb9352217436475d1`。

实际CPU读取显示，缺独立generation配置时默认只用text_config的248044（文本结束），而tokenizer的对话结束是248046。H45b显式保留二者作为生成停止符，且三项准确身份都要匹配；不改本地模型文件或删改生成文字。原v1严格检查不放松，新配置单独opt-in，原输出保留。26 CPU已过，独立窄审待；后续单次复验预算/输入与H45相同，没有新模型搜索/训练/物理。

17:53独立窄审通过：LMFE结束符在有效集合内，有限动作trie只在完整候选后放行两个EOS，无截断修补；26 CPU过，下一冻结源码后单次复验。原v1及现有训练不动。

## H45b真实结果（完整结果纠正尾日志误读）

380b9de于17:54提交，supervisor3440248/worker3440256，实际worker81.745s/监管91.162s，3条完成、第4条OOM。17:57聊天/计划曾仅凭tail误判为0完成，现明确更正，不隐去已经生成的弱回答。

| 请求 | 输入/输出token | 生成时间 | 结果 |
| --- | ---: | ---: | --- |
| 初始规划 | 1175 / 150 | 54.436s | 语法合法，但把关闭夹爪写成对radio的close子目标，并重复navigate |
| 初态观察 | 2813 / 62 | 2.299s | 目标不可见，与RAW一致 |
| d91近场观察 | 3275 / 96 | 3.509s | 错报不可见；本人在原head RAW可清楚看到红色radio |
| d91动作 | 5908 / 未生成 | — | FLA/Triton L2norm autotune申请256MiB触发自有allocator上限 |

前三条均通过语法不等于有用。54s包含首次内核/调优等冷启动成本，不能与后两条不同请求直接当模型纯速度对比。峰值allocator记录4591.701MiB、采样进程5278MiB、最低空闲2206MiB；失败后恢复7489MiB，四训练保留。OOM来源是原生FLA Triton调优，不可假定关闭Torch Dynamo就能消除它。

本地全包`artifacts/agentic-vlm-goal-20260918/h45b_shared_bundle_v1`；result SHA `44088517b0296f90abc9c46e14d8e2e63bee35de6b59d75a9fde3145baa69a1a`，supervisor SHA `2a6f58f01cd8b0e5fa874417096f75384753d15397a33e912eabb11df32e6058`。没有新模拟器回合/成功率，本轮不直接放行该2B配方；先评估现有4B量化是否提供更合理的容量/资源平衡。

## H46：同资源的4B NF4候选

已有冻结4B revision851bf6e及完整11文件SHA，现有bitsandbytes0.49.2/accelerate1.8.1，不安装升级共享环境。加载时NF4+double-quant，BF16计算，vision与lm_head不量化；加载后核真实Linear4bit/NF4状态、vision排除和唯一可见GPU，不使用自动跨卡映射或CPU/disk offload。实现依据[Transformers官方bitsandbytes接口](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)及本机5.7.0 quantizer代码实际支持的skip-module路径，不据此承诺无精度损失。

登记一次H46：与H45b相同4请求/30图/320、seed17、greedy、600s/外层900s、4864+512MiB且保留2048，0新训练/物理。只比较可部署候选，容量与量化同时变了，不作单因素归因。31 CPU过、独审待，实际4B模型尚未加载。量化路径在`from_pretrained`内就分配GPU，完整预算检查也明确前移到它之前；原BF16路径不改。

18:12：31 CPU及修后窄独审通过；增加并验证真实compute/vision/tied output BF16 dtype、nested double-quant，错误dtype三分支均拒绝，不只凭config作标签。4B的EOS两ID已CPU确认一致于H45b策略。H45b本地与远端完整result/supervisor SHA也已核同。

H46实际824f659/3442782/3442790于18:12:55提交，worker17.415s/0生成，在构造回执读取可选`hf_device_map`时AttributeError；前面的真实量化/dtype/placement检查已走过，但没有完整生成结果，不能算候选通过。修复只容许该元数据缺失，仍逐参数要求cuda:0并记录实际集合；31 CPU/独审通过，同配方H46b新run单次复验，不覆盖旧证据或放宽保护。

## H46b完整结果与H47单因素诊断

18:26核实H46b eacdc0f已经exit0：worker129.742s/监管139.878s，四条格式通过，采样进程峰值5318MiB、最低free2166MiB，退出后7489MiB，四原训练保留。实际248层NF4、视觉与输出BF16、所有参数cuda:0。4B部署通过这四条资源检查，但不能保证任意后续输入或模拟器共存。

规划1175/128token、59.105s，比2B的重复navigate/误用close合理；初态观察2813/107、22.402s，目标不可见结论合理但场景描述不完整；d91观察3275/96、7.555s，仍错误声称目标不可见，几乎逐字复用了系统提示的负例；d91动作5908/19、17.143s生成right/forward/coarse/base，与原动作一致。四条是保存态独立调用，动作仍使用原保存的公开上下文，**不能当作本轮observe到act闭环**。

本地全包`artifacts/agentic-vlm-goal-20260918/h46b_shared_bundle_v1`；result SHA`f9465e8d1ba87be4bea8bdcd330ac6c6b35044336b434eb684b4608581391728`、supervisor SHA`bedd4fed55d12845be4208194137152c55a0f3b56cbedcbf72e81094570e8626`双端核同。没有新物理/官方SR。

H47假设：完整负例JSON诱导小模型复制答案。唯一负责人Codex-parent，拟用同4B版本/量化/320/seed17/greedy，d0和d91各作原提示与中性字段说明配对，共4调用，600s/外层900s和原显存保护不变。仅替换包含固定答案值的那一行及对应引导语，其他原始输入/全部图像/输出解析保留，不删除低置信度约束、不注入对象坐标或特权标签。原提示负例保留；新代码CPU/独审后固定commit，结果逐条人工看。不声称两个开发状态能证明泛化，更不把格式/定位通过当完整成功。

18:32：36 CPU、真实两对30图保真检查和独立窄审通过，无剩余阻塞。原始和有效公开请求分别可定位，原profile行为不变。整段负例替换包含新的中性引导语，因果结论只能针对这整个干预，不能进一步区分删负值与引导语的贡献。

## H47结果：不采纳新提示

cb8ced5、3446566/3446574，四调用完成/exit0，worker106.461s、监管116.300s，峰值5038MiB、最低free2446MiB。原提示d0/d91分别58.323/7.737s，逐字复现H46b；新提示7.805/7.957s仍都不可见，且hazard:null令两条schema失败。d91新回答将场景说成只有天花板/地板，仍漏看head的明显radio。没有证据支持仅换负例块能解决问题，保留负结果，不上线。

全包`artifacts/agentic-vlm-goal-20260918/h47_shared_bundle_v1`，result SHA`c98d02ac09e05fc197bd262d91a75854c31f8cb26ec84f25a3bc7cedf7f7a145`、supervisor SHA`31da22053d20ca7fb5f7b37112ec5dbb8f379566d0c645ed58e820158e07f8c9`。CPU processor实核6/9幅对应600/900image token，pixel张量有限且有方差，回执`artifacts/agentic-vlm-goal-20260918/h47_processor_cpu_check.json`；排除模板阶段图像完全丢失，但不证明理解正确。

## H48登记：最小当前图像定位能力

唯一负责人Codex-parent，主假设是接口/多图负担而不必然是基础视觉完全无能。冻结原4B NF4、320、seed17/greedy；两个原公开子目标分别当前head一RAW图和三相机RAW配对，4调用、600s/900s、原显存限额，0reset/训练。只传原目标文字和当前RAW，无历史、机器人标记/坐标或GT。只输出静态visible/view/target_uv/note，严格单独验证，不能当持有、co-moving或完成证据，也不直接部署actor。

与H46/H47相比减少了多个因素，不能归因于某一种上下文；H48内部唯一组间差别是1幅/3幅当前RAW。四条全部保留人工审，不按输出选样。新源码/单测/独审固定后单次运行；目前尚无新调用或完整成功。

18:43：42 CPU/真实两对输入/独审通过，旧original与neutral行为不变；静态结果没有进入actor的代码路径。自由note需逐条人工检查，不以JSON合法替代语义审核。H47双端两SHA核同。

## H48结果与最后一个静态对照H49

H48 c81a90c、3448413/3448423，4调用完成、worker88.405s/监管99.263s，峰值4534MiB/最低free2950MiB。d0单head318token正确认出壁炉/TV/长凳并说目标不可见；三RAW532token却visible:true与view:none/UV:null矛盾，被拒绝。d91两路均认出红白radio，但单head输出[.54,.72]落在物体上方桌面，三RAW[.55,.78]在物体边缘，不能将其作精确中心/抓点。首冷调用51.13s，后三条4.51/5.18/5.27s。本人对照全部4输出/RAW，未做输出修补；识别改善不等于定位或完整闭环成功。

完整本地`artifacts/agentic-vlm-goal-20260918/h48_shared_bundle_v1`；result SHA`94d0eec1e70fe72d21b63dac0fd2f6ef73c92218fe8e19675e761ede35ee8a40`，supervisor SHA`dae4a92fb45bd30cd1d60e70c1681f3f7abf3906d141666b9c4b1c2a67c3d0cc`。跨旧接口多因素变化，只能说简化接口有助于这两态的识别，不能把因果细分到history/guide/文字某一项。

H49预登记（本轮最后静态对照）：负责人Codex-parent，假设仅提高当前head完整视野的分辨率320→640能改善定位，同两状态各320/640配对4调用，其余静态prompt/4B NF4/greedy/seed17/600s/900s/显存保护均不变。只有显式single-head profile准许640；不会放开旧9图高分辨率。不改已保存图、不裁出人工目标、不加入人工位置，不上线actor。结果后汇总接口取舍，不自动再刷这两个状态的提示。

18:51：44 CPU/真实两对原请求与像素核对/独审通过；旧probe拒绝任何case级分辨率override，H49严格单head，显存和时间保护未放宽。H48双端主SHA一致，四训练仍原PID。

## H49回答与本轮工程取舍

18:58收尾更正：只读重连与完整小包传输已经成功，四原训练3294346–3294349均保留、所有自有探针已结束；H49监管exit0/97.159s、峰值4534MiB/最低free2950MiB、退出后GPU2 free7489MiB。本地完整`artifacts/agentic-vlm-goal-20260918/h49_shared_bundle_v1`的result SHA`ff9080388764ca63ec03014e2a60b8d329180e14864f2674e1dbf5f8296cc0b2`和supervisor SHA`a08b83ce9eebaf7fba8ddddd380bc997c754fb0832ceebe10c10576b731bbb8b`均与远端一致；下段18:56连接失败仅为历史，不再待归档。44 CPU及独审已闭合。

2a9b62f、supervisor3450186/worker3450194：远端result已读到complete/87.931s，四条格式通过。d0两尺寸都正确否认目标可见；d91 320仍为[.54,.72]，640变成[.63,.72]。本人对应720×720原RAW核对，二者纵坐标都在radio上方桌面，升分辨率未解决此次定位。两尺寸热调用约4.26–5.28s，首49.57s为冷启动，不拿它估稳态Hz。18:56归档连接超时，完整小包/监管最后状态与SHA待，不重跑实验；本轮静态调用已经结束。

当前有依据的结论：4B NF4可处理登记的共享显存请求；复杂观察接口下会漏认已见物体，简化接口改善识别；但识别、定位和完成反馈是三个不同能力。只删负例或升分辨率都不足以解决当前失败。这里只有两个反复使用的开发状态，不能外推50任务泛化或SR。

下一工程方向（尚未实现/部署）：

1. 当前目标识别与运动前后反馈分工，每次只传任务相关视角/字段；保持未知反馈为未知，不能用看见目标代替抓取/完成。
2. 使用基于公开当前图像的有限编号区域，再在该区域用RGB-D提出可拒绝的表面候选。模型选择候选，几何模块负责像素/3D映射；不使用GT对象框、不按d91手工加坐标偏移、不自动把最近深度点认作目标。
3. 新目标锁定须语义表面确认。现有`RefinedGroundedPolicy.refine`会在深度有效且未触发近接触复核时跳过候选选择；深度平滑只证明那里有表面，不证明它是指定对象。这里是需检查的接口前提，不是已经通过新物理实验确认的全部失败根因。
4. 先CPU契约/旧路线兼容与少量异质保存态人工审；下一实验另写预算，不继续无限刷这两个状态。模拟器显存还未测得安全共存依据；不降低原H44资源门硬启动，不停止队友训练。只有资源验证通过才做零专家/旧策略前缀的完整回合。

本轮没有新增完整模拟器回合，**没有新的完整成功率，也未完成>0%官方完整成功目标**。所有探针输出都不直接进入实际actor，原H44控制实现未被热改。
