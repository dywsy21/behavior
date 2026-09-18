# H-09：Show-Harness 小模型语义动作 SFT

负责人：Astra/max 子代理。独立分支 `feat/vlm-sft-showharness-20260918`，基点 `6da8c80fb94e748c479ddd558290c6315f6e163d`。主代理负责 H-08 harness，本文只维护 H-09。当前是小规模方法验证，不是50任务大训练。

**最终结论：实际训练与验证已完成，但本轮没有证实闭环任务收益。** 2B LoRA在A100上600更新/18.26分钟完成，方向分类从5/192到180/192；然而速度延续规则已达170/192，操作子集仅2/18到7/18。四个注册物理回合全部官方false、没有夹爪命令或持物；微调radio反复转向，plates朝厨房台面前进，均不满足当前目标。保留全部负结果，不追加原配方训练/回合。机器可读轻量交付见`configs/vlm_sft/h09_final_result.json`。

## 最新状态

- 2026-09-18 21:53（北京时间）：四回合、100模型调用/1939新控制全部完成；两radio各448专家前缀单列。四视频SHA与服务器一致，967帧全解码，Astra亲看18个分层视频帧（精确索引见下）；完整逐决策图像/输入/控制/诊断和轻量回执已在本地。74735服务已退出，8918空、GPU1为0MiB；H09根2.8GiB、全盘余89GiB。数据/训练/live/诊断和最新零调用分析均获主代理独立review，21 H09 CPU与232原CPU通过。剩余为未验证能力与下一块数据设计，不存在待启动/待完成的本块训练或物理回合。
- 2026-09-18 21:44（北京时间）：前三物理回合完成，均官方false/无持物。radio-FT40/961/143.761s全为底盘负yaw，实际累计276.98°、39/40输入低base速度；480视频帧全解码并看0/120/240/360/479，确实转离radio而未抓取。plates-FT24/433/77.951s全前進，实际world路径1.315m后BASE_TRACKING_FAILED安全停。最后plates-base同975852a/同原预算已起，不增加回合/训练；终态/视频审核待。
- 2026-09-18 21:32（北京时间）：首物理radio_base_v1终态12决策/166新控制＋448专家前缀/43.854s；12次LEFT_FORWARD，9次执行达到命令目标、3次连续可达性/自碰撞拒绝，按注册条件停机。官方false，不能把移动命令完成当GRASP成功。radio_ft_v1同975852a/同预算开始；视频下载与实际位移审计进行中，后两个plates尚未启动。
- 2026-09-18 21:26（北京时间）：975852a双工程门完成：radio20命令/355新控制/55.423s，plates20/410/75.512s，gate_ok均true、同digest7993ec29；底盘后移遮挡拒绝如实保留，开合不当作抓取。本人已看两门head/right-wrist初始原图，客厅radio/厨房冰箱灶台场景一致。最终2B+adapter服务74735/8918/GPU1、cap160已起，原四回合按radio-base、radio-FT、plates-FT、plates-base顺序运行；不再训练。
- 2026-09-18 21:10（北京时间）：192测试同协议比较完成，98.753s。原始5/192、FT180/192；速度规则170/192、上一动作114/192。base子集3/174→173/174（规则170），arm/gripper2/18→7/18（规则0、history3），仍缺可靠操作能力。FT反向3/同部位错方向6/错部位3；base/FT中位延迟0.140/0.325s，包含各自生成token长度差异，非纯LoRA开销。配对同对3、仅base对2、仅FT对177、均错10。未改权重/看test选ckpt；下一双工程门和4物理短回合。不能从静态总93.75%声称任务有效或视觉理解增强。
- 2026-09-18 21:06（北京时间）：train_v1实际完成600更新/1095.469s，训练源140c47dabfe7d15ac5674e7f45874703aaa2241c，PID61793退出；67,276,800可训练LoRA参数，18.182GiB峰值。基座权重SHA `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`，600最终adapter SHA `b5a125ed14dc06c82a7ae7fd288d8c7202d90e2210cc1d1c2f7195f3daac15e1`，identity SHA185d96bd…。开始eval_test_v1（975852a），192条实例留出、原始/微调同input/grammar、base/非base分层与速度/上一动作持久性规则；不看测试选checkpoint。神经与实际物理效果仍待完成。
- 2026-09-18 20:56（北京时间）：train_v1实际270/600更新、510.87s，源140c47d/PID61793/GPU1，200检查点已存；峰18.18GiB。独立local-skill服务/runner完成本地16CPU，待固定review与双工程门。闭环登记`configs/vlm_sft/h09_local_pilot.json`：新桥接2×20命令/768控制/1200s工程门、0模型，再原4配对40决策回合；仍仅GPU1，训练完成后顺序。固定合法技能不读取expert skill；非base环境碰撞不认证；完整任务成功与局部持物诊断分开。
- 2026-09-18 20:46（北京时间）：真实两更新gate_v1完成（bff206c），native/custom CE同1.0453935861587524、误差0；冻结梯度为空、372 LoRA张量更新、回载logits差0，18.71GiB峰值，热态1.995s/update。随后执行器审计发现TORSO语义不一致，**未开始v3正式训练**；ec5c012将TORSO从监督/推理候选统一去除，历史截断并重建text。v4=1199/193/192、训练各任务238/222/739，图片全部核SHA复用、14已审阅非TORSO案例，manifest SHAaa2ca08f…。下一从基座新建train_v1最多600更新，预估20–30min；三任务新prefix/mask会先实际检验。首块仅base/arm/gripper方向，不验证躯干策略。
- 2026-09-18 20:39（北京时间）：严格v3完成1755训练/259验证/288测试；额外60候选被拒（task0/1/3=9/35/16），分组不变。图片manifest SHA `0973d88818a5295766249c07f77ca223812a9f77cb23088f0f98666ec689ce41`，机械门全过，21已查看面板全部保留，审核记录`configs/vlm_sft/h09_data_v3_review.json`。准备GPU1两更新真实门（≤1200s），此门只是工程验证，不是有效性训练。主代理已完成codec/prepare/modeling/train/evaluate独立静态review，无剩余阻塞。
- 2026-09-18 20:37（北京时间）：第三轮review指出TORSO/BOTH和rotation分支中途反转风险，已统一全路径方向/反转/离轴与双臂逐帧一致性；12CPU正负例通过。本人已查看v2的21张三视图current/+16面板，任务/活动技能匹配、可见夹爪开合/底盘方向一致；小幅末端/躯干运动无法仅靠图像定量确认，独立FK门提供坐标审计。下一生成严格v3并记录受影响量，复用原图不扩任务/episode。0训练/新物理。
- 2026-09-18 20:27（北京时间）：主代理独立review发现endpoint-only可能漏掉中途手臂 excursion、base只验前8帧、原留出需整instance去重、列表ID和history stride边界。`0cf316a`已改为全16帧方向/本体轨迹纯度、原5%所有instance排除、逐对象清ID、强制stride≥horizon，10CPU反例通过。未对旧data_v1训练；新data_v2索引25.23秒完成：train1779/val262/test293，分组SHA不变；任务0/1/3原窗口2106/5579/12747，严格映射准入485/615/2125（抽样上限前），保留全部拒绝分类。新抽帧仅复用核SHA旧图片/提取缺失图，原数据不覆盖。训练第二轮review要求native labels loss与自定义CE数值核对、EOS非UNK、回载logits核对，`1a76197`已实现，待真实GPU门。0训练/新物理。
- 2026-09-18 20:21（北京时间）：独立坐标/时钟/原标签SHA门通过：48episode×3时刻×双EEF/三相机，机器人本体FK重建误差最大3.374µm/1.247µrad；61D EEF确为机体系，源标签SHA666f8fc0…与正式release一致。2668样本×三视图共8004张256²图完成，所有视频PTS与指定帧在半帧容差内；数据596MiB，盘余93GiB。人工分层current/+16帧面板生成中。训练实现已写为同一推理prefix追加原生动作token、仅监督answer、冻结vision/projector、全LM线性LoRA；尚未GPU前向/训练，待人工审核与实际两更新/恢复门。
- 2026-09-18 20:16（北京时间）：`a201e09`双端7CPU通过。首轮CPU索引23.20秒完成，48来源episode按instance冻结：训练2006、验证298、测试364样本；分组SHA `3d6cb946fab72c156ae320cda2ae6f4a524dbcfb1c49875c50ae67f608a138aa`，索引位于服务器`vlm_sft_showharness_20260918/data_v1`。多数专家操作是混合运动而被拒；训练目前以底盘和躯干为主，不能宣称已覆盖完整抓取技能。独立robot-only FK核对真实61D末端/相机坐标与30Hz时钟的审计在运行（源90ee7ae，≤1200s）；三视图256²抽帧在运行（源a201e09，≤1200s）。0训练/新物理。标签是专家16帧内**运动方向**的保守投影，幅度不等于执行器固定步长，此边界会单列报告，不称严格复现原连续轨迹。
- 2026-09-18 20:12（北京时间）：已 fetch 团队 `origin/main=33677bd`；基点包含该提交。主仓有主代理未提交改动，未 pull/切换/覆盖；新建独立 worktree `/home/wsy/behavior_worktrees/vlm-sft-20260918`。GPU1 独占核验为 A100 80GB、0MiB，旧 PID38931 已退出、8918空。`/mnt/sdc1`余94GiB，登记本块新增≤8GiB且始终保留≥80GiB；不下载大权重。尚未训练。
- 已核官方原论文、项目及训练配置；已确认专家数据含同帧 RGB、23D动作、61D本体、每臂实际末端位置/姿态与原始技能区间。准备检查连续16帧运动能否无歧义归入通用微动作；混合运动、边界和未知样本会保留排除原因，不能伪造教师标签。

## 原论文与复现边界

[原论文 §3.4、§5.1、附录§7.1–7.2](https://arxiv.org/pdf/2609.10522)：FT使用Qwen3.5-2B，通过原生文本预测语义动作；LM线性层rank64 LoRA，冻结vision/projector。附录设定7.9k单臂样本、40epochs、LR1e-4、cosine/warmup0.1、bf16、256²视图、effective batch32，报告**单H200少于2小时**；不是27B或任意机器人/数据规模的时间保证。采集输入和动作通过共同接口严格配对。机器人与BEHAVIOR R1Pro、移动底盘、长任务明显不同，不能直接外推完整任务成功率。

[官方训练配置](https://github.com/showlab/Show-Harness/blob/main/train/configs/qwen3_5_2b_lora.yaml)当前写30epochs、`freeze_multi_modal_projector: false`，与论文40epochs/冻结projector存在差异。本轮明确采用冻结vision/projector，不称字节级复现。代码启用fused CE以避开大词表全序列logits开销；本轮须先量真实吞吐。

[官方训练接口说明](https://github.com/showlab/Show-Harness/blob/main/train/README.md)特别指出训练与服务chat-template和历史字段要一致；本轮从同一函数构造训练prefix和推理prefix，并逐token核验，仅assistant动作被监督。遵循原生词表、不另加动作head。

## 首块预登记

主要假设、数据/训练/物理预算固定在 `configs/vlm_sft/h09_first_block.json`。基座复用已有Qwen3.5-2B，GPU1；最多600更新/3小时，effective batch16，rank64；200/400/600仅小adapter检查点，最多新增8GiB。真实内存与估时在小batch门后补入。

选择task0 radio、task1 trash、task3 plates三类；每task12训练、2验证、2测试来源episode，原每task末10个留出episode绝不进训练，原开发实例task0/138和task3/242排除训练。来源instance与episode分组冻结后才抽帧；最多3456训练样本，不重建大集。测试用于一次最终比较，验证用于检查训练与选择checkpoint。所有未来动作/物理字段只生成离线target，不进入actor；actor只接收当前RGB、本体、自然语言任务/技能和过去动作。

闭环初始预算：同协议2起点×未微调/微调，各≤40决策/1280新控制/1200秒，GPU1顺序运行。原前缀单列。两个模型使用完全相同的控制、安全门、提示与重置规则；成功只认官方物理判据，另报告局部行为、拒绝、动作和延迟。数据/映射审计失败就先纠正具体根因，不消耗此物理预算。

## 本块验收

- [x] 来源分组冻结、同状态微动作映射和机械检查。
- [x] 分层直接图像/标签审核及可定位记录（Astra视觉审阅，不冒称外部人工标注）。
- [x] 实际训练mask、梯度、保存/回载与吞吐门。
- [x] 完成有界训练与adapter SHA。
- [x] 留出未微调/微调比较及静态失败分类（物理失败分类待闭环）。
- [x] 同协议物理闭环和视频审核（负结果完整保留，未扩大预算）。
- [x] 主代理独立代码审查（数据codec、mask/reload、训练、live/serve/runner与write-only特权诊断边界）。

## 首块已完成的训练与静态结果

训练run：`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_showharness_20260918/train_v1`。独占A100 80GB GPU1，1199训练行、有效batch16、600更新（9600次有放回样本抽取），rank64/alpha128/dropout0.05，LR1e-4/60步warmup/cosine；仅语言侧67,276,800 LoRA参数，vision/projector冻结。监督严格为当前推理prefix后原生动作token+EOS，其余全部mask。最终优化时间1095.469s，不含前序准备/模型加载；峰allocated18.182GiB、reserved21.211GiB。不把此小数据预算换算成论文7.9k×40epochs的训练时间。

不可变来源：训练源`140c47dabfe7d15ac5674e7f45874703aaa2241c`，过滤数据源`ec5c012e246ac24db3ac4510381b0d428c84bb1c`，data_v4图片manifest `aa2ca08f17f255f4738cc148dd5807b91f6b0306962f36e2817a2a077d772565`，原始source-instance分组`3d6cb946fab72c156ae320cda2ae6f4a524dbcfb1c49875c50ae67f608a138aa`。最终adapter SHA见最新状态；其余中间adapter不按test筛选。权重/优化器/图片不入Git。

`eval_test_v1`使用975852a，192测试样本来自6个独立来源实例/episode（在三个已训练任务内，并非新任务泛化）。两个模型共用同一processor/input tensor/41符号有限解码器，逐样本交替模型顺序；没有更新权重。静态当前技能来自同状态专家注释、对两模型相同，因此它是**给定正确当前意图的方向分类**，不包含高层规划准确率。

| 对照 | 全部正确 / 192 | 底盘正确 / 174 | 手臂/夹爪正确 / 18 |
| --- | ---: | ---: | ---: |
| 原始2B | 5 | 3 | 2 |
| 600步LoRA | 180 | 173 | 7 |
| 当前底盘速度规则 | 170 | 170 | 0 |
| 重复上个已执行动作 | 114 | 111 | 3 |

FT相对同基座多175个正确，但相对速度规则只多10个；其中底盘仅多3、操作多7。FT对左臂四个测试例全部错误，right-left四例也全部错误；操作正确集中为right-up 4/4、right-down 2/2、right-close 1/2。稀少操作样本不支持稳健抓取/跨任务能力结论。训练集合中1086/1199是底盘，规则已解释大部分总分；尚未做去图像消融，不声称视觉收益。单样本模型延迟中位数/95分位：原始0.140/0.170s、FT0.325/0.336s；输出token长度及LoRA路径都可能影响，不能仅归为adapter开销。

完整轻量训练/静态证据归档SHA `d10986f4bb7d7a7857d121165dc9a28198a4df9a4737b170858126700c5f368a`，已下载并核对至本地`artifacts/h09_training_eval/receipts.tar`；含原逐样本预测、逐更新样本ID/梯度/loss、mask/native loss/回载门、manifest与数据来源。物理闭环独立报告，不能用此表代替任务效果。

额外零调用CPU分层（`stratify_static.py`，a10d23a；低速度定义为当前底盘XY速度<0.02m/s且yaw速度<0.03rad/s，阈值未优化）：训练低速127/1199、验证23/193、测试20/192。测试非低速172条全是底盘动作，原始3/172、FT172/172；低速20条含2底盘/18手臂夹爪，原始2/20、FT8/20。空动作历史78条原始3/78、FT66/78；非空114条原始2/114、FT114/114。低底盘速度并不表示手臂静止，分层只是关联而非图像/本体因果消融。专家移动中的方向延续占主导，而部署微动作之间停顿，是闭环需检验的分布差。

## 独立图像审核补录

22:02补充（Codex主代理，**训练后的独立复核**）：在已有4例之外亲看其余10张最终v4保留的面板，共14例/84个前后三相机视图，涵盖三task、双臂准备、单臂方向、左右合爪/开爪和底盘运动。未发现明显错图/对象上下文或可见运动方向矛盾；细微平移不以像素估计代替FK/全路径数值门。它不弥补类别/意图覆盖不足，也不认证抓取或放置成功。索引、具体界限与两FT视频首末复核另存[父线审核回执](../../configs/vlm_sft/h09_parent_visual_review.json)；没有访问静态test图片、修改训练、重选checkpoint或补跑。

主代理在原36张已保留面板中独立直接查看4例，未见图像与方向/开合标签明显矛盾：selection index6 `t0_e12_f1472` RIGHT_DOWN对应约14.6mm下移；index7 `t0_e133_f1952` RIGHT_OPEN可见开爪，但未据此判断PLACE成功；index23 `t3_e787_f10256` BASE_YAW_MINUS背景转向一致；index24 `t3_e773_f6640` BASE_BACK视野后退一致。本地路径为`artifacts/h09_review/review/`，完整文件名见`h09_data_v4_review.json`补录。旧selection源自v1/v3，文本仍可能含TORSO/旧图像路径；仅核SHA相同的图像与有效标签继承为v4审核证据，实际训练41符号/无TORSO文本以过滤后的v4 manifest与机械prefix检查为准，未改写旧证据。

## 已完成的物理配对

四回合及服务均固定`975852aa0f012987084dc9a1d49e1d1783122163`，执行实现digest `7993ec29399233c0ba287c1a955fa9bdbf576d283e9271578054a65dd1a2e17b`；分析源`938fc87e6660cf95c9f307d49addcfe3b23a79a5`。共享原`6da8c80`的SafeServo/机载输入/底盘depth-veto，**不是H-08动态GroundedController同源策略**。两场景工程门先通过，分别355/410新控制、55.423/75.512s、0模型调用；策略新预算仍各40决策/1280新控制/1200s，环境seed0/policy seed41，顺序radio原始→radio微调→plates微调→plates原始。

H09 radio只有448专家控制前缀；H08匹配持物验证起点另接旧策略362控制，且模型/控制策略/目标阶段均不同。**两线不得直接比较成功率或将父harness改进归因于此adapter。** 本文有效的模型对照仅是H09同2B基座的adapter-off/on。

| 回合 | 决策/新控制 | prefix后秒数 | 实际行为和停止 | 官方成功 |
| --- | ---: | ---: | --- | --- |
| radio原始 | 12 / 166 | 43.854 | 12次LEFT_FORWARD；9执行、3可达性/自碰撞拒绝后停；左EEF端点路径87.8mm | false |
| radio微调 | 40 / 961 | 143.761 | 40次BASE_YAW_MINUS；实际累计yaw −276.98°，预算停，未执行GRASP | false |
| plates微调 | 24 / 433 | 77.951 | 24次BASE_FORWARD；底盘端点路径1.315m，末次BASE_TRACKING_FAILED安全停 | false |
| plates原始 | 24 / 379 | 96.318 | 24次LEFT_FORWARD；21执行、3拒绝后停；左EEF端点路径205.6mm、底盘不动 | false |

100次模型决策中没有OPEN/CLOSE；全部事后assisted持物审计为空。命令TARGET_REACHED只表示机器人完成该微动作，不表示抓取/导航目标完成。radio最终官方谓词均satisfied=[]、unsatisfied=[0]；plates均satisfied=[0,3]、unsatisfied=[1,2]，与工程起点相同。固定GRASP和NAVIGATE不是完整任务规划器，因此这里不能据0/4构造总体任务SR估计。人工视频看到plates微调朝微波炉/台面接近，没有转向餐桌；未设计独立导航距离成功阈值，不伪造局部导航SR。末次命令速度积分约19.9mm而预期为60mm，触发原跟踪门；没有读取接触真值来确定唯一物理受阻原因。

两对重置的初始robot world pose与提供给actor的proprio均完全相同；原始头图不是逐字节相同，radio/plates平均绝对RGB通道差分别1.595/1.103（0–255尺度，最大73/53），相机画面直接核对为同场景起点。故是同配置物理配对，不冒称物理首帧像素完全相同；静态配对则确实复用同一input tensor。world pose/assisted attachment仅write-only落盘作事后分析，未回注actor、动作分支或停止条件。距离/转角是决策端点观测之和，非连续扫掠路径，也不含末尾单步安全保持后的新增观测。

在线模型延迟median/p95秒：radio原始0.143/4.166（首次冷调用影响尾部）、radio微调0.334/0.346、plates微调0.222/0.227、plates原始0.140/0.149。解码符号长度不同，不把延迟差全归LoRA。物理actor处于低base速度的次数为11/12、39/40、24/24、24/24，总98/100；训练仅127/1199、静态测试20/192为此状态。该分布差是证据，但没有做图像/本体/历史因果消融，不能认定它单独导致不服从指令。

## 数据覆盖与负结果解释

原计划按每任务来源episode配额选样；严格过滤后并不任务均衡，更没有对**实际部署意图×动作类别×近静止/阶段起点**设置准入门。此缺口不能用“训练见过task0/3”掩盖：

- radio GRASP确切指令只有11条训练样本、5来源episode；10条低base速度、全部空history，其中RIGHT_CLOSE仅1条。task0另外213条是NAVIGATE、10条PLACE、4条PRESS。不能声称充分训练了GRASP。
- plates固定`NAVIGATE breakfast table`没有直接训练样本。相近NAVIGATE目标为fridge270、plate221、bowl51、drop in sink54；家具导航目标table/dining table/breakfast table均未出现。桌子概念并非完全未见：49条GRASP以breakfast table为source，另272条到plate/bowl的相关导航。物体目标和家具目标不是可无条件合并的同义标签，结论只是部署子目标粒度缺直接监督。
- 静态180/192主要是专家连续运动状态下的方向延续；当前速度规则170/192已解释大部分分数。FT比规则的操作增量7条、底盘增量3条仍是真实计数，但不支持可靠视觉操控结论。物理radio在GRASP下反复转向，直接证明这次模型没有正确执行指令；不是所有失败都应归因于控制器或模型规模。
- 源16帧/30Hz方向投影与部署固定微动作幅度/时钟不完全一致；许多混合操作被严格拒绝，113条非底盘训练标签不足以覆盖41符号及完整技能。没有HOLD/失败/完成正标签，不伪造它们补数；也没有用这些失败模型轨迹自循环作SFT教师。

训练前已针对独立review做过有界纠正：全窗口反转/离轴检查、整instance留出、ID清理、历史因果约束、实际native loss/回载校验，以及排除执行语义不一致的TORSO监督/候选/历史。正式600步后未追加训练或改prompt挑结果。这一负结果不否定有充分覆盖的harness-native SFT；也不为直接扩大到50任务或改用27B提供证据。

下一块如果获注册，**必须先**按实际部署子目标、动作类别、近静止与阶段起点分层审核覆盖；采集/映射严格同harness动作和时钟，覆盖真实停止/恢复决策及未知状态，而不是仅增加连续运动中间帧或同样本训练轮数。与现有数据负责人协调，不自行重建50任务或手写任务专用教师。尚未验证：去图像/去本体因果消融、未见任务/场景泛化、完整动态技能切换、可靠抓取与完整任务成功。它们不属于本块已完成的结果。

## 可复核交付与视频

主代理已将四条原始视频另存到本仓[视频目录](/home/wsy/behavior/artifacts/vlm-sft-showharness-20260918/videos/)，文件为`radio_base_v1.mp4`、`radio_ft_v1.mp4`、`plates_base_v1.mp4`、`plates_ft_v1.mp4`，SHA与下表原件及轻量结果逐一一致。父目录另有原训练/评测`receipts.tar`、`final_receipts_v1.tar.gz`和两个分析JSON，全部校验；不是重新编码或新评测。下表子代理worktree原件仍保留。

训练配置`configs/vlm_sft/h09_first_block.json`；物理注册`configs/vlm_sft/h09_local_pilot.json`；轻量结果`configs/vlm_sft/h09_final_result.json`。最终adapter目录为`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_showharness_20260918/train_v1/adapter_0600`，权重SHA `b5a125ed14dc06c82a7ae7fd288d8c7202d90e2210cc1d1c2f7195f3daac15e1`。数据/权重/全部原始证据仍在服务器H09根，没有新基座下载或覆盖。

本地四视频位于`/home/wsy/behavior_worktrees/vlm-sft-20260918/artifacts/h09_physical/`，每个run下还有完整逐决策图像/输入/反馈/特权离线审计；`video_review/frame_*.png`是按下表视频帧索引顺序提取的检查图。Astra直接查看18帧，非声称外部人工标注或逐帧人工观看；全部967帧另做解码检查。

| 本地视频 | 完整帧数 / 秒 | 本人直接查看的零基视频帧 | 可见结果 |
| --- | ---: | --- | --- |
| [radio原始](/home/wsy/behavior_worktrees/vlm-sft-20260918/artifacts/h09_physical/radio_base_v1/rollout.mp4) | 82 / 5.467 | 0,40,81 | 左手在桌左侧前移，radio留在桌面 |
| [radio微调](/home/wsy/behavior_worktrees/vlm-sft-20260918/artifacts/h09_physical/radio_ft_v1/rollout.mp4) | 480 / 32.000 | 0,120,240,360,479 | 从radio依次转向阳台、壁炉、电视、厨房，未抓取 |
| [plates微调](/home/wsy/behavior_worktrees/vlm-sft-20260918/artifacts/h09_physical/plates_ft_v1/rollout.mp4) | 216 / 14.400 | 0,54,108,162,215 | 持续接近微波炉/厨房台面，未转向餐桌 |
| [plates原始](/home/wsy/behavior_worktrees/vlm-sft-20260918/artifacts/h09_physical/plates_base_v1/rollout.mp4) | 189 / 12.600 | 0,47,94,141,188 | 只伸左手，头部场景基本不变 |

视频只含自主suffix、每2控制采1帧/15fps，省略推理等待；不能以视频秒数代替实际wall time。两radio的448专家前缀各自另计。审阅包仅省略重复约46MiB robot_calibration.json，其完整远端文件保留且SHA为radio `cd3a071749fc1457f75669e32828ad7377097e4740a4a7d4f66da99c9d2352a1`、plates `598d23bc69d61ac139f312e62bdd0b867d87ed33a6bc11b71817741bc0a2a886`；本地radio部分下载以`.partial-download`标明，不当完整文件使用。

四视频SHA、逐run result SHA与帧审核索引均列轻量结果JSON。完整轻量收尾包`final_receipts_v1.tar.gz` SHA `9d3082a3e8c828669d0a4116a3bfed29de3f26c9c37cec8c5b4f7d055030dea4`；物理汇总`physical_summary_v1.json` SHA `74cfcb73f1115e084c599c2c60f283fc9f36116705a295a989ad76d3455b8849`；零调用分布/意图审计SHA `77550cdea2fc5c16b6f52ea2ffc65bf9571f6fb0a6215385ab44237329d0182d`，均已本地核对。主代理负责最后Git集成，不热改任何旧实验源，不合main。
