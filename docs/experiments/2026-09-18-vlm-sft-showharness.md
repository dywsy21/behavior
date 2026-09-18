# H-09：Show-Harness 小模型语义动作 SFT

负责人：Astra/max 子代理。独立分支 `feat/vlm-sft-showharness-20260918`，基点 `6da8c80fb94e748c479ddd558290c6315f6e163d`。主代理负责 H-08 harness，本文只维护 H-09。当前是小规模方法验证，不是50任务大训练。

## 最新状态

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

## 尚未完成

- [x] 来源分组冻结、同状态微动作映射和机械检查。
- [x] 分层直接图像/标签审核及可定位记录（Astra视觉审阅，不冒称外部人工标注）。
- [x] 实际训练mask、梯度、保存/回载与吞吐门。
- [x] 完成有界训练与adapter SHA。
- [x] 留出未微调/微调比较及静态失败分类（物理失败分类待闭环）。
- [ ] 同协议物理闭环和视频审核。
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

主代理在原36张已保留面板中独立直接查看4例，未见图像与方向/开合标签明显矛盾：selection index6 `t0_e12_f1472` RIGHT_DOWN对应约14.6mm下移；index7 `t0_e133_f1952` RIGHT_OPEN可见开爪，但未据此判断PLACE成功；index23 `t3_e787_f10256` BASE_YAW_MINUS背景转向一致；index24 `t3_e773_f6640` BASE_BACK视野后退一致。本地路径为`artifacts/h09_review/review/`，完整文件名见`h09_data_v4_review.json`补录。旧selection源自v1/v3，文本仍可能含TORSO/旧图像路径；仅核SHA相同的图像与有效标签继承为v4审核证据，实际训练41符号/无TORSO文本以过滤后的v4 manifest与机械prefix检查为准，未改写旧证据。
