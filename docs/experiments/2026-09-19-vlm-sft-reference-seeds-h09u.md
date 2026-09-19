# H09U：完整原专家回放与可审 seed

唯一负责人 Astra 子代理；2026-09-19 06:04:34–06:34:34 BJT，CPU ≤1800s、准备 ≤30MiB、0 新 reset/物理控制/模型/训练。父代理维护总计划并审查；本文件和 template 均不是物理授权。

## 已完成的真实来源准备

准备源 `0df85fc`，robo `.../vlm_sft_native_teacher_20260919/h09u_reference_prepare_v2`；manifest SHA `d4311e887582e3259a835f83664e02126ea9abc8df69ff45d6a6046ae95317f2`。双端完整17文件 **774,831 B**；manifest 中其余文件合计771,074 B。本地完整副本为独立 worktree `artifacts/h09u-reference-prepare-v2`。原失败 v1 保留，失败发生在重复分支检查、未生成回放结果。

|原 TRAIN 来源|完整 prefix|完整技能段|12 稳定 +1 hold 后总控制|
|---|---:|---:|---:|
|task0 / episode66 / instance70，PRESS 左手、右手支撑|1170|1170→1610，440|1623|
|task1 / episode310 / instance192，GRASP 右手|164|164→452，288|465|

总计 **2088 控制 / 2 reset（仅提案，未执行）**。全 episode float64 state/action 数组与所有选定标签字段 SHA 精确复现 H09R：radio `f8775ce1ba25dd041a2fba46b3e7c94bcff92dc52c232d15bfd3e75c2f270404`；trash `06780c386db3be8d3d14f045b46af346ad9f6ee9b3d8fdfc933741bc9dd06e84`。H09R TRAIN 实例排除、原逐 task5% 全实例、H09 val/test、开发138/242仍强制；不是凭 `original_demo` 字符串豁免。

radio labels 1958 行/1954 frame：low1954、high4；trash2031行/2019 frame：low2019、high12。显式选 original_demo/low、同选中分支重复即拒绝；`label_selection_audit.json` 保存分支统计、全部源 SHA、episode-filtered 原行索引；`segment_labels.json` 保留所选区间的 high/low 原行，源数据未修改或去重。440/288 个 frame 均逐行核相同意图、区间、low supervision mask、action horizon、quarantine，不用端点替代完整段。

## 修复两个父审问题

seed 不再只验一个任意64hex文件。`validate_seed_release` 绑定源 task/episode/instance/完整数组标签SHA、segment边界、verb/hand/target/destination/support/payload/frame 与精确 SE3；绑定 authorization 指定准备 manifest；复算连续实际控制的严格 oracle 与 object-local pose；核 result、完整 private trace、四个完整 RGB-D/self geometry capture 的 hash/时钟及独立 review。review 必须明确看过源/当前/终态图像和 contact/update/control 账本，绑定 seed/identity/pose SHA。没有自动审批、没有把本 helper 当真实性密码学证明。

PRESS 改为该目标实例专属 `_update` 观察器，非全局类补丁；每次原方法恰好调用一次，原参数、返回值、异常身份保留，finally 恢复。观察失败不能跳过原更新，而在读取证据时失败。读取已固定源码的直接 `value`/counter，避免 get_value 缓存改变观测时序。原安装 `toggle.py` SHA `a7a88f4a55242816ab930fe5336cfd94b26e2aaf6e2d33aa0239ed47eca2fd3b`，回放及自动 PRESS collector 在 reset 前检查。

每个真实 update 前只读查询原相同 marker sphere、两手各自 target contact、其他机器人 marker 命中，保留所有命中。连续5个真实 update 必须指定手 marker+target contact、其他手/机器人均不命中，原 counter 连续且 false→true 恰为5；缺链/重复/未观察边沿 UNKNOWN。**5 updates 不等于5控制**。PRESS seed 使用发生边沿的真实 update 前 EEF/marker-parent pose，不以整个控制结束姿态冒充。随后仍须12个真实控制保持 true；右支撑手 target 相对稳定条件不变。这是严格局部归因定义，不代表按键力/所有接触动力学已被证明。

## 可执行入口和失败边界

`native_teacher_reference_replay.py` 用当前官方 factory、同源原0起点 reset，逐条执行完整 native23 prefix+专家段；每个控制先记 issued，再记 completed，逐控制私有 contact/pose/真实 toggle update 与 oracle。原 state61 的 q/grip 每步交叉核验（.02rad/.005m 上限，不做重新标定/调尺度）；失配停止。prefix和原专家段不是离散 native BC，seed 本身也始终 `training_eligible=false`。

完整段第一条前按 raw 形状预留全部余下三次 full capture、private/control ledger、每30 controls 三张256px缩略原图和 cleanup；发布 capture 前完成整包字节检查。初态、段尾、12稳定尾、final hold 全部 raw 三视图/depth/FK/hash；render-only 不推进控制，若 q/grip改变停止。不会把 UNKNOWN/失败/annotation尾/官方 episode提前终止改成功。所有成功还须父人工审图/接触账本后才能供自动教师使用。

异常后有剩余控制且非terminal时显式保持当前q和最后命令grip；普通 step 尚未建立、全部写失败时仍尝试该 hold。issued/completed区分 step异常可能已局部发生；次级记录/teardown错误不覆盖首错。terminal后不额外step。最终成功要求 exact总控制及 final hold 后局部结果仍成功。不能保证进程/驱动硬崩溃时物理hold完成，只能如实留账。

原专家 replay 不运行 SafeServo 来改写原动作，因此不得宣称每个专家控制经新 native guard认证；同源工程门是环境/几何初始化检查，不替代后续每步私有接触/状态审计。只有后继 native teacher 使用原41符号、fresh depth/actual selfboxes、SafeServo geometryTrue 逐动作预检。PLACE_IN仍在 reset前封锁；没有伪造corners_inside。

## 下一块精确提案（等待父审/授权）

优先短 task1：1 reset，164+288+12+1=465controls，reset后≤900s，run≤80MiB；然后单独协调 task0：1 reset，1170+440+12+1=1623，reset后≤2100s，run≤96MiB。0模型/训练、GPU1独占顺序，失败无retry。按真实 task1 prefix约.385s/control，纯控制proxy约179s/625s；private读取/接触额外开销未知，900/2100是硬限，不是完成时间承诺。此前task0初始化/长prefix失败保留，此回放是新的明确预算，不重用旧reset余额。

实际720²+480²×2 RGB-D的单capture保守上界9,276,026B：radio完整段剩余证据预留53,358,446B，trash45,002,606B；加初态、compressed calibration、prefix trace仍小于96/80MiB。累计≤384MiB必须包含原85,022,736B及后续准备/旧失败，不能换目录逃预算。SDA当前仅约81GiB、保80GiB，父已提出另一个2.7TiB空闲NVMe路径；本票未创建或移动。若迁移新准备，需按新绝对prefix路径重建window与manifest、重验SHA、重新授权，不直接复用本manifest去别的路径。NVMe新root也必须显式登记旧累计占用/独立块总额，不抹旧失败。

当前 native collector 的6primitives/200新controls不改：1cm动作，通常18执行+12停稳。抓取最低1次接近+close+4次1cm抬升=6条、约193controls（含初始12和final1），只有极近对齐初态才可能够；接近L1距离12cm则约17条/523controls；worst per primitive40+12ticks，24条上界1261加prefix，不应偷偷用200名义跑。seed成功后再提 **单实例≤24 native primitives/1280新controls/reset后≤2100s** 的新先导；原工具尚未改接新预算，需父另票与review，绝不此刻启动。

方向/姿态仍可能不可达：carryTrue限制旋转，仅1cm平移，seed orientation偏离会停止，不能无限平移或把缩距当正确。先导须至少真的完成局部技能，再决定数据产率。≥240正确native与每verb/实例/近静止/阶段起点门不降低；未来≤400步2B新LoRA、有限base/FT静态及6局部配对沿H09T已登记路线。尚无新seed、无新正标签、无新训练/效果，本块只把实际来源及回放/归因闭环做成可审实现。

## CPU 验证

当前60 SFT tests（含新11组）/0.232s，331 harness/5.201s。新增覆盖原_update返回/次数/异常/恢复、measurement失败不影响原更新、另一手归因/断链/重复、high/low与同branch冲突、原SHA/quarantine、source q失配、任意seed/跨hand/pose/准备SHA与证据损坏拒绝、完整seed/review正路径、真实runner finally AST在全写失败时保首错/hold以及terminal/控制上限不step。均合成CPU测试，不冒充实际物理成功。

06:30 BJT 远端固定 `9b52faae182ed446453ffe5d8aca63529636712f`，`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/vlm_sft_h09u_9b52faa`：60 SFT/0.531s；实际两源 load_reference、原TRAIN排除、四维绑定/全部文件SHA、现有两安全门flag+digest、真实官方 factory 原前缀逐值均通过，inactive template拒绝。未构造session。安装scene.idx/robots、robot.is_manipulation接口及toggle源码SHA只读核过。轻量原始计数见 `configs/vlm_sft/h09u_cpu_validation.json`。

父审核且另行生成 exact-code 有效授权后，入口为该干净固定目录下：`env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python scripts/vlm_sft/native_teacher_reference_replay.py --prepared /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/h09u_reference_prepare_v2/task_1 --authorization <父另发有效授权绝对路径> --output <父登记的新运行绝对目录> --gpu 1`。task0需另一份 `[0,66,70]` 授权及1623控制预算。当前 template明确false、缺code/reviewer/gate路径，不能直接启动。剩余：父独立代码审查、真实接口/时钟先导、成功seed人工审核、native新预算实现/审查、数据门、新训练和配对效果，全部尚未完成。

## H09U-P1：父终审后唯一 task1 参考回放（已失败退出）

2026-09-19 06:36 BJT 父逐文件终审和独立60测试通过并合入，明确放行一次task1。06:38:45提交GPU1、固定9b52faa/239cb591；唯一输出 `vlm_sft_native_teacher_20260919/h09u_reference_task1_v1`，相邻`.log`。有效授权 `authorization_h09u_reference_task1_v1.json` 双端SHA `44e2c2cd4131d5081d4cea464886cf652593aa2c0385fe8ecdd0306cb68454f1`，reviewer `Codex-parent-H09U-final-review`。164prefix+288完整技能+12稳定+1hold=465控制、reset后900s、80MiB、累计384MiB含旧失败、磁盘余80GiB、0模型训练、失败不retry。task0与native teacher未获授权。

启动前实际CPU source/factory前缀、TRAIN排除、toggle依赖、两门flag/digest/完整SHA通过；source干净。GPU1无进程，父156556仅GPU3，队友GPU0不动。root实占85,836,638B、磁盘余87,032,434,688B，已预留完整80MiB后仍≥80GiB。Python **PID179879**（启动壳179878），远端实际进程起点06:38:59 BJT，已见OG初始化。当前仅提交初始化，绝不当成seed成功；终态、真实控制、全证据和父手审待。

06:42:08 BJT 失败并退出：原164×23 prefix逐值精确，source q误差18.68327mrad、grip37.387µm，均在原20mrad/5mm门内；保存before后，`PrivilegedReader`误用native scene name查询BDDL-keyed `task.object_scope`，`trash_can_116`未找到。完整专家段 **0控制**、private ledger空、0 seed。最终hold真实完成，向量=before当前q＋最后原grip，总165控制；首错误保持，`failure.wall_s=66.703707`是hold前时刻，不冒称整段总wall。没有retry。

完整17文件6,585,894B＋相邻log均双端SHA通过；本地 `artifacts/h09u-reference-task1-v1/run`，before三原图/depth/selfboxes、完整压缩标定、trace、failure及hold齐全。本人已看三图：桶在地面、双手打开，不是已完成GRASP。此次错误前没有expert skill执行，也无hold后RGB-D（错误cleanup只保存控制回执），不伪造终态图证。末root92,428,246B含旧失败、free87,006,310,400B、GPU1回0MiB。轻量实际账本及所有关键SHA见 `configs/vlm_sft/h09u_p1_failure_identity_audit.json`。

### 有界身份适配修复（06:44:32–06:54:32 BJT，CPU≤600s）

父另批准0reset/模型/训练的最小修复。已直接读安装factory与`BehaviorTask`：303行按BDDL instance建scope；524–553行从scene `inst_to_name`读取native name并赋 `scene.object_registry('name',name)` 返回的实际对象，当前安装版不是需猜测解包的代理。Evaluator283–318行加载原TRAIN192 TRO时仍用BDDL key。实际原scene cache `metadata.task.inst_to_name`明确 **`ashcan.n.01_1 -> trash_can_116`**，原192 TRO也有该key；scene/TRO SHA已固定在审计JSON。原source标注的native name正确，错误是teacher查键接口。

修复仅 `native_teacher_og.resolve_bound_native_objects`：从已经绑定的task scope值精确匹配native `.name`，要求唯一BDDL别名、metadata同名、scene registry返回同一对象、实际links/states/prim_path存在；target/destination/payload角色分开，重复同物拒绝。未绑定、重复alias、registry替身、模糊/category/BDDL名冒作native名均拒绝；无scene-wide候选搜索、不unwrap未证明的wrapper、不改原source、无新对象姿态进入actor。精确身份回执只写private frame。

64 SFT/.231s、331 harness/5.025s通过；新增4组包含真实同构的ashcan映射正例与上述负例、actor投影拒绝identity字段。旧9b运行源未改，失败完整保留计预算。修复尚未经真实reset验证，必须父独立审及另登记一次新参考回放；未自行追加物理或宣称seed可用。

## H09U-P2：身份修复后唯一参考回放（已失败退出）

2026-09-19 06:55 BJT 父独立审26e修复通过并合入，逐读安装BehaviorTask/scene/TRO、唯一native映射与64测试，另行明确放行一次task1。运行源码固定 **26e476eaa7c5143f1e64d79f649d9003ef6aad7b**，新干净Git worktree `.../git_worktrees/vlm_sft_h09u_26e476e`；旧9b源及P1失败均未修改。executor仍239cb591；prepared仍d4311e88，不重建来源。

07:00:25 BJT 唯一提交GPU1，Python **185102**（launcher185097），输出原root `h09u_reference_task1_v2`及相邻`.log`。新的有效授权 `authorization_h09u_reference_task1_v2.json` SHA **60d4204ab4e384e57b4ab44b1a5b455534ce6a4702a3a9df3f450e3bfab7d5ea**，reviewer `Codex-parent-H09U-identity-final-review`；164prefix+288完整专家+12tail+1hold=465控制、reset后900s、单run80MiB、原root累计384MiB含P1及此前全部失败、保80GiB空闲、0模型/训练、无retry。没有task0或native teacher授权，成功seed仍必须父人工审后才能使用。

启动前64远端测试/.528s通过，source clean；实际完整manifest/8源文件、TRAIN排除、同digest两门flag/hash、安装toggle依赖、官方factory任务名/mode/instance/seed和164×23 prefix逐值全部通过，未在CPU预检构造session。root文件实占92,429,849B、free86,993,231,872B，完整80MiB预留后仍≥80GiB；GPU1空闲、GPU0队友及GPU3父服务不动。已见真实OG初始化，尚未以进程启动或身份修复声称产生seed或有效训练数据。

07:04:44 BJT 已停止并退出：身份解析真实通过，三条private frame均唯一绑定 `ashcan.n.01_1 -> trash_can_116`；实际164prefix+**1专家控制**后关节状态门失败。固定20mrad门测得最大 **43.879509mrad**，在q[10]（左臂第7关节）；grip35.193µm仍在原门内。没有调门或retry。显式final hold真实完成：总 **166控制=164+1+1**，私有账本连续164/165/166；三帧均双手held/contact=false、无新forbidden、IN_PROGRESS，不是GRASP成功。0seed、0native BC、0模型/训练。failure中的165控制和127.983529s是hold前首错快照，最终hold回执166单列；不能把进程退出码或不完整技能当成功。

完整 **18文件6,593,455B**＋log双端逐文件SHA全部一致，capture内7文件hash、完整压缩标定roundtrip也通过。本地 `artifacts/h09u-reference-task1-v2/run`；本人亲看before三原图：目标桶在地面、双手打开，没有已抓取证据。没有故障后RGB-D，不冒称最终视觉状态已认证；final hold私有pose/contact仍留存。末原root99,028,328B（含全部旧失败），free86,971,916,288B；本次Python和launcher均退出、immutable源仍clean。GPU1当时仍报419MiB但无本次PID，不擅自终止任何其他进程。关键SHA/原始计数见 `configs/vlm_sft/h09u_p2_failure_clock_audit.json`。

只读时钟诊断：新旧两次control164的q完全相同；它与source frame164最大18.6833mrad、与frame165最大5.8459mrad。control165的实测q由final_hold命令的当前q保留，与应比较frame165差43.8795mrad、与frame166差5.9124mrad；prefix164条和first expert action23均逐值精确匹配prepared源。这提示必须查原始state/action对齐、控制延迟/工厂时钟，但仅两个实际边界**不足以证明可平移一个frame**。原source/20mrad门保持，未按最近邻改时钟，未向actor灌入未来state。后继应先从原数据导出与安装控制循环证明契约，再有界登记；目前无成功seed，不能诚实量化其native可达性或启动采集/训练。

## H09U 来源时钟审计与最小验证分离（CPU，未新回放）

父07:10:24 BJT另登记≤900s/0reset、模型、物理、训练；读完证据后明确要求来源与物理成功分开验证，不机械统一+1。已核官方原e310/raw11920/instance192完整2402帧：61D state、23D action与G05子集逐值完全相同，float64 state后action字节SHA `09854140…0aec6`。下载缓存固定Hub revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`，30Hz timestamp也一致；不是我们prepare截取错位。

直接读取安装版三层一手实现：`hdf5_data_wrapper.py:118–149,481–541`把reset state与每步post-state收集后按N截断，HDF5 state[i]因而是action[i]前；`data_wrapper.py:377–381,701–744`逐i恢复state[i]、有contact时以1000Hz小步action[i]取观测；`lerobot_data_wrapper.py:307–331,435–458`又把action[i]配前一history观测。实际writer方法AST在无OG/无物理fixture执行得到`(S0,A0),(S0+tiny,A1),(S1+tiny,A2),(S2+tiny,A3)`。该组合**能产生额外一拍滞后**，符合P2观测，但公开metadata未固定生成该数据的导出代码commit，本票没有本episode原HDF5；因此不把当前安装实现当历史导出的完全证明。未更改共享Git安全配置、源或数据。

整段定量而非单点拟合：task1 288对相邻原q，max-joint变化中位7.002mrad、P90 22.554mrad，33对>20mrad；task0 440对中位4.542mrad、P90 23.443mrad，50对>20mrad。仅离线比较command关节目标与source state的lag0/1/2/3：lag2整体较接近，但task1仍28/287、task0仍8/439超过20mrad。这不是实际回放误差、不能据此选最优帧或声称lag2已正确。完整原计数、安装三文件SHA和writer输出见 `configs/vlm_sft/h09u_source_clock_audit.json`。

结论：旧20mrad是我们添加的**导出轨迹逐帧复现门，不是机器人关节安全阈值**；在此来源上不足以作为真实动作失败判据。新最小修改不移动source索引、不重写原manifest/标签：动作23、原实例/seed、完整区间、SHA及实际issued/completed控制时钟继续精确；固定source帧的有符号q/grip差连同每步实际q/grip只写私有账本，不参与成功或动作选择。actual control必须仍等于prefix+原source index，错钟硬拒绝。

真实安全/物理判定独立：每个普通控制前及每条物理measurement检查有限actual q/grip、校准native关节硬上下界（仅1e-5rad数值容差，绝不clip）；实际FK、未知接触、非法新接触、目标身份/held/提起/相对稳定、PRESS因果、12tail、末hold、时间/控制/容量门均不放宽。参考原动作仍不是SafeServo逐步预检的BC，也不是完整环境碰撞认证；原prefix接触未作全程认证的限制不变。新的seed只有真实整段局部oracle成功并由父看完整实际图/ledger后才能使用，源姿态差过大仍须父判断是否已失去目标语义或局部pose可迁移性。失败P1/P2不追认为成功。

67 SFT/.239s、独立分支冻结331 harness/5.197s通过；实际P2已保存control164/165的q通过真实标定硬限但保留18.683/43.879mrad诊断。新增实际runner `measure` AST核：43.879mrad只记诊断、仍IN_PROGRESS，错source时钟/新非法接触继续拒绝；NaN、越硬限、畸形source负例及actor白名单泄漏拒绝通过；原finally首错/hold回归保留。新代码尚未部署或新增reset，需父独立审和新的精确授权后才进行一次参考技能先导，不扩到native采集/训练。

## H09U-P3：验证分离后唯一参考回放（局部成功，seed待独立审核）

2026-09-19 07:25 BJT 父独立审 `affc1e3a3e5f0841d427d422e01d68db323ea9ad` 通过（67/.249s）并合入 `fed20ef`，明确另放行一次 task1/e310/i192/seed0。新不可变源码 `.../git_worktrees/vlm_sft_h09u_affc1e3`，executor仍239cb591、prepared仍d4311e88；新授权 `authorization_h09u_reference_task1_v3.json` 双端SHA **00a798b1c148e7d6f3359f408d77fc44c4cf2059cf5ce2a15282de282594b4a1**，reviewer `Codex-parent-H09U-clock-final-review`。

启动前远端67/.556s、完整来源/排除/8文件SHA、factory164×23前缀、toggle依赖、同digest两门flag/hash全部通过；source clean。原root累计99,030,094B、盘余86,958,698,496B，预留80MiB后仍大于80GiB。GPU1无其他实质任务，但父GPU3模拟器在各卡建立的200MiB上下文仍保留，不擅停。07:27:19 BJT 唯一Python **PID196049**（launcher196046）已启动，输出 `h09u_reference_task1_v3` 和相邻log；07:31已见真实初始化及prefix执行，尚无终态。

本次仍仅1reset、164prefix+288expert+12tail+1hold=465控制、reset后900s、run80MiB、原累计384MiB含所有失败、0模型/训练；不重试、不启动task0/native collector。原P1/P2保留，不因验证分离追认为成功。父H18新FK只读审查与本次运行分离，绝不热改aff源。实际完整账本、全部图像及真实成功seed仍待，本条不把启动或跨过旧source-q门当局部技能成功。

07:34:03.796 BJT 远端真实 `result.json` 生成，**REFERENCE_LOCAL_SUCCEEDED**；Python/launcher随后退出。本例完整465控制=164prefix+288expert+12稳定+1独立末hold，0模型/训练。前452条23D来源动作逐值完全相同；464 ordinary issued/completed与private逐步动作一致，12尾控制保持段末q及最后grip，final hold465保持尾末真实q且没有误OPEN；302条private连续164..465、固定source索引未偏移或搜索。首CLOSE397，首严格局部成功433，至465连续33条SUCCEEDED；目标实际抬升216.067mm、手从close抬升198.447mm，末12条相对手位姿最大变化12.42µm/0.00313°。这些是原专家参考技能的真实局部物理结果，**不是官方完整任务SR、native BC或新VLM效果**。

所有302私有测量无新增forbidden contact、q在真实硬界内，最小边界余159.5µrad；原source289帧q也都在界内，原452动作target有frame446/q16恰在上界但未越界。实际固定source误差最大74.34mrad/开口9.11mm仍保留为诊断，不隐藏动力学偏差。参考回放未使用SafeServo改写专家动作，也不证明原prefix所有场景接触安全。seed始终 `QUARANTINED_pose_seed.json`、`training_eligible=false`，必须父独立看源/当前/过程/终态与账本后另写批准receipt。

完整 **70文件20,782,245B** 和相邻log双端SHA全部相同；13组capture/59个绑定文件、四full capture控制164/452/464/465的q/grip/原depth数组hash全部通过，完整压缩标定roundtrip仍原canonical `b7511abe…65fd5`。本地 `artifacts/h09u-reference-task1-v3/run`；本人亲看全部12张full RAW与27张过程图（另有四页只加相机/时刻文字的缩略拼图 `full_captures.jpg`、`process_1/2/3.jpg`）。可见右手接近地面桶沿、闭合提起、末尾保持，不能从图片单独替代接触/稳定判据。原root累计119,815,902B、盘余86,923,771,904B，仍在384MiB/80GiB门内；GPU1无本次进程。全部关键SHA/计数见 `configs/vlm_sft/h09u_p3_reference_result.json`。

### native起点可行性（只读，不新增控制）

使用成功object-local hand pose、**同control实际目标pose**构造当前goal，world-base由实际手pose与机器人FK交叉恢复（两手推导最大元素差5.39e-7），这些只进离线teacher诊断。control164原GRASP阶段起点需要base系[+4.21,+11.34,−36.73]cm，L1=52.28cm、姿态10.59°；按1cm轴向网格名义54步，现6条/200控制和此前拟议24条都不足，carryTrue禁转也不能消除姿态差。当前DOWN/LEFT/FORWARD三个原native单步通过fresh depth+实际selfboxes+geometryTrue预检，绝不代表整个路径能走完。

额外固定右7关节/左臂和躯干不动、原SafeServo bounded IK/hand-body guard最多64迭代：control164/344/374的最终位置残差分别289.6/24.2/13.2mm；control396仍双手全开、held/contact均false，到目标L1仅20.74mm、姿态4.845°，5迭代到0.418mm/0.120°。这不是离散路径、环境避障或停稳后的可达性证明；1cm网格与4mm近位阈值也可能造成终点无法精确到达，不能据IK成功直接采正标签。

父已在07:45左右另放行≤1200s/0reset/模型/训练的最小CPU支持票：对确认全开无载手才使用原41中3°旋转，平移仍1cm；明确396条**付费专家前缀**，它不算原GRASP phase-start覆盖。未来候选12 native/420新控制/900s/100MiB仍仅建议，必须稳定代码独立审和新物理登记；原240覆盖门、新训练及成对实际效果均未完成。此处以成功seed推进后继，而不拿一个近终点种子冒充足够训练数据。
