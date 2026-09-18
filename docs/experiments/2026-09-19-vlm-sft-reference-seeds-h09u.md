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

## H09U-P1：父终审后唯一 task1 参考回放（运行中）

2026-09-19 06:36 BJT 父逐文件终审和独立60测试通过并合入，明确放行一次task1。06:38:45提交GPU1、固定9b52faa/239cb591；唯一输出 `vlm_sft_native_teacher_20260919/h09u_reference_task1_v1`，相邻`.log`。有效授权 `authorization_h09u_reference_task1_v1.json` 双端SHA `44e2c2cd4131d5081d4cea464886cf652593aa2c0385fe8ecdd0306cb68454f1`，reviewer `Codex-parent-H09U-final-review`。164prefix+288完整技能+12稳定+1hold=465控制、reset后900s、80MiB、累计384MiB含旧失败、磁盘余80GiB、0模型训练、失败不retry。task0与native teacher未获授权。

启动前实际CPU source/factory前缀、TRAIN排除、toggle依赖、两门flag/digest/完整SHA通过；source干净。GPU1无进程，父156556仅GPU3，队友GPU0不动。root实占85,836,638B、磁盘余87,032,434,688B，已预留完整80MiB后仍≥80GiB。Python **PID179879**（启动壳179878），远端实际进程起点06:38:59 BJT，已见OG初始化。当前仅提交初始化，绝不当成seed成功；终态、真实控制、全证据和父手审待。
