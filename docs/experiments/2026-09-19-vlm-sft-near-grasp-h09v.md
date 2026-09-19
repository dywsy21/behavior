# H09V：真实seed后的近抓取教师支持（CPU，尚未物理）

Astra独占SFT；父独占harness/总计划。2026-09-19 07:45:46–08:05:46 BJT，≤1200s CPU、0reset/控制/模型/训练。期间按父要求分离只读审H19（368/5.876s＋24预算和3未知载荷负例），没有合入或热改父harness。部署执行器仍239cb591，原aff P3源码不变。

## 已有实际成果与不可外推边界

P3原专家GRASP完整465控制真实成功；父35da7cd独立看39图并复核完整控制/物理账本，批准**object-local pose seed**，不是native BC、phase-start或完整任务SR。原批准文件按Git完整原字节纳入，SHA `06fbd9aa…436a`；新本地 `validate_seed_release` 已以真实70文件成功执行。原run/seed字节均未改。

父发现跨CPU/BLAS复算同一位姿尾数差8.8818e-16。仅“完整ledger重新计算矩阵 vs 固定seed矩阵”采用atol1e-12/rtol0；SE3/有限检查、outcome完全相等、seed文件SHA、spec↔seed精确相等、review digest仍严格。1e-15复算差过、1e-8拒，spec改1e-15或seed只加换行仍拒；不是物理阈值放宽。

## 最小执行支持

`PoseTeacher(...allow_empty_rotation=True)`仅新near schema启用。原41符号及base坐标解释不改：fine平移始终carry=True/1cm；fine旋转仅在该手真实object-agnostic `is_grasping(arm)`为FALSE、target hold为FALSE、当前无任何外部手指接触、command OPEN在[.999,1]、实际开口与完整开口标定相差≤0.5mm且≥49.5mm、从未发CLOSE时可用3°。不能只用“没抓住指定目标”推断空手。安装robot.py1957–2001明确candidate=None返回any-object状态，UNKNOWN保留；这些新增字段仅在PRIVATE teacher frame，actor白名单不变。

选择器和实际 `native_preflight`两处检查资格；fresh RGB-D/current self geometry与原SafeServo geometryTrue保持，另一手/躯干关节和grip不随active手旋转而改变。**现LocalDepthGuard非BASE不是完整环境碰撞认证**；新旋转仅经已有门、机器人IK/全手-躯干自碰撞/路径/执行反馈检查，不伪称持物或未知空间完整避障。关闭后永远不再借empty模式旋转；其他loaded/unknown情形不放宽。

新near专属授权限定1reset、396专家prefix、≤12native/420新控制含初始12停稳和末hold、reset后900s、run100MiB、原累计384MiB、保80GiB、0模型/训练。旧manual授权仍3例/3候选/200/30MiB/100MiB，默认旋转仍禁用。新授权模板明确false，缺commit/准备SHA等，不能启动。

在动作前保留完整after+settle+trace预算；容量不足不执行下一条，不要求用满12条。按既有raw形状一个macro预留约20MiB，实际三组完整capture通常约10MiB；100MiB可能在12条前触发保守容量拒绝，不能把候选数上限当保证完成数。每条先核完整ticks+12settle+最终hold余量。near新增issued账本，env.step抛异常前已经发出的CLOSE仍锁存供finally保持，issued/实际completed区分并计控制上限；新增真实step AST负例保首异常、无假completion、不OPEN。末hold仍独立尝试、无retry。只有完整local oracle与末hold都成功、整条轨迹后审后才可能释放BC；所有当下record仍隔离。

## 来源绑定与真实CPU准备

`native_teacher_near_grasp.py`不读/复制大数据集：从已固定d4311e88原完整TRAIN参考的8文件核SHA与实例排除，重建 `prefix164 + segment[0:232]` 得396×23，逐frame选择original_demo/low、同GRASP/原164→452边界；高层重复不覆盖，选中分支冲突拒绝。当前reference frame为396，但原skill_start164保留，保证批准seed身份不被假改成新技能起点；prefix末双OPEN必须实际为[.999,1]。来源图/未来动作仅教师参考，不投影给actor。

本地真实准备 `artifacts/h09v-near-grasp-prepare-local-v1` 78,438B；manifest `4b3cd89a4093bb4eb0ec8b56a94517c734b6e20293d3f2a50538c6b65896819d`，prefix SHA `f92baf548a56b651e13e99f3519a6a5b1df98177fefde4c74dc1012a5f4a1447`、shape396×23、末grip[1,1]；verify重新从原source推导逐值相等。window完整字段（task/mode/instance/seed/max_steps817/前缀路径与SHA）也严格重建比较。实际源训练排除保持，未读静态test来优化。远端准备必须在新Git不可变源用远端绝对output重新生成，manifest SHA会不同，不拿本地路径hash冒充远端。

CPU入口：`native_teacher_near_grasp.py --reference <原h09u_reference_prepare_v2/task_1> --output <全新h09v_prepare_v1> --start-control 396 --reference-manifest-sha d4311e887582e3259a835f83664e02126ea9abc8df69ff45d6a6046ae95317f2 --counts <固定source/configs/vlm_sft/h09r_train_feasibility_counts.json>`。随后spec用父批准seed的精确goal_pose和pose_evidence SHA、原GRASP语义与reviewer；active release必须另绑定spec SHA、远端prepare SHA、当前commit、原两安全门与预算，再走原 `native_teacher_collect.py --teacher-config ...`。本票未生成active授权或调用collector。

## 6307220时已识别的离散几何门（历史，后继proposal见下）

P3控制396的连续right-only IK可行不等于41符号路径可行。其初始goal位差[3.650,15.082,2.004]mm，理想1cm格点的最近误差仍**约6.45mm > 原4mm接近门**；3°旋转不能改变EEF位置。实际12tick停稳后的状态尚未获取，不能假设误差恰好减小。现compiler只给严格降低cost的动作；网格局部最小返回空列表→停止，另有同pose最多4次/12primitive上限，绝不来回无界刷、降低4mm或把close/距离当成功。

因此此实现可审，但**不建议直接用它赌396停稳后恰能进4mm**。最小后继应单独审“通用候选抓取区域内允许有限一次close尝试”的proposal条件：几何条件只决定尝试，不是正标签；实际指定目标接触+独立提起+相对稳定+末hold仍是唯一局部GRASP终判，失败整轨迹零正BC。需要真实抓取区域/开口/目标几何与尝试预算依据，不按这一实例调4mm数值。本票没有实现/授权该更宽proposal。

原240条/6意图与跨实例/近静止/phase-start覆盖门不降低；一个paid396近终点先导只验证局部teacher接口。后继仍须可迁移真实采集→新2B≤400steps→base/FT有限同预算闭环，不重启旧H09负配方，也不把此CPU支持当训练效果。

父另明确：上述旧门继续约束“广覆盖”结论，不作为所有窄问题永远不能训练的理由。若新near先导物理通过，可另行**前瞻登记GRASP-only停顿纠正实验**，独立实例组划分、有限新SFT及同预算base/adapter物理对照，只对该局部范围给结论，不冒称六类覆盖/原起点SR。当前尚无native完整GRASP轨迹，不能以P3 expert seed或两条手工缩距标签填数；后继具体样本数和预算须在真实先导产率已知后登记，本票未放训练。

最终CPU：75 SFT/0.290s、冻结分支331 harness/5.186s通过；新7+1组涵盖object-agnostic未知/数字0/接触/开口/命令latch、默认旋转禁用、base系3°/平移1cm、前缀真实重建与实例排除/close latch、near预算及false几何flag、真实step异常已发CLOSE保载、不可达网格停止。另真实P3父review helper与396准备完整核验通过，executor digest仍239cb591。无新reset/模型/物理/训练；准备78,438B，远端尚无新prepared或运行源。

## H09V-G：通用格单元的一次尝试（仅CPU增量，待独立审）

2026-09-19 08:08:59 BJT开始，≤600s CPU，0reset/控制/模型/训练。父已独立审630全增量、75测试及真实near396来源/release通过；本块仅追加教师proposal和负例，未改共享harness、原aff运行源或既有数据。Git已fetch并在干净独立分支ff-only同步；origin/main仍33677bd。父单独维护共享plan/TEAM。

旧4mm且角差≤4°的精确到位分支完全保留。额外的 `allow_grasp_cell_attempt` 默认False，只有新near授权显式True可启用：GRASP、此前从未CLOSE、真实已知空手与全开证据全部满足、角差≤4°、当前位置差≤1cm正方格单元的半体对角 `sqrt(3)*0.01/2=8.660254mm`，并且**全部6平移＋6旋转的未过滤几何cost都没有严格下降方向**时，才可提出一次CLOSE。阈值来自通用离散动作格，不按该实例调，不是扩大原4mm分支或物理成功门。

几何邻居先于空手资格/IK/深度/执行筛选计算，所有12项及当前cost/位置差向量/角差落入 `PRIVATE_proposal.json`；`request.json`只在actor外的私有字段引用。采用直接严格 `< current_cost` 检查，而不是原动作排序的1e-6下降幅度：即使方向只改善2nm、排序返回空，也不得CLOSE。安全门将所有可改善动作否决时，实际collector直接失败；不能把“无安全候选”改写为“几何局部极小”。一次格单元CLOSE提案已经锁存，未执行/被否决/中断后再请求都停止；已执行但未建立真实指定目标接触仍沿原no-retry失败。其他verb不使用格单元分支。

这是一次**尝试许可**而非正确性证书。只有真实目标身份接触/held、实际抬升、连续相对稳定及最终hold全通过原LocalOutcome，再经父整轨迹后审，才可能将该段作为BC；仅CLOSE/缩距/几何局部最小都不能产生正标签。当前0条新native完整GRASP、0新训练。旧P3仅pose seed；未触及原六类240覆盖标准或另行前瞻登记的GRASP-only实验边界。

本块验证：81 SFT/0.325s、原冻结harness331/5.032s通过。新增6组涵盖一次提案/禁止重试、未知或持物/未全开/非法flag/其他verb、未获旋转资格但仍存在几何改进、2nm改进不能被排序容差吞掉、越界/角差、旧4mm路径不变、AST实际collector深度否决后不追加CLOSE、私有字段actor泄漏拒绝及只闭爪不抬升不成功。额外固定seed190919生成800个CPU几何状态，直接执行Git固定630旧selector与本版默认cell=False，对两种rotation模式完整候选顺序逐项相同；不是物理/神经运行。准备/seed/schema来源与41动作幅度未改；fine平移carry=False/True均为1cm（coarse才3cm），本实现仍只对明确空手旋转采用False。inactive模板仍 `authorize_collection:false`，尚无新部署或物理授权。

## 08:17:55 BJT起：新不可变123837d远端准备（≤300s，0物理）

仅显式fetch并创建 `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/vlm_sft_h09v_123837d`，完整commit `123837daf73d5a5aff2667b2a03da482b2f4b88e`、clean detached；原aff/父0d均未pull或热改。远端81 SFT/0.636s＋331 harness/13.605s通过，运行digest仍 `239cb591f178099f20e9a9ba6d7cd3ce04aa5ccfe183840ebe1ff78ac2f40b9b`。

原experiment root下新 `h09v_near_prepare_v1` 共78,412B（远端绝对路径与本地不同），manifest SHA `b146d41baa3329657c609886fa1fdffad78696f76abf3ee82f7a99a68161d400`；prefix仍 `f92baf548a56b651e13e99f3519a6a5b1df98177fefde4c74dc1012a5f4a1447`，reference `9c59cfcea4a38621c14168edcc65571d9cad83f59112db0e26cfe2d22c705bb1`，window `5cc1aa9f3d2b95bdb007c5d612a50c3fe51dcb23d647dd99bcace0ff4348381b`。真实factory加载 `picking_up_trash/train/192/seed0`、396×23前缀逐值相同、末双OPEN、max_steps817；原skill_start164和reference_frame396保留，前后重核全部来源SHA。只调用window loader，未构造session/reset。

spec SHA `6c2cccb811b51db3f388f853b02aa6ed1757415a8494511f630a41a559765217`；实际P3seed `bc8865c719e84ab73b43bb2f0ba7cbaa975ce0bf1612ab1053f6073804cf4b7d`、父review `06fbd9aaf34ecf60613872084835cbe627291a603858522211ef0fa0b3a0436a`，完整physics/图像/evidence/recomputed-pose release helper通过。两H15r1门flagTrue/239digest和SHA分别2460d102…35e19、b7117a5e…7f75保持。

只生成 `authorization_h09v_near_v1.INACTIVE.json`，SHA `1dbaf6b9b0da2b5b64dfacd445f2f6f6c0b87b70b35c50782d26103ed33ef38d`，`authorize_collection:false`，实际require_release明确拒绝；reviewer仍未填，不能执行。原始CPU回执 `cpu_preflight_h09v_123837d.json` SHA `5b9de636f9bfb332d8dc435b3ef0bf1afc8916df13dd84eca5d00afd88995873`，本地完整小副本在 `artifacts/h09v-near-remote-preflight-123837d/`。08:20:00 BJT核原root文件累计119,896,803B（另CPU回执少量字节随后落盘），sdc1余86,878,801,920B、NVMe余2,936,491,474,944B；完整100MiB预留后仍保≥80GiB，旧失败全部计入原384MiB。GPU1在部署前0MiB/无进程，父GPU3服务与GPU0队友不动。尚待父独立增量审查/另一次明确物理放行；0新reset/控制/模型/训练。

## 08:27:09 BJT：唯一H09V原生近抓取先导（历史启动，已容量停止）

父08:25已独立读固定123增量/实际collector，81/.286s＋600额外几何边界过（13 CLOSE提案逐核）并明确放行一次。准确runtime仍123837d/239，新run原root下 `h09v_native_task1_v1`，Python PID213880/GPU1；启动命令与GPU/时间落 `launch_h09v_native_task1_v1.json`，日志相邻 `.log`。只用396精确专家前缀＋≤12原生primitive/420新控制含初12停稳和末hold、reset后900s、100MiB run/原root含失败384MiB/余80GiB、0模型/训练、无自动retry；不是phase-start、官方SR或VLM效果。

新active `authorization_h09v_native_task1_v1.json` SHA `08162a7213da6dc66e6b905b509c772fdc3c69b164eb2ab8aee042d59b5430d1`，绑定123源/b146准备/6c2spec/06f父seed-review与原两安全门；authorizer `Codex-parent-H09V-123837d-independent-final-review`。原inactive与所有旧源/失败不改。启动前完整require_release/factory/来源/种子/预算再次通过；free86,878,404,608B、原root119,899,132B，GPU1仅父模拟器跨卡209MiB上下文、80,943MiB空，不停止其他进程。此处只记录真实提交，完成与实际抓取证据仍待；所有新record继续隔离，父亲自看图/物理账本后才可能放BC。

## H09V真实终态与完整证据（08:36:45确认退出）

第9个macro开动前，完整后验预留被100MiB run门拒绝；原failure wall412.484551s为reset后首错时长，非初始化全过程。PID213880退出，无retry。8个macro全部TARGET_REACHED：LEFT、YAW+、LEFT、ROLL−、YAW+、CLOSE、UP、UP；前6个各18控制，两个UP分别28/22控制，每macro另12停稳。396专家prefix逐值相同；266普通native＋独立末hold=267新控制，总663，所有ordinary issued与completed完全一致。最后右grip−1保持，无OPEN。没有result成功文件、零完整GRASP正BC/新训练/官方SR。

CLOSE提案误差6.308293mm/2.355°，全12未过滤邻居无降cost、三轴残差均≤5mm，符合新一次性格单元条件。首CLOSE实际tick559；目标identity held＋contact从573..663连续91帧。两次上抬手实际world-z位移9.642571/9.611607mm，停稳后目标相对初态上升9.433657/18.665373mm；末hold目标升18.407673mm、手从首次CLOSE升19.046932mm，未达到原30/25mm门。末12帧相对手位移变化仅2.9198µm、角变化0.0009746°，但不能替代缺失的抬升证据，最终仍IN_PROGRESS/stable_ticks=0。全部256连续私有状态中0新forbidden；255非初始oracle回执逐项重算一致。

完整本地 `/home/wsy/behavior_worktrees/vlm-sft-native-teacher-20260919/artifacts/h09v-native-task1-v1/complete/h09v_native_task1_v1/`，246 run文件89,949,654B＋3附属文件均双端SHA相等；25套/75 RAW图、75 depth数组hash全过，相同控制时刻的capture q/grip与PRIVATE行差0。teacher_00..07有before/after/after_settle全图；teacher_08只有before(native266)，其后hold267只有真实PRIVATE状态，不能冒称有末hold新RGB。本人看初态3RAW、02/04/05/07停稳head及05/07右腕，支持桶沿闭爪后随手小幅抬动，不单凭图认定阈值成功；父另亲看11图并独核完整账本。8份record全保留隔离，容量终止不否定教师方法，也不追认成功。

关键SHA：failure `598dc79e38d3b88690ba8addb03a4d71292a19b840a30f3eb39fdcbaf82ab08e`；final_hold `bc761a5b24b68e0d4d4e2922fc3060bfad59ae7e018e32555e5d2560281fc237`；native/issued两个文件同 `dc69f268164c6f49335d7b88eac1ae38210488689adf27709389357c3a209513`；PRIVATE trace `06509316aa6a031cf880cb60e07277468d46dc159098cf970fb8c7de4e4638d8`。完整清单 `h09v_native_task1_v1_hashes.json` SHA `f0ccdda21bb3b850f5cc00031322df8bbde98c06f450693b5033134451eb7c66`；本地原数值审计 `h09v_native_audit.json` SHA `dbbc009663378e6480e656bf8344c9c7c790e063a1b88d3b63493a529ba7ffed`。

实际每capture 3,374,422–3,446,200B，每完整macro 10,208,914–10,317,697B。8对相邻after_settle/before只有depth.npz/proprio/selfgeometry三文件字节完全相同，RGB与sensor/capture收据不同，不能声称整capture同hash。当前格式单capture保守上界9,276,026B，每macro后验预留19,879,156B；12个完整macro＋396prefix账本＋4MiB清理/附属余量＋已知完整标定压缩件的上界358,805,493B（约342.18MiB）。这以固定传感器形状及已核标定为条件，不以本次压缩率保证所有未来输入。父下一H09W仅CPU容量profile票，见[容量跟进](2026-09-19-vlm-sft-capacity-h09w.md)，原run/100MiB授权/123运行源不改。
