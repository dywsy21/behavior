# H09S：停顿态原生教师的最小先导

2026-09-19 03:36–03:56 北京时间，Astra/max 独立工作树；父代理拥有 harness。预算 ≤1200s CPU、0 GPU/模型/控制/reset/训练、新增 ≤20MiB。基点 `33cfa33`，已 fetch；未 pull 或修改父活跃源码。H09 真实训练及四负闭环、H09R 数据门全部保留。共享 plan/TEAM 由父代理记录，本分支不编辑。

## 本块交付与实际结果

### 04:13 后独立审查修正（CPU，尚未采集）

父审发现初版未实际开启 H15 安全开关，且仅绑定 reference、未绑定窗口和前缀全部字节；**初版 7f4ccb3 不得用于采集**。本次 ≤900s CPU/0 GPU、模型、控制、reset、训练修复块从干净独立分支显式 fetch 并 merge `b6f0845`，再同步父固定手 API 修复 `5cfbb89`；不改父工作树或活跃源。

修订后 `SafeServo` 强制 `ServoLimits(robot_geometry_guards=True)`，`LocalDepthGuard` 使用与该次 RGB-D 同一不变 q/指口区间捕获的真实机器人盒。审批等待不推进 physics，执行前再核 q/指口和控制时钟；试算回执记录开关、深度门与绑定观察。两工程门必须 `gate_ok is True`、`robot_geometry_guards is True` 且完整 executor digest 相同；旧 H13 门或 H15 初始化失败门不能放行。

授权固定 `preparation.json` SHA `415d5e96eb85290c9433c61052712d32e2644f3d4d60460930b35635ad876873`，进而校验窗口、完整 prefix.npy、teacher reference 原字节和实例/技能首帧；前缀路径必须指向该准备目录。官方 factory 装载后再核任务名、train/instance/seed、全部前缀数值和文件身份，均在创建 evaluator/reset 之前。原 `prepare_v1` 不修改、不重建；本地媒体副本不能冒充远端绝对前缀路径。

每个 before/after/after_settle 保存三 RGB、原始三视图 `depth.npz`、`robot_self_geometry.json`、传感器回执和 `capture.json`。后者绑定全部文件 SHA、depth 数组 SHA、完整机器人 q/指口及 prefix/native 控制计数，不进入 actor。图像/深度按实际编码字节在落盘前检查每 run 30MiB，沿用总100MiB与清理余量，不扩大预算。前审 request 绑定 before capture；记录仍全部 quarantine，人工后审和 H09R 240条覆盖门不变。

新增负例：旧/false-flag 工程门、错误 digest、窗口/参考/前缀字节漂移、同形状错前缀、错任务、畸形当前几何及捕获中 q 漂移；正例验证真实安全/保载传播、深度回读/hash。33 项 SFT CPU 测试通过；首版29项为历史。父已看3任务×3视角×0/8/16帧共27源帧，仅确认原专家参考可审，不是对尚未发生的停顿态批准。

04:23 真实 CPU 收尾：合入 `5cfbb89` 后完整331 semantic/5.153s及33 SFT/.063s通过；固定手 API 单独6项/.016s通过。`6c9034f` 的来源验证代码经 Git blobs 在远端 CPU 执行，0.2445s 完成原3个准备目录清单/窗口/前缀/参考核验与真实 factory 的全部4807控制数值比较，均通过且未导入 omnigibson，0 reset/控制。最后留证补充模型 calibration digest，并将单个 primitive 的实际控制条数与后置12步 settle 分开记录，不改变执行。执行器 digest 保持239cb591…；旧准备数据完全未改。所有新增代码/报告仅在作者 worktree，父共享计划由父同步。新物理/正标签/训练仍为0。

使用 `configs/vlm_sft/h09s_authorization.template.json` 填真实最终 collector commit、**新通过**工程门绝对路径和 reviewer，再由负责人显式授权；模板默认 false，不能直接启动。父 API 修复执行器 `5cfbb89` 预期 digest 为 `239cb591f178099f20e9a9ba6d7cd3ce04aa5ccfe183840ebe1ff78ac2f40b9b`。最终 review、新同源双门和 GPU1 交接前不运行。拟议命令如下（没有执行；task_0、task_1、task_3 分别显式运行，逐个完成审核，无自动批量循环）：

```bash
env PYTHONPATH=src /mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python scripts/vlm_sft/native_teacher_collect.py --prepared /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/prepare_v1/task_0 --output /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/pilot_v1/task_0 --authorization /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/authorization_v1.json --gpu 1
env PYTHONPATH=src /mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python scripts/vlm_sft/native_teacher_collect.py --prepared /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/prepare_v1/task_1 --output /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/pilot_v1/task_1 --authorization /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/authorization_v1.json --gpu 1
env PYTHONPATH=src /mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python scripts/vlm_sft/native_teacher_collect.py --prepared /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/prepare_v1/task_3 --output /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/pilot_v1/task_3 --authorization /mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/authorization_v1.json --gpu 1
```

以下保留首块准备记录，不将本次代码修复写成新训练效果。

### 04:30 条件先导：服务器准备完成，等待显式放行

父已独立审核最终 `1509b7ae38c3d49d6a025cb689a75c64e1b22ff2` 并合入，04:28 条件登记三实例先导。本块只做 ≤600s CPU 准备：显式 fetch 后建立干净独立 Git worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/vlm_sft_native_1509b7a`，实际 runtime commit 为1509、digest239cb591…；后续仅文档提交不替换该不可变源码。robo 33 SFT/0.182s、331 harness/13.694s通过。实际部署源核原准备清单、3窗口/参考及全部4807前缀和官方factory任务绑定，0.0686s通过，未导入OG、0 GPU/reset/控制/模型/训练。

证据在 `/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/preflight_1509b7a/source_preflight.json`；同目录 `authorization_inactive.json` 的SHA为 `2ff9e14a6643fbfa03d42b00cbf37ea0ff2e3271073478ac091c337668d42976`，`authorize_collection=false` 且实际被 require_release 拒绝。余盘89,167,646,720B，原prepare_v1不改。两个小文件本地副本在本worktree `artifacts/h09s-runtime-1509b7a/preflight`。**尚未启动collector**：需父完成新H15双门/首图/动作审阅与退出核验、停止旧132309并交接GPU1，再显式放行。逐实例等待父人工教师就绪，不自动启动下一例。

### 04:35 task0 唯一先导运行中

04:34 父完成两新门退出/实际链及首图审阅，确认旧132309退出并独占交接GPU1；显式只放行task0。`authorization_first_three.json` SHA `1348181d7f11112f74df7a82650a43c1a12dbea4b2b0948ec81c8e921d72449c`、gate开关及digest均在启动前核验。04:35:24 BJT提交PID156956，runtime仍1509，run=`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/pilot_v1/task_0`，日志同级task_0.log，launch记录同级task_0_launch.json。启动前GPU1空；TRAIN70/e66/f1170、单次reset/1170前缀/≤200新控制/≤900s、≤3人工候选/每次120s，0模型训练，预算不扩大。目前仅初始化，未称reset/停顿/采集成功；task1/3不自动启动，原始记录仍全部隔离。

### 04:40 task0 基础设施失败停止；04:42 有界 CPU 修复

task0已经退出，并非仍运行或成功采集。标定JSON **47,345,882B**（包含完整底盘视觉网格）超过每run30MiB，旧1509仅在前缀/settle后检查磁盘，真实规模未被先前小fixture覆盖。发现后核唯一PID并SIGINT停止；记录304条已完成专家前缀、0教师动作/候选/标签、0模型训练。OG自己的SIGINT handler直接shutdown，**无result/failure/final_hold回执，不声称safe hold已执行**；信号时可能在途的step不能补造完成计数。一reset已消耗，task0不重跑，task1/3暂停。原失败47,511,519B全保留并计入累计100MiB，不能换目录绕过。

失败证据 `pilot_v1/task_0/budget_operator_stop.json` SHA `158970e4607fa05ce33c0ed8f5cbc1230c5967347a8ccffefcbf1cf4c42e814e`，trace SHA `b0bac7f5f53e4de5c5a2d7d7da49f26ea2d5604bf8098a75d001d0ac8e417882`，标定原字节SHA `bbf369bc6ec7cb6d0fcfd35cffd8465790e5fd39857f609a8cbf6138c0a35fd6`；轻量文件及日志本地 `artifacts/h09s-runtime-1509b7a/pilot_v1`。这不是PRESS动作失败或SFT效果证据。

父另登记≤600s CPU/≤20MiB派生、0物理/模型/训练修复。在线控制器始终使用完整内存model，没有按robot_calibration.json路径回读依赖；仅将持久化改为 `robot_calibration.json.gz`，保留全部字段，另存raw字节SHA、gzip SHA、原canonical model SHA与两种大小；`load_calibration`可验证并还原完整模型。真实失败JSON只读压缩初验3,127,231B/0.501s、解压逐字节相同，未删除旧文件。

`ArtifactBudget`在所有正常二进制/JSON写前核字节，prefix每step也先为trace预留空间，避免先超额落盘再浪费前缀。每run最多29MiB正常数据+1MiB清理，仍≤30；总root由SHA绑定的prepared位置确定，递归包含旧失败/日志/准备/授权/派生等，最多97MiB正常+3MiB清理，仍≤100。目录改变不能绕过；少于预期样本即停止，不放宽容量。标定写入前先建立实际reset夹爪latch，以便预算异常进入原finally保姿态/保夹爪hold。36项CPU测试通过，包括大mesh无损往返、原始超限文件写前拒绝、跨run累计、以及实际collector初始try/finally的预算异常注入：0prefix、1hold、错误和hold回执均保留。真实大标定最终writer/loader与归档核验见后续补记；物理仍暂停，旧1509源码不热改。

### 04:55 修复证据完整归档；新固定源部署完成，未新采集

修复 runtime 固定 `68daccaa55cbd23002a19711226c20230a4ef22b`，独立父审36 SFT/0.197s及331 harness/5.057s通过后合入。真实失败标定在新 writer/loader 中4.640s完成：原47,345,882B写前拒绝且无文件，完整gzip3,127,231B，解压逐字节及模型全字段均相同。raw SHA仍`bbf369bc…35fd6`；gzip SHA `1d6d77250c010bad7b99940c19e78761d188007e302c998f2cfbdf79be3eca07`；canonical model SHA `7426d08fef24cc7e938f0ae4d80041db5c9b8c6faffc3ab42a014e9336603414`。这验证存储修复，不是新物理安全或标签正确性证据。

完整原失败目录、日志和launch压缩归档为`artifact_budget_fix_68dacca/task0_failure_full.tar.gz`，3,167,823B，SHA `94c4f14d70d427db1926ba0b7460126a4c387a8a50d464ec5d3701bb95f53952`；原四文件大小/SHA及304prefix/0native计数在同目录`validation.json`。远端根仍`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919`；完整派生目录已下载到本worktree的`artifacts/h09s-artifact-budget-fix-68dacca`，归档和gzip SHA双端一致；未在本地展开47MB原标定，原远端证据未删除。新增派生约6.30MB，含原失败的全root累计约54.62MB，仍计入100MiB总限，剩余普通写额度约47.09MB，不能将剩余两例各30MiB都视为可用。

另获≤300s、0reset/模型/控制/训练部署票：显式Git fetch并新建干净不可变`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/vlm_sft_native_68dacca`，robo36 SFT/0.443s通过，未热改1509。真实原清单SHA415d5e96…及全部4807前缀通过官方factory逐值绑定，CPU0.379s、未导入OG；旧1509授权被新源拒绝。证据`preflight_68dacca/source_preflight.json` SHA `cba2ef7d3925dcbec3845d94ec8f46e0122cfcb8c4d61224d9d3f1f3994f904f`。executor digest仍`239cb591f178099f20e9a9ba6d7cd3ce04aa5ccfe183840ebe1ff78ac2f40b9b`；GPU1只有约214MiB上下文占用、无本票采集进程，盘余88,422,387,712B。**task1/3未启动**，等待新授权SHA及父逐例人工放行；task0不重置。后续文档commit不替换已审runtime身份。

### 04:59 task1 原剩余单次先导启动

父已读实际writer/loader与部署回执、独立审查通过并显式放行仅task1。核新`authorization_remaining_two_68dacca.json` SHA `599298a34957c3f971ae2cbae2597baa56b71f71a7c22803ca268c49f0178795`与原清单后，04:59:09.866 BJT提交唯一PID163349/GPU1，run=`pilot_v2/task_1`，日志/launch同级。源固定68dacca，TRAIN192/e310/f164、164专家前缀、≤200新控制（含停顿/最终hold）、reset后900s、≤3候选、每次人工等待120s，0模型训练；每run30MiB/global100MiB计原失败不变。启动时盘余88,355,119,104B、pilot累计54,623,709B。当前仅初始化，非采集通过；父承担当前及前后人工教师，所有记录隔离。task3未放行、task0不重置，遇基础设施故障暂停remaining。

### 05:09 task1 结束；05:11–05:21 仅CPU容量、吞吐和教师路线审计

PID163349已退出。真实164prefix＋91native（54候选、36停顿、1最终hold）=255控制，0模型/训练。三个独立审批均在120s内落盘，文件时间差93.217/95.497/112.310s，三次各18条实际向量与总trace逐值一致。前两RIGHT_LEFT达到目标并各停顿12步，右EEF侧移9.543/9.549mm；父亲看各before/after/after_settle三视角、源和轨迹后分别后审通过。两record digest为`e44140ed…05d9b`/`91982752…48089`，两post canonical digest为`04867711…19b87`/`d3e6ec2e…3c30`；实际`release_reviewed_record`再次通过，保存为`artifacts/h09s-runtime-68dacca/reviewed_task1/`的两条候选。它们仅同实例/同GRASP意图/同RIGHT_LEFT，不是抓取成功，也不足任何新训练发布。

第三18条已执行，但在`sample_02/after/robot_self_geometry.json`写前容量拒绝；其after仅三RGB和depth，没有完整capture/后停顿/record/持久化finish反馈，**不补造TARGET_REACHED或第三正标签**。failure首错误保留，reset后425.513s；`final_hold.json`真实completed=true、native91，零base、双grip1。第三原审批仍保留。run实际30,390,700B（28.983MiB），未越30MiB，正常29MiB余18,004B不足再写约31KB自体盒；写前拒绝与清理余量按设计生效，但没有事前为整段后验预留空间，导致真实动作缺完整证据。task3暂停，不重置task0/1。

83个run文件和最终log已全量下载，逐文件SHA/大小均匹配远端：本worktree `artifacts/h09s-runtime-68dacca/pilot_v2/task_1/`，相邻log/launch；清单`task1_evidence_cpu/final_artifacts.json` SHA `314d9fef76896077f20cd022858ac3affa7bf86b0b0e35a99fa300abdfe0d6af`。旧失败仍计入全root85,022,736B（本审计派生前）；97MiB普通额度只剩16,689,136B，绝不把历史超限追认为合规。本票只新增小JSON/报告，无新控制/reset/模型/训练。

每个完整capture实测3.348–3.429MB，其中三RGB约1.07–1.09MB、depth.npz约2.24–2.31MB，自体盒约31KB。两次`after_settle→next before`的时钟/q/grip完全相同，**depth.npz、proprio、selfgeometry三个文件逐字节相同**，共重复4,637,920B；RGB三路数组全不相同，sensors/capture也不同，不能称整帧重复。可做无损内容寻址复用完全相同的payload，并保留每观察独立时钟/图像/引用SHA；或另审“0physics且q/grip相同才直接复用已拍快照”的接口，但本票均未实现、未改旧证据。下一采集器必须在动作前预留整个after+settle回执的保守容量，不只保证单次write。

时间证据：标定receipt至首before=65.048s，涵盖FK检查＋164prefix＋12settle＋capture；两次实际12settle+capture为5.347/5.423s。扣其均值得prefix代理速率0.36380s/control，3473前缀约1263.5s，另有标定约12.83s及动作/审核。**这不是逐tick计时，task3场景吞吐可能不同；但不支持900s可行，更不能靠先reset试运气。** 文件原时间与计算在`task1_evidence_cpu/capacity_timing.json`。一次有现实余量的后继应另登记同未用task3单reset、最多1个完整候选、reset后1800s及按实测整段预留至少24MiB；连同原85MB证据，累计上限需前置重定为例如120MiB，旧失败仍标失败。此为未批准建议，需代码/授权显式绑定新预算与整段预留、独立审查；现在不改源、不执行。

#### 可到新训练的唯一建议：特权离线教师完成局部技能，再抽审完整轨迹

可以用仿真特权几何/原专家作**离线教师**，不必240条都在线人工审批。教师使用同状态对象/容器/按钮元链接、原同技能轨迹的对象相对路点作提案，逐次从同41符号选动作，并经完全相同native安全门真实执行。距离或专家路点误差只作搜索代价，**不能直接发正确BC**。只接受从未完成到真实局部完成、保持任务支撑/载荷约束、无未解释接触/跌落且前后证据完整的同意图轨迹，再分层人工审图/稀有类别全审；失败整段隔离，恢复若需要必须有新合法意图及实计控制，不把失败原意图配另一技能动作。对象ID、隐藏位姿、未来路点、谓词结果只在教师/出处字段，现actor投影不变。

本票读到的真实API及边界（未新仿真验证）：安装`object_states/toggle.py`的ToggledOn带togglebutton元链接，真实手指接触和marker重叠连续5更新才翻转，可用**原生false→true及保持**作为PRESS候选终判，绝不用set_value写成功；`inside.py:240`读取的是对象中心落入容器体积，不能单独证明整个物体/盘上食物被稳定放好，需释放后连续物理更新、目标支撑/内容保持与载荷约束；`robots/robot.py:1957`在physical模式用ControllerView状态，实际实现另核candidate接触（其“忽略candidate”注释已落后实现），仍不能独自证明目标稳定持有，需目标身份接触＋真实小幅提起/迁移后物体相对手稳定，不能看闭爪自报。H14已核RigidContactAPI可读碰对/物体pose，但覆盖/caching仍需沿已审contract，纯接触亦非成功。

旧sim_runtime的`g05/utils/memlite_native_fm_contract.py:569`确有`evaluate_installed_subgoal`，动态依赖`PhysicalSkillOracle`；有UNKNOWN/IN_PROGRESS/SUCCEEDED、因果完成及placement streak接口，明确不是官方整任务SR。**当前68源并无该模块，不能把旧adapter存在当即插即用且全部谓词可靠**；后继要固定其实现SHA、依赖和各意图正负例，校验GRASP身份/保持、PRESS因果翻转、PLACE支撑/释放后稳定，再只接教师侧。现同41符号能否在这些起点完成操作也未证明；特权几何不能绕过IK/碰撞/持物约束或瞬移状态。

有限路线（均尚未授权）：先独立CPU实现/审查教师与三类终判，随后一个三类局部技能先导，最多3来源reset、12000prefix、3000native、3GPU小时、300MiB，不branch restore/重跑；每类至少有一个完整技能成功且人工审整条，任类不能完成就定位并停，不直接做600条。只有此门过，才登记12 TRAIN来源（每task4）＋6全新按实例留出来源；排原5%全实例/H09留出/138/242，预先核总前缀≤30000；≤18reset、20000native、600条候选、单GPU≤10小时、≤6GiB完整证据。按当前代理速率prefix约3.0h、20k native粗估3h、600条多帧存储/初始化/教师计算另留余量；这些是上限而非成功保证。旧6小时/1GiB建议已不适合保留完整RGB-D证据的真实吞吐，必须重新登记，不能沿用。当前盘余约82GiB无法直接新增6GiB且保80GiB；需父先明确可用数据盘/已核可归档空间，不能删除旧证据凑预算。

达到原240正确native准入门（GRASP/PRESS/PLACE各≥40、各≥4来源、各近静止≥20/阶段起点≥10、CLOSE/OPEN各≥20并各≥4来源、BASE≤20%）后，才用现2B另起adapter≤400更新/1GPU小时；不续写旧H09。相同新来源协议base/FT静态≤300调用＋3异质起点配对共6短闭环、每例30决策/960控制/600s，报告局部物理指标与全任务SR分别。人工可改为按意图×动作族×停顿/起点抽审且稀有格全审，仍保上述数据覆盖/完整轨迹正确性门，不用两个同方向样本支撑泛化结论。

实现的是 **人工当前状态教师的可执行采集流程**，不是把混合专家动作重新解释成纯动作：`native_teacher_prepare.py` 做严格 TRAIN 来源与前缀转换；`native_teacher_contract.py` 编译只供审阅的方向候选、绑定人工审批与隔离记录；`native_teacher_collect.py` 可在停顿后导出三 RGB，零 physics 等待审批，执行已有 `token_to_action`/`SafeServo`，导出实际 23D 轨迹、立即及再停顿后的图像，默认全部 quarantine。没有新的 VLM、自举标签或自动正确性证书。

初版 `973ba76` 已由 Git 同步；远端 CPU 直接执行该提交的 Git blobs，复用 `semantic_joint_90a7c20` 中未变的 common/live/prepare 依赖，未创建或热改运行源码。真实准备 **3.353s，799,441 字节**，含三个前缀、9 段三视图原专家 17 帧视频和完整定位/来源回执；0 模型/控制/reset。目录：`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/prepare_v1`。后续修订增加稳定停顿后图像、近静止 EEF 门及保守未知负载约束，不改变已准备的数据。29 项 SFT CPU 测试通过（原21＋新8），新采集器尚未在 simulator 中运行，不能称端到端采集已通过。

03:55 收尾：三个生成窗口及所有4807条前缀已经真实安装的 `load_official_oracle_window` / `frozen_window()` 逐一装载通过，确认未导入 omnigibson、未启动物理。本地完整小包为 `/home/wsy/behavior_worktrees/vlm-sft-native-teacher-20260919/artifacts/h09s-prepare-v1`（844KiB）；3前缀SHA、9视频SHA及全部17帧解码通过，可直接在各 `task_N/expert_{head,left_wrist,right_wrist}.mp4` 人工审阅。还没有任何当前停顿图或正标签，不把源视频审核当采集审核。

| task / episode / TRAIN instance | 同源技能首帧 | 前缀控制 | reference SHA256 |
|---|---|---:|---|
| 0 / 66 / 70 | PRESS，1170 | 1170 | `204f5fc4b14d5b85bec38cabf9ad75594fcaaeee76e91b9b31cd9bc35de5bdcb` |
| 1 / 310 / 192 | GRASP，164 | 164 | `7347c6434b2b49c22f360595954cb11f0029b0f950fc59b0d0609fd672ad216c` |
| 3 / 629 / 30 | PLACE_IN，3473 | 3473 | `b7c60d2666aaa996a83b0c6df351f0c5123dcb1a78ab2dc8debf1668a682b1d9` |

来源是在 H09R 已审核 additional_train 中按 `sha256(h09s:teacher:1:task:instance)` 预选，未看测试或执行结果选例；原5%全部实例、H09 val/test全部实例、138/242保持排除。同实例所有后继分支仍属 TRAIN。逐源重新验证 H09R 完整 states/actions/labels 的 SHA，17帧逐行同技能/low mask/branch 与 quarantine；三个原窗口均不通过旧 pure-window codec，未把它们提升为 SFT 标签。task0 只提出6个混合成分方向；task1/3源底盘在动，未来 EEF 的机体系不可直接对齐，**拒绝空间排序**，不猜坐标变换。

## 谁是教师，何时标签有效

父代理承担独立人工前/后审：每次必须查看当前三视图、本体、合法任务/清理身份后的当前意图、原专家三视图片段与准确帧号，以及已执行历史。原注释只说明原轨迹此时的技能；经过12控制停顿、对象滑移或一次新 primitive 后，不自动认为仍成立。教师逐次重新确认同一意图、判断当前状态下的正确下一步并解释依据；若失效就拒绝、结束该先导，**不自动改子目标或恢复**。

原混合运动仅给 proposal aid。即使候选安全、缩距、原生动作到达、夹爪闭合，都不能自动成为正确 BC。实际执行后还要看完整前后/再停顿图、23D轨迹与回执；后审独立说明动作是否符合该意图、是否破坏支撑/载荷约束、是否存在意外接触。未知即隔离。`release_reviewed_record` 只验证人工判定的证据绑定，不证明审阅者判断必然正确，也不宣称技能/全任务成功。它保留 source_group 在数据出处中，actor 投影仅任务、意图、当前机器人本体、当前RGB身份和真实已执行符号，不含源帧/ID/未来/场景真值/审核结论。

第一小先导只开放现41符号中 **23个平移/开合/HOLD**，并一律 `carry=True`；不从夹爪宽度猜未知负载为空，不尝试 BASE/旋转。41符号和动作幅度/时钟本身未改。父 H14 已发现轮–桌与指–躯干碰撞，**当前采集硬阻塞在 H14 修复、独立审查及新同源双工程门**；不得用旧90a7c20执行器采“正确BC”。授权必须绑定采集器commit、执行器digest、全部准备reference SHA及工程门。布尔审核字段是操作者授权/判断，不是程序生成的物理证明。

## 最小使用流程（当前不启动）

1. 已准备的 `task_0/1/3/window.json` 可由原 `load_official_oracle_window` 装载；通用 OfficialEvaluatorSession 支持这三个 task 名，不需要写逐任务控制器。模板仅复用通用R1Pro配置；不复用开发138/242的动作或语义。前缀来自各自原 TRAIN 的 `actions[:frame]`，最后图像对应执行下一原 action 之前的状态。
2. 修后同源门与授权通过，分别运行 `native_teacher_collect.py --prepared <task目录> --output <新目录> --authorization <审核授权JSON> --gpu <已交接空闲卡>`。当前没有授权JSON，故不会自行启动。初始前缀q与原帧相差>0.02rad拒绝；固定12控制保姿态/零底盘停顿后，以过去4帧q/EEF平移与旋转/指口和本体速度做原尺度近静止筛查，不追加停顿凑数。原生base_qvel仅作小速度筛查，不积分、不当真实位移。
3. 程序写 `sample_00/before/{head,left_wrist,right_wrist}.png`、request、`WAITING_FOR_MANUAL_APPROVAL.json`。人工在120s内原子落盘同目录 `approval.json`。字段为 `request_sha256, reviewer, decision, token, intent_still_valid, judged_correct_next_action, reason, reviewed_current_and_source_views`；decision为approve/reject，reason必须解释当前状态，不能写裸PASS。前缀/停顿不冒充41符号历史。等待没有env.step/render/自动恢复；审批后检查q/grip未变化。
4. 至多40控制执行已审核的一次原生primitive，不超过现有门；保存独立不可变 `native_execution.json` 和 before/after/after_settle 图。失败不重试。`QUARANTINED_record.json`不是训练数据；后审需绑定其digest、reviewer/reason及 `correct_for_current_intent, reviewed_full_before_after_and_native_trace, not_based_only_on_safety_or_distance`。另需decision=approve；缺任何项不得经release helper进入候选数据集。只记录动作正确性，不补造 SUCCEEDED/FAILED 标签。

## 仅拟议的首三实例预算与停止条件

**尚未批准物理，也未批准训练。** 三实例各一次reset；前缀总4807控制；恢复/restore/重新前缀均0，不当免费分支。每实例≤3个人工候选、≤200新控制（包括所有12控制停顿和预留1最终hold）、reset后≤900s，单个审批等待≤120s。合计至多9条隔离记录、600新控制、45分钟单空闲GPU、新增≤100MiB（每run30MiB提前停、保留10MiB总清理余量），全盘≥80GiB。不使用GPU0/2，不与父/队友抢卡。非有限、来源漂移、近静止失败、意图失效、审批过时、原生拒绝/未完成、官方终止或任一预算到顶立即安全停，无自动扩大或重跑。

先导有效性是三种阶段能否**真实停顿、可审阅地重新教一个动作、准确执行和隔离**；允许0正标签，不能降门凑数。即便9条全经人工审核，也远未达到H09R的240条/每意图40条/近静止与起点/实例覆盖门，不能直接训练。保持下一阶段原门；通过先导后才另提有限采集块，再做真实新SFT与配对验证，不原样重训H09。

## 剩余限制

这是最小实现与真实源转换，不是已验证的模拟器collector；源视频已生成但本块没有人工看片给它们正确性盖章。新raw RGB和停顿状态必须在获批物理先导中得到。自动可执行性没有等价于任务动作正确性；手工审核本身也会犯错。原生近静止门不是物体静止或全环境避障证明。当前官方 `evaluate_installed_subgoal` 评估的是技能整体物理结果，可能UNKNOWN/IN_PROGRESS，不能给任意中间单步发正确性证书，因此本实现不调用它来自动放行BC。硬安全修复与同源工程门、父代码review、数据人工前后审均仍待。
