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
