# H09T：同执行器离线教师与完整证据事务

负责人：唯一 Astra 子代理 `vlm_sft_showharness`。2026-09-19 05:25–05:55 BJT 有界 CPU 块；0 新 reset、0 控制、0 模型调用、0 训练。父代理维护共享计划。独立分支 `feat/vlm-sft-native-teacher-20260919`，父执行器保持 `239cb591f178099f20e9a9ba6d7cd3ce04aa5ccfe183840ebe1ff78ac2f40b9b`。本报告不是新采集授权。

## 真实起点与本块结果

H09 的 600 次训练和四次闭环负结果保持，不重训旧配方。H09S task1 已审核两条同实例 RIGHT_LEFT 接近动作，非抓取完成；第三条实际执行但证据预算失败，仍隔离。旧 task0 304 prefix/0 native 失败与旧 85,022,736 B 总占用不能删除或不计。

本块实现了真正的 collector 可选入口 `--teacher-config`，不是伪造数据生成器；没有运行该入口。未授权、seed 缺失/不一致、TRAIN 组排除失败、旧安全门、PLACE_IN 不完整体积接口都在 reset 前拒绝。所有结果先隔离，代码没有正 BC 自动导出/训练接口。实际 OG 读取适配器尚未经过新物理实例验证，因此不能据 CPU 测试宣布采集可用或有效。

## 1. 执行前预留完整事务

`native_teacher_artifacts.py` 根据 raw CHW RGB / HW 浮点深度形状而非历史压缩率给 DEFLATE/PNG/ZIP 保守上界。capture 先在内存编码全部文件、一次核总量，写临时目录后 rename；不再出现已发布的半个 capture。动作前预留完整 immediate+settled capture、最多 52 个控制的 trace/teacher telemetry、记录；原 cleanup 余量独立。普通写入不可挪用 reservation；动作/settle 控制预算也先整体核验。容量不够则动作前结束，不为了凑 3 条而执行。

实际保存 task1 sample00 的 raw 大小核验：head 720×720、腕各 480×480、深度 float32；完整旧 capture 3,412,025 B。新单 capture 上界 **9,276,026 B**，完整动作证据预留 **19,879,156 B**。旧第三条前仅约 **3,305,149 B** 可用，必在动作前拒绝。该 before-third 估算排除第三 after 与故障/hold 文件，不冒充精确历史时刻磁盘计数。新保守上界可能更早结束第二条；这是不扩大原 30/100 MiB 的明确代价。

保留完整标定 gzip/canonical SHA、当前实际 self boxes、raw depth 和全 capture hash。每个 step 和动作事务前核 80 GiB 余量。预留是单 collector 的应用内预算，不是对别的进程抢占磁盘的 OS 级承诺；IO/硬件异常仍隔离，并保首错误/最终 hold。

## 2. 教师提案、当前意图与 actor 隔离

`native_teacher_policy.py` 只接受与原单个 skill verb/target/destination/hand 一致的 spec；TRAIN 由已审核 H09R 72 源原始数组/标签 SHA 和实例排除核验，额外排除 H09 val/test、原 5% 全实例、开发 138/242 与新留出组。不重新统计静态测试。

实际对象/参考手位姿只由 `native_teacher_og.py` 读取，供离线教师及私有证据。候选按 object-local 已审核目标姿态计算，并经原 fresh RGB-D、actual self geometry、SafeServo(robot_geometry_guards=True) 独立重查。缩距只排序，绝非正确性标签。一次拒绝可检查同状态剩余有限候选，最多 12；执行失败、UNKNOWN、无进展或预算终止后不恢复、不重置、不再搜索。

仍是 41 符号/native23：首版保守 carry=True，因此原执行器禁止旋转，本教师明确不提旋转/base/BOTH；不把旧 rotation token 改成别的轴/尺度。目标姿态不能在这一子集达成就停止，不假称全 41 已被覆盖。未来如要用已验证空手转腕须另登记与父同源执行契约，不在本块偷改。

部署样本只从 `actor_input` 白名单构造：原任务文本、清洗后当前意图、本体状态、三张当前图 hash、真实执行 history。target UID、隐藏位姿、oracle verdict、未来专家状态、候选成本、source group 只在 private/出处字段。CPU 负例检验 private 字段不能混进 proprio。暂未接旧 live.py 的 GRASP-radio/NAV-table whitelist；后继 PRESS/PLACE/task1 物理配对必须先新增通用合法意图接口并独立审查。

停顿改变状态，所以原 annotation 不自动证明停顿后意图仍正确。首次物理先导需父审核初态与 seed/任务条件；随后自动执行的整条轨迹必须有终判及分层图像/trace 审核。初始接触基线完整保存但不能证明其无害；oracle 拒绝新 robot 非目标手指接触，已有支撑/接触仍是重点人工检查项。

## 3. 局部终判，不冒充任务 success

`LocalOutcome` 需要每一个连续实际物理 tick。同 tick 重读不积累稳定性，缺帧/倒序/冲突拒绝；缺状态 UNKNOWN。teacher 私有结果明确 `official_task_success=None`。

- GRASP：初始双手均确认未持目标；发生实际 CLOSE，指定目标身份接触/held 真；目标比初态升 ≥3cm，手比 close 时升 ≥2.5cm；连续 12 tick 相对手稳定（4mm/3°）。闭合命令或 bool(UNKNOWN) 不算成功。
- PRESS：初始 toggle 确认 false；真实 false→true 边沿同 tick 有活动手指–该目标接触，再持续 12 tick true。原始 true 不给新 credit。若另一手支撑，持续要求 held 与目标相对支撑手稳定。
- PLACE_ON：初始持有，实际 OPEN、实际开口 ≥45mm、双手均不持目标、关系及目标–支撑接触为真、目标线速 ≤1cm/s、角速 ≤.05rad/s、12 tick 目标位姿稳定（3mm/2°）；配置的 payload OnTop 始终保持。
- PLACE_IN：CPU oracle 还要求全角点体积证据。安装的 Inside 只验 AABB 中心，当前 reader 故意给 UNKNOWN，collector 在 reset 前阻断该类。未以中心命中替代全部放入。

上述严格判据仍是离线局部操作定义，非官方全任务判据。接触时机/高层意图/碰撞遗漏等实际 API 风险必须用先导证据审查。所有 partial 或失败轨迹都不释放正标签；最终 hold 也检查结果是否保持。

## 4. 原 oracle 版本与差异

只读检查服务器旧 `GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime`：PhysicalSkillOracle 的 GRASP 只依赖 held；PRESS 因果条件由外部 precondition 传入；PLACE 的三次正判断不自己绑定物理时钟、速度、支撑或 payload。这些不足以直接充当本轮标签。四个文件/许可证 SHA 固定在 `configs/vlm_sft/h09t_offline_teacher.json`，依赖为 numpy、原 skill protocol 与 contract。旧 LICENSE-G0.5 为非商用/有限专利条款；本块没有复制旧实现，只做原始独立实现并复用当前 Git 内已审基础设施。

安装 API 已只读核对：robot.is_grasping 的 physical 分支实际上核 candidate 接触（docstring 有过期描述），enum UNKNOWN 不能取 bool；contact API 是 row/col 非对称注册，因此复用父 `audit_pairs` 双向查询；ToggledOn 由连续真实手指 marker 接触更新，绝不调用 set_value。对象/EEF/base 全局位姿仅在 private reader。payload 或体积/状态接口不可用必须 UNKNOWN。

## 5. 从 seed 到真正新训练的有限路线（未执行/未授权）

种子不能来自 annotation 尾或现 17 帧。可执行的 CPU 种子编译器 `native_teacher_seed.extract_seed(spec, rows)` 输入逐真实控制的 full23 + 私有观测，重跑严格因果 oracle；成功后 GRASP 取目标局部 EEF pose，PRESS 取触发边沿 toggle-link 局部 EEF pose，PLACE 取 destination 局部最终目标 pose。专家动作仅产 seed，**不是 native BC**。seed 仍需来源 SHA 和父独立图像/接触审核。

下一必要入口是另登记的全原专家段回放适配器：复用官方 factory 从 0 重放完整 prefix，再执行同 episode 全 skill 区间，逐 env.step 记录上述 teacher ledger；完成前不可截取 annotation 终点作正例。确切来源：t0/e66/i70 f1170→1610（440）；t1/e310/i192 f164→452（288）；t3/e629/i30 f3473→4756（1283）。prefix 合计 4807、skill 2011，加每例最多12稳定+1 hold，**3 新 reset / ≤6857 控制 / 每例≤2100s**；不是挪用旧 task0 已消耗 reset。完整段尚未提取/固定新 SHA，回放适配器尚未实现，PLACE_IN 体积适配尚未审，这是明确后继工作，不伪称已有源可直接执行。任何失败一次停止，不自动重采。

先导之后才能登记 native 教师小块；当前 collector 的 200 新控制/900s/30MiB/100MiB 不因此报告自动变大。尤其 task3 的 3473 prefix 已有约1263s 吞吐估计，旧900s不可直接启动。若要每例6条完整证据，新保守实际形状建议**每 run96MiB、累计384MiB（含旧85MB）**，须父新授权并显式改配置读取，不可换目录绕过旧累计。首候选能否完成技能未知；旋转受限/未成功立即停。

训练数据门保持 ≥240 条、每 GRASP/PRESS/PLACE ≥40、各≥4 TRAIN 实例，近静止≥20/阶段起点≥10，OPEN/CLOSE各≥20且跨4实例、BASE≤20%，新留出按实例隔离。6条/run即使全有效也至少40次 reset；失败率未知，不能承诺“3实例即可凑齐240”。建议先以3次参考回放+至多2次 native先导测真实产率，再独立决定总采集块，最多480候选/80reset的成本上限只是待审提案，不能自动开跑。按真实 task1 capture约10MiB/动作证据估算240条约2.4GiB，480尝试约4.8GiB另加标定/视频，需另做≥80GiB磁盘余量协调。

只有新数据门与人工审查通过，才从现有2B重新初始化新 LoRA，最多400 updates、effective batch16、GPU≤1h，一套预注册配置、原mask/native-CE/reload实门，不搜索参数。静态 base/FT 同协议≤300调用；3异质实例×base/FT 共6局部回合，各≤30decision/960control/600s。报告真目标接触/提起、toggle因果、释放支撑，以及失败类别；不能用 loss/合法输出/缩距代替实际效果。需要新的高层意图接口与同源工程门后才可运行。

## CPU 验证及剩余风险

49 SFT tests/0.217s，331 harness/5.195s，0新模型/物理；包含容量拒绝前无动作、普通写不能侵占 reservation、不可压缩 raw bound、真实 collector 故障 hold、重复 tick 不加 credit、UNKNOWN enum、因果 toggle、lift/支撑/payload、TRAIN实例隔离、teacher字段泄漏、seed尾帧假成功否决。实际保存 task1 raw shape 和容量已复核；未对新 OG adapter 做真实 reset，因此运行时API、contact覆盖、hand/本体框架数值绑定仍要父独立审查与有界先导。

本块不是训练完成或正效应声明；交付稳定源供父审，下一票才能修复/采集。H09/H09S 旧失败和两条真实已审接近标签全部保留。
