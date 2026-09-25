# H63：保留真实两指位置的可移植手指几何

2026-09-25 10:31北京时间；owner Codex；分支`feat/semantic-agent-grounded-20260918`，实现commit待固定。

## 唯一假设与范围

按压合同审计发现：当前`RobotState.gripper`是每只手两个关节位置的平均值；q18只含躯干/双臂，`grasp_regions`只能使用完全张开参考。平均开度相同的两个不对称手形不能共享接触几何，夹持空隙中心也不能当手指接触点。因此先保存命名的真实finger joint proprio，导出各手指相对于EEF的机器人资产参考与关节螺旋轴，并支持纯CPU FK。

本票不训练模型、不产生环境轨迹、不读物体/接触真值/成功判据。保留原23D动作和`gripper`均值兼容接口；旧数据不凭均值补造两指位置。新增几何显式区分资产夹持条带与真正碰撞表面；缺失输入或未支持手型不能静默使用open参考。

## 接口和验证

- 当前proprio按原机器人joint名字传递，避免左/右指索引互换或对称假设；`RobotState`旧调用保持可用。
- 新校准数据独立version，空间坐标只用robot-base/EEF/finger frame。原生Jacobian从COM移到link origin后再换坐标，不能重复包含躯干/手臂运动。
- CPU覆盖参考pose、任意EEF姿态、独立/不对称finger变化、有限差分、非法/缺失/越界输入及旧接口；需要独立审查。
- 后续新仿真中与原生finger位置比较才可启用press策略；当前没有原生多开度实测。资产夹持条带不是已验证的物理按压接触面，不据此放行按压或报告成功。

## 预算、停止条件和后继

单次本地CPU回归≤120s；0GPU/模型调用/训练/仿真重置，未新增任务/实例/seed，不作为成功率分母。发生旧接口回归或机器人数据契约不符先停止修正；不修改活跃robo源码/环境、不自动启动物理。

后继为真实native FK检验、按压接触面及固定参考选择、距离/动作前瞻一致、持物手保护和可观察效果验证；最终仍须原始reset的完整官方任务成功。资源复验另需用户答复和冻结预算，不能借本票放宽旧H59限制。

## 10:40实现状态（北京时间）

已实现15个新增CPU用例，连同原相关共66通过（0.537s），`diff --check`过；初次mock原生Jacobian张量少一个维度导致4个测试错误，修正fixture后通过，没有改期望掩盖控制问题。全量回归及独立审查进行中，尚无GPU/物理结果。

`--finger-kinematics`默认关闭，只在新grounded运行中导出校准、保留各指位置、逐帧记录`robot_finger_geometry.json`并与native FK比较；相应gate标记必须一致。渲染前后、保存帧复用及重定位快照保留独立两指身份；不把同均值当同手形。现有press控制参考尚未切换，几何条带显式标为非接触面/非持物证据。

10:43验证：全量首轮627的实际捕获AST回归因新args闭包而失败1项，修为直接根据当前state的命名finger字段决定绑定检查，不修改该旧接口调用方式。补真实state_now opt-in/legacy、实际SearchReanchor八槽快照复制及同均值不对称变化反例；17个新测试，相关41/41、全量629/629通过（16.392s）。独立审查尚待，原生多开度/实际碰撞面与press策略仍未验证。

## 10:53独立审查结论与交接（北京时间）

独立审查复跑41相关CPU通过，但发现两项实质阻塞，**目前均未修复**：

1. `run_v2.implementation_digest()`只覆盖v2目录及runner，遗漏新增finger数据源`control.py/og_backend.py`和直接运行依赖`run_sim.py`。这些文件改动后旧gate仍可能被接受。后续应绑定全部实际运行依赖，用仓库相对路径及内容生成稳定摘要，并测试改内容/增删/改名、迁移checkout与无关文件变化。
2. `self_odometry.make_frame/validate_frame`仍只保存和验证q与平均gripper。capture前后检查不能防止后续把保存帧用于另一种同均值手形。审查已实际复现该替换仍返回有效digest；需将命名finger位置贯通保存帧版本、validator、odometry/substep、controller、runner及reanchor，同时覆盖legacy兼容和缺失输入拒绝。

上文“保存帧保留两指身份”的表述仅对已扩展snapshot及capture时检查成立，不是self-odometry全链路验收。629 CPU通过不代表这两项已修复。当前转处理用户最新Zetta请求，全部H63源码修改保留、开关默认关闭，不部署、不合入；恢复H63时先修两项、补实际反例测试并重新独审，再考虑native物理检验。0新增物理结果或SR。

## 11:03修复验证（北京时间）

按G-AV1 goal续接，两项已实现修复：实际递归semantic_robot代码、runner及run_sim以相对路径/内容SHA绑定；self frame新增命名finger的v2版本，沿RGBDMotion、SubstepMotion、controller、reanchor、runner gate/中途/末帧及completion观察传递实际值。当前值缺失、帧降版本、同均值不对称替换、校准名字/范围/均值不符均拒绝；旧模型的旧frame不加新字段。

新增12用例覆盖实际runner/digest AST、真实RGB-D estimator/controller调用、Substep末帧交付、实际reanchor及schema反例。相关63/63通过（0.525s），全量641/641通过（16.502s）；原审查者复核进行中。代码commit待独审后固定，开关仍默认关闭；未native多开度/真实碰撞面核验或启动新仿真，不宣称按压改善及完整成功。

11:05最终CPU/复审：独审另指出named state+无finger校准旧model可生成v2，已加拒绝及逆向真实make/validate反例。现13新帧/digest用例，相关64/64（0.399s）、全量642/642（16.108s）通过；原独审最终确认三个问题均关闭、无剩余实质代码发现，独立最新30/30指/帧回归通过。默认关闭策略与native/物理/SR未验边界不变。
