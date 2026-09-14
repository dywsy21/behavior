# 原生task-AR500：一次有界物理评测与人工复核

2026-09-13 22:44（北京时间），owner Codex。**执行链已核验；256个模型控制内未达到局部抓取成功。主要现象是手臂接近保持原位、夹爪开合未对准对象，以及间歇机身运动，不是持续原地打转。** 不能将此短测写成完整任务成功率0%，也不能由它唯一确定训练层面的根因。

## 本次实际运行

- `robo:/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/ar_native_prefix_pilot_v1`，源码989a575，290 CPU测试通过。原生task-only AR500权重SHA `639e64aeeb251113b807751f234077595165e65e9dd9e3b66cd7c4661f9df963`，静态schema完整动作解码；不是无约束AR已经学会格式，不是A4-FM，也没有接MEM-Lite高层。
- radio train121/instance138、env seed0/policy seed17；官方重置后原448控制前缀，再执行16 chunks、共256模型控制，到预算停止。actor只接任务及真实六帧历史/本体，GRASP对象与物理oracle仅供评估器。0训练更新、0新标注release；没有追加回合。
- `actual_rollout/collection_result.json` SHA `5a200786ea42317c97327431eed3cbf794355be00eac53b7bf4fa38fb426757d`，状态complete指采集作业完成；末态IN_PROGRESS、官方未终止。supervisor3191511、service3191531、sim3196694均已退出，私有服务由owner正常停止（-15），其他训练与旧服务未动。

## 视频和人工检查

本地视频：[rollout.mp4](/home/wsy/behavior/artifacts/experiments/2026-09-13-native-ar-prefix-v1/actual_rollout/rollout.mp4)。本地/远端SHA均为`e845f688b7d7a45a1805261ff0af0d707f38cb59334e4b39752efd47c768e5d0`。720×720、15fps、353帧/23.533秒；约前14.933秒是原演示前缀，后8.533秒才是本次模型接管，不把前缀表现算策略能力。

Codex主agent亲自查看17个头部相机关键帧448、464……704，以及448/480/528/576/640/704处左右手腕共12张，**29张不同视图**；另高分辨率重看头部448/704，不重复计数。收音机在头部视图中仍留桌面；腕部主要朝桌沿/地面/桌面，夹爪有开合但未形成对准对象的接近抓取；机身/视角前移倾斜，夹杂转向，没有持续绕圈。

- [头部关键帧](/home/wsy/behavior/artifacts/experiments/2026-09-13-native-ar-prefix-v1/model-segment-contact-sheet.png)，SHA `fb26124a5b3bf686b4bfaa1299d109adca14724abf19132da91c0f262b02810c`。
- [双手腕关键帧](/home/wsy/behavior/artifacts/experiments/2026-09-13-native-ar-prefix-v1/review/wrists-contact-sheet.png)，SHA `6d574b42648f78c72b7ac33ffd170ca2c15c7b0d6937351753f3e6e6e2d90b4d`。
- 逐帧选择与人工结论在本地`review/manual_visual_review.json`。保留原完成回执中的“待人工审查”历史字段，由这份新记录完成审查，不篡改旧结果。

## 动作、历史与物理回执交叉检查

只读脚本`scripts/experiments/review_native_ar_prefix_pilot.py`已在真实复制结果上运行通过，无模型/仿真/优化；`review/machine_checks.json` SHA `ce039f5c4d6970d70ee1fdee155cfc4e7f2a94fe38dedc95ddfe822d0d762f60`。

| 核验/现象 | 实际证据及解释 |
| --- | --- |
| 传输与执行 | 16次神经生成完整32×23；实际256条控制逐位等于对应输出前0:16和物理记录，底盘/躯干/双臂/夹爪保留；没有跳过前5步、补零动作组或FM兜底 |
| 真实历史 | 16次请求共96个历史锚点hash与捕获的相机/本体逐一一致，物理时钟448＋16i，actor没有收到评估器技能/未来控制 |
| 局部结果 | 初始0、前缀末448、16次chunk末，共18个保存的物理边界均双手未持有、IN_PROGRESS；没有保存每一物理帧的held序列，不能把边界检查扩写为逐帧证明 |
| 手臂目标 | 每chunk动作目标减去该chunk开始的实际关节角：右臂RMS约0.0020–0.00345rad，左臂大多约0.0021–0.0025rad，仅一chunk约0.00587rad；最大单值差左0.02048/right0.00947rad。说明解码目标本身接近保持，而非仿真丢掉了大幅接近指令；这不是相对专家动作误差 |
| 夹爪 | 两手指令均出现-1与+1，观察到开合。不能仅由首chunk都+1声称整段夹爪恒定或从未闭合 |
| 底盘yaw | chunk0/10/14约-0.30，其余多数接近0。全段角速度命令积分-0.47448rad，不是实测机器人朝向变化；不以此声称实际转了多少圈 |
| 规则约束 | 每次仍覆盖4–7个原argmax，完整60-token结构是schema强制的；没有额外safe-clamp。不能用格式通过替代内容学习 |

## 结论与下一步

本轮把“执行器没执行正确手臂/底盘动作、历史锚点错、动作起点错”与“当前策略生成的动作缺少有效接近运动”分开了，证据偏向后者。训练、任务条件与离散内容学习的具体因果还不能只凭此回合确定；该原生task-only配置不具备A4的相同技能条件/训练历史。A4此前464模型控制抓住，而此处仅256，**不是等预算胜负对照**。

不延长或重跑这份已诊断策略；继续原队列的marker500和FM/joint/KI筛选，再根据真实内容/自由生成决定后续AR或CoT候选。`C1`记录里继承的`outcome_supervision_eligible=true`和`active_bundle=GRASP`是评估器元数据，**不是actor意图或已放行标签**；本次仍`training_admissible=false`，队友不可直接回灌为意图—动作SFT正例。模型与视频等artifact不入Git，代码与本摘要走Git同步。
