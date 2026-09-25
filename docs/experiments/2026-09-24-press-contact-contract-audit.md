# 按压接触接口静态审计（未做新物理验证）

2026-09-24 22:53北京时间，负责人Codex。H55独立场景检查运行时做只读源码核验；没有改活跃源、增加sim/model、调用对象状态或执行动作。

## 直接证据

robo安装`/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson/object_states/toggle.py` SHA `a7a88f4a55242816ab930fe5336cfd94b26e2aaf6e2d33aa0239ed47eca2fd3b`：

- `global_update`从机器人各手指link集合查物理contact pairs，建立与手指发生contact的对象集合。
- `_update`仅对该集合里的对象进一步查toggle marker区域与机器人finger rigid body overlap；两条件成立才累加`robot_can_toggle_steps`，断开则清0。
- `CAN_TOGGLE_STEPS=5`，计数恰达到5才翻转状态。这里是**状态更新次数**，不直接换算成5个actor决策/5个控制指令或固定秒数；requires_closed的额外条件仍照官方实现。
- 上述只是规则源码审计。没有读取当前任务marker位置、对象pose、contact真值或toggle状态给actor；后续也不能把这类特权信息作为部署策略输入。

当前仓库`src/semantic_robot/v2/grounded_harness.py`的observe及`_expected_distances`不区分press与pick，都以`model.grasp_centers(q)`计算target距离与候选运动后的距离。`kinematics.py`的grasp_centers是两边机器人assisted-grasp contact strips均值，不是单个finger表面；`og_calibration.py`也将其明确标为closing-region centre。

`harness.py`的press palette有平移/旋转，没有与按压工具准备对应的gripper open/close阶段。`grasp_regions(q,gripper)`仅适用于校准的完全张开夹爪，开度变化即弃权；portable q18没有单独finger DOF。因此不能直接把旧完全张开的contact点用于闭合夹爪，也不能把EEF原点默认为指尖。

## 结论边界

这是**几何参考点与实际接触条件之间的静态接口不匹配风险**。足以要求修正press的几何合同，不能据此认定过去每次失败已经到达按钮，或说这就是全部失败的唯一原因；可见性/定位、navigation/IK、持有判断与预算仍可能先失败。H55没有actor，本票也没有新的SR。

## 后续通用接口方向（尚未实现/未授权新物理票）

1. 明确区分“夹持空隙中心”与“可用于接触的手指表面点”，press进度/候选前瞻使用同一个、明确标记的接触参考点。机器人几何从本体asset＋当前proprio/FK导出，目标仍只来自允许的RGB-D/VLM，不读对象marker姿态或仿真contact。
2. 参考点需随真实finger aperture更新；校准缺失/指形未知/可能持物时不能静默复用open手几何。是否需要闭手准备应由机器人接触工具合同明确，不以去掉task0抓取提示替代此实现。
3. 单次触碰与保持、撤回/重新观察要区分；动作发出/距离归零不当作effect。可观察效果用于下一决策，完整成功仍仅由官方结果报告。禁止直接设置toggle值或把内部5次计数接进actor。
4. 先CPU刚体变换/开度/单双手/held-arm保护反例，再单个有界局部工程检查；确认资源与视觉输入可靠后才另登记零前缀闭环。不为task0手写按钮轨迹或为50任务各写控制器。

当前优先仍是H55取得真实RGB-D并核共享资源；本审计不改变正在运行的H55代码/预算，不启动H54b或新训练。

## 2026-09-25 12:52北京时间续查：保持动作与效果时序

当前主源95a7bfe；H67单体原生检验独立运行中，未热改它或启动另一个场景。安装toggle.py SHA仍为上文a7a88f4a，规则没有新变化，不将此次复读说成新发现。

新核对实际执行调用链：`v2/servo.py::begin`对HOLD仍设置12/18/24个控制tick，保持RUNNING；`run_v2.py`的accepted循环照常逐tick调用`servo.next_action→step`。因此“HOLD只有墙钟等待、不推进物理”这一疑点**不成立**。

但`v2/harness.py`的INTERACT分支只在`effect=True && last_action.move != "hold"`时进入VERIFY_EFFECT。未来显式的press-hold如果继续被当成普通HOLD，保持期间出现的可见效果不会立即进入验证。普通HOLD也用于安全停止/预算与感知等待，不能简单全局删除这个条件来放行模型自报。

后继press状态机应为“工具准备→接近→受限接触推进→有计数的保持→撤回/再观察”保留专用执行回执；效果仍来自合法新观察，成功仍由官方终态单独统计。H65的HOLD几何豁免是现有安全停止路径，不应未经校验就转用为主动press-hold许可。本次仅静态定位接口依赖，未实现该状态机，也未证明历史失败发生在此分支。
