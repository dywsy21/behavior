# H65：固定的实体手指按压参考

2026-09-25 11:41北京时间；owner Codex。当前实现分支`feat/semantic-agent-grounded-20260918`，新源码commit待固定；前置真实H64导出0f8321d/JSON SHA `5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04`。

主假设：以实际外露手指表面、而非夹持空隙中心计算当前距离与动作预测距离，可以消除press接口的几何参考错误。仅此一个接口改动，不同时换VLM、训练、目标定位器或成功判据。

## 设计与验收

- H64独审发现258/3696三角重心埋入另一片>0.1mm，最深3.034mm；任意三角/片AABB不等于并集外露表面。共面三角可被cooker换对角线，因此引用保存显式link-local XYZ、来源SHA、link/mesh，面编号仅作authored来源，不冒充PhysX身份。
- 合并同片共面三角为有界凸平面patch，内缩1mm后从当前允许RGB-D目标求近点；要求距同手其他凸片的半空间外侧至少1mm（保守排除接缝/埋藏/另一指）。没有合法面则弃权，不回退空隙中心。1mm是工程裕量而非接触/成功阈值，未按本任务结果调参。
- 每个press子目标只绑定一次工具局部点；后续named finger proprio/FK只移动同一点。缺校准、点被另一指遮埋、活动手已知或疑似持物时不得执行。另一持物手继续原有保护。
- 同goal的新目标必须仍在固定面正侧，point→target线段不得与同手其他凸片的1mm膨胀半空间相交；无可选patch的小凸片也能遮挡。目标转背侧或通道不合法时阻塞/HOLD，不为继续移动而自动换点。该判据只涉及同手几何，不证明环境可达或目标身份。
- 新开关默认关闭；非press/旧运行默认逐项保持。接线后当前距离、候选真实joint-plan后的距离、日志都必须使用同一参考；无joint-plan的旋转不猜参考位移，夹爪开合预测不复用当前开度。
- 仅CPU实现/单次测试≤120s、真实保存机器人几何审查≤120s；0新模型调用/训练/SimulationApp/reset/动作。独立代码复审后固定Git。没有新的仿真预算/共享显存放宽，也不把本票作为官方SR分母。

后续必需：native多开度FK、PhysX cooked/rest/contact offset及单次实际按压检验，随后完整官方reset、零前缀的策略回合。当前仅authored参考不是接触成功，且不包含“准备—触碰—保持—撤回”的完整交互状态机；该后继未完成不得宣布press问题全修复。

## 2026-09-25 12:06北京时间实现状态

开关`--press-finger-surfaces`要求named finger FK/current robot guards，严格绑定H64完整JSON SHA；gate/运行manifest/result绑定同一资产，reset后核四指link/joint与active calibration。controller/prompt使用实体参考，当前与真实joint-plan预测同点；粗方向排序的平移估计不是执行认证。候选接受前与runner实际servo.begin后/任何控制前检查每个关节计划点及其中点，VLM延迟后的q/finger/负载/goal变化也拒绝。base只检查端点，不冒充连续扫掠。

30新增CPU（含两个独审反例、无patch遮挡、同goal变点、失败轨迹中间点/有效末点、握物/缺命名状态/开合未知等）通过0.444s；全量681项中676通过、5本地USD缺SDK明确skip，15.819s。真实H64保存JSON已成功编译全部32片（0.660s），但没有新的native当前姿态或动作验证。12:08最终独立delta复审30/30（0.819s）及diff通过，无剩余可复现代码阻塞；确认held/world目标语义与候选/执行前路径。仅CPU接口验收，未部署、新模型/训练/仿真皆0，官方SR没有新分母或成功。
