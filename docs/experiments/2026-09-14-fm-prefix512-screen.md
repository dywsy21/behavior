# 三种FM500配方：同窗口局部闭环筛选

负责人Codex / M-02-L1。为后续50任务训练筛选方法，仅是开发训练实例的固定技能诊断，不是自主完整任务成功率。当前未完成全部三臂。

## 固定条件与真实状态

原radio episode121/instance138、env seed0/policy seed17，448真实原演示控制前缀，固定正确GRASP；每臂最多32请求×16动作=512模型控制，预测32/执行0:16，真实23维控制及六帧图像历史。三个模型均从A4独立训练500步且完成全权重/优化器核验。共享原神经服务/归一化/物理判据，不把原演示前缀算作策略能力。

| 配方 | 实际模型控制 | 局部物理结果 | 人工/机器核验 |
| --- | ---: | --- | --- |
| control500 | 512/512 | 32个chunk边界均IN_PROGRESS，无因果抓取成功；34个保存物理边界均未持有 | 512动作、192历史锚点、34边界通过；本人看17头部＋12腕部视图 |
| Beta分层500 | 512/512 | 完整回合结束，结果待全记录核验 | 视频/回执已开始传到本地，人工审核待做 |
| 执行段加权500 | 尚未完成 | 不预填结果 | 12:02:04专用8788服务59091启动，待实际物理 |

control的右手确实会向radio靠近，早期最大chunk目标关节变化RMS约0.232rad、实测0.244rad；后期在目标附近微调（后22个chunk右手目标变化RMS约0.0023–0.0165rad），未形成稳定抓取/抬起。图像与物理边界支持这一描述，不能据有限视图排除瞬时接触；没有持续原地绕圈的视觉证据。

原A4曾在相同来源448前缀后464动作抓住，是历史参考。此处不能未经实际前缀状态比较，将单个随机物理回合直接解释为500步训练导致退化；本次三份500模型才是主要配对对象。后续报告三臂实际回放状态差异，不假设同seed保证逐位相同。

## 证据与资源

- 服务器根：`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/fm_screen_prefix512_v1`；各臂子目录含service/socket/rollout/completion，所有旧失败保留。
- 原编排7b806b6 / 137 CPU passed；原manifest SHA `d37907aab5b3bb72706bade8ed2909a78379f4c189a3512a94ab1d789d0ce4bb`。FM神经服务原文件SHA `c4bc099d1ac5d4d96e311498d6d8c8309900f8cb1e91c140be2f1d901057109f`。
- control result SHA `cd356486a4377ff76b87366fc194eaee8913d8625882933a9646dcd2dec7b6a6`；video `e8c51bde9278a854fd03950ea16b202956ae6707eb80c2f08cc8def5560bcbd4`，本地/远端一致。
- 本地control视频与审查：`/home/wsy/behavior/artifacts/experiments/2026-09-14-fm-screen-prefix512-v1/fm_control_v1`。机器记录SHA `db7f5cf8f075c25bf533740c4c6385f6a24cbbe554ed08d7aab7c39ab5882df6`；本人29视图记录SHA `ffe94b20b915fa7b7fe96ee649865b06dccccc2ca7cad3cb8d0024fe9e51f14a`。
- 首臂完成后，复用8786的端口检查报Address already in use，后两臂未启动。检查时旧PID均退出、端口已无连接，短暂连接残留是推测而非捕获的TIME_WAIT真值。
- d5f0aec / 145 CPU passed仅恢复未执行Beta/exec，各自独立8787/8788；11:56:09 supervisor34585提交，recovery SHA `1b58e96dcb6fa8fccd25bbc14ff0aba6fe415cad0f700c3c4bbfe4a44ccbf0df`。不重跑control，不改旧manifest、不自动重试。
- 每臂GPU1，启动模型前需40GiB空闲、仿真前需25GiB；磁盘保留120GiB，不停止既有四卡训练/其他服务，共卡墙钟不用于方法加速比较。

没有新训练、标签release或恢复教师数据。CE/FM诊断、实际动作误差、局部抓取和完整任务SR分别解释；这轮证据不足以直接堆叠Beta与执行加权。
