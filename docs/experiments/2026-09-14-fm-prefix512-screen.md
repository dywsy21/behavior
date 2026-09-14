# 三种FM500配方：同窗口局部闭环筛选

负责人Codex / M-02-L1。为后续50任务训练筛选方法，仅是开发训练实例的固定技能诊断，不是自主完整任务成功率。12:11三臂物理、全记录核验和本人87视图审核全部完成。

## 固定条件与真实状态

原radio episode121/instance138、env seed0/policy seed17，448真实原演示控制前缀，固定正确GRASP；每臂最多32请求×16动作=512模型控制，预测32/执行0:16，真实23维控制及六帧图像历史。三个模型均从A4独立训练500步且完成全权重/优化器核验。共享原神经服务/归一化/物理判据，不把原演示前缀算作策略能力。

| 配方 | 实际模型控制 | 局部物理结果 | 人工/机器核验 |
| --- | ---: | --- | --- |
| control500 | 512/512 | 32个chunk边界均IN_PROGRESS，无因果抓取成功；34个保存物理边界均未持有 | 512动作、192历史锚点、34边界通过；本人看17头部＋12腕部视图 |
| Beta分层500 | 512/512 | 32个chunk边界均IN_PROGRESS，无因果抓取成功；34边界未持有 | 512动作、192历史锚点、34边界通过；本人看17头部＋12腕部视图 |
| 执行段加权500 | 512/512 | 32个chunk边界均IN_PROGRESS，无因果抓取成功；34边界未持有 | 512动作、192历史锚点、34边界通过；本人看17头部＋12腕部视图 |

control的右手确实会向radio靠近，早期最大chunk目标关节变化RMS约0.232rad、实测0.244rad；后期在目标附近微调（后22个chunk右手目标变化RMS约0.0023–0.0165rad），未形成稳定抓取/抬起。图像与物理边界支持这一描述，不能据有限视图排除瞬时接触；没有持续原地绕圈的视觉证据。

原A4曾在相同来源448前缀后464动作抓住，是历史参考。此处不能未经实际前缀状态比较，将单个随机物理回合直接解释为500步训练导致退化；本次三份500模型才是主要配对对象。后续报告三臂实际回放状态差异，不假设同seed保证逐位相同。

实际前缀比较：Beta和exec的六个历史锚点中，所有保存本体字段对control差值均为0；三路RGB均不是位级相同。没有声称对象全状态或渲染完全一致。两候选同样是接近radio后在其旁上方停留微调；32/32模型chunk边界均IN_PROGRESS，未持有/未抬起的物理记录与所看图像一致。这轮没有显示局部成功收益，**不直接将Beta与执行加权组合，也不为这个radio窗口再扩seed/步数**；单例失败不足以否定跨任务方法。

## 证据与资源

- 服务器根：`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/fm_screen_prefix512_v1`；各臂子目录含service/socket/rollout/completion，所有旧失败保留。
- 原编排7b806b6 / 137 CPU passed；原manifest SHA `d37907aab5b3bb72706bade8ed2909a78379f4c189a3512a94ab1d789d0ce4bb`。FM神经服务原文件SHA `c4bc099d1ac5d4d96e311498d6d8c8309900f8cb1e91c140be2f1d901057109f`。
- control result SHA `cd356486a4377ff76b87366fc194eaee8913d8625882933a9646dcd2dec7b6a6`；video `e8c51bde9278a854fd03950ea16b202956ae6707eb80c2f08cc8def5560bcbd4`，本地/远端一致。
- 本地control视频与审查：`/home/wsy/behavior/artifacts/experiments/2026-09-14-fm-screen-prefix512-v1/fm_control_v1`。机器记录SHA `db7f5cf8f075c25bf533740c4c6385f6a24cbbe554ed08d7aab7c39ab5882df6`；本人29视图记录SHA `ffe94b20b915fa7b7fe96ee649865b06dccccc2ca7cad3cb8d0024fe9e51f14a`。
- Beta result `e6d697b83d87d859818f3fe094013acb2c47bccea5264b7f23c9208ed517cf90`；video `0eaea9a6945970fb5d9a800eeb0e39330b12c8847538cbf83388e679fb165226`；machine `59f5b3b7ae416c832c86356e47efcf0d2e29c4c9309ecaea077d2f25c8d76566`；本人29视图记录`d0c12546a3df597d91347a9ec8ccd7d9d1b858ae2c0ce8453a9b3cef249100e7`。
- exec result `ffd62353feaa1d1accbaf68c5cccfb470ab291c3f8cd24eb20ee3aeab14376c2`；video `b899ae0ea1d1f145cfcd450a6dfcc12ace9fe9c00a5a810b15f19d5bfb539947`；machine `4bcdf212b56bea1fa5d849bb7c89d2d6f300838df08ee080ad36ae6ba25d1df2`。两臂视频/审查位于同一本地根下对应模型名子目录，视频SHA均与服务器匹配。
- exec本人29视图记录SHA `0233341bf33e20d7d8d64f6dcd87a9aace93157bdc52c0735b83e119947ce1c8`。三臂审核JSON及拼图已同步到服务器run下`review_by_main_20260914/<arm>/`，逐文件SHA与本地相同，供队友直接检查；不覆盖历史completion。[轻量汇总](results/2026-09-14-fm-prefix512-screen.json)已入Git。
- 首臂完成后，复用8786的端口检查报Address already in use，后两臂未启动。检查时旧PID均退出、端口已无连接，短暂连接残留是推测而非捕获的TIME_WAIT真值。
- d5f0aec / 145 CPU passed仅恢复未执行Beta/exec，各自独立8787/8788；11:56:09 supervisor34585提交，recovery SHA `1b58e96dcb6fa8fccd25bbc14ff0aba6fe415cad0f700c3c4bbfe4a44ccbf0df`。不重跑control，不改旧manifest、不自动重试。
- 每臂GPU1，启动模型前需40GiB空闲、仿真前需25GiB；磁盘保留120GiB，不停止既有四卡训练/其他服务，共卡墙钟不用于方法加速比较。

12:10确认本次恢复supervisor、Beta/exec服务与仿真均已退出；control三个进程之前退出。旧完成回执中`personal_review_pending`保留原时点，新完成的`review/manual_visual_review.json`是后续真实审核证据，不篡改历史字段。

没有新训练、标签release或恢复教师数据。CE/FM诊断、实际动作误差、局部抓取和完整任务SR分别解释；这轮证据不足以直接堆叠Beta与执行加权。
