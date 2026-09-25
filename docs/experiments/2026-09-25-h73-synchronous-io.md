# H73：显式物理时钟与同步 RGB-D

2026-09-25 16:10北京时间，owner Codex，源待固定。对照H72的真实漏步和旧depth证据，不重新调底盘增益/视觉比例/成功门。

主要假设：把控制和渲染解耦，逐次保证1/30s物理推进，并等待原生annotation完成，可恢复微动作与观察的一致性。只改I/O执行协议，原控制参数、24动作、RGB-D里程计/目标门、720/480相机、PT4/16与任务保持。

- 默认关闭的新`synchronous_io_v1`；每env.step使用render_on_step(False)，检查真实physics index＋4/time＋1/30s。不能用重复env.step补足漏步，也不能把尚未执行的命令算完成。
- 原三render_product只附加ReferenceTime，禁止新建/改分辨率。原sim.render先同步Fabric；Replicator1.12.27 `step(delta_time=0, pause_timeline=False, wait_for_render=True, rt_subframes=4)`等待当前图。原生time/index必须不变，三相机ref时间须匹配当前物理时间；不以像素改变当freshness。上述API对应关系仍需native验证，失败记录真实值后停，不造匹配。
- 仿真时钟/相机时间不是场景特权状态；不读取物体、接触或绝对位姿，不向actor泄露H72私有真值。失败停止原gate，不放宽容差。
- task0 TRAIN138/seed0/reset→load→reset/0专家或旧policy前缀，0模型训练。本轮单次≤1200s＋30清理、24决策/1536controls；GPU3≤24GiB，自有辅助各512MiB，全卡8GiB余量，CPU72–75。达到预算停止，禁止同配置自动重跑。
- run `/mnt/nvme_tmp/robodojo_agentic_20260925/h73_synchronous_io_v1`，runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h73_synchronous_io_v1`。双端CPU、独审、依赖和资源预检后才从新Git worktree唯一启动。
- 验收：每control时钟、每capture零物理推进＋3路reference time回执、原24gate完整通过。局部工程通过仍不是完整SR；后继另一任务gate/actor另登记。

依据：[与服务器一致的Replicator1.12.27 API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html)。已读安装orchestrator.py SHA c297093ae4100f5fc20f576f0b71d0dc4c765b2135f2a945764bbe63889a7c42；初始化控制timeline/async/material加载，须记录/检查副作用，不把调用成功当数据已新鲜。

当前仅准备，0新GPU/训练/run；H72全部原件保留。

16:23实现/回归：新增synchronous_io.py、launch_h73.py、目标/launcher测试，run_v2默认关闭且要求6-control/reserved-stop profile。物理context覆盖env.step及后处理，默认关闭时保持旧后处理在render context外；正常/异常保持统一逐调用时钟验证，不能重复env.step补帧。

原异常优先，失败后时钟不可读/恢复设置/日志失败不得盖住primary；ReferenceTime attach部分失败仍逐路清理，close幂等且发生于SDK仍活着时，成功result前封存唯一close＋journal SHA。监管检查相邻事务时钟连续、每capture与三sensor receipt绑定。视频用最近verified图并注明对应时间/帧保持，未称每control有新图。

完整semantic720/715通过5本地SDKskip28.385s，60重点3.231s过；初次全量AST测试缺新增closure造成13错误已修后重跑，另一次邻接模块名写错已纠正。最终独审/源固定/native API真实验证待。legacy fixed-four-render receipt新增unverified标记；旧SFT completion仍要求旧屏障协议，不将H73回执无检查混入训练，本轮未训模型。

16:25最终独审通过：独立38/38，父最终123邻接0.996s及diff过；异常保真/生命周期/时钟连续/默认关闭兼容问题全闭合。GPU0队友3564916/12548MiB、1/2/3空，准备固定源/Git后robo预检。仍0launch，实际ReferenceTime帧关联/AA/PT/timeline副作用与控制效果待原生验证。
