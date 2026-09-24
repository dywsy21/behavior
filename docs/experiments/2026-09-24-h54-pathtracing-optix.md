# H54：原相机分辨率的PathTracing/OptiX共享资源检查

负责人Codex；2026-09-24 22:18北京时间登记，分支`feat/semantic-agent-grounded-20260918`。当前50 CPU通过、独审中，尚未部署/提交GPU；准确源码commit在launch/plan固定。

## 假设与证据

H53b无viewer仍在机器人/渲染初始化时超过主卡4096MiB增量门，未完成reset或观测；全704资源样本/8件证据封存b11932b。冻结OG源码`Simulator._set_renderer_settings`强设`RealTimePathTracing`，日志明确警告A100不支持当前DLSS-RR。

[NVIDIA RTX特性表](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_architecture_support.html)列明A100的DLSS-RR为NO、OptiX Denoiser为YES，并提醒非RTX显卡运行SDK没有支持保证。因此**只确认不同降噪组件的能力，不能据此宣称A100获得完整Isaac/OG官方支持**。[实时2.0](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html)及[legacy](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt_legacy.html)均说明RR非可选；不采用“只换legacy就能避开RR”的错误推断。[Interactive PathTracing](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)有独立的OptiX降噪路径。

唯一主要假设：改为原生分辨率PathTracing/OptiX后，三路RGB-D能够在原共享显存门内形成可用图像。**不保证节省显存/更快；渲染模式、采样与降噪构成整套兼容profile，不分开归因某个参数**。不是任务SR试验，也不是部署actor或新训练。

## 固定输入、变化与不变量

同H53b原始task0 TRAIN138/seed0、R1Pro23D、原window/robot/所有14依赖、Isaac5.1.0.0/OG3.9.1开发运行时；额外冻结`omnigibson/__init__.py` SHA`c7867c236051fd8994c75973284b5e88b2637c5b8ab1fece9c6e29bdbd9ca03f`及`envs/env_base.py` SHA`ab3e0cefd46a583a8fa9e8ccd42e5cc29590f0b9efddfb0a6f34dab74e4f6c7c`。正式比赛版本/提交资格不据此改变。

新入口`probe_scene_pathtracing.py`、私有接线`shared_pathtracing.py`；原H53/H53b/空Kit入口不默认改renderer。所有配置从Git固定，无CLI任意开关。

- PathTracing；后处理AA=0，显式pathtracing DLSS关闭。
- 每帧4spp、累计上限16spp；最大bounce4、specular/transmission6。
- OptiX开启、temporal mode关闭、blendFactor0（仅降噪后图像）。这不是用DLSS低分辨率上采样模拟原相机。
- 无旁观者相机，机器人head720×720/双腕480×480、RGB+linear depth不变；不resize/remove已创建传感器，不改机器人几何/碰撞/控制/物理频率，不删场景对象、材质或光源。
- 保留同纹理缓存0.01/16MiB、GPU3主设备/禁止多卡渲染、CPU72–75/Kit4线程。新的冷私有runtime，不复用其他run缓存，不更新已安装软件。

图像光照积分、噪声与细节分布可能改变；只能称兼容性候选，不能与旧RT2按“完全相同观测条件”混算性能或SR。4spp/16累计是否足够看清任务小物体需人工验，非空图像不等于可部署图像。

## 启动边界与实际证据

同原OfficialEvaluatorSession。官方`Environment`在装scene之前动态调用`og.launch`；该alias须实际等于冻结`simulator._launch_simulator`，首次进入须og.app/og.sim均空。进程私有one-shot wrapper先原样转发原调用与参数，保留Simulator构造/physics/renderer/Fabric/MDL等所有初始化；原函数返回后，检查返回值正是og.sim、app存在、**实际scenes为空且viewer不存在**，再设置profile。不在constructor中删掉官方物理设置，不热改活跃camera。

记录原renderer必须RealTimePathTracing、新十项设置实际读回；外层reset完成后和三路捕获后再核profile，任何漂移失败。失败、重复、嵌套、context逸出、跨context重试均拒绝；退出恢复og.launch引用，不在活跃相机存在时反向切renderer。共享安装文件不写。无模拟器特权状态进入actor；这里只读场景空/自身设备身份来证明初始化边界，未调用策略。

预算与H53b相同：**一次新进程/一Session、600s活动监管＋最多30s自有清理**，外层官方reset→load138→reset、4render-only、一次三RGB-D捕获；0actor/专家前缀/旧策略前缀/模型/训练。官方内部settling照旧，不宣称0物理初始化。启动四卡free≥7168MiB，运行≥3072，GPU3新增且自有≤4096MiB，辅助各≤512MiB，原四训练PID/UUID不变；越界只停止自有child，不修改训练。

通过须原H53完整来源/外层真实事件/RGB-D字节/深度有效比例/4刷新0控制/关节不变/退出0/资源样本验收，加实际PathTracing设置与三RAW人工核验。全量小包/远端SHA、GPU真实active表均核对。不以CPU通过、launch提交、文件存在或换了renderer配置当通过。若仍失败保留原证据，不自动加显存/时间或重复运行。

## 位置

- 预定run：`/mnt/nvme_tmp/robodojo_agentic_20260924/h54_pathtracing_v1`
- 预定runtime：`/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h54_pathtracing_v1`
- 预定本地完整包：`artifacts/agentic-vlm-goal-20260918/h54_pathtracing_bundle_v1`

当前未创建远端run/runtime、无GPU/模型/仿真/训练。源/预算/CPU测试及修后独审通过后，只提交这一票；真正零前缀完整任务成功仍是后续独立验收。

22:22北京时间补充：已核安装版SimulationApp构造会把`totalSpp`重置为每帧`spp=4`，故CLI16本身无效。现于明确拥有的empty app创建完成后重应用profile，再执行原严格设置读回；原OG的RT2覆盖则仍在空sim边界重应用。场景/相机存在后只校验，不再设置。52项目标CPU通过，包括两阶段覆盖及完整mock worker/原reset链；原582项semantic_robot回归通过。初审未见其他边界阻塞，修后独审待。`/rtx/post/aa/op=0`只指禁后处理AA，不把它混称路径追踪的采样pattern键；OptiX设置为true也不独自证明native实际执行了OptiX，真实Kit日志/图像仍须验收。

22:23修后独审闭合，52/52目标CPU/语法/diff通过，无实质阻塞；开始固定commit和远端同测试/16依赖/四卡资源门，未提交H54进程。
