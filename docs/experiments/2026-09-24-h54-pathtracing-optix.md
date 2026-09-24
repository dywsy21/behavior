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

22:24源码已固定`7e706be67514c261aef94917b1bfa561478f2f80`/远端`git_worktrees/shared_scene_7e706be`，52远端CPU/16依赖过。但实时GPU0出现PID3480326 C+G39MiB，非注册四训练，严格身份门拒绝；未调用launch、未创建run/runtime、无H54 worker。正在只读核身份，原四训练仍各73644，不发信号/绕过检查。

22:26实际提交：短暂3480326在查身份时已自行退出，用途未知；22:24:55及之后仅原四训练，原严格身份/显存门重新通过，未增加白名单或停止任何外部进程。确认预定目录不存在后唯一launch UTC14:26:05.076513，supervisor3485875，固定source7e706be；run/runtime现已创建，原预算不变，真实图像/资源/退出结果待验，不重提。

## H54实际未通过及归档（22:29北京时间）

监管33.874304785s，3485875/3485882已结束；worker native退出0但worker receipt明确failed，监管正确拒绝成功。empty app已实际读回全部18项（原8＋PT10）设置；原og.launch返回后`/rtx/rendermode`却是`RaytracedLighting`，触本实现过窄的`RealTimePathTracing`前置断言，未执行第二次apply。即源码曾set RT2，并不证明多次native update后这个可变状态仍RT2；不能以这次失败否定PT/OptiX能力。

外层reset/load、RGB-D、actor/前缀/模型/训练均0；未载任务scene。42样本全部四训练各73644MiB、退出原余量恢复。GPU0/1/2/3自有峰456/416/416/568MiB，增量470/422/422/597，最低free7018/7068/7068/6891；不是显存门失败。所有source/runtime/run保留。

全部8件在本地`artifacts/agentic-vlm-goal-20260918/h54_pathtracing_bundle_v1`，4主SHA双端一致：

- launch `4caf0ccb5368c741f300559c20a5ff74a9605f92ca7f3cea663300ba9f6f6f4b`
- supervisor `8cc71e19a3403560bf9a03e2d6a46884e6bf7cdd1a8f4f4d2adcfbb08cd8a942`
- worker `86f5c3dee15935daeecb5af8538c91b6fdb7cc5df3be977447c19d62b5dbff05`
- kit.log `baf67982f7cf36b10d42a08957f22fac9bca4ec93d469dfd3389b8d9f7c1dba4`

正只读查设置来源；factory/官方Evaluator/既有chunk wrapper无renderer模式赋值。后继若修前置模式检查，仍须保持准确source/alias/空scene/无viewer、一调用、最终PT严格读回及所有资源门，不接受unknown配置，另用新run记录，不掩盖这次负例。

## H54b：修正前置mode断言的单次后继（22:32登记）

负责人Codex，唯一变化是允许初始化结束时实际观察到的合法`RaytracedLighting`，与原允许的`RealTimePathTracing`并列；不放行unknown/null，不更改最终目标PT十项/资源/来源/alias/空scene/无viewer/唯一调用。来源推断更正：OG虽在setter写RT2，其后还会play/stop和native app更新，**不能用最终可变设置值证明是否执行了原构造**。完整Kit日志没有记录每次设置writer，具体把mode改回的原生路径未确定，不声称已证明某个SDK回调负责。已核安装Kit UI将这两值列为合法实时renderer，官方factory/wrapper未改mode。

同一冻结外部环境/16依赖/原始TRAIN138/seed0、同PT/OptiX4spp/16累计/原三相机/物理/冷新runtime/CPU4；**另一次600s/1Session/原4096主512辅助3072余量，0actor前缀模型训练**。原H54源7e706be/33.874s失败全保留，既非重用run也不是修改正在运行的源码。入口仍`probe_scene_pathtracing.py`，新准确commit与run启动后记录。

新run `/mnt/nvme_tmp/robodojo_agentic_20260924/h54b_pathtracing_v1`，新runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h54b_pathtracing_v1`，本地包 `artifacts/agentic-vlm-goal-20260918/h54b_pathtracing_bundle_v1`，现在均未提交/创建。新增legacy前值接受、后值漂移拒绝及完整worker链回归，修后独审和双端同测过才唯一提交；不是放宽实际renderer验收。失败后不自行加资源。

22:35优先级调整：H54b独立复审/54 CPU过，但未部署/提交GPU，暂缓执行。另发现原r1pro配置三路1080初始化、环境完成后wrapper才变为head720/腕480，属于比渲染模式更直接的启动缓冲浪费。先做独立H55相机最终配置前移、最终分辨率/物理不变，原renderer对照；不把H54b写成已运行或把H55改善归给PT。H54全部证据/源码保留。
