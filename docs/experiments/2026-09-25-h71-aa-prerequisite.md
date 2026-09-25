# H71：非DLSS抗锯齿的配套选项修复

2026-09-25 14:54北京时间，owner Codex。H70实际source3642dd7，运行508.514s后在同一原session初始化失败；已捕获第一台相机clipping_range更新中的native app.update将`/rtx/post/aa/op`从0改3，其他9个登记PT值不变。其Python栈不是C++backtrace，H70没有读到limitedOps旧值，不能提前宣称该flag因果性已证实。

依据：服务器安装Replicator1.12.27的`core/scripts/settings.py:140–143`在非DLSS/DLAA选择时先清`/rtx-transient/post/aa/limitedOps`；[NVIDIA1.11.4变更记录](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/CHANGELOG.html)也注明此修复。[NVIDIA设置API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/API.html)将post AA0定义为None、3为DLSS。旧profile只设置aa/op=0而漏了配套限制。

唯一变化/假设：在原两个空app/空simulator边界，先设置limitedOps=false，再设置原aa/op=0，能避免首相机render自动回退。继续原PathTracing/4spp/16total/OptiX、三原720/480相机、物理、机器人和reset顺序；不改默认AA值、不设置多项候选、不在观察者/每帧自动写回。加入该flag读回/变更追踪/最终检查，若仍变或另有漂移则失败。`a100_full_v1`仍表示原目标图像配置，具体修复由新commit/digest绑定，旧源不改且不能用旧gate放行。

安装依据文件SHA `e4901268a16048062fe5c33bb3507752a907cb17fa7f4b6797835d8687816006`，精确路径为`/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim/extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/settings.py`；本票不修改该文件、不调用会切换为RT的set_render_rtx_realtime，只复用其非DLSS配置前置条件。

- task0 TRAIN138/seed0，reset→load138→reset，0专家/旧policy前缀、0模型/训练；H64资产SHA保持。
- 单次scene/session，24现有基础gate动作、≤1536controls；动作阶段≤1200s，总外墙钟≤1200s＋30自有清理。H69/H70已知冷启动约510s，本票为后续真实24动作留时间，并非延长原已结束run。失败停止，不自动重试。
- GPU3≤24576MiB、启动free≥32768/运行每卡free≥8192；自有SID在0/1/2辅助分别≤512MiB，0/1队友正常启停；2/3启动无外部进程。4CPU72–75，原资源门完全保持。
- run `/mnt/nvme_tmp/robodojo_agentic_20260925/h71_aa_prerequisite_v1`、runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h71_aa_prerequisite_v1`；新固定Git源码，无共享安装或活跃目录写入，不借旧shader缓存。
- 验收必须实际无设置漂移、原完整gate_ok/24动作/测量门、原reset计数以及人工原三视角；只越过H70断言不足以称harness通过。若输出为黑图/深度缺失则按真实失败处理。不验证press/VLM语义，更不称完整任务SR。

当前本地一行配套设置＋新单次入口已实现，CPU/独审/新source固定后再运行。

14:59独审要求关闭来源漂移：安装Replicator/settings.py的精确路径/SHA现已进入共用PT与compatible profile的DEPENDENCIES，父launcher与worker各自identity均检查，24安装文件缺失/改字节拒绝。新增对应CPU反例16目标过0.055s，原完整回归703/698pass5skip29.912s在此依赖门delta前；最终邻接/独审待，0launch。

15:03最终增量：113邻接0.970s，独立16 renderer、11 PT、3 compatible、18 native-gate和diff通过，无剩余实质代码阻塞；父/worker来源硬门确认有效，未修改安装文件，0H71 GPU。下一固定运行源、robo实际24依赖及资源预检后唯一原生运行。

15:06唯一提交：运行源0189148d7a77cb86420925f3e21f8a485ffca3f0，新clean Git worktree `aa_prerequisite_0189148`；真实113 CPU1.944s、24安装依赖、资产/资源门通过，两新run/runtime不存在后才创建。launch UTC07:05:54.900657、supervisor3578340；GPU0队友3564916/12548MiB保持，1/2/3提交时空。真实初始化/终态待，不重提。

15:12只读补核（不改变当前实验）：[官方API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/API.html)把post/aa/op列为RealTime设置，PT采样滤波则是pathtracing/aa/op；[PT文档](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)也区分此项。H70原kit日志仅证明该字段漂移使我们的断言停止，不能由此直接声称实际PT图像已坏或DLSS执行失败。安装源检索有多处官方测试使用limitedOps=false，但没有找到首camera native更新的Python setter。H71仍严格按原条件完成，若失败先区分真实renderer故障与检查合同过度限制，不中途放宽、也不据“PT文档没写”就认定完全无影响。

15:15实际进展：supervisor running/564.172s，worker3578347；native_profile/worker均reset2/load1且无差异，首HOLD18ticks TARGET_REACHED（EEF误差约0.8/1.4微米）。两个追踪通知仅spp/totalSpp同值恢复、AA0/limitedOps=false保持。候选已越过原初始化故障，但后续24gate、有效RGB-D和终态资源仍待，不提前称harness全部通过。

15:18初始图人工审：本地`artifacts/agentic-vlm-goal-20260918/h71_initial_preview_v1`是5件局部预览，不是完整run。本人看全部三RAW，头图可见壁炉/房间和厨房方向、目标收音机尚不在视野；两腕大部分是机器人本体近景、不是黑屏也不能当目标。PT初始图较软，后续视觉里程计及模型语义仍须实测，不能由有限深度替代这些验证。

RGB形状分别720²/480²/480²，非零像素比例均1；深度三个float32视图有限正数比例均1，头0.8415303–6.9996419m、左腕0.0766893–0.8394450m、右腕0.0766885–0.8424187m。这里只排除空图/空深度，不独立证明外参/尺度。

5件双端SHA256核同：

| 文件 | SHA256 |
| --- | --- |
| CURRENT_HEAD_RAW.png | 5ad14d1099add12405661379d2811ed9c2bc585850039341de8acd66df97f235 |
| CURRENT_LEFT_WRIST_RAW.png | e0541d9164b067ad33ccb3877144f2be01def8144e488940c110e54d3467dd97 |
| CURRENT_RIGHT_WRIST_RAW.png | 84d0ab94446810f1b7e0d5f9ba7e4b0f81d4a401fdc947984af0e5246b7a30fa |
| initial_depth.npz | 3981e8732b47746fdddd5ebd68613b2eee931c81566520929602198e18b6bcd1 |
| manifest.json | 3fafb6ff98b0fff1a3d2cd8427682bc3b27ff524a7b26f455bdd2799374c318c |

## 终态：15:20–15:24北京时间

原唯一run共732.7008375860751s，worker退出0，supervisor严格failed。12决策/232controls（末1安全保持），10个TARGET_REACHED、decision09躯干上被预检拒绝/0执行，decision11底盘后退BASE_TRACKING_FAILED。因此更正早期口头“前11动作通过”：不是11次实际执行成功。reset2/load138一次及最终renderer验证均完成，4设置通知仅同值spp/totalSpp，AA0与limitedOps=false保持；本票修复已实际越过旧失败点，但完整控制gate未通过。

底盘目标-60mm，速度积分raw=-57.2719mm（不能当真值），RGB-D整段=-30.9136mm、29.0864mm残差触发原12mm门；0模型/训练/专家或旧policy前缀，不是完整SR回合。监管/worker均已退出，自有显存全部释放，GPU0队友3564916/12548MiB保持。

全包本地`artifacts/agentic-vlm-goal-20260918/h71_native_gate_bundle_v1`，527件208990160B。双端将所有相对path/bytes/sha256按路径排序、JSON sort_keys=True/separators=(',',':')所得聚合SHA `06d00843c1db781e5ec041ff6c7b12cf23bfd84d9a85688babbafd47f71eb3c9`相同，逐文件身份被全量核验。关键SHA：result `e3f7550f8044d13bbc400d1046336aebf3d4a972422d10ab6bd93e6bf50fc41f`；supervisor `b590d9b1bec88ad9c801edd6951a66e69ad99ed713fa3016a08fbf5dd355cb1a`；视频 `d4637364e89af9382f11abb431eecab0dadfdbeea1ab7445be3b36ddf962b373`；动作011链 `769a169a5969429d4ba71854328675ef6e35b89b550f473e14c633505b897180`。

已验证的采样线索：control219与225的head深度逐bit相同（raw depth SHA `d0b9bc05f9d3071ac25081e7f1ca5cb3a0b9f045a0a207fb92a05bfb363a26f2`），RGB MAE0.0402469/255；中段估计1.34微米、560inliers，前段-5.8368mm/后段-25.0781mm。receipt虽然均写4render屏障且snapshot48→49，这不是原生渲染帧时间戳。不能只凭新snapshot/非同RGB噪声宣称fresh，也不能据非零命令宣称真实身体必然位移。下一只读检查SDK帧更新/时钟并CPU重放；0新launch、不降低位移门。
