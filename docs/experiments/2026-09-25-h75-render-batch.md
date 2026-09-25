# H75：把渲染批次与冻结物理状态绑定

2026-09-25 17:20北京时间，唯一owner Codex；实现/源commit/真实结果待。H74已封存，不能用其三份全局ReferenceTime证明三相机新鲜。

主要假设：原生逐render-product的SdFrameIdentifier，加上Replicator同步采集和RGB/depth原产品绑定，能排除旧批次/错相机/混合批次；physics与Fabric参考时间不再直接比较。

- 不改物理、相机分辨率、底盘增益或原24动作门。保持每control实际4 ticks、1/30秒。
- 只在既有三个render product附加原生帧标识节点。检查RGB/depth原annotator确实绑定该产品，节点确实接该产品的PostProcessDispatch；无对象/私有位姿进入actor。
- 一次零物理推进的prime并丢弃，不作为actor观测。每次force-Fabric render后取基线，再同步capture；新scheduled、全局完成及三个RP帧时间同批且严格超过基线/上一批，整数有理数比较。读取/复制全部RGB-D后再次核同批与物理时钟，禁止撕裂观测。
- 逐RP帧标识不是每个像素buffer自带独立时间戳；RGB/depth完成仍同时依赖已安装Replicator同步API及真实产品绑定。安装同版本文档/原生实跑验证，不把mock当SDK已支持。
- API依据：[Replicator1.12.27](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html)、[SdFrameIdentifier0.6.13](https://docs.omniverse.nvidia.com/kit/docs/omni.syntheticdata/0.6.13/GeneratedNodeDocumentation/OgnSdFrameIdentifier.html)、[逐产品事件](https://docs.omniverse.nvidia.com/kit/docs/omni.syntheticdata/0.6.13/GeneratedNodeDocumentation/OgnSdOnNewRenderProductFrame.html)。新增安装源pin，不改共享SDK。

## 唯一原生预算

先CPU、独立审查、固定Git、robo独立worktree/同源测试与资源预检，再一次task0 TRAIN138/seed0原reset→load→reset、零前缀gate。最多1200s＋30s清理、24gate动作/1536controls、0模型调用/训练更新；任一时钟/批次/原动作门失败即停，不自动重提。

GPU3自有≤24576MiB、0/1/2辅助各≤512MiB、全卡≥8192MiB余量，CPU72–75；队友0/1任务保留。新run `/mnt/nvme_tmp/robodojo_agentic_20260925/h75_render_batch_v1`，runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h75_render_batch_v1`。task3 gate/actor尚不启动。

验收只代表harness的普通gate，不是完整任务成功率。新PT微调标签仍须实际动作时钟和人工图像检查；此开发实例不回灌训练。

17:29实现首回归：新增render_batch模块，真实frame节点/exec与renderResults同dispatcher/两buffer产品绑定，全局与各相机帧末尾重读；一次prime弃帧，每次force-render后基线，复制后batch/physics/hash复核。异常读仅在异常清理路径允许弃掉pending以执行唯一安全hold，不能正常绕过。62目标CPU0.594s/diff过，完整回归/独审待；0新native/模型训练。

17:33独审/CPU：父完整737（732pass/5SDKskip）27.865s、144邻接1.050s，独立38/38/diff过；H74归档独重算相同。安装Annotator._add_auto_sync_gate对唯一PostProcessDispatch输入不插入额外gate，attach把HydraTexture转真实path列表。仍只放行一次原生诊断，不声称API实跑或RGB-D逐字节独立帧号已证；新source/双端预检进行，0launch。

17:34非阻塞测试缺口也已补：真实runner AST异常闭包验证弃读在唯一安全hold之前、close在后，且弃读发生于原异常上下文。最终738项/733pass5skip27.723s、149邻接1.498s全过；未增加行为或预算，准备固定source。

17:36唯一启动：robo独立clean render_batch_c54ade1固定c54ade102f6675e00e03649cb71b2ccfd182dbcd，149同源CPU3.019s/原安装依赖＋6同步pin/资产/资源过。digest `7bad2e9021b6a48dd298e571b213c19e763453b8c593ba871afbc404f0c765f8`；launch UTC09:36:06.136403、supervisor3603301，run/runtime已创建。真实worker/初始化待；0模型训练，保留队友GPU0/3564916，所有源/预算固定，不热改或重提。

17:47实际同步路径已运行：worker3603308、119controls/27capture-read/1prime/1initialize全部completed，共175journal行。第27次baseline schedule427/30、global431/30，三RP旧frame427；完成后schedule/global/三RP437/30精确相等，原生type=FrameNumber，三产品分别Replicator_02/head、Replicator/left、Replicator_01/right，RGB/depth实际hostPtr绑各自产品；physics509/time4.241666887886822全capture/read不动。证明安装API与当前批次契约已实际通过这些采样，不代表完整24gate或任务SR，原预算继续。
