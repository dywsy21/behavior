# H55：首次创建即最终RGB-D尺寸，避免live相机重建

负责人Codex；2026-09-24 22:45北京时间登记，分支`feat/semantic-agent-grounded-20260918`。67项CPU通过，独审中，尚未启动GPU；准确source commit由本票首次launch记录并同步plan。

## 唯一假设

原冻结`r1pro.yaml`给全部VisionSensor 1080×1080；官方Evaluator完整构造Environment后才instantiate RGBDFullResWrapper。现用chunk runner保留此顺序，逐相机加depth、依次设置height和width；冻结VisionSensor这两个setter即使值相同也会detach、destroy/recreate render_product并各render4次。H53b虽最终要求head720/腕480，启动期间并非这个尺寸。不能以最终图像合同推断启动缓冲已优化。

**将最终RGB-D配置前移到相机首次创建、后续只读校验，有机会降低初始化显存峰值，在原共享资源额度内完成原始任务观测。** 3499200→979200是相机像素总数，不是实测GPU节省比例；渲染、几何、材质等开销还在。对照H53b同原renderer/无viewer，H54b PathTracing候选暂缓、不叠加。

## 实现与不变量

- `scripts/semantic_robot/shared_camera_config.py`仅CPU导入，复制resolved evaluator config；依据robot.name与eval.camera_sensor_names构造3个per-link key，复制整个VisionSensor条目后仅设head720×720/双腕480×480、rgb+depth_linear。原class entry与原YAML不改。安装版Robot的per-link优先class且不做两者合并，故不能只复制两尺寸而丢其他kwargs。
- 严格拒绝不是原1080/原RGB-proprio、未知wrapper、已存在其他sensor overrides、缺少/重复/外机器人camera角色的输入；除这3个新增sensor_config条目及wrapper target外完整config深比较一致，记录原/生效config SHA。
- 私有Hydra target `wrap_preconfigured`构造原EnvironmentWrapper，校验原3个视觉传感器及initial `_load_config`/current尺寸、模态、实际render_product存在；**不设置任何live尺寸/模态、不reload space、不删camera、不调用控制或渲染**。保留`_behavior_deferred_space_reload=True`，既有冻结chunk runner仍在原Evaluator完成articulation handles之后reload一次space。
- evaluator构造完成、原Session最终reset后、4次render-only捕获后均校验sensor/render-product实际对象未替换，全部最终观测仍为head720/双腕480 RGB+linear depth，不允许附加segmentation等特权模态。
- `probe_scene_preconfigured_cameras.py`独立profile默认关闭，H53/H53b/H54入口原行为不变。H55与PATH_TRACING同时启用直接拒绝；原scene/物体/光照/材质/控制器/底盘质量/相机光圈/频率/23D映射及官方内部settling不改。
- 最终分辨率与物理配置相同不意味着逐像素/轨迹比特一致；初始化渲染数量与深度启用时机改变。成功后仍需人工看三原图/深度，不能只靠nonblank判可用。

## 固定输入与预算

原task0 `turning_on_radio` TRAIN138/seed0、同原window/robot SHA、Isaac5.1.0.0/OG3.9.1开发运行时；文件名f448并不执行演示前缀。无权重/训练数据变化，无actor/model部署。冻结所有H53b外部依赖，并增加robot.py、vision_sensor.py、sensor factory、eval utils、原RGBD wrapper、EnvironmentWrapper的准确SHA（入口代码为权威清单，启动逐一核验）。

一次新worker/1Session，**600s活动监管＋最多30s自有清理**；原外层reset→load138→reset、一次三路RGB-D捕获、4次render-only；0actor控制/专家前缀/旧策略前缀/模型调用/训练。原四xhz训练PID3294346–3294349/四UUID保留；启动各卡free≥7168MiB，运行free≥3072MiB，GPU3新增及自有≤4096MiB，辅助每卡≤512MiB。任何未知GPU进程、训练身份变更、资源越界/超时只停止自有worker，不放宽白名单或显存/时间重试。

CPU72–75/4 Kit线程、纹理streaming0.01/16MiB、多GPU禁用/主GPU3不变；新冷私有runtime，不触其他运行、安装、缓存或旧证据。Git固定独立worktree，不热pull。

## 验收与位置

67项本地CPU涵盖原H52/H53/H53b/H54回归、完整H55 mock worker/原reset链、配置非相机完整保持、modality/initial1080/对象替换等负例及禁止live setter。还需独审、远端同测和真实OmegaConf载原YAML的CPU转换验证，最后原GPU身份/资源门才唯一提交。

通过需worker完整真实事件/receipt、3RGB-D字节及深度覆盖、4render/关节不变/0actor、实际初始与最终相机尺寸相等且对象身份不变、退出0/全部资源样本过门、三RAW人工审。完整失败也归档，不以worker exit0、launch或本地目录存在当通过。本票**不产生success rate分母，不证明完整任务成功**；后续仍需零前缀官方物理成功。

- run：`/mnt/nvme_tmp/robodojo_agentic_20260924/h55_preconfigured_cameras_v1`
- runtime：`/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h55_preconfigured_cameras_v1`
- 本地全包：`artifacts/agentic-vlm-goal-20260918/h55_preconfigured_cameras_bundle_v1`

当前上述运行/目录未创建，准确source/PID/实测结果随后记录；H53b/H54所有负例和H54b待运行源码保留。

22:48更新：初审发现基类Wrapper的属性代理语义也参与receipt存储，已将`utils/python_utils.py` SHA `2d6ba4bc43b07a4e266f3b7048de2cd72efcb39f817acadeafb9488577a88869`纳入冻结和profile回归，总21外部依赖。原67目标CPU＋582接口回归通过，修后窄独审待；尚无GPU进程。

22:49修后独审通过，独立67/67与diff检查过，无其他实质阻塞；按原预算进入Git固定/远端CPU及真实配置转换预检，不把mock通过当相机创建或显存通过。

22:49实际启动：固定`4cd5b0adda144455e4c66495d050d26b001f9852`/远端`git_worktrees/shared_scene_4cd5b0a`，67远端CPU/21依赖/原YAML真实OmegaConf深复制与roundtrip/Hydra target解析均过（全程未导入torch/OG/Isaac），实时原四训练/显存门通过。唯一launch UTC14:49:25.195096，supervisor3489219，run/runtime现已创建；源不可热改，原预算不变，真正scene/三RGB-D/终态待。

## 实际结果与归档（23:02北京时间）

H55未通过：3489219/3489226已退出，监管602.738772275s，worker -15，600s wall budget触发（error字符串沿用底座H52标签，不代表本票未载scene）。原scene日志360.992s Imported scene，原机器人构造随后开始；Kit563.718s出现内部240×240 render resolution及A100 DLSS-RR不支持警告。末600.251s外层reset/load/捕获/actor/前缀/模型/训练均0，预配置已实际传入原Evaluator，但wrapper后验与三RAW未形成。

743资源样本全核：GPU0/1/2/3自有峰456/416/416/3317MiB，整卡增量峰470/422/422/3349MiB，最低free7018/7068/7068/4138MiB；四训练样本全部原PID/73644MiB，退出各卡free约7489MiB。**未触显存门，不等于整个初始化最终峰值已可容纳；不能与更早因显存终止的H53b作最终峰值等阶段比较。** 也不能把A100警告时间关联当作证明超时由该插件唯一造成。未临时延长时间、增加显存或重跑。

完整8件本地`artifacts/agentic-vlm-goal-20260918/h55_preconfigured_cameras_bundle_v1`已取回，四主SHA与远端一致，原source/run/runtime保留：

- launch `c87a1f86904572c897e99c1fa54201f5382baef4adc8dca5fb6fa78580b7f09c`
- supervisor `87c9371879bf02e4bef0eb09139c5e9b98a8afe44b07cfb8f80fd0aaa740c79a`
- worker `12dcd92b854998b9eb1645f9bcc688715d265b4621bd2efec497a67a6bb3eda3`
- kit `9b26483b70ba48de8355fe807b10e95f2ba50ee11fd556e623093334b5830a06`

后继考虑H56：在H55同相机前移配置之上仅换已有独审的PathTracing/OptiX兼容profile，单独登记，保持原600s与显存门、冷新runtime和physics。此时H55是camera-config固定的对照；不回到1080的H54b，也不在H55结果里混称PT作用。H54b继续未运行，H56在CPU/独审前不启动。
