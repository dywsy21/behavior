# H57：不删视角，降低三路RGB-D缓冲

2026-09-24 23:28北京时间预登记，负责人Codex，分支`feat/semantic-agent-grounded-20260918`。准确代码commit由launch冻结；当前仅CPU实现/验证，未启动GPU。

## 假设与范围

H56在创建render view附近，主卡整卡增量由约1.4GiB快速升至4466MiB，触4096MiB/3072余量保护；完整三路RGB-D尚未获得。不能由时间相关性分解BVH、纹理、渲染buffer各自占比。主要假设：只降低实际相机像素，保留三个视角和H56渲染器，可以在原共享显存门内取得可用观测。

head720→512，双腕480→320，总979200→466944像素（减少约52.3%）；不预言显存同比下降。不是缩放已拍好的图，也不删除相机；在构造前设置，wrapper仍只读、严格核初始/当前尺寸及sensor/render-product对象身份。所有焦距、孔径、裁剪、姿态/视野相关配置及physics复制不改，RGB和线性深度尺寸一致。捕获前后从真实sensor读K，核有限/正焦距/主点范围/相同K，不拿720的旧内参硬套512。

这种改变会降低小按钮等细节和深度空间采样密度，**不能宣称与720/480的信息完全等价**。三张RAW和实际深度必须人工审，后续actor必须使用本次真实标定；H57本身不是成功率评测。若原资源下仍失败，先审原因，不继续逐级盲降像素到画面失用。

## 固定实验票

- 新入口`probe_scene_reduced_cameras.py`，显式`shared_v1`；H53–H56默认仍`full_v1`。H56的PT/4spp/16total/OptiX、禁viewer/私有runtime不变；23外部依赖SHA均冻结。
- 原task0 TRAIN138/seed0/同window/robot，原OfficialSession reset→load138→reset，0演示/旧策略前缀、0actor、0模型、0训练；不改变任务实例/对象/物理/官方成功判据。
- **单次600s活动＋最多30s自有清理**，1Session/1组三RGB-D/4render-only；GPU3整卡增量及自有≤4096MiB、辅助每卡≤512MiB、运行全卡free≥3072/启动≥7168。四原训练PID/UUID不变，不停止/修改训练或放宽门。
- 新冷runtime，CPU72–75/Kit4线程/原texture限制；不拷贝缓存或热改旧源码。不是延长H56或同配置重跑。
- CPU负例、完整mock构造/reset/capture链、独立审查、远端同测和真实YAML转换/23依赖/资源门后才允许一次launch。原capture字节/有限深度/物理静止校验与全部success字段保留，新增实际内参验收和明确`input_resolution_changed=true`。

输出`/mnt/nvme_tmp/robodojo_agentic_20260924/h57_reduced_cameras_v1`，runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h57_reduced_cameras_v1`；本地全包预定`artifacts/agentic-vlm-goal-20260918/h57_reduced_cameras_bundle_v1`。当前均未创建，无新物理结果。

23:31独审发现并修复真实接口缺口：首79目标及582旧接口虽过，OnboardRGBD构造仍硬编码原720/480，mock替身掩盖了与512/320冲突。为真实adapter添加显式三尺寸mapping，默认保留官方尺寸，未知键/bool/非正或非整数拒绝；不删尺寸校验、不从live sensor自动猜测允许值。增加真实adapter构造及4render/read检查，完整worker mock现在也断言实际参数。修后重测与复审待，仍未launch。

23:33修后独审通过，独立复跑81/81与diff过；真实adapter/完整worker参数、profile恢复/三视角/renderer/物理及资源门均无剩余阻塞。下一冻结commit/服务器同测和原YAML转换/23依赖/资源门，尚未运行。

23:34实际launch：`73c11236c36cf63b789a158050ac89a1b38b483f`/`git_worktrees/shared_scene_73c1123`，双端81、修后582旧接口、独审及原YAML/OmegaConf/Hydra/23依赖/资源门均过。UTC15:34:22.948928/supervisor3503152，原输出与新runtime创建，实际图像/内参/资源终态待。**原launch继承了H53b的`actor_camera_resolution_changed:false`旧标记，漏覆盖；本次明确改变了相机实际尺寸**，其`input_resolution_changed:true`、`shared_v1`/head512双腕320及后续实际shape才与实施一致。保留原launch/source，不能篡改运行证据；下一本地修metadata并补断言，不重跑本次。
