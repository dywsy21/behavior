# H53：共享训练资源下的一次原始任务场景/RGB-D检查

负责人：Codex；登记2026-09-24 21:29北京时间。分支`feat/semantic-agent-grounded-20260918`，准确执行commit在launch回执及plan登记，提交前不运行。此票承接H52b空应用通过，**不是完整任务评测、不是Zetta演化复现，也不训练模型**。

## 单一假设与对照

H52b已证明显式GPU3/禁多卡/低纹理缓存的空Kit可以启动；还不能证明实际BEHAVIOR场景、三相机和物理可共存。本票只验证：保留相同资源限制和已有官方初始化流程后，一个真实原始任务场景能否产生非空、有效、无机器人运动的RGB-D。

对照为已封存H52b空启动，不重跑、不作任务成功率对照。正式任务成功仍需后续单独预算的原始起点闭环和官方终止判据。

## 固定输入与版本

- 模拟器：已有Isaac Sim5.1.0.0/OG3.9.1开发运行时，不称为v3.9.2正式挑战提交。
- 原始任务：`turning_on_radio`，TRAIN实例138，seed0。只复用冻结window的任务/机器人/实例配置；路径名字中的`e121-f448-grasp`**不代表本轮执行448步前缀**。不调用`window.frozen_window()`，不加载/执行专家或旧策略动作，也不使用window里的语义子目标来规划。
- 机器人R1Pro，真实23控制维；原head720×720、左右腕480×480，已有`rgb/depth_linear`模态，不重配传感器。
- 无权重、无模型服务、无优化器、无训练数据构造或训练；只保存公开机器人相机和自身关节状态。
- `probe_scene_startup.py::EXTRA_DEPENDENCIES`冻结OG simulator、官方Evaluator、旧Session factory/runtime、render wrapper/trace helper、window/robot和两个已有apps资源的SHA；公共监管另冻结SimulationApp源码和experience。启动各阶段均核依赖与干净Git，固定解释器。
- OG源码SHA`d800c2832f24962c440c4781ebfdb4ea78c74aac37d2c74ee7894ae430f1e2a9`；factory`5492910dac0a75aac9e8ca4e71f5b762aeff4e129c37d6ed4d8c995c312e5f70`；window`93731a793ed550bae225e148d872b97d42e331a709e2fd0cd33bd1ccf76ccbd8`；robot`a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93`。其余完整SHA在源码字典，不从新下载依赖。

## 执行边界与停止条件

仅一次新进程/一Session：沿已有Evaluator构造、一次API reset、load frozen instance138、再一次显式final reset，然后四次render-only刷新，取一次三路RGB-D，退出。

这里的“两次reset”是调用层计数，**不是声称引擎只运行两步或0物理初始化**：官方内部加载/settling的原有步数不改，总墙钟限额约束。actor控制、expert prefix、old-policy prefix、模型调用及训练步数均0。不调用隐藏物体状态/完成predicate来选择动作。

- 600s监管活动预算，最多额外30s只清理自有child；禁止自动重试/复用run。
- GPU3主要渲染/物理；原训练PID3294346–3294349/四UUID严格核对，未知进程或训练身份变化即拒绝/停止自有child。
- 启动四卡free各≥7168MiB；运行各≥3072MiB；主卡新增及自有进程≤4096MiB，辅助卡≤512MiB。**不继续提高H52b辅助额度**，也不同时装VLM。
- 与H52b同纹理流设置0.01/16MiB，同GPU实际8键读回。CPU72–75/Kit线程4；私有portable/cache/temp/appdata。不更新共享环境/活跃源码、无全局shutil或已安装文件patch。
- `shared_og_startup.py`只在本进程OG simulator模块保留原`_launch_app`完整流程，将已存在且双端SHA一致的两次apps复制变成逐次核验no-op；调用已验构造器。未知复制、config/experience漂移、重复/嵌套/失败后重试一律拒绝；离开context的闭包失效；恢复所有临时引用。

## 验收

当前作者36 CPU通过：原15监管、9桥接、10场景，以及2项原RGB-D刷新/不重配相机回归。首轮runner独审发现depth实际数组/回执与init reset/load计数缺口，现已补三路深度键/shape/dtype/RAW哈希/有效比例、实际4 render调用、外层Session reset/load/reset序列观测，并绑定sensor/backend实际Git导入；修后独审待。外层计数代理只包装新建Evaluator实例，内部settling仍保持原有。远端必须同测试及依赖/实时资源预检通过，才唯一提交。

21:36北京时间修后独审闭合，独立36 CPU全过，无剩余实质阻塞。下一固定代码并作远端同测试/资源门，不把CPU通过当场景通过。

实际通过必须同时满足：监管退出0/训练保留、所有限额无越界、一次启动/构造/Session/final reset/捕获准确、外层reset→load138→reset三事件完成、两copy验证/0安装写入、三原分辨率RGB-D/RAW哈希、实际4 render-only与回执相符、head非空且有效深度比例≥10%（双腕各>0）、捕获前后自身关节无变化（最大差≤1e-5）、全部小件hash核同并人工检查图像。只通过CPU、构造app或worker文件存在都不算通过。

无模型或机器人执行，`success_rate=null`；即使通过也不能声称完整场景+VLM同时共存、动作能力或任务成功率提高。后续联合资源预算必须考虑Kit在非主卡的辅助context，不能把两个独立探针余量简单相加。

## 位置与结果

- 新结果预定：`/mnt/nvme_tmp/robodojo_agentic_20260924/h53_original_scene_v1`。
- 新私有runtime预定：`/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h53_original_scene_v1`。
- 本地完整小包预定：`artifacts/agentic-vlm-goal-20260918/h53_original_scene_bundle_v1`。
- 实际run尚未启动，以上预定目录不当作已存在；实际结果和源码commit启动后追加。

21:37北京时间实际启动：源码`ab01d2777084cfa8ccf72a56748a0fb6a296b92d`，独立`git_worktrees/shared_scene_ab01d27`。双端36 CPU、修后独审、依赖与实时资源门过；上述新run/runtime现已创建。唯一launch UTC13:37:43.218613，supervisor3469426，真实场景与终态待验。
