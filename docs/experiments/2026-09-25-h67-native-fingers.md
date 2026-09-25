# H67：R1Pro机器人单体原生手指几何核验

负责人Codex；2026-09-25（北京时间）预登记，代码commit在启动前固定。不是任务评测，不生成微调数据；无任务/实例，seed0，无权重。

## 唯一主要假设

H63/H65的portable FK在实际不同手指开度、左右不对称开度、手腕转动后，与原生PhysX读取的link pose一致。只在导出参考姿态上自洽不能回答这个问题。所有1912个H64声明的碰撞顶点也随原生和portable变换分别运输，比较位置；不将它叫作PhysX cooked接触面。

## 固定实验

使用原安装OmniGibson3.9.1 / IsaacSim5.1.0.0、H64同一R1Pro3.8.2 USD/机器人定义和竞赛`eval/r1pro.yaml`。控制器/真实23维动作定义、自碰撞、assisted grasping、初始关节和原CPU PhysX保持；仅诊断场景为`Scene`＋地板＋机器人，不载房间、任务对象或skybox。机器人观测modalities和sensor允许列表均为空，构造前关闭viewer，不创建相机、不调用VLM。

原30Hz控制/渲染、120Hz物理保留。沿用H56已审的进程私有PathTracing/OptiX启动配置，避免A100不支持的渲染路径；没有图像可据此宣称视觉性能不变。私有启动桥只把已校验相同的安装资源复制变成no-op，不修改共享SDK。

九个样本依次为：

| 样本 | 左两指范围分数 | 右两指范围分数 | 左/右最后手腕关节偏移rad |
|---|---|---|---|
| reference_open | .9/.9 | .9/.9 | 0/0 |
| symmetric_mid | .5/.5 | .5/.5 | 0/0 |
| symmetric_narrow | .1/.1 | .1/.1 | 0/0 |
| left_asymmetric | .2/.8 | .5/.5 | 0/0 |
| right_asymmetric | .5/.5 | .2/.8 | 0/0 |
| both_asymmetric | .2/.8 | .8/.2 | 0/0 |
| wrist_rotated_open | .9/.9 | .9/.9 | +.12/−.12 |
| wrist_rotated_asymmetric | .2/.8 | .8/.2 | +.12/−.12 |
| return_reference_open | .9/.9 | .9/.9 | 0/0 |

分数由各原生joint自己的实际上下界转换，索引由原生controller定义取得，不假设full-q顺序或两指均值。每次全关节位置赋值、清零关节速度，调用一次原生`step_physics`后取实际q/link/J。原native setter清除articulation缓存并reset控制器；如果控制器或物理使实际非对称开度未达到，覆盖门必须失败，不能用请求值替代测量。

只在第一个实测姿态生成一次机器人参考校准；不调用依赖相机的生产calibrate、不为其伪造相机，输出标记`not_deployment_calibration`。余下姿态不重新拟合。主对照是原生link APIs，而不是另一份portable实现。

## 验收与预算

- 开度覆盖最大偏差≤5mm；身体q18最大偏差≤.003（原生混合单位，仅覆盖门）。FK各左右EEF/躯干和四指位置误差≤0.25mm、角度≤.001rad；所有声明碰撞顶点的最大位置差也≤0.25mm。阈值在观察结果前固定，非接触/成功判据。
- 每次CPU回归≤120s；一次原生运行≤600s，含冷初始化，原监管额外清理≤30s。不自动重试或追加样本。
- 原四训练3294346–3294349、主GPU3自身及增量≤4096MiB、其他各≤512MiB、运行各卡余量≥3072MiB、预检≥7168MiB不变；4CPU72–75。只允许停止自有worker，未知进程/身份改变/越资源门即失败。
- 最多一个Environment、一次校准、9次关节赋值、9个显式物理步；初始化内部load/reset/物理单独说明由600s约束。0任务实例/官方reset、0政策动作、0模型/训练、0RGB-D。不把校准teleport称为无物理动作。
- 任何几何/姿态覆盖/API失败即停止，保留已测样本与错误，不因部分通过改写为全部通过。监督器终态及清理结果也是完成条件。

输出：`/mnt/nvme_tmp/robodojo_agentic_20260925/h67_native_fingers_v1/`；私有runtime同日`robodojo_sim_runtime_20260925/h67_native_fingers_v1/`。源脚本`scripts/semantic_robot/probe_native_fingers.py`，测试`tests/test_native_fingers_probe.py`。

H64输入JSON SHA `5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04`；USD/定义/安装源SHA在脚本固定并在每个启动阶段校验。完成后把小体积结论、全量产物路径/SHA和失败样本保留到本文件与plan。

## 尚未证明

代码/CPU测试乃至H67原生通过，都不是完整任务成功或按压有效。尚需PhysX实际cooked几何/接触offset与接触执行状态机检查、视觉—目标语义对齐和原reset零前缀闭环。H66的数据捷径问题也不会被几何校准解决。

当前状态：9新CPU通过0.797s、39启动/保护回归通过0.705s、全semantic681中676过/5本地SDK skip（16.935s）。独立9/9过1.466s、无可复现阻塞；仅提示compare抛异常前coverage证据可更完整，已加即时保存。原生run未启动，不将本地mock当SDK验收。

12:42（北京时间）**启动前登记更正**：原覆盖门0.1mm改为5mm，但FK误差0.25mm/.001rad不变。只读SDK源码证实smooth gripper的reset no-op将两指目标平均，compute_control广播后送joint position targets，一步物理可能产生真实跟踪漂移。本票应比较实际姿态的FK，而不是检验控制器跟踪teleport目标。5mm范围仍使10%/50%/90%三档彼此分离，20%/80%非对称两指至少相差20mm；完全恢复均值的15mm偏差仍被拒绝。变更时0新app/robot/原生样本，未在看结果后调门；原始0.1mm登记留此明确更正。实际gripper controller SHA新增到启动依赖。

## 原生单次启动

源`95a7bfeefdb14ca80ada14c16865f2d0a9033b6e`；远端干净worktree `git_worktrees/native_fingers_95a7bfe`。服务器9/9 CPU过2.362s，17份依赖SHA/实际YAML/资源门过，UTC04:46:35.487310（北京时间12:46:35）唯一启动，supervisor3553910/worker3553917。当前运行中，未得到九姿态结果。

12:53只读日志显示空Scene在相对00:05:56.782导入，随后开始R1Pro构造；进程6分27秒时累计CPU17分48秒，仍在做初始化。此时不能按worker的粗粒度`constructing_app`字段误判仍未创建app：pathtracing回执已证实空simulator构造成功。尚未越过run_samples入口，0关节样本/0VLM。冷启动开销是实测现象，具体耗时分解尚未完成，不把它当显存已够完整场景或任务有效。
