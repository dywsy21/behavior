# H64：从真实机器人碰撞资产定义按压参考

2026-09-25 11:31北京时间续接；owner Codex，分支`feat/semantic-agent-grounded-20260918`，源码commit `0f8321d5679b9d2bab6b360e20d2dec2914c9e2f`已push。前置H63为`11dfe4b`，原642 CPU及独审通过，仍默认关闭、未native物理验收。

## 假设与范围

现有press进度和候选预测都用两指夹持空隙中心，不能保证实体手指接触目标。R1Pro实际USD显示每根手指是多个带非均匀scale的`convexHull`碰撞分片；不能把visual mesh、AABB角点或assisted-grasp条带当作真实按压面。

本票先导出四指资产声明的碰撞分片及各自finger-link局部坐标，所有缩放和中间变换恰应用一次。结果不含场景物体、按钮marker、物理contact、任务成功状态或实际当前finger角度；**authored convexHull不是运行时PhysX cooked表面或接触offset的验证**。后续固定接触点应关联finger/link/mesh/面内坐标，并用H63当前proprio/FK同时计算距离和候选后的同一点，不能每次偷偷换成另一根手指来制造进步。

## 输入与预算

robo原资产目录：`/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/`。

- `usd/r1pro.usda` SHA `6029617cdd3aefce981428058a1c82cffe6af20ae61dc5be1da7334342c3bc52`；约126MiB。
- `r1pro.yaml` SHA `63f841cffd5c499102416a797a22fa5b4cbaf146539df7dde8a4a1053e2326f1`；3339B。
- 原安装内USD0.24.5与`omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311`目录；不安装/修改环境，只给新诊断进程配置包和动态库路径，禁止写字节码。
- 一次实读，外层`timeout --kill-after=10s 60s`强制60秒活动时间/10秒清理，2CPU、CUDA不可见，输出≤2MiB；0模型调用/训练/SimulationApp/仿真reset/任务实例/动作，不属于SR分母。不修改原资产，不覆盖已有输出。

## 验证状态

本地34相关CPU过（新增4个纯数值/名称/拓扑测试），4个真实SDK内存stage测试明确skip。真实SDK测试覆盖root/link位姿消除、嵌套mesh非均匀scale、错误单位/缺link/不支持approx/禁用collision，以及visual与其他link不进入输出。**须在robo实际SDK下8/8通过，源固定并独审后才进行唯一实读**。

独审进行中，已按反馈固定SDK版本/来源，强制超时由外层监管提供，脚本末尾elapsed仅是附加检查。真实读取结果、手工抽查和后续press接线尚未完成；不把CPU测试或资产导出称为按压改善/官方成功。

11:20修审更新：按独审补collision到named finger之间的RigidBodyAPI归属检查，拒绝嵌套独立刚体（包含collision自身）。增加真实SDK两种嵌套body反例，远端门现为9/9；上条8/8是增加反例之前的预算/计数，未曾声称通过。全量先前650中646通过/4本地SDK skip，不含本次修复；最终回归待。

11:22修后651本地回归中646通过/5 SDK明确skip（17.244s），独立最终复审确认无剩余实质代码阻塞。下一固定源运行robo真SDK9项，只有全过才唯一读取资产；输出仍不能称cooked物理接触面。

## 实际结果与人工检查（2026-09-25 11:38北京时间）

robo新干净源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/finger_collision_0f8321d`已固定。原生SDK9/9无skip通过0.672s；唯一实际导出exit0/PID3549580/5.9399s，四指各8分片，共1912顶点/3696三角。11:34:50确认导出进程已退出、原四训练仍在，源保持clean。结果`/mnt/nvme_tmp/robodojo_agentic_20260925/h64_finger_collision_asset_v1/result.json`为657333B，完整复制本地`artifacts/agentic-vlm-goal-20260918/h64_finger_collision_asset_v1/result.json`，双端SHA `5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04`。

主agent用新增工作树审查脚本`audit_finger_collision_asset.py`（SHA `46adf23ca1a43fc389341e34265904a317a72470f4de5ca2f208175eb58daa7c`）在本地CPU重算全部32片，1.221s，0新USD读取/模型/仿真。各片封闭边计数异常为0，最大凸平面误差6.82e-10m、最大三角体积与凸包体积相对差1.12e-9；这只验证声明网格数值，不验证PhysX cooking。数字与753402B图/报告在同本地目录`inspection_v1/`。

本人已看四指全部XY/XZ/YZ及3D投影视图：各指为相对link坐标的弯折多片实体，不是夹持空隙中心；两手link1形状一致，link2为相对方向、局部有小几何差异，不能复用单手的片编号/坐标或把颜色当跨指语义。范围约39.1×71.1×93.2mm，无米/毫米或二次缩放量级异常。此检查没有当前EEF/finger开度、target或动态接触，不能给出“哪个面一定能按按钮”结论。

独立真实产物审查完成：32封闭凸片、Euler=2、每边2面、无重复面，变换/体积数值通过。发现258/3696面重心位于同指另一片内部>0.1mm（最大3.034mm）；16/32片的共面三角划分与另一凸包算法不同，right finger2/col6还有非极值共面顶点。故不能把任意三角当外露面或把authored面编号当PhysX identity。

下一已进入H65：固定finger/link/mesh及明确link-local XYZ，合并共面patch、内缩/排除重叠，不依赖cook三角编号；随H63真实named关节/FK计算当前与候选的同一点。native多开度/cooked与真实按压仍待，完整SR尚无新证据。原数值报告manual_review字段保留生成时PENDING状态，本节是父人工检查的补充记录，不回写原产物。
