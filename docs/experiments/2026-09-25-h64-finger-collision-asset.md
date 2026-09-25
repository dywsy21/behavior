# H64：从真实机器人碰撞资产定义按压参考

2026-09-25 11:18北京时间；owner Codex，分支`feat/semantic-agent-grounded-20260918`，源码commit待固定。前置H63为`11dfe4b`，原642 CPU及独审通过，仍默认关闭、未native物理验收。

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
