# H79：修复时钟后的原始起点 VLM 闭环

**19:00北京时间状态：用户新指令转向扩大数据及至少五小时规模的微调，本实验暂停、0launch。** 启动器本地11目标CPU/完整738基础回归（733pass5SDKskip）通过，但独审未结束；真实H77 runtime的旧tree_bytes拒绝Kit生成的screenshots目录，须修后独审再考虑恢复，不能把本文件预算视为正在运行。

2026-09-25；唯一负责人 Codex；实现/审查中，尚未启动。H78视觉组件训练是独立实验，不把其adapter自动部署，不改变已有27B模型或actor提示。

主要假设：H75/H77已通过的真实四物理tick控制和逐相机同步接口，能让现有grounded VLM动作—观察反馈可用，从原始task0 TRAIN138/seed0无专家/旧policy前缀进行自主操作。保持完整harness digest `7bad2e9021b6a48dd298e571b213c19e763453b8c593ba871afbc404f0c765f8`、两门全部开关和H64指尖资产；不改变物理、成功门、模型或按实例手写动作序列。

模型保持旧27B目录 `semantic_agent_v2_20260917/models/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，其真实config为Qwen3_5。完整本地文件SHA将只读核验/固定；BF16、thinking=false、HF5.7与固定结构化decoder817f944，原三路RAW/局部图、640上限、无特权对象状态。比较旧原reset轨迹仅用于开发诊断，不宣称独立盲测或严格同渲染分布A/B。

先CPU与独立审查、Git干净固定源码、模型/两门身份/安装依赖与资源预检。H78结束释放GPU2后才提交一次：96决策、3072真实controls（含最终安全hold）、2400s重置后策略循环；总3600s含模型加载/场景初始化/验收，另60s清理两自有进程组。模型最多215调用（与runner上限一致），0训练；GPU2模型≤65536MiB、模拟器辅助≤512MiB，GPU3模拟器≤24576MiB，0/1各自有辅助≤512MiB且团队任务不动，全卡留≥8192MiB。CPU模型48–51/仿真72–75；新run≤8GiB、私有runtime≤16GiB、NVMe余量80GiB。任一资源/同步/模型错误或到预算即停，不自动重置/重试/扩轮。

终态分别报告：是否完成本次工程运行；实际动作/控制/模型调用、具体失败链；官方环境done.success/goal_status（不是模型自报），局部抓持不能替代任务SR。一条开发实例成功也只能记1/1开发试跑，不外推50任务总体。新run预定 `/mnt/nvme_tmp/robodojo_agentic_20260925/h79_synchronous_actor_v1`，runtime同stem的 `robodojo_sim_runtime_20260925`；source commit待固定。H75/H77原证据与所有训练数据保持不变。
