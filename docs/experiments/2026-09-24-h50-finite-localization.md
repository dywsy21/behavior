# H50：有限区域与表面选择，替代小VLM自由坐标

Owner：Codex。当前双端73 CPU及修后独审通过；19:45北京时间单次模型对照已提交，结果待验，不能报告定位提升或完整成功率。

## 假设与接口

H48/H49发现4B能识别红白收音机，但自由UV会落在它上方的桌面；单纯升分辨率没有解决。H50检验一种通用接口：先从当前图像3×3编号区域中选择，再从该区域内均匀4×3采样、且通过原RGB-D有效性检查的表面候选中选择。两步均有null，不返回未经模型确认的区域中心，不按最近深度自动赋予对象身份。

`src/semantic_robot/v2/finite_localization.py`提供可调用的`locate_target`。每个明确给定的相机最多2调用、最多3相机，当前探针仅head。原始图与标记图独立，候选坐标来自原生像素及标定，模型只输出有限ID。没有对象框、仿真物体状态、旧模型回答、手工偏移或答案坐标。

返回值是**VLM静态位置声明**，不是物体真值：可能仍然选错背景。有效深度不是语义证明。所有enclosed/co_moving/supported/effect均null，不生成左右手接触点、holding或官方成功。时序反馈由独立公开运动证据模块处理；这轮没有改其阈值，也未部署新actor。

捕获ID、目标、所有输入RGB-D、机器人q/夹爪、标定及实际live FK数组均绑定；回包绑定到当前请求。`for_frame`拒绝跨捕获消费，即使像素相同。它验证数据一致性，不自行保证真实模拟器拍摄新鲜度；调用者仍需现有render barrier。deadline在callback前后检查，真正中断由外层推理worker/监管完成。

独审发现只检查spec hash会漏掉live FK数组原地修改，已修并加五类变异测试。15新接口测试＋5旧affordance通过，核心修后独审通过。探针首8新测试＋旧共享44测试通过；窄审另发现共享范围门不等于H50精确冻结，已补GPU/UUID/PID、全部预算/seed和模型全清单/量化/EOS身份锁定及反例，修后回归/复审待。机器人完整标定约45.2MiB，故仅SHA固定的标定文件允许64MiB，其余文件限32MiB；没有扩大GPU预算。

## 单次配对实验（运行中）

固定配置：`configs/semantic_robot/h50_finite_localization_probe.json`。实现提交`82b7ef0c5c9982c35aa8e755bbfc5f94f5038c8d`；robo独立源码`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/shared_small_vlm_82b7ef0`，73 CPU与全部真实输入prepare通过。

唯一launch UTC2026-09-24T11:45:12.591784，supervisor3454291；spec SHA`da4ce6dc8d84fd316cda5306d9440eaa7721d36fa8f8a708a4f092dda419359e`。run `/mnt/nvme_tmp/robodojo_agentic_20260924/h50_finite_localization_v1`及同stem回执/log，独立cache `robodojo_vlm_runtime_20260924/h50`。提交非成功，完整输出待验。

| 查询 | 历史捕获 | 目标/人工先验检查（不进入模型） |
| --- | --- | --- |
| radio_surface | H38 task0 train138 decision084 | 红白radio可见；与旧d91同一episode，仅开发回归 |
| refrigerator_handle | H38 task3 train242 gate decision000 | 冰箱及长把手可见；考察细部而非整个冰箱 |
| plate_not_visible | 与上一行同一捕获 | 当前head没有餐盘；必须允许拒绝 |
| trash_can_surface | task1 TRAIN114 prefix989 teacher02/after_settle | 木色垃圾桶可见；有演示前缀，仅静态定位输入 |

这只有三个捕获、四个目标查询，既不是四个独立实例，也不是盲测、完整任务或成功率评估。本人在调用前检查了三张原始图像；不把这些图像用于训练或从结果中挑样本。

每查询一次H48最小静态自由UV基线，再最多两次有限选择。同Qwen3.5-4B revision`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、冻结NF4/BF16、原模型全文件SHA、320图像上限、greedy/seed17。两套接口的图像数/提示/输出形式不同，因此检验整套接口设计，不声称仅ID编码的单因素收益。基线结果不进入候选生成或有限选择请求。

最多12调用（4基线＋8有限）、600s worker/900s监管、0新控制/reset/训练；GPU2仅共享原训练3294348，Torch allocator4864MiB、额外512MiB、至少2048MiB空闲，遇资源身份变化/OOM/限额只结束自有worker。没有MPS硬隔离，也不保证零训练吞吐影响。输出使用新目录，不重试覆盖。

将保存每个完整请求、原始/实际缩放图像指纹、原始回答、全候选及选中坐标、资源采样。验收必须逐条查看选中点是否落在所问目标/把手上、不可见时是否拒绝、有效深度与跨视图检查、调用/显存/延迟；不能只看JSON是否合法。失败后不盲目增加网格密度、重刷相同提示或直接放行物理。

## 完成与未完成

- [x] 核心接口/弃权/绑定/静态与时序边界，CPU及独审。
- [x] 四目标的所有公共源文件hash/render barrier本地核验。
- [x] 有界配对探针与CPU回归。
- [x] 探针独审、固定源码、服务器CPU及资源检查。
- [ ] 单次全部输出人工审核和完整资源证据归档。
- [ ] 决定接口是否可部署，独立验证共享模拟器资源和控制兼容性。
- [ ] 零专家/旧策略前缀完整回合的官方成功；goal仍未达成。
