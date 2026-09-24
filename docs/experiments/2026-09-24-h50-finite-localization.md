# H50：有限区域与表面选择，替代小VLM自由坐标

Owner：Codex。H50已完成且人工语义审不通过，不部署；正在准备一次输出选项契约修复H50b。没有新的完整成功率。

## 假设与接口

H48/H49发现4B能识别红白收音机，但自由UV会落在它上方的桌面；单纯升分辨率没有解决。H50检验一种通用接口：先从当前图像3×3编号区域中选择，再从该区域内均匀4×3采样、且通过原RGB-D有效性检查的表面候选中选择。两步均有null，不返回未经模型确认的区域中心，不按最近深度自动赋予对象身份。

`src/semantic_robot/v2/finite_localization.py`提供可调用的`locate_target`。每个明确给定的相机最多2调用、最多3相机，当前探针仅head。原始图与标记图独立，候选坐标来自原生像素及标定，模型只输出有限ID。没有对象框、仿真物体状态、旧模型回答、手工偏移或答案坐标。

返回值是**VLM静态位置声明**，不是物体真值：可能仍然选错背景。有效深度不是语义证明。所有enclosed/co_moving/supported/effect均null，不生成左右手接触点、holding或官方成功。时序反馈由独立公开运动证据模块处理；这轮没有改其阈值，也未部署新actor。

捕获ID、目标、所有输入RGB-D、机器人q/夹爪、标定及实际live FK数组均绑定；回包绑定到当前请求。`for_frame`拒绝跨捕获消费，即使像素相同。它验证数据一致性，不自行保证真实模拟器拍摄新鲜度；调用者仍需现有render barrier。deadline在callback前后检查，真正中断由外层推理worker/监管完成。

独审发现只检查spec hash会漏掉live FK数组原地修改，已修并加五类变异测试。15新接口测试＋5旧affordance通过，核心修后独审通过。探针首8新测试＋旧共享44测试通过；窄审另发现共享范围门不等于H50精确冻结，已补GPU/UUID/PID、全部预算/seed和模型全清单/量化/EOS身份锁定及反例，修后回归/复审待。机器人完整标定约45.2MiB，故仅SHA固定的标定文件允许64MiB，其余文件限32MiB；没有扩大GPU预算。

## 单次配对实验（已完成）

固定配置：`configs/semantic_robot/h50_finite_localization_probe.json`。实现提交`82b7ef0c5c9982c35aa8e755bbfc5f94f5038c8d`；robo独立源码`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/shared_small_vlm_82b7ef0`，73 CPU与全部真实输入prepare通过。

唯一launch UTC2026-09-24T11:45:12.591784，supervisor3454291/worker3454299；spec SHA`da4ce6dc8d84fd316cda5306d9440eaa7721d36fa8f8a708a4f092dda419359e`。run `/mnt/nvme_tmp/robodojo_agentic_20260924/h50_finite_localization_v1`及同stem回执/log，独立cache `robodojo_vlm_runtime_20260924/h50`。全部12调用已完成，exit0；模型阶段108.580s/监管124.841s、峰4534MiB、最低free2950MiB，退出恢复7489MiB，四训练仍73644MiB。有限选择热调用0.745–0.862s，只是生成延迟，不包含CPU准备/整套控制环，不能当实际机器人Hz。

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
- [x] 单次全部输出人工审核和完整资源证据归档（语义不通过）。
- [ ] 决定接口是否可部署，独立验证共享模拟器资源和控制兼容性。
- [ ] 零专家/旧策略前缀完整回合的官方成功；goal仍未达成。

## 全部人工语义审（19:53北京时间）

本人检查全部12原始回答、三个完整当前RAW、四张裁剪/候选图；不是只检查JSON。三可见目标区域都有目标，但后续选点全部不正确；其中细把手是候选覆盖失败，即使选择器完美也应弃权。

| 目标 | 自由UV | H50有限选择 | 人工判断 |
| --- | --- | --- | --- |
| radio | [.53,.72]，物体上方桌面 | region7→ID10 [390,680] | 桌面；ID7清楚在radio，ID6在其左边缘 |
| refrigerator handle | [.75,.35]，门面偏离细把手 | region5→ID5 [570,360] | 门面；此网格没有点覆盖把手，应null |
| plate not visible | 正确不可见/null | region0→ID0 [30,40] | 墙/抽油烟机背景，严重假阳性 |
| trash can | 错说不可见（note还提到背景垃圾桶） | region7→ID5 [330,600] | 地面；ID2在可见桶内壁 |

本地完整包`artifacts/agentic-vlm-goal-20260918/h50_shared_bundle_v1`。result SHA`8b51daa0b653d740c9fe76ddbae422b552070210301c3a9eecd5e772d629121d`、supervisor SHA`6957ebaef7f75b7462f5d27867ecbd3988085bfb5a42fa675debec44abc57d8a`已双端一致。没有新训练/模拟器reset，所有模型答案均未进入actor。

## H50b：一次实际输出契约修复复验（未启动）

发现可定位实现缺口：region/surface提示要求从列出的JSON中选，但新请求text没有列出JSON，只有解码器收到了allowed；旧policy会明确列出。补`choice_suffix`让模型可见文本包括与decoder完全相同的候选（含null），传输前检查一致；缺失/错列表反例拒绝。不改原图、网格、点生成、语义描述、解码grammar、模型或控制阈值。不能先假定这就是全部错误的原因。

同四原query，配置`h50b_explicit_choice_probe.json`，case全文canonical SHA固定`5846788db1516fd08b8a341947466dbcd4320c3041cb99937260bfc328a257ea`。最多8finite、0baseline重跑、600/900/4864+512/2048/原GPU2训练身份/seed17保持；不读取旧输出进模型。第一层图和公共源完全相同；如果本轮区域选择不同，第二层裁剪自然不同，不强行复用旧答案指定区域。0训练/reset，无重试或追加样本。

75 CPU过（21核心/旧affordance＋10探针＋44旧共享），修后窄独审通过。先核这一工程遗漏，结果后再决定定位工具设计；不增加网格密度或针对四图手调位置来制造正确率。H50本地12条call与result逐项一致、全部28图原生及实际320像素SHA均核同。
