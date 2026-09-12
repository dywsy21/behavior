# 第三个接管窗口：旧A自行抓取，未调用纠正候选

2026-09-11；本人直接视觉复核，无子代理。task2、episode479、官方train instance97。

结论：1972个真实演示前缀动作之后，旧A5000执行448个控制，在frame2420取得GRASP pillar_candle_91的物理成功。由于A已经满足局部条件，独立G0.5接管被正确跳过。**这不是A2的成功、不是专家纠正、不是整个万圣节任务成功，也不是自动训练准入。**

本人查看了80.73秒视频中0–80秒每5秒的17个非空面板，并进一步查看实际最终事件的6个历史时刻×3相机，共18个相机面板。右腕视角显示夹爪逐渐接近并闭合于蜡烛周围；头部视角里目标很小且局部遮挡，所以不能只凭头部视频否认/确认抓取。

最终物理凭据独立记录：目标pillar_candle_91唯一映射到candle.n.01_1，右手held=true、左手false，连续18个真实physics frame成功，超过6帧门槛；初始/前缀后未满足，policy_causal_success=true。全流程只记录30个C1事件，最后一个为成功，不存在G0.5纠正动作。

同一旧A和窗口在先前C1记录中没有成功，这次变化不能归功于A2训练。已完成前缀端点的实际文件比对：frame1972的全部六个本体状态通道逐位相等，三相机图像却不同，像素RMS差约1.15–1.46（0–255尺度）；语义bundle相同。首块16×23实际动作也不逐位相同，RMS差0.000541、最大绝对差0.002719。该比较没有隔离图像、模型后端及未记录物理状态，不能断言仅渲染噪声导致最终成败，但足以说明不能将这两个回合当成完全相同输入的复验。保留两次结果，后续需重复评测；不使用这次成功覆盖原未成功样本。

比对记录：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c2_w6_restart_input_comparison_v1.json`。其中完整descriptor比较包含文件引用路径，不能把路径不同本身当成实际模型输入不同；上面的图像和本体结论来自实际NPZ数组比较。

本例可作为新增真实C1反馈候选，但仍需原始记录的全量reader/split/因果检查与数据集级审查；不因为本人看见抓取就直接放行整个反馈训练集，尤其目前FAILED/其他技能类仍缺失。

[完整本地视频](/home/wsy/behavior/memlite-results-20260911/C2-halloween-train-e479/rollout.mp4)

![实际六帧三相机](/home/wsy/behavior/memlite-results-20260911/C2-halloween-train-e479/c2_halloween_e479_final_history.jpg)

原始目录：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c2_same_live_state_pilot_v1/c1v2-matched-t2-train-e479-f1972-grasp`。

最终物理凭据SHA：`9fed9f60533c0580edaa681f80fe15207c323536746abd9e8eced18b7d5dea9e`；最终原始观察SHA：`9cae864823cd461f008aca158e83f9e854333e6ba0b38370fe2cd2d6796955f8`。
