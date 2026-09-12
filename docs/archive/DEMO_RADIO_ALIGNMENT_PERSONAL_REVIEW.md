# 原始收音机演示物理回放与首帧图像：本人复核

2026-09-11，train episode121 / instance138。这项检查只回答原始动作、技能标签和本次物理回放是否相容；不是模型成功率，不是失败状态专家纠正，也没有数据准入。

原始动作序列先NAVIGATE [0,448)，再GRASP [448,1122)，PRESS [1122,1226)，最后PLACE_ON [1226,1562)。原始23维控制未经修改，两次独立回放都在971步达到六帧稳定抓持。有效v2在1193步被官方谓词判定整任务成功并终止。此时PRESS刚出现物理成功，诊断的六帧稳定窗口还没凑够；应保留两种判定各自语义，不伪造额外帧或否认官方终止。

本人查看原始演示与回放在0/448/640/970/1122的三相机对照，以及额外1/16/32帧对照；还检查两次回放全程的非空视频面板。关键发现是回放第0帧头相机看向沙发/收音机，而原始画面看向壁炉；第1帧即恢复相近视角。后续关键帧的目标、机械臂接近与抓持过程在视觉语义上相容。两手相机未见同等方向错位。本体感觉有小量偏差，不能称物理逐位复现。

源代码证据与陈旧图像缓存相符：非灯光任务reset后直接使用env.reset返回的缓存观察，没有像灯光任务那样额外render；模型adapter只是再次投影该缓存。后续将独立验证render-only刷新，必须不改变完整物理状态或物理时钟，不得用零动作/settling偷推进仿真。当前未把该候选接入正在跑的A2评测。即使修好首帧，也不能预先声称数千步后的抓取失败都会解决。

v1保存视频和完整物理trace，但Kit shutdown使外层finally未落result/actions，保留原不完整状态；v2在上下文退出前持久化，1193条实际保存动作逐位等于原始动作，完整结果可核验。

[有效回放视频](/home/wsy/behavior/memlite-results-20260911/demo-radio-train-e121-v2/rollout.mp4)

[第0帧原始与回放对照](/home/wsy/behavior/memlite-results-20260911/demo-radio-train-e121/demo_radio_original_alignment_panels_v1/compare_f00000000.jpg)

[第1帧对照](/home/wsy/behavior/memlite-results-20260911/demo-radio-train-e121/demo_radio_initial_alignment_panels_v1/compare_f00000001.jpg)
