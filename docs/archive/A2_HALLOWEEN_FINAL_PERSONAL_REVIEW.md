# A2-5000 + B-final：万圣节完整开发评测

2026-09-11，本人复核，无子代理。task2 / public_test instance301 / 环境seed0 / policy seed17，20682/20682实际动作，5164.299秒，正常以完整官方动作上限结束，task_success=false。没有演示前缀。这是反复使用的development实例，永久eval-only，不是独立盲测或整体SR估计。

本人核验保存动作与官方执行trace逐位相等，20682个物理诊断帧无缺口。高层调用162次：前2048动作16次NAVIGATE；后18634动作146次OPEN_DRAWER，始终为bottom_cabinet_rhdbzv_0的left部位，没有转入装饰物GRASP或放置。全程未出现确认持有对象。

本人观看了精确抽取的30幅非空全程画面：视频帧0、360、720、…、10080和10341；15fps下对应0、24、48、…、672和689.4秒。画面反复朝向电视柜、壁炉、餐桌与沙发，伴随底盘移动及转向；没有看到有效打开抽屉或收起装饰物。这里不是“已经抓完只差高层切换”的情形。图像只能支持视觉行为描述，不替代独立物理后置条件。

按每16动作前记录的位姿估计，累计路径47.070米、绝对旋转4082.646度、净位移0.751米。打转仍然严重，前两个任务上的旋转减少并未推广到此任务。低层每16控制的FM平均618.717ms；整回合耗时还包含仿真、初始化、高层及诊断，不全部归因神经推理。

诊断oracle全UNKNOWN：导航缺统一物理判据；具体开抽屉的资产/部位证据也尚未验证。因此不能把“0个已确认局部成功”写成已证明物理开抽屉成功率0%，更不能生成18634条FAILED训练标签。官方整任务未成功则有明确记录。

两个问题要分开解决：低层在给定OPEN_DRAWER时仍产生持续巡回移动，须通过保留真实历史的oracle-low试验和动作条件干预检查技能服从；高层持续同一子目标且目前没有已训练物理结果头，须构造真实完成/失败反馈。诊断oracle的绑定缺口是反馈取证问题，不声称它本身导致当前模型打转，因为特权诊断没有输入模型。

本轮五任务截至此记录为前三个完成且均未成功，后两个仍待完成；不提前报告0/5。结果SHA `f8193762a4944abb6e3706f3290617b00848232c0eed6c7261201a2a1391098e`，物理trace SHA `f9e5da644b7300ef3eaa5d7a944b2b38bf1764f9bd35d336e0ba5df5c2630936`。

[完整视频](/home/wsy/behavior/memlite-results-20260911/A2-5000-Bfinal-halloween/rollout.mp4)

[本人查看的30帧面板](/home/wsy/behavior/memlite-results-20260911/A2-5000-Bfinal-halloween/review_30_exact_frames.jpg)

[实际逐技能分析](/home/wsy/behavior/memlite-resume.L6rqZ6/a2_final_halloween_actual_episode_analysis_v1.json)
