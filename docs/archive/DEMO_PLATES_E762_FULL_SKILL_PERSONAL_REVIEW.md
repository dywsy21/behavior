# 完整原始演示回放：盘碗 train e762 / instance242

2026-09-11，本人完成检查，无子代理。原始23维控制实际执行10709步，逐位等于原动作；完整物理trace连续，v3在每次step之前复制观察。9项审计反例通过，并用已完成的radio实际回放交叉验证审计入口。

这次是原始演示回放，0模型调用/优化器更新，不是学习器成功率、不是C2同失败状态纠正、没有教师认证或数据准入。执行到原始动作上限，**没有触发官方成功终止**；局部技能成功不能替代整任务成功。

本人查看全部6张面板，27个实际时刻×3相机=81个非空画面，涵盖每个已到达的技能边界以及各首次稳定成功。图像可见：开冰箱、右手分别把两只盘子放进冰箱、左手和右手分别持碗、最后依次把碗放进水槽。放置时的“已释放”依赖完整独立held/relation物理记录，不仅凭一个仍贴近碗边的夹爪画面猜测。

实际获得7个当前技能内的因果稳定成功段：plate_93放入冰箱3055；plate_94抓取4877、放入冰箱6111；bowl_91抓取8089；bowl_92抓取9313、放入水槽10276；bowl_91放入水槽10554。全trace为807帧SUCCEEDED、2344帧IN_PROGRESS、7558帧UNKNOWN，没有FAILED。

一个重要边界反例：plate_93的原GRASP区间为[2039,2317)，全部仍IN_PROGRESS；右手首次确认持有出现在2321，已经进入NAVIGATE区间，直到3050释放。当前段内oracle不能因此倒填2317为抓取成功，也不能声称该演示没有抓起第一只盘子。原动作与标签按half-open区间完全一致，说明这至少是“注释技能边界≠本轮物理完成时刻”的真实例子；是否还含回放动态差异，需要独立比较，不能推成所有原始数据整体错位。

开/关冰箱诊断返回 `open_parent_identity_unverified`。现有oracle要求冻结资产身份与父对象内的关节映射，但当前构造并未提供真实静态绑定。磁盘原资产明确存在right_door/left_door及openable_joint_ids；在与实际活关节、方向、资产字节绑定验证前，仍保留UNKNOWN，不凭画面/原标签直接放行。

持有变化均与开门、抓取、放置阶段相对应；未发现可据此确认的意外掉落证据。不能为了补FAILED类别将主动释放改成失败。本段仍 `training_admissible=false`，下一步是独立反馈数据接口和真实类别覆盖，不是直接回灌低层BC。

结果manifest SHA `43999186c59ac8803e940a8239d8959e6cbde7e8b879caa698b9a5c69b7cdd1e`；实际动作SHA `afb025d0c20f48209ef02d8ba0adce2f4b613031931ffadd9f2231c4c145ac39`；物理trace SHA `a2f67832cf666ff1bd5a7197f5ee61531f7a7c9336b11875d967ea603e73e9b5`。

[原始动作回放视频](/home/wsy/behavior/memlite-results-20260911/demo-plates-train-e762-v3/rollout.mp4)

[完整逐技能分析](/home/wsy/behavior/memlite-resume.L6rqZ6/demo_plates_e762_complete_skill_analysis_v1.json)

[全部人工检查面板清单](/home/wsy/behavior/memlite-results-20260911/demo-plates-train-e762-v3/personal_full_skill_review_v1/receipt.json)
