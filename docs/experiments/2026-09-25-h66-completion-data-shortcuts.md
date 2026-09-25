# H66：状态微调究竟需要多少视觉信息

2026-09-25 12:10北京时间；owner Codex。仅CPU诊断，未注册新训练。

H09AB已真实完成120更新，但8态诊断的history规则也能全部猜中。该小集不是独立盲测，不能据此说VLM已经学会“看出抓到了”。本票在原39条TRAIN状态上直接量化非视觉捷径，不增删标签或动作，不接入部署actor。

输入固定H09AA states SHA `2fd27d707ad86e7fd6c09662000e99af26c0b1d686bbe9ff92b3c6aa8604fa99`：5条轨迹，task1的TRAIN114/192两个来源实例，34 CONTINUE/5 REQUEST_VERIFY。沿既有合法sidecar验证，所有图像SHA/私有label provenance都禁止进入特征；原eval i1/i71及H51四态不访问。

比较：多数类、exact recent history查表（未见历史回退训练多数类；平票CONTINUE）、末尾连续RIGHT_UP次数阈值（只在训练fold按balanced accuracy选）、q18/两指平均开度的固定尺度1NN（关节1rad、开度0.05m，不从测试fold拟合尺度）。每次将一个TRAIN来源实例全部留出；另列完整TRAIN内的history/无图公开输入标签冲突，仅为可记忆性诊断。

输出必须保留两fold原始计数、每条真实标签与预测、最近邻来源及跨group证明；报告正类召回/误报与balanced accuracy，不能只报高占比CONTINUE的accuracy。共2个相近TRAIN实例，不称新独立验证集、视觉因果证据或完整任务SR。

预算：每次CPU检查≤120s；固定源后单次真实39条统计≤60s、2CPU、输入≤1MiB/输出≤1MiB；0模型/训练/物理。任何SHA、schema、group或行数不符停止；不改原文件，不生成可训练的新sidecar。

下一：依据真实混淆选择视觉对照数据来源。成功/失败动作历史应尽量匹配；失败轨迹只可提供经独立验证的状态/感知监督，不能直接复用其失败动作作正向BC。图像不够明确就保留UNKNOWN，不硬填成功标签。新训练须另定具体模型、资源上限和闭环对照。

12:15准备完成：7目标回归0.013s、当前源全SFT188/188/5.656s，独审7/7/0.012s及diff通过，无可复现阻塞。CLI失败分支已静态核读，但未宣称其全路径动态集成测试；真实统计尚未执行，下一固定clean源码后外层60s硬超时运行。

## 实际结果：2026-09-25 12:16北京时间

唯一真实统计exit0/0.171s，源`a79907016c42eb3784438eec495bb41e1a5975bf`已push。输出`artifacts/agentic-vlm-goal-20260918/h66_train39_shortcuts_v1/result.json`，SHA `89b8a864d6daef148aafded92dcc4e9a98d4a1cbd00738b04d16ce553f7e0587`。39条来自原TRAIN114（21态）/192（18态），本人逐核全部预测及邻居来源，无同实例泄漏。

| 非视觉方法 | 正确数/39 | 请求召回 | 对CONTINUE误报 | Balanced accuracy |
| --- | --- | --- | --- | --- |
| 总猜CONTINUE | 34 | 0/5 | 0/34 | 50.00% |
| 历史查表 | 37 | 3/5 | 0/34 | 80.00% |
| TRAIN折拟合末尾UP阈值 | 38 | 5/5 | 1/34 | 98.53% |
| 固定尺度proprio 1NN | 32 | 2/5 | 4/34 | 64.12% |

两fold独立在TRAIN选择的阈值都是3。唯一阈值误报为`1f8b2f307ee20b02_09_continue`，即W396第三次UP后、第四次UP前；原同tick标签保持CONTINUE。Exact history及完整无图公开输入的异标签冲突组均为0，因此这批数据并不要求模型必须使用视觉才能拟合。**这不是证明VLM完全没用视觉，也不是98.53%成功率。**

后继决定：不直接续训原39；优先从已审TRAIN轨迹寻找“相同/近似完成动作历史，但未握住、滑落或不应继续动作”的真实状态。失败动作不会转正向BC；不清楚的视觉状态保留UNKNOWN。若现有轨迹无这类对照，则如实记录缺口，另注册小批公共感知/状态数据而非偷偷使用旧eval。最终需图像对照与真实闭环两种检验，不能只看状态CE。

## 既有TRAIN失败归档覆盖核对（12:45北京时间）

只读根目录`/home/wsy/behavior_worktrees/vlm-sft-native-teacher-20260919/artifacts/h09y-resume-20260921/native_complete`的8个`failure.json`来源；逐核原request、native_execution的feedback及控制起止（不是把request存在当执行成功）。

- i192 p0380、p0380_ws45、p0380_ws45_cd1_b2、p0388均没有CLOSE/UP执行。
- i192 p0388_ws45只执行CLOSE；p0392的CLOSE本身TRACKING_FAILED。均无执行UP。
- i114 p0953_ws45_cd1_b2超时前两次UP分别37/25控制tick、TARGET_REACHED；不是完整任务或独立抓取成功。
- i114 p0993只一次UP、28tick、TARGET_REACHED；最终私有local verdict仍IN_PROGRESS，真实握持/接触记录不使它自动成为完成正类。

这8条都没有“执行≥3次UP但视觉未完成”的直接对照，因此不能简单追加来声称H66揭示的末尾UP捷径已经解决。本次未构造/重标/训练，也未将失败动作作BC；后继需要新状态覆盖，或明确限制非视觉输入的训练消融并检验其视觉依赖。

## 13:25北京时间：视觉监督可观察性的具体人工复核

Codex重新按“能否从图直接给出这个label”而非“私有标签是否计算正确”检查了TRAIN192/W396最后一组相邻状态：`1f8b2f307ee20b02_09_continue`（tick693）及`..._terminal`（tick723）的head/right_wrist四张原RGB。定位映射为原`completion39_ccb16f2_manual_local_paths.json`；本地原件在`artifacts/h09w-native-task1-v1/complete/native_task1_v1/teacher_09/{before,after_settle}`，完整路径前缀`/home/wsy/behavior_worktrees/vlm-sft-native-teacher-20260919/`。

本人看到：两态头图均为右手贴在桶沿、桶口可见，腕图均是桶沿位于两指之间的近景；单帧没有直接显示“已累计12个稳定物理tick”，也不能可靠量出抬升高度阈值。它们不是明显“未夹到”和“夹到了”的视觉正负例。此结论不重标原合法私有标签，也不是说图完全相同或有陈旧帧。

实际原动作回执是18控制的RIGHT_UP、右EEF上移9.77094mm；再含12控制settle，q18最大变化0.0814122。像素检查head MAE1.23367/255、right_wrist3.54707/255，明显非字节相同。没有新增模型或物理运行。

后继微调标签应区分两层：视觉支路监督`目标可见/部件位置/手指与物体关系/短时间的共同运动或滑落（看不清=UNKNOWN）`；是否满足执行窗口、是否应该结束由另一个使用实际执行回执的验证协议负责。先准备真正可见的状态对照及跨来源实例拆分，不能拿原39条改个prompt就宣称纠正了视觉捷径。当前只完成这四RAW的可观察性复核，未释放新训练集或新权重。

## 14:46北京时间：闭爪失败不等于视觉抓空（Codex人工再审）

趁H70真实场景初始化，只读原TRAIN八张RAW：i192/p0392 `teacher_00/{before,after}/{head,right_wrist}`、成功来源i192/W396 `teacher_06/before/{head,right_wrist}`、i114/p0989 `teacher_01/before/{head,right_wrist}`。失败来源根为上文`native_complete`，W396根为`h09w-native-task1-v1/complete/native_task1_v1`。未读取旧eval，也没有构造新训练标签。

本人看到p0392闭爪后桶沿处于两指之间，和W396闭爪后的腕图关系相似；头图也能见到右手贴在桶沿。因此不能把这条旧run的`TRACKING_FAILED`直接当“抓空/已滑落”的视觉负例。闭爪前后图有明显指形变化，非陈旧字节。这一观察只确认可见的相对关系，不能确认握持力、离地或稳定物理tick，均保留UNKNOWN。

原`native_execution.json`（SHA `0f9e08bac4dbedb151cc0b3484d5e6bee5c3247303142fb811816b21f8943208`）记录18control闭爪，右EEF位置误差2.98331mm/姿态1.29924°、平均指开度5.62875mm，status=TRACKING_FAILED，holding=UNKNOWN，且**没有新版gripper_execution回执**。它不能被事后伪装成新版GRIPPER_COMMAND_COMPLETED。当前源码已有独立闭爪命令完成协议；此次不是重复宣称新增该修复，而是避免给后继视觉数据引入旧执行器语义造成的错标签。

图像SHA可定位：p0392 before头/腕 `c1cc3ee5`/`b1872132`，after头/腕 `65a2ae79`/`65c08f82`；W396 teacher06头/腕 `99c6e14c`/`68f666ac`；i114 teacher01头/腕 `8dc33651`/`891866ab`（均SHA256前8，原文件保留）。这个三状态候选不能提供视觉“成功/失败”对照；后继应采集真实抓空/共同抬升/滑落短序列，并将执行器版本和物理标注独立留在离线provenance。0新增样本发布/训练/模型请求。

## 15:09北京时间：接近阶段的多视角可观察性（Codex人工检查）

本人检查另外八RAW：TRAIN192/W396 teacher00、05的before头/腕，TRAIN114/p0969 teacher00、02的before头/腕。W396 teacher00腕图主要是桶外壁/地面、05已看到桶内/桶沿；114两态均看到桶内和长孔，指尖只有图像边缘的局部轮廓。对应头图可以定位右手和桶，但精细两指接触区很小/部分遮挡。因此这些图可支持“目标可见/部件出现”之类标签，不足以自动支持精确接触/握持/完成标签；不是新发现陈旧图或训练已有效。

路径根同上：W396 `h09w-native-task1-v1/complete/native_task1_v1`；114 `h09y-resume-20260921/native_complete/native_t1_i114_p0969`。各before头/腕SHA前8：W396/00 `fd680d6f`/`036586a9`、W396/05 `4addd785`/`3fd7c775`、114/00 `ac093da5`/`4bfa6a12`、114/02 `6387681d`/`c9043fed`。没有重标、训练集发布或新模型调用。

输入实现也明确区分：旧native SFT actor把三原图各缩至256，当前grounded agent服务允许640且有局部观察。不能把它们当相同视觉输入或同一执行协议。后续视觉监督先建立多视角/短时证据及UNKNOWN的协议；单纯给旧39条换分辨率不能解决不可观察标签和历史捷径。

## 16:30北京时间：H72实测故障对新微调采集的约束

本轮只读追到具体调用：`native_teacher_collect.py:step` 使用render_on_step(True)，随后计数prefix/native control并调用`teacher_reader.read(prefix_count+controls)`；`LocalOutcome`虽称physics tick，实际传入的是逻辑control序号，12个settle也是12次env.step。H72的新PT分支已实证18次调用仅推进44/72预期physics ticks，并且RGB-D可滞后真实状态。这足以禁止把新PT分支照搬进旧采集器而不校验时钟，但**不能倒推此前不同renderer的全部TRAIN标签错误**。

后继数据版本必须同时具备：每实际control的原生before/after clock；每3路图的同状态ReferenceTime；命令成功与物理/视觉结果分离；稳定窗口按真实simulation seconds和采样间隔解释，不把逻辑步数冒充物理tick。同一记录既有动作时钟也有相机时钟，错位或漏推进直接隔离，不用训练弥补。

旧39条及旧120步权重保持原版本、原结论，不重标或伪装成新协议。新训练仍先解决可见监督与同历史负例：目标/部件可见性、相对位置、短时共同运动/滑落，遮挡=UNKNOWN；不让视觉模型猜不可见的稳定tick阈值。H73的I/O实跑是此前置工程验收，尚未构造/发布新数据或重训。

## 16:59北京时间：新视觉数据的来源与九图人工检查

本人核H09R counts原件SHA `d94850ebfeadeef8b28f618e543ebae7cb7a18bd127c5d79ba6f4c18c0f015e3`及筛选代码。原5%留出、H09 val/test和开发task0/i138、task3/i242已经排除；但其旧`additional_train`仍列task1/i1和i71（之后才成为native eval）。现有native_dataset另传HELDOUT正确阻止它们；**不能把旧cohort单独当未来新数据的训练许可**。后继来源仍须并集排除所有后来留出/开发实例，不能用H71–H74保存态扩充训练。

本次只从既有合法参考片抽9张首帧检查，不重放仿真、不生成动作/完成标签。来源root `/home/wsy/behavior_worktrees/vlm-sft-native-teacher-20260919/artifacts/h09s-prepare-v1`，task0/i70/e66/frame1170，task1/i192/e310/frame164，task3/i30/e629/frame3473；三个reference SHA和九个clip SHA全部与原preparation/reference对上。原片是384像素人工预览、17帧，不能冒称已满足720/480 RAW训练契约。

| 来源 | head本人所见 | left_wrist本人所见 | right_wrist本人所见 |
| --- | --- | --- | --- |
| radio/i70 | 红白收音机在右手旁清楚可见；按钮细节不足 | 桌面/反光/少量机器人边缘，无所问收音机 | 近距离外壳和白边可见，不能据此标出按钮 |
| bin/i192 | 图下边缘只露一小段棕色桶，单图身份不充分 | 桶口/桶沿和内壁明显可见 | 左上角部分桶口/内壁可见，目标不完整 |
| plate/i30 | 盛食物餐盘、右手及敞开冰箱可见 | 机器人自身结构/地面，无所问餐盘 | 食物/盘沿及指尖近景可见，不能由单帧证明夹紧或稳定持有 |

这组三任务多视角来源有“全物体可见／局部可见／不可见”差异，适合作为后继部件/可见性监督候选，而不是GRASP完成或正确动作标签。仍欠额外来源实例、同类目标的视角内正负对照、原生分辨率与完整人工标注；没有发布新训练集，也没有把9图检查称为全部数据审完。

新本地预览目录`artifacts/agentic-vlm-goal-20260918/h66_train_source_preview_v1`，文件`t{0,1,3}_{head,left_wrist,right_wrist}.png`。按head/left/right顺序的PNG SHA256前8分别为：t0 `6a621995/51985cdd/b7fba290`；t1 `21708c62/6622187b/47a7b350`；t3 `e955ba83/523201b4/7bbbf45d`。不改原视频/reference，不回灌开发/旧eval，不据此声称微调或SR提升。
