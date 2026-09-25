# H66：状态微调究竟需要多少视觉信息

2026-09-25 12:10北京时间；owner Codex。仅CPU诊断，未注册新训练。

H09AB已真实完成120更新，但8态诊断的history规则也能全部猜中。该小集不是独立盲测，不能据此说VLM已经学会“看出抓到了”。本票在原39条TRAIN状态上直接量化非视觉捷径，不增删标签或动作，不接入部署actor。

输入固定H09AA states SHA `2fd27d707ad86e7fd6c09662000e99af26c0b1d686bbe9ff92b3c6aa8604fa99`：5条轨迹，task1的TRAIN114/192两个来源实例，34 CONTINUE/5 REQUEST_VERIFY。沿既有合法sidecar验证，所有图像SHA/私有label provenance都禁止进入特征；原eval i1/i71及H51四态不访问。

比较：多数类、exact recent history查表（未见历史回退训练多数类；平票CONTINUE）、末尾连续RIGHT_UP次数阈值（只在训练fold按balanced accuracy选）、q18/两指平均开度的固定尺度1NN（关节1rad、开度0.05m，不从测试fold拟合尺度）。每次将一个TRAIN来源实例全部留出；另列完整TRAIN内的history/无图公开输入标签冲突，仅为可记忆性诊断。

输出必须保留两fold原始计数、每条真实标签与预测、最近邻来源及跨group证明；报告正类召回/误报与balanced accuracy，不能只报高占比CONTINUE的accuracy。共2个相近TRAIN实例，不称新独立验证集、视觉因果证据或完整任务SR。

预算：每次CPU检查≤120s；固定源后单次真实39条统计≤60s、2CPU、输入≤1MiB/输出≤1MiB；0模型/训练/物理。任何SHA、schema、group或行数不符停止；不改原文件，不生成可训练的新sidecar。

下一：依据真实混淆选择视觉对照数据来源。成功/失败动作历史应尽量匹配；失败轨迹只可提供经独立验证的状态/感知监督，不能直接复用其失败动作作正向BC。图像不够明确就保留UNKNOWN，不硬填成功标签。新训练须另定具体模型、资源上限和闭环对照。

12:15准备完成：7目标回归0.013s、当前源全SFT188/188/5.656s，独审7/7/0.012s及diff通过，无可复现阻塞。CLI失败分支已静态核读，但未宣称其全路径动态集成测试；真实统计尚未执行，下一固定clean源码后外层60s硬超时运行。
