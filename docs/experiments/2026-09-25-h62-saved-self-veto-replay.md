# H62：真实保存观测上的self-veto回归

2026-09-25 00:49北京时间，负责人Codex，尚未运行。本票不是新的模型评测或任务成功率。

## 唯一假设与固定输入

H61的公共接触定位门应在真实机器人标定和RGB-D上拒绝已知本体点，同时不把所有非本体点当目标。用已有结果作一次只读CPU回放，不添加新模型答案/人工点击/目标标签，不从已见点调阈值。

- H51真实模型输出4条native point，原result SHA `3769f632eb2593237e8403d6e7d3b8c3d5ac75cdc5e87a5220a8414ff54723f6`；回放原始文本，逐条核同capture binding、RAW像素、请求SHA，原4 case全部保留。这里只调用本地解析与定位函数，**不再请求VLM**。
- H60全部12查询/10框的1440个原采样像素，result SHA `eb6b85bab705eb12a60a4aee5680fe034ef3f76a8d56877ac6105758a1e08d20`。这些是均匀几何诊断点，不是VLM预测的接触点，更不是1440个独立任务样本。原771个已知base-surface点只检验新入口是否拒绝；其余点不得当真目标正样本。
- 公共RGB-D/proprio/本体标定仍用H60本地v2完整30件，沿H51/H58原SHA及frame binding递归校验。未取新服务器数据/对象GT/奖励/结果，bin保存态仍有原1038步演示前缀，不能用于零前缀成功率。
- 沿H61现有6mm exact base-surface检查和公共`localize_target`，不改阈值/动作/模型。原H51人审已知handle点偏离、absent plate幻觉照实保留；即使geometry有效也不能改称语义正确。

## 预算与验收

本地CPU4核内/库单线程、外层硬timeout60s＋15s仅清理，0GPU/仿真/reset/动作/训练，4条缓存模型回答＋1440像素定位；不启动或等待新的远端worker。独立runner/负例/审查后冻结Git单次执行，输出新`artifacts/agentic-vlm-goal-20260918/h62_target_self_replay_v1`，原文件不改。任何已知本体点仍valid、原输入绑定/计数/缺件不符即完整保留失败并停止，不重刷。

全部点的有效/拒绝原因及4条真实point前后都报告，不挑成功子集。验收只能证明H61在既有真实传感器输入上的接线行为；不能当新准确率、物理效果或完成goal。四原训练保持，仿真仍等待用户协调或资源空闲。

00:53准备更新：runner `scripts/semantic_robot/replay_target_self_veto.py`实现，原request receipt/SHA完全一致才回放缓存文本；原H60格点/box/三维深度点逐核，已知self仍valid即failed。6新负例和相关共66 CPU通过（0.571s），全量回归/独审中，尚未真实回放。主报告分别标注4条旧真实模型输出与1440个均匀几何样本；后者不伪装为模型预测或目标正确标签。

00:54全量612/612 CPU过（14.646s），diff过；原两result SHA/4case/12row/1440pixel/771self只读核同。独审进行中，无新模型、物理或回放结果；输出已确认被ignore，待固定commit后单次执行。

00:55独审通过：独立相关46/46/diff过，原输入/请求/1440采样/771self计数及CPU边界核验无阻塞。冻结本次Git后唯一运行；外部timeout的真实退出码是最终依据，若硬杀保留部分JSON但不宣称completed，也不由running字段推断进程仍在。

## 实际结果（00:56北京时间）

源码`454c91c0f687451a97e7f9b6850b7434941dc132`，PID3668125，唯一外部timeout执行exit0/9.161746198s（未触60s）。结果`artifacts/agentic-vlm-goal-20260918/h62_target_self_replay_v1/result.json`，SHA `e977f15aafa4754c0221c995bd58f34017260bf984c8e82e9c4748cf951ac60e`。4缓存回答和全部1440像素完成，0新模型/控制/reset/仿真/训练。

| query | 原采样数 | 原本体数→仍valid | 现几何valid数 | 其他拒绝 |
| --- | ---: | ---: | ---: | --- |
| radio head | 144 | 0→0 | 115 | 29局部深度 |
| radio left_wrist | 144 | 123→0 | 21 | 0 |
| radio right_wrist | 144 | 128→0 | 12 | 4局部深度 |
| handle head | 144 | 0→0 | 123 | 21局部深度 |
| handle left_wrist | 144 | 131→0 | 12 | 1局部深度 |
| handle right_wrist | 144 | 129→0 | 14 | 1局部深度 |
| bin head | 144 | 0→0 | 122 | 21局部深度、1跨视角深度矛盾 |
| bin left_wrist | 0 | 0→0 | 0 | 原无检测 |
| bin right_wrist | 144 | 0→0 | 134 | 10局部深度 |
| absent plate head | 0 | 0→0 | 0 | 原无检测 |
| absent plate left_wrist | 144 | 131→0 | 13 | 0 |
| absent plate right_wrist | 144 | 129→0 | 14 | 1局部深度 |

合计771已知本体点均被拒绝；其余669点中580仍几何valid、88局部深度拒绝、1跨视角矛盾。六个本体误检腕框仍留下86个几何valid点，它们是几何残余，不是该框声称的目标，也没有被人工标为目标正样本。因此不能把“在框内且不是本体”作为直接动作依据。

| 原真实native point | 旧/新几何结果 | 保留的原语义判断 |
| --- | --- | --- |
| radio [288,554] | valid→valid | 原点在radio机身；不是抓姿证明 |
| refrigerator handle [539,259] | valid→valid | **原点偏到门面/饮水区旁，不在细把手；self-veto不能修复这类语义错点** |
| bin [460,686] | valid→valid | 原点在桶内壁；不是可抓桶沿或完整任务成功 |
| absent plate [705,252] | invalid→invalid | 仍由深度拒绝，模型原答案不是正确弃权 |

主agent核全12行及4原point前后，无新提示/阈值/模型输出，无挑选重试。结论仅为H61的几何门在既有真实传感器数据上生效且未误拒这两个已知正确区域点；语义部件定位、完整机器人几何和实际接触仍未解决，完整零前缀成功率仍无新证据。下一不能把非本体残余自动升格目标，物理复验仍需资源可用。
