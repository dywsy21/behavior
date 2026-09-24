# H60：检测框内样本是否落在机器人自身

2026-09-25 00:13北京时间，负责人Codex，准备/CPU实现；未执行实测。H58十个固定输出框不修改，不增加检测器/VLM调用，不做新的成功率评测。

假设：腕图六个明显本体误检，可由公开机器人asset/current proprio/当前RGB-D的几何对应检测出来。先报告重合比例与未知项，不预先宣称统一阈值已可部署，不把“非本体”变成目标/接触/持有/完成判断。

- 输入固定H58 `result.json` SHA `7aba22853b9458f06205abcdc3543ef166378a4dc7648bfc590ef0776854668c` 和H51配置SHA `76b0e7e366ff921c1dbf30b57d280833e3476a4c7698d5979684b45089347f61`；12查询/10框，原4来源状态，不是12新独立样本。
- 用原public文件白名单/逐SHA/recursive capture/sensor byte checks。为本地CPU复用，单独复制30个明确文件≤150MiB至新`artifacts/agentic-vlm-goal-20260918/h60_public_inputs_v1`，保留原服务器相对路径，不复制outcome、对象姿态/segmentation、reward、模型答案以外新标签。
- 框不修剪或改输出；仅在图内交集按固定12×12像素位置取真实深度（相同像素去重），原0.025–3m域，允许invalid/missing。用冻结相机K/current joint FK投回机器人坐标，沿已有`ChassisSurface`的6mm精确三角距离；只覆盖原base_visual_surface，不冒充全部手指/躯干mask。
- 输出深度有效/本体重合数、比例、原框、当前帧binding、标定来源与missing；没有完整标定则unknown，不fallback为“非本体”。不设已见数据挑出的reject阈值、不采用box中心或返回action。
- 本地CPU4核以内/单线程库、最多60s/12条/10框/每框≤144点；0GPU/模型调用/新模拟器/训练。先CPU负例/独审、冻结commit后单次运行，全10框统计审查。H59独立仍按原预算运行，四训练不动。

若几何不足，保留缺失/负例再设计完整本体mask；若可分离，只为后续通用可见性/候选表面约束提供证据，不能据此宣布闭环成功或直接部署。

00:16准备：v1未压缩30文件传输在55s截止，tar Unexpected EOF；保留部分文件，不能当输入就绪。先前prepare在第二case不完整时被原SHA/大小门拒绝，未做几何实测。后继仅改为压缩流到全新`h60_public_inputs_v2`，仍30白名单/150MiB，递归原SHA全通过才用。本地`proposal_geometry.py`/runner已草拟，无自动阈值/action，CPU负例/独审待。

00:21输入已核：v2压缩传输成功，30件/157646923B=150.343821526MiB，实际超过预估150MiB约0.344MiB，这是准备预算偏差而非“150MiB内完成”；保留文件，不追加来源/不重复制。四case的recursive SHA、RGB-D/proprio/传感器回执、frame binding全部原样通过，均有156655三角robot base surface及FK。原v1部分副本不使用。28/28 CPU/diff过、独审中，10框几何实测尚未执行。

00:27修审更新：独审指出连续xyxy框的上界应保留width/height，且像素应floor而非round，否则最后一行/列或1像素框会被错误采样。已修正半开像素约定，新增末行/列、右下角、1像素、部分负边界和完全越界负例；目标29/29 CPU通过。仅采样域修正，原检测框/阈值不变，窄复审中；未执行实测。拟冻结本次Git后，由外部timeout严格执行60s（额外15s只作清理）的单次本地CPU运行。

00:28修后独审通过：独立复跑29/29及diff检查无剩余实质阻塞。v2全部30文件和四case绑定已在准备阶段核定，实际150.344MiB偏差如上；下一固定commit、核输出目录不存在，再唯一启动。本次启动/结果文件将记录真实source_commit，不用文档自引用SHA。

## 实际结果（00:29北京时间）

固定源码`9a76d08f9243ac08fc1f01aae1b8e12a42598525`；本地唯一运行PID3660660，外部timeout60s/清理15s，exit0、6.428561535s，12查询/10框、每框144个深度有效样本。0模型/新仿真/训练；不是成功率评测。

| 原query | base surface重合 / 有效样本 | 比例 | H58既有人审（不是新独立标签） |
| --- | ---: | ---: | --- |
| radio head | 0 / 144 | 0% | 正确目标区域 |
| radio left_wrist | 123 / 144 | 85.42% | 本体误检 |
| radio right_wrist | 128 / 144 | 88.89% | 本体误检 |
| refrigerator handle head | 0 / 144 | 0% | 两把手共同区域，未消歧 |
| refrigerator handle left_wrist | 131 / 144 | 90.97% | 本体误检 |
| refrigerator handle right_wrist | 129 / 144 | 89.58% | 本体误检 |
| bin head | 0 / 144 | 0% | 正确可见区域 |
| bin left_wrist | 无框 | 不适用 | 正确弃权 |
| bin right_wrist | 0 / 144 | 0% | 正确桶区域，不能按大框全拒绝 |
| absent plate head | 无框 | 不适用 | 正确弃权 |
| absent plate left_wrist | 131 / 144 | 90.97% | 本体误检 |
| absent plate right_wrist | 129 / 144 | 89.58% | 本体误检 |

完整result：`artifacts/agentic-vlm-goal-20260918/h60_proposal_geometry_v1/result.json`，SHA `eb6b85bab705eb12a60a4aee5680fe034ef3f76a8d56877ac6105758a1e08d20`。主agent核全部10框和两空结果，不挑选只报好例。

结论：这批腕图误检确实主要覆盖机器人底座外壳，公开本体几何能提供直接排除依据；单靠检测置信度不够。**没有从这12查询拟合reject阈值，没有部署mask，也未证明完整本体排除/目标识别/闭环成功。** 下一应把已知本体排除接在候选表面/点层，而不是把未重合残余像素自动当目标或采用框中心。
