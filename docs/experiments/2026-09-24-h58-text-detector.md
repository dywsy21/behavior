# H58：用公开小型检测器提出目标区域，不让VLM凭空写坐标

2026-09-24 23:34北京时间，负责人Codex，准备中。完整goal仍是零专家/旧策略前缀的官方任务成功；本票只是定位部件筛选，不把检测框/模型置信度视为接触点、持有或任务完成。

## 选型证据与边界

SAM3官方[代码](https://github.com/facebookresearch/sam3)与[权重](https://huggingface.co/facebook/sam3)支持文本条件分割，是值得尝试的候选。但robo既有认证对revision `3c879f39826c281e95690f02c7821c4de09afae7` 的config请求返回401 GatedRepoError，已知项目模型根没有SAM3；已异步询问用户合法已有权重路径，不绕过授权或代填个人资料。

先验证作者公开的[Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)，不是SAM3复现。官方repo [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)，Apache-2.0、文本条件检测；其框不能替代分割mask，更不能把框中心无条件当接触点。若对目标存在性/区域有用，后续供VLM选择、RGB-D表面筛选及独立时序反馈使用，不直接接入actor。

robo只读官方HuggingFace API/HEAD已核：revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`，ungated/config可访问；safetensors 689359096B、LFS SHA256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`。只下载safetensors/必要JSON与vocab/README，不下载或执行pickle权重、远程Python代码。

当前既有VLA解释器Python3.10/torch2.7.1+cu128/Transformers4.57.1已CPU导入GroundingDinoForObjectDetection；实际postprocess参数是`threshold`/`text_threshold`，不是旧模型卡的`box_threshold`。参考[官方Transformers接口](https://huggingface.co/docs/transformers/model_doc/grounding-dino)，同时以安装版签名为准。无需升级/改训练环境；推理关闭custom kernels以免编译进共享缓存，先CPU版单批。

## 准备预算与拟定单次测试

- 下载单个revision/精确文件白名单，磁盘上限2GiB、600s下载上限；独立新目录 `/mnt/nvme_tmp/robodojo_grounding_models_20260924/grounding-dino-tiny_a2bb814`。完成必须核权重SHA及全部文件manifest，目录存在不算完成。不写仓库权重或改环境。
- 准备沿用H51冻结的四个开发查询，各取当前head/左腕/右腕，共12条。只有RAW RGB及原public目标文字入检测器；无GT框、特权state或旧模型答案。四head用于已有定位失败对照，新增八腕图检查跨视角/不可见目标，不能把它们称12个独立实例或盲测。
- 文本统一按原目标小写并句点结尾；固定一次，不按结果改写prompt/阈值。阈值拟0.4/0.3、单批、官方processor默认尺寸/FP32 CPU，4线程/CPU68–71、最多600s/12调用、0GPU/0sim/reset/训练。精确代码、全输入/权重SHA在执行前固定；当前未推理。
- 全12条输出/RAW人工审，记录假阳性、漏检、目标框是否覆盖真正目标及耗时；不挑好看的结果。CPU可用后才另登记GPU共享资源和在线接口；CPU时间不是GPU部署Hz。

当前H57工程检查使用GPU3/CPU72–75，H58准备不与之新增GPU竞争；四原训练不改、不停。没有新的SR证据。

23:37下载阶段：唯一timeout父3504210已退出，公开9文件13s下载日志完成，模型目录659MiB；完整SHA核验进行中。既有GPU上只有原训练与H57，尚未加载检测器或新推理。没有升级Torch/Transformers。

23:43：全9件SHA核闭合，权重匹配官方LFS，精确manifest在`configs/semantic_robot/h58_text_detector_cpu.json`。`probe_text_detector.py`已实现固定12输入、recursive原SHA校验、真实RAW像素回执、FP32 CPU/自有600s timeout、单次claim和全预测保存；不调用GPU/模拟器，输出`/mnt/nvme_tmp/robodojo_agentic_20260924/h58_text_detector_cpu_v1`、独立缓存`/mnt/nvme_tmp/robodojo_vlm_runtime_20260924/h58_cpu`均未创建。单测/独审/服务器真实prepare在执行前完成，尚未推理。

23:50初审：20 CPU检查过，但非权重8件manifest未精确锁定、GNU timeout结束时无负责写terminal的父进程，属于真实准备缺口。修复为全9件canonical摘要冻结、独立Python supervisor监管600s及15s清理并原子记终态，补正常/异常/TERM/KILL用例；复审前不launch。范围与12输入/CPU/不接actor不变。

23:55修后作者23/23 CPU与diff通过：全9件manifest canonical SHA锁`046922670d986c772d797b0276079ec530456db2afe73fe198aec78e9e9104f7`，token/commit/spec/保留父PID及单次claim，supervisor独立记600s活动＋15s清理终态，完整12预期ID/exit0/CPU标记才通过。正常/partial/缺件/异常/TERM/KILL/spawn失败均有用例，超时保留原worker最后状态，以supervisor终态为准。修后独审待，尚未部署/推理。

23:58最终窄复审通过：另补TERM/INT handler、Popen赋值短窗信号遮罩、worker解遮罩；cleanup异常仍写terminal、重复停止不打断15s清理，恢复旧handler。pending-signal/cleanup异常用例也过，双人23/23、diff过，无剩余实质阻塞。下一固定Git及真实服务器prepare后单次运行，尚未推理。

23:59唯一launch：冻结`01e60457f14c5533de4f9493024fba1c52689e5a`/`git_worktrees/text_detector_01e6045`，双端23/独审/9权重及12原RAW像素SHA prepare过。UTC15:58:39.789044/supervisor3510163，输出与runtime创建。600s/12/CPU68–71等原预算不变，全部输出及人工验收待；不重复提交、不热改源、不称goal完成。

2026-09-25 00:01北京时间终态：原3510163/3510168 exit0/reaped，监管84.558186875s、worker83.237971349s，12/12完成，加载12.670870597s，172249090参数/CPU FP32/custom kernels关闭/CUDA未初始化。固定processor800最短边/1333最长边，当前各正方RAW被缩放到800；每调用5.225–6.466s。三head有框、缺席plate head无框，但多个腕图近全画面框，完整12 RAW人工语义审/包SHA核验尚待，暂不部署/不报告SR。

## 全12 RAW人工检查（2026-09-25 00:04北京时间，主agent）

完整19文件在本地`artifacts/agentic-vlm-goal-20260918/h58_text_detector_bundle_v1`；每条RAW像素SHA与原输入/输出回执一致，全部亲自查看，不只抽head。各条目录即下表ID；精确坐标/置信度/文本/时间都在result.json，没有修框或重推。

| ID | 原图与检测核验 | 结论边界 |
| --- | --- | --- |
| radio_head_d060__head | 0.834，紧框覆盖桌上红白收音机 | 可作为物体区域提议，未定位开关/接触面 |
| radio_head_d060__left_wrist | 0.535，把近场黑色机器人壳体框成radio | 明确本体误检，目标不可见 |
| radio_head_d060__right_wrist | 0.742，同一类本体壳体被框成radio | 明确本体误检 |
| refrigerator_handle_h44_d012__head | 0.509，框覆盖冰箱并排两根银色把手 | 有用的部件区域，但未区分左/右或单一接触表面 |
| refrigerator_handle_h44_d012__left_wrist | 0.755，本体大块壳体框为handle | 明确本体误检 |
| refrigerator_handle_h44_d012__right_wrist | 0.744，同类本体误检 | 不因高置信度采用 |
| bin_i71_d00__head | 0.728，框覆盖底部被图边截断的棕色桶 | 正确可见区域，未证明可抓取 |
| bin_i71_d00__left_wrist | 无框，画面主要地板/少量机器人，未见桶 | 正确弃权 |
| bin_i71_d00__right_wrist | 0.620，框覆盖左半图确实存在的棕色桶 | 大框在此是可用区域；不能一概按面积删框 |
| plate_absent_h44_radio_d000__head | 无框，未见盛食物盘 | 正确弃权 |
| plate_absent_h44_radio_d000__left_wrist | 0.480，机器人壳体被称plate | 明确本体误检 |
| plate_absent_h44_radio_d000__right_wrist | 0.563，同类本体误检 | 明确本体误检 |

这不是12独立测试实例，也不是检测mAP或任务SR。4条输出有区域用途（其中handle仍需左右/表面消歧）、2条可见性弃权合理、6条本体误检；不把这个计数报成跨任务准确率。主问题不是生成JSON或未缩放坐标，而是腕部被机器人自身遮挡时的目标幻觉。下一在允许的本体asset/current proprio/RGB-D中构建self-exclusion与可见性门；不靠对象GT，不盲删大框、不把框中心当抓取点。是否足以改善VLM选择及闭环仍需后续验证，当前不部署。

证据SHA：result `7aba22853b9458f06205abcdc3543ef166378a4dc7648bfc590ef0776854668c`；supervisor `41612e4086e1fdaa1b8b11b3d912cd1085664f14b026fd4b6f154b0e55972ea9`；launch `c5606062dcfe50c722e342d1d3f3ac8d0a5545c7da6048a4a868a72f8fe85a95`，均双端核同。原4训练保留、无检测GPU进程。
