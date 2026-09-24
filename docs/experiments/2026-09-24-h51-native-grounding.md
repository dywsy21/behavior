# H51：Qwen原生点／框协议对照

Owner：Codex。2026-09-24。当前状态：20:26单次真实GPU探针运行中；不是成功率评测。

## 唯一主要假设

H50b仍有粗网格漏掉细把手及目标外选点。H51检验直接使用模型原生0–1000 `point_2d`/`bbox_2d`协议能否改善当前RAW上的定位。依据为[Qwen官方grounding示例](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)及[Qwen3.5对应问答](https://github.com/QwenLM/Qwen3.8/discussions/56)。原生提示、system、schema均变化，比较的是整套接口，不把结果单独归因于坐标尺度。

`src/semantic_robot/v2/native_grounding.py`每次只给模型一张完整、无标记的当前RAW和公开目标名称；没有旧答案、人工坐标、仿真目标位置、成功判据或机器人状态文字。单个点按width/height换算为原生像素，然后独立走当前公开RGB-D检查；边界1000映射到图外的点拒绝、不剪裁。box为半开像素区域，不取中心、不生成接触、持有或完成证据。

空数组表示弃权；多个目标保留歧义、不选第一个。错字段、重复键、float/bool坐标、越界、反向框拒绝；仅允许完整单层JSON代码围栏这一传输规范化，保留原文。点/框和capture/goal/RGB-D/标定/live FK绑定，跨帧或回调变异拒绝。深度有效不证明属于所问对象，仍须人工语义审及真实闭环。

## 冻结输入与对照

配置：`configs/semantic_robot/h51_native_grounding_probe.json`；四case canonical SHA `5f6847b6ca2f38794e76268d6286b3f13bb4a3602053cddeee3b150a30b2279b`。所有直接文件或递归capture清单均按SHA核验，只读取公开传感器、机器人本体和目标文本。原图在调用前已人工检查。

| 固定query | 保存态 | 人工调用前检查／限制 |
| --- | --- | --- |
| radio_head_d060 | H38 radio fullstart decision060 | 收音机可见；与H50为同一TRAIN138历史episode的不同帧 |
| refrigerator_handle_h44_d012 | H44 gate_plates decision012 | 细长把手可见；不是整个冰箱门 |
| bin_i71_d00 | H09y eval_t1_i71_base decision00/before | 底部部分截断的垃圾桶；有1038步演示前缀，仅诊断，不是零前缀 |
| plate_absent_h44_radio_d000 | H44 gate_radio decision000 | 当前图没有所问盛食物餐盘，应弃权；是新目标查询而非盲测新场景 |

四个查询不同于H50原四query，但都是既有开发态，并非四独立盲测episode。每条恰三独立调用：原自由UV基线、原生point、原生box；后者不读前者答案，box不指导本轮point。不重跑旧H50/H50b，不做新采集或训练。

模型沿用Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、原11文件SHA、NF4/double-quant/BF16、冻结vision/lm_head不量化策略，单张图上限320、greedy/seed17。原UV最多320输出token，原生点／框各128；提示长度不同，不声称纯单变量或直接比较整机器人Hz。

## 预算及停止条件

- 最多12请求、600s worker/900s外层监管，0模拟器reset/控制、0训练。错schema/截断/资源门失败保留原回答并终止，不修答案或重试。若提前失败，未执行的case如实留未测，不从成功子集报整体正确率。
- GPU2，UUID `GPU-3e4fda8c-536e-5899-e877-b8be97032fe0`，仅允许共享既有训练3294348；Torch allocator上限4864MiB、非Torch allowance512MiB，保留至少2048MiB。进程身份、模型、预算全部固定。未知进程/资源变化只停自有worker，不动训练。没有硬GPU隔离，不能承诺训练吞吐零影响。
- 运行前固定Git并在robo新建独立source worktree，沿用已有只读环境；不安装依赖、热改旧服务或拉取活跃源码。新run拟为`/mnt/nvme_tmp/robodojo_agentic_20260924/h51_native_protocol_v1`，cache拟为`/mnt/nvme_tmp/robodojo_vlm_runtime_20260924/h51`，当前尚未创建。

## 验收与后继

核心14新＋21既有、探针12、共享44，共91 CPU通过；20:24探针窄独审通过并独立复跑91项，无实质问题。四真实输入本地prepare/render barrier/递归SHA通过，binding已固定；spec SHA `76b0e7e366ff921c1dbf30b57d280833e3476a4c7698d5979684b45089347f61`。保存每次完整原始输出、请求/原生图/实际服务像素SHA、推理与资源回执。本人的逐项审核必须区分：点是否属于目标／细部、框是否包住指定目标、不可见是否拒绝、RGB-D是否通过。框好而点错不能算抓取可用；格式通过不能算语义通过。

仅在真实结果支持时接入现有agentic闭环；不能以本静态结果完成goal。共享模拟器显存仍需独立验证；最终验收保持从原reset起点、0专家/旧策略前缀的完整官方物理成功。任何小范围局部效果与完整SR分开记录。

## 结果

固定源`06360d389de3d75422cc35906b2675515832922d`，远端`git_worktrees/shared_small_vlm_06360d3`同91 CPU/四输入prepare通过。唯一launch UTC12:26:08.432985、supervisor3460478，spec SHA与上文一致；上述run/cache现已创建。实际输出、终态资源和全量人工审核待，不重复启动。
