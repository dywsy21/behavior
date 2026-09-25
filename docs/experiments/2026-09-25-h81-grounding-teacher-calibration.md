# H81：扩数据前的视觉定位teacher校准（准备中，未启动）

2026-09-25 19:25北京时间，owner Codex。承接用户“更多数据、至少五小时微调”的指令；先检验标签来源是否可靠，不把模型生成的描述当作接触、抓持、动作或任务成功真值。

主要假设：现有27B基座能在原始演示图上提供有用的可见性与2D框监督，质量/批吞吐足以扩充小VLM训练。仅限本次81个既有人审TRAIN图；H76的27个val图及H80新val/test全部不用来改prompt/筛teacher。不得借已知标签作为输入提示。

## 拟定协议

只输入一张CURRENT_RAW及普通语言目标查询；与H78一样头图缩至640、腕图保持480，不输入task/episode/frame/view名、历史、演示进度或特权状态。输出严格JSON `{"visibility":"present|absent|uncertain","boxes":[[x1,y1,x2,y2],...]}`，坐标按图像宽高归一化到0–1000；present需要至少一个对应可识别目标的可见范围框，absent/uncertain为空。多目标允许多个框，不能武断选唯一目标；盘背无法确认食物时不确定。

teacher先用已在服务器的27B：目录名虽为Qwen3.8-27B，实际配置为Qwen3_5ForConditionalGeneration，revision1d4bf0f…c60c0、全部30文件身份由h79_model_identity.json固定。不加载H78小adapter、不升级依赖。2B后续训练与部署采用同一新视觉定位输入前缀，不能拿旧三选一协议权重直接当微动作actor。

## 校准预算与验收

- 一次独立source/run，GPU2独占≤70GiB、其余卡0占用，CPU56–59；0物理控制/训练；81原图（9训练实例）各一次，另16张相同TRAIN图以第二批大小重复，至多97条生成，max_new_tokens192。GPU3保留，GPU0/1队友不动。
- 1800s包含载入/生成/整理、另30s清理，≤512MiB新结果、≥80GiB磁盘余量。单条异常、EOS缺失、JSON/框非法保留并计失败，不补调用直到答对；不以格式合法代替视觉正确。
- 固定81人审可见性混淆矩阵、P/N/U精确率/召回、实际输出长度、batch4与batch8每图耗时及显存。本人逐张查看至少全部P预测框和可见性错例，另N/U分层样本；保留图ID/原标签/预测/框审记录。框目前没有人工金标，不编造IoU。
- 只有质量足以进入“可审核伪标签”阶段才另登记H80全量标注；若teacher幻觉/漏检明显，停止自动扩标，先改数据来源或标注方法。97条并非数万伪标签的质量保证，扩标后仍需父本人按5task×3视角×遮挡/远近/正负不确定分层审查，并对数据按实例/像素去重。

当前代码与GPU监管尚在准备，未新增模型调用/训练。后续五小时有效训练预算要根据新输出长度的实测吞吐、合格独立图像数来固定，不照搬H78短分类速度，也不以延时/重复小集凑时长。

19:41工程门通过：`visual_grounding.py`共享前缀/严格框协议，`calibrate_grounding_teacher.py`仅TRAIN批量生成，`launch_grounding_calibration.py`身份/资源/退出验收。父完整229 SFT CPU8.262s、8目标1.900s，独审8/8 1.859s并放行一次登记校准。拒重复JSON键、EOS前special/后非pad，保留完整generated token IDs、终审用固定processor重新decode；P/N/U/invalid混淆与逐batch秒/token数重算。预定run `/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h81_grounding_calibration_v1`，冻结源码待Git固定后填。尚未调用模型/生成框标签。
