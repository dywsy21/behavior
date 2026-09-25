# H82：训练来源参考图辅助的标注校准

终态（2026-09-25 20:35北京时间）：source234d6f89f9abc98996354d31c25a8b1a19d759af，监管completed/exit0/436.930s，97生成/0训练控制，自有GPU释放。69/81=85.19%，P precision28/29=96.55%/recall28/32=87.5%，N precision37/42=88.10%未过原90%门。三task0/1/3分别25/27、23/27、21/27；U4/8，不能合并N/U冒称通过。**不准全量自动扩标。**

本人复查全部81图、29个P图的30框，记录`configs/vlm_sft/h82_parent_teacher_review_v1.json`；t3_i96_e678_f006750_right_wrist框只含披萨而漏盘子，已单列拒绝。其他框有部分遮挡粗框，无IoU金标。predictions SHAa231bbab2c4fa66f7f8fda304f2ef73c73ab0eff8c6ac1f9fb3bfc6fc7fb5cf1，本地小包`h82_reference_results_v1`、人工图`h82_parent_review_sheets_v1`。下一H83是新来源上严格双teacher一致筛选，另注册/不修改H82阈值或冻结金标。

以下保留运行前登记。

2026-09-25 20:05 北京时间；唯一负责人 Codex。**准备中，0 新生成/训练/控制。** 代码 commit 待审查后固定；新 run 预定 `/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h82_reference_calibration_v1`。

## 假设与固定对照

H81 纯文字类名 teacher 在模拟图上的识别仅 41/81，主要漏认侧面/近景及混淆机器人外壳。H82 唯一改变是：为同一 27B 提供两个目标外观 TRAIN 裁剪与一个机器人反例，再标注当前原图。参考图来自 H80 本人已审的 TRAIN 来源组，与 H76 校准的九组完全不同；不得从 val/test 选择参考。坐标只能指当前图，参考本身没有位置答案。

当前类别仍为 radio/wastebasket/plate holding food；不因通过而默认 task2/4 的其他类别也可靠。保留 H81 同一 81 TRAIN 图片、16 次固定重复、batch4/8、seed41、原图缩放及严格 JSON/框输出。原始 H76 标签冻结，绝不为提分改标签。参考图只用于 teacher 标注，未来学生仍是 CURRENT_RAW＋query；这是标注信息增强，不是给部署 actor 输入特权状态。

## 预算和准入

- 一次新 Git worktree/run；GPU2 ≤70GiB，CPU56–59，1800 秒含载入/生成/验收，另 30 秒清理；97 生成、192 新 token/样本、1800 输入 token、512MiB 产物；0 训练和控制。队友 GPU0/1 与 GPU3 不动。沿用 H81 同一权重/环境完整身份。
- 必须先本人审七张实际裁剪、单元检查与独立代码审查；每张原图 SHA、来源组、裁剪坐标和缩后像素保留。模型输入不能含校准答案/任务时刻/相机名。
- 第一层门：格式完整，固定 81 图正确率至少 80%，P precision 至少 90%、P recall 至少 75%、N precision 至少 90%；U 太少，不靠合并 U 与 N 提分。若未过，停止这套方案的全量自动扩标，不无变化重试。
- 即便上述过，也只进入“可审核伪标签”阶段；本人检查所有 P 框及错误输出，然后为 H80 新来源/其他对象单独分层审查。没有框金标不报 IoU，没有闭环不报 SR。明显错框/跨图坐标混淆不准入。

## 后续五小时训练

20:16 工程准入：父238完整SFT/15目标CPU与独立15目标通过；来源门从自报split升级为冻结H80 manifest/source_plan/ledger逐ref校验，实际裁剪尺寸/像素SHA锁定；终验重建81个真实processor输入token。本人已查看全部7实际裁剪（`h82_reference_crop_review_v1`），只批准teacher参考用途。尚未新调用，下一固定Git及robo真实环境/图像预检。

H80 已有 38,400 个 TRAIN 原图候选，不等于全部合格监督。后续依合格样本数、类/实例/近重复分布与新输出长度的实测吞吐固定训练步数，保存中间权重及独立 eval；不以同一小集重复或空等凑五小时。若参考 teacher 仍不够好，先更换可信标签来源，而非放宽质量要求。
