# 同状态接管第二例：收音机episode91

2026-09-11；本人执行及视觉复核，无子代理。此为train-only局部纠正尝试，不是完整任务成功率。

**结论：不准入成功纠正数据。** episode91 / 官方train instance97；先执行467个演示前缀动作，再由旧A执行1280个实际控制，仍未完成GRASP。独立原始G0.5从source frame1747同一状态接管，再执行1280步，到frame3027仍未确认抓取成功；不把超时改成FAILED类，不认证教师，不声称复放通过。

## 本人实际抽查

- 视频100.93秒、1514帧、15 fps；前缀约0–15.57秒，A约15.57–58.23秒，G0.5约58.23–100.90秒。
- 本人查看0–95秒每5秒的20面板，并查看58.20/60/65/70/85/100秒六个放大面板。这是关键帧检查，不是逐帧看完视频。
- 前缀抵达收音机桌边，A阶段有视角/接近调整，后期右手出现在收音机前；接管时右夹爪在物体近侧，随后沿物体上缘/正面继续调整。录像没有证明稳定夹持或完成开机，不能把接触/推转误作抓取成功。
- 本窗口未呈现持续大角度原地转圈；与第一训练收音机实例一样，主要现象是近距离操作停滞，但两者不能外推为所有任务都同一根因。

## 物理记录交叉核验

全部1280条纠正记录的source frame1748–3027连续；1280/1280为IN_PROGRESS，目标没有被任一手确认持有。两手confirmed/unknown列表全空、official terminal=0；第一查询前后状态逐位不变。记录均隔离于policy输入且training_admissible=false。

首例同样未成功，两例都保持隔离诊断。当前仅能得出“这个独立G0.5候选在这两个真实未完成状态下没有救回抓取”，不能用2例给出总体恢复率或宣布方法优劣。

[本地完整视频](/home/wsy/behavior/memlite-results-20260911/C2-radio-train-e91/rollout.mp4)

![全程抽样](/home/wsy/behavior/memlite-results-20260911/C2-radio-train-e91/review_0_95s.jpg)

![接管细节](/home/wsy/behavior/memlite-results-20260911/C2-radio-train-e91/review_takeover_58_100s.jpg)

原始目录：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c2_same_live_state_pilot_v1/c1v2-matched-t0-train-e91-f467-grasp`；未修改原结果、动作或oracle证据。
