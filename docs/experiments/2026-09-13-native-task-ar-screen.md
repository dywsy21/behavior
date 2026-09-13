# 原生G0.5 task-only AR：500步筛选结果

Owner Codex，五任务方法筛选，不是50任务正式训练。run：`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/ar_native_task_fulltrain_v1`；训练源码6e2587b，spec SHA `2f01682d6912b14cd7c8d4d6371694cd81ea1303756968d3722bda68811c42af`。

## 配方与完成证据

- 原生G0.5 base SHA `072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327`、新零函数LoRA/新Adam；不接A4或前驱权重。任务＋六帧三相机/真实proprio，部署不要求MEM-Lite技能或teacher，未加CoT。原生词表不注册HL_END，state编号252188。
- 原950/50逐task切分，seed41、四A100/global16、每rank4 worker、LoRA1e-5/50 warmup/cosine500，动作专家/原词表冻结，纯CE、FM监督为0。5步保存门后从原生base独立执行正式500。
- 正式500更新/8000真实train抽取、192 Adam、冻结不变和完整模型/优化器/四rank RNG保存回读通过；2026-09-13 21:27:03北京时间状态complete，未实测新进程完整续训。rank0峰值reserved23,005,757,440 bytes。
- 最终`formal/checkpoints/step_500.pt` SHA `639e64aeeb251113b807751f234077595165e65e9dd9e3b66cd7c4661f9df963`，`formal/checkpoint_inspection.json`passed。2222863/2724340已退出。后继Beta曾按既有依赖开始preflight，随后在四卡入口因继承空CUDA可见性失败；不影响已完成native结果，恢复细节见plan最新记录。

## 效果与限制

| step | 固定80 CE | 五task各一窗无约束生成完整动作 |
| --- | --- | --- |
| 0 | 17.8859887 | 0/5 |
| 100 | 14.3310336 | 0/5 |
| 200 | 8.4513430 | 0/5 |
| 300 | 6.9902122 | 0/5 |
| 400 | 6.7992888 | 0/5 |
| 500 | 6.7387826 | 0/5 |

最终每task CE为6.5545993/6.7749644/6.8628391/6.5801337/6.9213763。所有终点输出仍37 tokens：双臂残差块后结束，漏lower_body及双夹爪；完整动作保护性检查拒绝，没有下发仿真，**不能给出新SR或说AR已可用**。最终eval SHA `eb796e8170e003d34485e4ecec06631e8f8427d915aec6705b7da86f0ecb667f`。

辅助FM参考初始1.4306554、终点1.4270732，来自未针对这组任务训练的原生FM专家与当前prefix；它不是AR动作误差，不能拿它和A4-FM的约0.197作AR控制优劣比较。

四rank各1000条原train来源回执与A4-AR500逐文件SHA相同：`c64cb087…`、`b22de4ae…`、`b1de49d5…`、`0e2fa483…`。说明两臂真实抽样来源/顺序相同，不是像素张量逐位对照；原生task与A4-skill同时改变了初始化和条件，不能把略低CE解释为去技能输入单因素收益。两者目前都没学会完整自由动作格式。

对实际回执的完整计数为8000抽取、950个不同train轨迹、每task各1600抽取；不是只用十行缓存完成500步，也不等于覆盖了全部技能段或8000个不同窗口。

## 后续

22:23更新：原生观察v3十状态通过后，0701fd1的`ar_native_actor_wire_probe_v1`一次真实神经/schema→原raw32×23→原msgpack检查complete，result SHA `2570ba7d943b39a05953bc7bcc69b216709955dafb2d3eda6575e45062d82c75`。完整1138/192、原生词表/统计/0:16全部核验，格式强制覆盖5次原argmax；没有teacher/技能/物理真值进入actor，0优化与仿真，不能当成功率。其首chunk raw yaw约-0.30，是否持续转动须靠真实闭环判断。

当前`ar_native_prefix_pilot_v1`固定989a575/290 CPU，私有服务3191531与实际socket身份门通过，唯一仿真3196694运行：同train121/instance138、env0/policy17、原448prefix、最多16 chunks/256模型控制、0训练/新数据release；GRASP与oracle只供评估器，task-only actor不接收。尚无物理结果，不将这个短预算与A4的464步局部成功直接当等预算胜负，或与完整任务SR混淆。

保留已声明的8行marker原切分候选与独立schema变体；[同窗口A4-AR诊断](2026-09-13-ar500-schema-diagnosis.md)已经表明强制格式后仍有动作内容误差。先完成真实观察→模型→23D逆变换和有限局部闭环，不能把工程门或总CE下降当成功率。原CoT输入/两临时更新门已做，正式CoT训练尚未放行；本轮不自动加5000步/全任务训练，也不从原生500覆盖当前A4-FM部署。
