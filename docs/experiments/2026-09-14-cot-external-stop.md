# CoT被外部停止：诊断与恢复边界

Codex / AR-01-CoT，2026-09-14 19:30北京时间。**CoT、EMA、tail当前均未运行，不是继续排队。恢复GPU工作需先与用户协调。**

## 已确认的根因

只读查询系统journal发现：19:18:02，账号`ssy`通过sudo执行`/usr/bin/kill -9`，目标明确包含进程组`2032144`，也列有`2032153`–`2032156`及旧服务`3543247`等组。2032144正是已登记的CoT formal torchrun主进程；19:18:07.727482原supervisor记录子进程返回`-9`。因此本轮退出有明确外部强制终止证据，不应改学习率、数据或模型代码来“修复”这个退出。

不复制日志中的其他用户工作目录/无关命令到Git；原记录可在robo按UTC `2026-09-14 11:16:00`至`11:19:00`的system journal、相关PID过滤定位。kernel/dmesg无对应OOM/Xid，user.slice及user-1003.slice的oom/oom_kill/oom_group_kill均0；用户memory.max为max，当时事后检查可用约839GiB。当前空闲内存本身不能证明历史情况，但人工kill记录已直接解释-9。未确定该账号停止任务的业务意图，已向用户确认，不把账号等同于某位已知队友。

## 对三项实验的影响

| 原run | 最后可靠状态 | 当前证据 |
| --- | --- | --- |
| ar_native_subtask_cot_fulltrain_v1 | smoke5完整通过；formal最后记录39更新/624抽取，进程被kill | 没有formal/checkpoints目录或最终inspection；不能从39步无损恢复，不推断进程终止瞬间未记日志的状态 |
| fm_trainable_ema_v1 | 仅等待，无smoke/formal更新 | 19:18:13.856850因缺少CoT正式inspection而failed |
| fm_tail_lr_v1 | 仅等待，无smoke/formal更新 | 19:18:21.404421因缺少EMA正式inspection而failed |

三项根均在`robo:/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/`，旧method spec、launch、status、日志、CoT初始eval与smoke权重均保留；没有删除或覆盖任何实验。已完成的FM、marker及LoRA审计结果未重跑、未受此结论否定。EMA/tail只是依赖失败，尚没有它们的训练效果。

关键只读证据SHA：

- CoT status：`db56d42d6c847bca3766f93d85edc6c2c9e353353f3f886c23ab9fccaa11c4fe`；formal/train_metrics.jsonl：`2e6092ccd55291f178fb0202a07e30e7a095db2f6030c57dfbc5c5e46efcc2c4`。
- EMA status：`f0ef755e93840e70d63f6d9fa9492603a084fba8158c4aba1e824799009734a2`。
- tail status：`26583d8f5972baaf5ae8f9b1be25ec3f0717897e7ded3da53c5c71beb8ed860b`。
- CoT已有smoke权重：`12ae130c87d73525c4269ca6c9fe6684ad94aff345a339db68ffb3ed8cfbad1a`；它不是formal39权重，不得替换路径假装续训。

## 恢复时必须先做什么

1. 由用户与停止任务的账号协调，确认GPU可重新使用。确认前不重启训练、等待器或被停止的旧服务，不以新run名字绕过人工停止。
2. CoT无正式checkpoint，需要明确重新开始及已损耗更新的预算处理；不能宣称精确续训或偷偷增加步数。保留原父、数据及初始配方，另行决定是否增加中间保存/可靠恢复能力。
3. EMA/tail各自原5＋500完全未执行。协调后若恢复，必须新run/固定commit/严格旧失败身份及新依赖；当前校验器只白名单v1，尚不支持新恢复，不能热改v1 spec或用伪造complete放行。
4. 新36模块LoRA覆盖目前只有选型和静态API核对，没有新实现/真实梯度门。恢复后仍按原父函数保持、旧192 LoRA完整恢复、新参数梯度的次序，不改变r8/alpha16/dropout0.05，不直接追加训练臂。

这次goal轮取得了故障根因证据，完整goal仍未实现；首次出现的协调依赖不标goal complete或blocked。等待用户方向后从本记录续做，不重答历史超参问题。
