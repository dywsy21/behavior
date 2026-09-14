# M-03：EMA接入前的真实实现检查

2026-09-14 16:52北京时间，owner Codex。**已完成源码核对与CPU标量恢复检查，尚未运行EMA策略训练，不是EMA收益结论。** 原KI/AR队列和已完成权重不受影响，没有修改共享环境、主训练入口或当前实验trainer。

## 三个已查实的接入条件

1. 当前`train_action_method_probe.py`和`train_fm_method_probe.py`都没有EMA更新/评估分支，因此改`use_ema: true`并不会自动给这些实验添加EMA。此前各500运行也没有可后处理的逐步EMA序列。
2. 主`finetune.py`虽然支持EMA，但将`cfg.model.ema.power`传入库的`beta`。当前配置0.67实际得到`beta=0.67`、库自身`power=2/3`、默认`update_every=10`；不能把配置字段名当成实际衰减公式，也不据命名差异推断作者意图错误。后续配方必须显式写清beta、power、更新间隔和起点。
3. 主保存入口只保存`ema_model.ema_model.state_dict()`，恢复也只加载该子模型；已安装库另有`step`和`initted`。仅平均权重无法恢复这些时钟，下一次update可能重新初始化平均权重。必须保存/恢复完整EMA状态及配置，再验证继续更新，而不是只看EMA tensor可加载。

库实际为`ema-pytorch 0.7.7`，源码`robo:/mnt/sdc1/robodojo/GalaxeaVLA/.venv/lib/python3.10/site-packages/ema_pytorch/ema_pytorch.py`，SHA `f87d751eb3e4fc2977adbb0d7a2fd765f110efb07723dd0b4ab5b3ff37a505d0`。`init_ema`默认deepcopy完整传入模型，包括冻结部分；尚未实测本策略的额外显存，不能把训练参数影子占用当成完整模型副本占用。后续须选择明确跟踪范围并做资源门。

## CPU恢复小例子：确实会丢掉平均历史

一个CPU标量权重依次赋1至35，每次调用原库EMA update，采用上述实际构造参数；然后在同一个后续值36上比较三条分支。没有真实策略前向、训练数据、optimizer更新或模拟器控制。

| 分支 | 恢复后step / initted | 更新前EMA权重 | 下一次update后step / 权重 |
| --- | --- | ---: | --- |
| 连续运行 | 35 / true | 16.8033695221 | 36 / 16.8033695221 |
| 完整EMA state恢复 | 35 / true | 16.8033695221 | 36 / 16.8033695221 |
| 仅EMA模型权重恢复 | 0 / false | 16.8033695221 | 1 / 36.0 |

前两条完全一致；第三条重置时钟，第一次update将已有平均历史覆盖成在线权重。此例的完整分支在第36次调用不更新平均，是因为库采用每10次调用更新的时钟；这正说明只比较“加载后的平均权重相同”不够。[精确数值和身份](results/2026-09-14-ema-preflight.json)。这里只做同进程CPU state_dict检查，未实测磁盘/新进程恢复或真实策略EMA。

## 后续边界

- 这轮没有选择EMA衰减值、启动EMA500或重复原FM500；不能把终点权重混合冒充逐步EMA。
- 真正对照需在线权重与EMA权重并列评估，原未加权固定80、真实动作接口及训练更新顺序不变。EMA改善评估可能来自平滑，也可能因滞后而更差，不声称训练loss会下降更快。
- 已有`get_scheduler.py`可复用warmup→平台→cosine，但当前实验入口明确使用cosine；日程对照须单项接入并固定步数/峰值/边界，不顺带与EMA或分模块裁剪组合。日程、容量及EMA方法效果仍待后续选择/实测。
