# H85组合动作SFT数据：全量构造与训练读取接口

负责人：Codex主代理；2026-09-26。当前是离线候选格式，不替换现有agentic单微动作接口或FM策略，不启动新训练。

## 数据含义

沿固定原生动作中间集的466实例、256,214个16步窗口构造，其中TRAIN原有212,500条、validation23,009条、test20,705条。不是新增256,214条演示；同一实例内的相邻窗口也不等于统计独立轨迹。原始float32状态61D和动作16×23完整保留，所有实例隔离/保护组保持。

新增每条样本字段：当前actor的JSON、可重建的全身动作JSON、实际模型前缀/回答/总token长度、token与目标文本SHA、超长隔离原因。图片仍从原视频按各相机独立时基读取当前帧，不复制大批PNG，不把未来图像或私有实例ID输入模型。目标是低层短时动作文本，而不是让VLM只输出一个LEFT/UP标签；与Show-Harness单微动作协议并不等价。

原23D映射、量化误差和推理执行约束见[组合协议](2026-09-26-h85-composite-action-protocol.md)。技能文本代表意图，不代表成功；没有补造抓取、接触或任务完成标签。

## 构造与终态检查

`scripts/vlm_sft/prepare_trajectory_sft.py`在新目录写`shards/{train,validation,test}`及`quarantine/`。前缀独立文本展开先与50例已验证真实processor的逐token SHA匹配；新视频reader同时解码150当前原图，逐像素SHA对已审原PNG一致后才批量构造。每条动作做全tick数值还原、token原文往返，序列超过回答1536/总4096时保存完整样本至隔离目录，不静默截断。

每个输出Parquet逐字段往返比较；最后重验输入、全体输出SHA/大小/行数、精确输出文件集合和全部源视频身份后，才写manifest。失败标记和缺失manifest均不能视为完成。最大1800秒、4 worker/8 CPU、4GiB新输出，正式命令须外层`timeout --kill-after=10s 1830s`，无GPU、模型权重加载、训练或机器人控制。

输出manifest状态为`ACTION_SFT_FORMAT_COMPLETE_CAPACITY_PENDING`，`training_eligible:false`，不是长训许可。原中间集、封存图审和真实编码run全部保留。

## 训练读取接口

`scripts/vlm_sft/trajectory_dataset.py`提供`TrajectoryDataset(root, manifest_sha256, processor, split='train', allow_prepared=True)`供明确的离线加载检查。当前pending状态必须显式opt-in；它不会根据truthy的“发布”字段自行放行。正式训练还需要独立完成模型forward/backward与吞吐检查。

读取时只缓存至多两个Parquet分片、三路视频容器；进程改变则清空句柄，适用于worker独立读取。每个item只返回CPU tensor，不返回含参考动作/来源路径的原始字典。它将三张当前RGB与actor编码，核实际前缀/回答token SHA、长度、回答＋EOS监督，与构造时结果一致。原图最终输入尺寸为head448、左右腕320。

`trajectory_modeling.collate`负责不同长度的左填充，padding和全部输入前缀不参与loss；`supervised_loss`仅在回答token计算CE。两个接口沿用已回归的因果位移，不引入FM监督或改变旧trainer。数据文件中的审计ID可用于分组采样和日志，但不能被串进prompt。

## 验证进度及容量边界

41项CPU定向回归与独审通过，覆盖原23D映射、token还原、因果输入、跨实例/跨相机/未来帧错配、真实大小/哈希绑定、长JSON CE、输出/源篡改、quarantine、Dataset split与CPU输出。已有真实2B processor的50例/150原PNG检查通过，完整输入＋回答最长2153 tokens；这只是样本级证据。

全量导出与新的原视频preflight：**待真实run结果**。全量Dataset实际item加载、GPU forward/backward及至少3小时有效微调容量：**待验证**。不要用CPU分词速度或原有视觉分类训练速度代替此动作协议的训练吞吐。

后续必须按实际可用TRAIN数量和真实训练样本/秒核算：三个小时需要`10800 × 实测样本/秒`次样本展示，并报告覆盖的唯一实例/窗口及epoch数。不能用极小数据反复重复凑时间，也不把高CE/低CE直接当任务成功率。当前只是为后续50任务实验筛方法，未授权全任务重建或正式训练。
