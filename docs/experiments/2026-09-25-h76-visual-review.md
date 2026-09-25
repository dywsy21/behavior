# H76：为可见性/部件定位微调准备原分辨率人工审查集

## 本人全部图审与保守标签（2026-09-25 18:10北京时间）

主代理亲自检查12张索引表全部108格（原腕图480、头部审图预览480），另重开4张720/480原图核小桶、radio出界、can遮挡、双盘歧义。逐36态×3视角的定位记录和标签在`configs/vlm_sft/h76_parent_visual_review_v1.json`，绑定原manifest SHA916ecc58…b19a7；未修改原不可训练的采集清单。

只批准后续有限的单图目标可见性微调：P表示可识别目标像素，N表示当前图无可识别像素（不推断物体不存在），U表示可见候选但身份/限定词不可核。数据、history、帧进度、相机名/实例号不作输入。对照/微调必须同一query+原图缩放到最多640、不放大480腕图；图表和跨视角人工推理不进入模型。此标签不是可执行接触点、稳定抓持或完成信号。

重点负/不确定例：task1 i278中期左腕是蓝色罐子而非桶；i194中期左腕只露棕色碎片，被橙罐遮挡，标U。task3末段大多只有盘背或头图底部白盘碎片，不能确认food，标U；i220中期右腕同理。i252首头图有两盘披萨：只可确认“至少一盘带食物的盘子可见”，不能生成唯一对象坐标、抓取点或伪造选择依据。

后继先测未微调2B视觉基线，再小预算固定步数训练/同一留出实例对照；只看视觉准确率/类别混淆与图像消融，不报完整SR，不自动部署。尚无新训练启动。

## 18:01原图完成（北京时间）

71a050d一次提取exit0/real7.09s、脚本6.962s；36态108图、720×720/480×480、图像22928169B，精确manifest/seal/来源/分组/SHA检查过。本地h76_raw_review_bundle_v1已收，双端聚合与本人逐图审查待。全部training_eligible=false，不等于已发布训练集；0模型调用、控制、训练。原run/source保留，不重提。

2026-09-25，北京时间；唯一owner Codex，源commit待。H75独立模拟器源保持冻结；本票只读原演示视频，不运行模型/仿真，不构造动作或完成真值。

主要假设：以图中可观察的目标/部件和UNKNOWN作监督，比让VLM猜不可观察的稳定tick或记动作历史，更适合作为当前grounded agent的感知微调入口。本次只检验数据覆盖，尚不承诺提升。

- SHA固定H09R counts和episode metadata/隔离区。只取additional_train；排除原5%/H09 val-test/开发138与242，以及后来eval的task1 i1/i71，不能只信旧cohort。
- 提取前资格更正：旧状态SFT已训练的task1 i114/i192另列`prior_state_training_groups`，本批也排除，防止未来拿旧adapter或熟悉的状态冒充新验证；不改变它们原本TRAIN身份。此次尚未提取/查看新图，没有按图像或模型表现挑分组。
- task0/1/3各按固定源实例hash取4实例（共12）。前三作候选视觉训练，第四作候选视觉验证；先分来源实例，再看图，不按模型效果挑样本。每实例取0/中间/90%位置，三相机共36态/108图；任何预选帧落quarantine就停止，不默换。
- 保留视频原像素分辨率与实际PTS、时间基、PNG SHA、源视频size/mtime及元数据SHA。源视频未做TB级全hash，不冒称已有视频内容SHA；输出像素有SHA。摄像头名/实例/私有语义只留provenance，后继actor输入不得含隐藏标识或标签来源。
- 完整parent人工看图并定位记录，遮挡/反射/看不清均保留UNKNOWN；不能把“无目标像素”自动称任务失败，不能由单图确认稳定抓握/真实任务完成。当前所有输出`training_eligible=false`，提取完成不是训练集发布。

预算一次CPU48–49、≤240s内限/270s外限、108图/≤1GiB、磁盘≥80GiB余量；GPU0、模型0、控制0、训练更新0。不触H75与队友活跃源码。新run `/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h76_raw_review_v1`，由Git独立源码执行，旧数据不覆盖。

固定外层监督命令为`/usr/bin/time -p /usr/bin/timeout --kill-after=5s 270s <现有VLM Python> scripts/vlm_sft/prepare_visual_review.py --output <上述新run>`，SSH侧另设330s连接上限，二者不同；终态记录真实exit code和time的real。内部计时从可选依赖import前开始，封存后再验240s；manifest的wall_seconds只记录其生成前时刻，最终进程用外层real。任何timeout/非零退出、failure.json、缺/坏complete.json均不算提取完成。须用validate_collection校验完整manifest/108个PNG逐SHA/36态三视角和9/3实例分组；目录存在、部分PNG或单独manifest不能通过。启动回执与失败原错保留，失败写证据自身报错不覆盖primary。

提取/人工检查后先看是否真有跨实例、视角内可见/不可见和可定位部件覆盖。没有覆盖就如实列缺口；具备后另登记短微调＋同split零微调对照，保持训练/推理输入协议一致。此集不是新独立完整任务测试，不把静态定位改善称SR提升。

17:53本地实现/审查闭合：10目标0.215s、全SFT198/1986.176s、独立10/10/0.188s与diff过。修正了首集成tuple/list持久化、失败写证据盖原错、部分文件误收及重新seal绕过source/预算合同；真实prepare mock覆盖成功108图、失败、SHA篡改、seal后超时、错误组/frame/split及缺启动回执。尚0真实提取/训练；先固定Git并核robo真实依赖/meta/quarantine，CPU过不当真实图已合格。

17:54唯一提交：source71a050db68bba7b92acc6baff524407eef68abd7/visual_review_71a050d，robo10测试0.350s，真实episode meta/quarantine SHA和36固定frame全过。av14.2.0/pyarrow22.0.0/Pillow10.2.0，实际外层补taskset48,49让import也受两CPU限制；270s timeout＋5s kill/330s SSH，stdout/time保存在新run旁侧.log。尚未确认108图完成/原始尺寸/人工标注，没有训练发布。最终split来源：t0 train135/31/70 val53；t1 train194/8/278 val185；t3 train96/146/252 val220。
