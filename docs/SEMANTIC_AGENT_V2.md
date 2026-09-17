# R1Pro语义agent v2：实现契约与验收

2026-09-17，Codex；独立于v1/G05/FM/MEM-Lite，当前分支`feat/semantic-agent-v2-20260917`。依据[失败审计](experiments/2026-09-17-semantic-agent-failure-audit.md)实现。**代码落地不等于闭环效果验收；实测状态以plan顶部为准。**

## 模块

| 文件（`src/semantic_robot/v2/`） | 职责 |
| --- | --- |
| `protocol.py` | 严格JSON微动作与可见事实：重复key/未知字段/NaN/歧义格式拒绝，无任意数值动作或GT字段 |
| `kinematics.py` | 18可控关节的局部POE FK/解析Jacobian，可序列化校准，纯numpy/scipy，不依赖模拟器 |
| `og_calibration.py` | 开发期从机器人局部姿态/Jacobian、固定相机安装位姿导出模型；不读场景物体/全局定位/奖励；与原生FK在新姿态比较 |
| `servo.py` | 关节界/每帧步长进入有界最小二乘，显式位置/姿态权重，FK线搜索/机器人胶囊近似碰撞检查；可达性预检、30Hz异常停机与可信执行回执 |
| `vision.py` | 当前原图＋机器人几何方向标记图＋动作前原图；图像尺寸与标定绑定，原图不被标记遮挡，没有GT目标标记 |
| `harness.py` | 显式目标/阶段、按阶段与手臂裁剪动作集、视觉/实际开度/实际微抬共同验证、有限恢复、无进展与往返振荡检测 |
| `policy.py` | 任务策略提示→VLM规划；每步先观察，再由harness更新阶段后选择有限动作；每次调用/输入有回执，失败不隐式重试 |

入口：`scripts/semantic_robot/serve_v2.py`、`run_v2.py`；CPU测试`PYTHONPATH=src python -m unittest discover -s tests/semantic_robot -v`。v1代码/旧试验原样保留。

## 动作和观察

动作示例：`{"part":"right","move":"up","scale":"micro","frame":"base"}`。部位、方向、粒度、参考系各有枚举和交叉校验；没有`L|BOTH`或新特殊token。未指定部位保持。普通抓取不开放另一手；`both`只执行共享base系的同步动作。

- 手臂平移micro/fine/coarse为2mm/1cm/3cm，旋转1°/3°/8°；底盘2/6/15cm，躯干3mm/1cm/2.5cm。
- `base`方向为机器人前/左/上；相机系为视图深处/画面左/画面上；`tool`为夹爪局部轴。坐标转换由固定的机器人几何模型执行，不让VLM猜四元数。
- CARRY由已验证持物及目标的level标志启用，限制步长/转腕；**保留当前姿态，不自动扶正倾斜盘、不承诺食物不滑**。
- 微动作标称12/18/24个30Hz控制帧；关节空间距离较大时按真实速度界预先延长，micro/fine/coarse硬上限24/40/64帧，超限拒绝。本体每帧检查，视觉在微动作之间重看。模型等待时模拟器暂停，视频不含等待时间。

观察JSON明确`visible/view/target_uv/enclosed/co_moving/supported/effect/hazard/note`。不可见时禁止编坐标；无法确认的布尔值用null。这里都是视觉模型的**观测声明**，不是仿真真值。

## 高低层与完成判据

VLM规划返回有限子目标，每项指定kind、target、hand、done_when、level。task0/3已有策略提示仍提供右手抓radio、左手按按钮或端盘注意点，但不提供隐藏位置、不生成固定动作轨迹。其他任务共用规划schema和阶段逻辑，不维护50套低层控制器。

阶段包括SEARCH、APPROACH、ALIGN、GRASP、VERIFY_GRASP、INTERACT、VERIFY_EFFECT、VERIFY_SUPPORT、RELEASE、VERIFY_PLACE、RECOVER。阶段与动作之间有真实校验：目标未找到时不开放手臂抓取；抓取未确认时不进入下一个目标；松手前再次确认支撑。刚闭合夹爪只进入验证阶段；第一次看到开关等效果后先HOLD复核，避免重复按键撤销效果。恢复必须实际执行一条新的成功恢复动作，不能拿上一步旧回执假装恢复完成。

持有证据必须至少同时有非空开度、视觉包围、前后图像随动声明和真实手部位移；宽度独立不能证明持物，模型自述也不能。置放要求松开实测及连续观察到支撑。跨目标保存持物对象的**观测性记录**，丢失证据回退，不引入MEM-Lite权重。最终计划耗尽仍标记`NOT_OFFICIAL_SUCCESS`，官方成功仅由隔离评估器记录。

恢复有3次预算；持续执行失败、无任务进展或取消位移的ABAB振荡进入恢复。不会一概禁止正常重复接近动作，不因危险而自动松开可能持有的物体。恢复失败则停，不空夹48次。

## 运动学、碰撞和失败边界

校准来自机器人自身参考关节位形的局部link姿态和运动Jacobian，转换为空间螺旋轴。远端policy可用保存的校准JSON及18维proprio执行FK/Jacobian，运行无需访问OG。校准必须在其他姿态与真实机器人比较；单纯解析Jacobian与自身FK差分一致是必要而不充分的验收。相机采用USD的+x右、+y上、-z前约定，投影与原图分辨率绑定。

**质心参考点不能混用：** PhysX线速度Jacobian在link质心，而pose是link原点。导出时读取机器人link自身的固定局部质心，使用`Jv_origin = Jv_com - cross(Jω, R_link·com_local)`，再构造POE和相机偏移Jacobian。忽略它时末端可能因质心接近零而看似正确，其他link/腕相机却跨姿态漂移。本轮真实门恰好捕获该错误，未通过放宽阈值处理。[NVIDIA工程说明](https://forums.developer.nvidia.com/t/differences-between-isaac-sim-and-mujoco-jacobians/252755)、[本机同系API说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)

执行器在关节求解过程中施加界和速度限制，使用FK预测检查改善与机器人碰撞距离，而非最后逐维裁剪。动作开始前有限迭代验证可达性；执行时偏差、持续限位或停滞触发零底盘速度/保持夹爪。回执区分TARGET_REACHED、UNREACHABLE_OR_COLLISION_BLOCKED、TRACKING_DIVERGED、JOINT_LIMIT_STALL、NO_MOTION_PROGRESS等；TARGET_REACHED只表示该微动作到达，不表示抓取/任务成功。

**有限时长计划必须与执行一致：** 在task3收拢姿态发现，32轮预检判定可达，但18帧逐次求解只能完成1cm命令中的1.5–1.7mm；去掉物理引擎仍复现。已改为先求终点，生成有速度界的quintic关节路径，逐点/中点检查机器人碰撞及笛卡尔位置/姿态通道，再执行同一条计划。物理滞后时整段统一缩步，不逐关节裁剪IK解；持续偏差仍中止。路径太长返回DURATION_LIMIT_EXCEEDED，不能承诺不到一秒必定到达所有姿态。终点位置容差按动作幅度为0.7–2.5mm，微动作完成不宣称数学零误差。

**碰撞能力有明确边界：** 当前是非相邻同臂/跨臂3cm半径胶囊近似及双末端8cm距离，OG仍启真实自碰撞。它不是全身精确mesh或环境碰撞规划器；RGB证据仍可能错，深度环境障碍约束尚需合法RGB-D接口与专门验收。不得用“控制器有保护”宣传为真实机器人安全认证。正式比赛v3.9.2部署未因本次v3.9.1开发测试自动通过，不热升级共享框架。

## 验证和模型

- CPU覆盖格式/传输外包装、FK差分/非零质心真值回归、像素坐标、真实收拢本体有限时长计划、限位重新分配、拒绝不可达、异常停止、躯干补偿、空夹与完成反馈、有限恢复及Python3.10兼容性。最新数量与真实门结果见plan。
- 真实gate在radio原448前缀经过的不同姿态及task3初态验证；gate结果与实际实现digest绑定。agent入口必须收到两个task都通过的回执，不能绕过失败门。
- 模型服务接受最多9张明确标签图，头图最长边640，腕图不放大；plan/observe/act分别有输出上限，只有动作使用当前阶段语法树。原始输出、输入像素hash、耗时和预算均记录，截断拒绝执行。
- 原Qwen3.5-4B作对照，官方Qwen3.8-27B固定revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`作强参考；没有训练或套用不匹配的Show-Harness adapter。下载55,586,036,737字节，文件大小校验完成；实际效果和GPU显存/速度待实测。
- 静态与物理试验限额在[plan](plan.md) H-06中；静态集按已有状态人工审核，不把模型回答或模板补全当机器人成功。
