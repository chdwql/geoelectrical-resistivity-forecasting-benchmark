# DLinear与PatchTST目标域模型卡

状态：DLinear与PatchTST配置已在验证期选择并冻结；G5全部U/M/X正式规格零失败，G6统计完成。

## 固定身份

- DLinear参考：`cure-lab/LTSF-Linear`提交`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`
- PatchTST参考：`yuqinie98/PatchTST`提交`204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- 参考代码许可：Apache-2.0
- 本地运行时：独立`environments/target_deep`，Torch `2.14.0+cu130`
- 固定种子：17、29、43
- 支持轨道：U、M、X

## 架构定义

DLinear保留长度25的移动平均分解，以及季节项和趋势项各自从730天到预测时距的线性映射；映射后的多通道预测由显式线性头融合为焦点测道输出。

PatchTST使用长度16、步长8的非重叠式滑动patch输入，通道独立共享三层Transformer编码器，`d_model=64`、4头；编码后的全部通道表示由显式输出头融合。该融合使M/X轨道的信息能够影响焦点测道预测，但属于披露的目标域变体。

## 合成能力证据

- 输入形状：4批次×730天×5通道；
- 输出形状：4批次×180天；
- 前向、反向、优化器更新均成功；
- eval模式重复推理最大绝对差为0；
- DLinear参数量263,166，PatchTST参数量5,340,980；
- 两模型合计探针峰值显存约119MB。

证据保存于忽略提交的`artifacts/model_probes/target_deep.json`。完整训练轮数和超参数将在G4仅用验证期选择。

## 真实数据契约与冻结配置

- 核心环境从冻结窗口表选择`train`与`validation`样本；
- 长表`values.parquet`保存标准化输入、训练标签及观测掩码，`samples.jsonl`保存窗口身份、锚点与尺度；
- validation样本仅含输入，不含任何`label`行；
- 剩余缺口使用逐时点历史扩展均值因果填充，目标标签按起报点前锚点和一阶差分尺度标准化；
- G3能力门使用20条目标序列各3个训练窗口、三个固定种子和2个epoch，该成绩未参与选模；
- DLinear与PatchTST在11003、12301两站U-90均2/2成功，各输出180个有限预测值；
- 两模型在两站M/X-90及U-30也均2/2成功；
- DLinear输出已通过统一评分器。G4最终将两模型均固定为20个epoch；DLinear学习率为0.0003，PatchTST为0.0001。冻结后使用截至2023年底的允许数据重拟合一次，测试期间不更新权重。
