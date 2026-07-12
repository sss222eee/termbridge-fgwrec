# 数据说明

本目录包含论文实验使用的四城市固定划分，以及复现最终方法所需的 term 特征。

- `official_split/`：Chicago、NYC、Singapore 和 Tokyo 的训练集、验证集、测试集及 ID 映射。
- `term_features/`：由 OpenSiteRec 训练可见信息确定性构造的品牌和区域 term 特征。
- `ODbL-LICENSE.txt`：OpenSiteRec 数据对应的 Open Database License。

原始数据集：[OpenSiteRec](https://github.com/HestiaSky/OpenSiteRec)

OpenSiteRec 数据及衍生数据库内容继续受 ODbL 约束，本仓库代码不会改变上游数据许可。MF 基线和本文方法始终使用相同的城市划分与评价条件。
