# TermBridge-FGWRec

本仓库提供课程论文《面向跨城市选址推荐的语义桥接与长尾自适应排序融合方法》的代码、处理后数据、MF 模型产物和实验结果。

项目在 OpenSiteRec 上复现 MF 基线，并加入三项改进：品牌 TermBridge 语义对齐、区域 FGW 迁移，以及城市--长尾自适应 RRF 融合。基线与本文方法使用完全相同的城市划分、候选集合和评价协议。

## 主要结果

| 方法 | 平均 Recall@20 | 平均 nDCG@20 |
|---|---:|---:|
| MF Base | 0.2428 | 0.1366 |
| TermBridge-FGWRec | **0.2534** | **0.1401** |

最终方法相对 MF Base 的平均 Recall@20 提升 0.0106，平均 nDCG@20 提升 0.0035。分城市、品牌流行度分组、消融和参数验证结果见 [`results/`](results/README.md)。

## 仓库结构

```text
artifacts/mf_bce_seed2024/  对齐后的 MF 模型、embedding、分数矩阵与指标
configs/final.json          最终随机种子和超参数
data/official_split/        所有方法统一使用的 OpenSiteRec 固定划分
data/term_features/         由训练可见信息构造的语义与结构 term 特征
results/                    最终结果表、搜索记录和论文图表
scripts/run_final.py        最终实验统一运行入口
src/otc/                    数据、模型、OT/FGW 与评价模块
paper/paper_zh.pdf          中文论文
```

## 环境配置

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 复现最终实验

在仓库根目录运行：

```bash
python scripts/run_final.py \
  --data-root data/official_split \
  --feature-root data/term_features \
  --score-root artifacts/mf_bce_seed2024 \
  --target all \
  --variants q25_valid_r_plus_2n \
  --rrf-ks 30 \
  --protect-tops 5 \
  --out outputs/final
```

MF 基线随机种子为 `2024`，训练内部代理验证随机种子为 `2026`。最终配置同时记录在 [`configs/final.json`](configs/final.json) 中。

## 数据与许可

实验使用 [OpenSiteRec](https://github.com/HestiaSky/OpenSiteRec)。仓库中的处理后划分及衍生数据库内容继续遵循上游 ODbL 许可，详见 [`data/ODbL-LICENSE.txt`](data/ODbL-LICENSE.txt)。使用相关数据或复现实验时，请引用 OpenSiteRec/OTC 原论文。

## 说明

- 已保存的 MF 分数矩阵可直接用于最终融合实验，无需重新训练基线。
- 大型中间搜索产物和重复特征版本未纳入仓库。
- 复现论文结果请使用统一入口 `scripts/run_final.py`；其余脚本按数据准备、训练和评价职责命名。
