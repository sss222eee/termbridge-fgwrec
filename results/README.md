# 实验结果

最终配置为 `q25_valid_r_plus_2n`：RRF 参数 `K=30`，`protect_top=5`，Tail/Mid/Head 分组比例为 25%/50%/25%。

| 方法 | 平均 Recall@20 | 平均 nDCG@20 |
|---|---:|---:|
| MF Base | 0.2428 | 0.1366 |
| TermBridge-FGWRec | **0.2534** | **0.1401** |

- `all_runs.csv`：小范围搜索的完整运行记录。
- `aggregate_runs.csv`：搜索结果的汇总统计。
- `city_summary.csv`：分城市结果。
- `group_summary.csv`：品牌流行度分组结果。
- `component_summary.csv`：组件和消融结果。
- `selected_weights.csv`：选中的融合权重。
- `EXPERIMENT_SUMMARY.md`：实验结果摘要。
- `EXPERIMENT_REPORT.md`：完整实验报告。
- `figures/`：实验图及图表索引。
