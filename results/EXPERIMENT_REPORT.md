# Robustness validation micro-tuning report

## Conclusion

Robustness validation mainly confirms the TermBridge-FGWRec final setting instead of producing a new large gain. The best configuration remains the q25 city-tail adaptive fusion with RRF K=30 and protect_top=5. Using the validation objective Recall+2*nDCG reaches the same final average result as the TermBridge-FGWRec best candidate.

## Best result

| Variant | RRF K | protect_top | Avg Recall@20 | Avg nDCG@20 |
|---|---:|---:|---:|---:|
| q25_valid_r_plus_2n | 30 | 5 | 0.2534 | 0.1401 |

Compared with the MF baseline average 0.2428 / 0.1366, the final method improves Recall@20 by +0.0106 and nDCG@20 by +0.0035.

## City-level result

| City | MF R@20 | Final R@20 | Delta R | MF nDCG@20 | Final nDCG@20 | Delta nDCG |
|---|---:|---:|---:|---:|---:|---:|
| Chicago | 0.2348 | 0.2469 | +0.0121 | 0.1460 | 0.1498 | +0.0038 |
| NYC | 0.1658 | 0.1694 | +0.0037 | 0.0884 | 0.0896 | +0.0012 |
| Singapore | 0.4383 | 0.4652 | +0.0269 | 0.2360 | 0.2448 | +0.0088 |
| Tokyo | 0.1324 | 0.1323 | -0.0001 | 0.0759 | 0.0762 | +0.0003 |

## Popularity-group result

| Group | Delta Recall@20 | Delta nDCG@20 |
|---|---:|---:|
| Tail | +0.0170 | +0.0057 |
| Mid | +0.0109 | +0.0032 |
| Head | +0.0023 | +0.0011 |

## Figure files

- figures/final_method_comparison.svg
- figures/city_delta.svg
- figures/group_delta.svg
- figures/ablation_comparison.svg
- figures/q25_rrf_heatmap.svg

## Suggested reporting choice

Use Robustness validation as a final micro-tuned version of TermBridge-FGWRec, not as a substantially new method. The report can say: after small-range tuning of group ratio, RRF K, protect_top and validation objective, the earlier TermBridge-FGWRec structure is stable; the improvement mainly comes from long-tail and mid-popularity items.
