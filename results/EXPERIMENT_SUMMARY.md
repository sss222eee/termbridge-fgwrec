# TermBridge-FGWRec Small Tuning Results

## Best Run

| Variant | RRF K | protect_top | Avg R@20 | Avg nDCG@20 |
|---|---:|---:|---:|---:|
| q25_valid_r_plus_2n | 30 | 5 | 0.2534 | 0.1401 |

## Top Candidates

| Variant | K | protect | Avg R@20 | Avg nDCG@20 |
|---|---:|---:|---:|---:|
| q25_valid_r_plus_2n | 30 | 5 | 0.2534 | 0.1401 |
| q25_sum | 30 | 5 | 0.2534 | 0.1401 |
| q20_sum | 30 | 5 | 0.2533 | 0.1400 |
| q30_50_20_sum | 30 | 5 | 0.2533 | 0.1400 |
| q25_sum | 30 | 6 | 0.2534 | 0.1399 |
| q25_valid_r_plus_2n | 30 | 6 | 0.2534 | 0.1399 |
| q30_sum | 30 | 5 | 0.2532 | 0.1400 |
| q20_sum | 30 | 6 | 0.2533 | 0.1398 |
| q30_50_20_sum | 30 | 6 | 0.2533 | 0.1398 |
| q25_sum | 25 | 4 | 0.2529 | 0.1401 |
| q25_valid_r_plus_2n | 25 | 4 | 0.2529 | 0.1401 |
| q20_sum | 25 | 4 | 0.2526 | 0.1400 |
| q30_50_20_sum | 25 | 4 | 0.2526 | 0.1400 |
| q30_sum | 30 | 6 | 0.2528 | 0.1397 |
| q30_sum | 25 | 4 | 0.2526 | 0.1399 |
| q25_valid_r_plus_2n | 25 | 5 | 0.2522 | 0.1398 |
| q25_sum | 25 | 5 | 0.2522 | 0.1397 |
| q30_sum | 35 | 5 | 0.2522 | 0.1398 |
| q30_sum | 35 | 4 | 0.2519 | 0.1399 |
| q25_valid_r_plus_2n | 25 | 6 | 0.2522 | 0.1396 |

## Component-Only

| Component | Avg R@20 | Avg nDCG@20 |
|---|---:|---:|
| local_mf | 0.2428 | 0.1366 |
| rrf_local | 0.2428 | 0.1366 |
| rrf_semantic_only | 0.1016 | 0.0527 |
| rrf_transfer_only | 0.0806 | 0.0364 |
| semantic_only | 0.1016 | 0.0527 |
| transfer_only | 0.0759 | 0.0345 |

## Best City Detail

| City | MF R@20 | TermBridge-FGWRec R@20 | Delta R | MF nDCG | TermBridge-FGWRec nDCG | Delta nDCG |
|---|---:|---:|---:|---:|---:|---:|
| Chicago | 0.2348 | 0.2469 | +0.0121 | 0.1460 | 0.1498 | +0.0038 |
| NYC | 0.1658 | 0.1694 | +0.0037 | 0.0884 | 0.0896 | +0.0012 |
| Singapore | 0.4383 | 0.4652 | +0.0269 | 0.2360 | 0.2448 | +0.0088 |
| Tokyo | 0.1324 | 0.1323 | -0.0001 | 0.0759 | 0.0762 | +0.0003 |
| Average | 0.2428 | 0.2534 | +0.0107 | 0.1366 | 0.1401 | +0.0035 |

## Best Group Detail

| Group | MF R@20 | TermBridge-FGWRec R@20 | Delta R | MF nDCG | TermBridge-FGWRec nDCG | Delta nDCG |
|---|---:|---:|---:|---:|---:|---:|
| tail | 0.3163 | 0.3332 | +0.0170 | 0.1686 | 0.1742 | +0.0057 |
| mid | 0.2388 | 0.2497 | +0.0109 | 0.1178 | 0.1210 | +0.0032 |
| head | 0.1478 | 0.1501 | +0.0023 | 0.1168 | 0.1178 | +0.0011 |
