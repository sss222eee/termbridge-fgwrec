from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RankingMetrics:
    recall: float
    ndcg: float
    num_brands: int

    def as_dict(self) -> dict[str, float | int]:
        return {"recall": self.recall, "ndcg": self.ndcg, "num_brands": self.num_brands}


def all_ranking_metrics(
    scores: np.ndarray,
    train_pos: dict[int, set[int]],
    eval_pos: dict[int, set[int]],
    k: int = 20,
    drop_train_overlap: bool = False,
) -> RankingMetrics:
    num_regions = scores.shape[1]
    top_k = min(k, num_regions)
    recalls: list[float] = []
    ndcgs: list[float] = []

    for brand, positives in eval_pos.items():
        if drop_train_overlap:
            positives = positives - train_pos.get(brand, set())
        if not positives:
            continue
        brand_scores = np.asarray(scores[brand], dtype=np.float64).copy()
        for region in train_pos.get(brand, set()):
            if 0 <= region < num_regions:
                brand_scores[region] = -np.inf

        if top_k == num_regions:
            ranked = np.argsort(-brand_scores)
        else:
            candidate = np.argpartition(-brand_scores, top_k - 1)[:top_k]
            ranked = candidate[np.argsort(-brand_scores[candidate])]

        hits = [idx for idx, region in enumerate(ranked[:top_k]) if int(region) in positives]
        recalls.append(len(hits) / len(positives))

        dcg = sum(1.0 / np.log2(rank + 2.0) for rank in hits)
        ideal_len = min(len(positives), top_k)
        idcg = sum(1.0 / np.log2(rank + 2.0) for rank in range(ideal_len))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    if not recalls:
        return RankingMetrics(0.0, 0.0, 0)
    return RankingMetrics(float(np.mean(recalls)), float(np.mean(ndcgs)), len(recalls))


def grouped_metrics(
    scores: np.ndarray,
    train_pos: dict[int, set[int]],
    eval_pos: dict[int, set[int]],
    train_counts: np.ndarray,
    k: int = 20,
    low_max: int = 10,
    mid_max: int = 30,
    drop_train_overlap: bool = False,
) -> dict[str, dict[str, float | int]]:
    groups = {
        "low": {int(i) for i, c in enumerate(train_counts) if 0 < c <= low_max},
        "medium": {int(i) for i, c in enumerate(train_counts) if low_max < c <= mid_max},
        "high": {int(i) for i, c in enumerate(train_counts) if c > mid_max},
    }
    out: dict[str, dict[str, float | int]] = {}
    for name, brands in groups.items():
        subset = {brand: pos for brand, pos in eval_pos.items() if brand in brands}
        out[name] = all_ranking_metrics(
            scores,
            train_pos,
            subset,
            k=k,
            drop_train_overlap=drop_train_overlap,
        ).as_dict()
    return out
