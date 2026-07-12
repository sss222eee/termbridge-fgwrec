from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.semantic import normalize_rows  # noqa: E402
from run_termbridge_fgw import (  # noqa: E402
    compute_region_fgw,
    row_stochastic,
    termbridge_weights,
)


PAPER_MF = {
    "Chicago": {"recall": 0.2494, "ndcg": 0.1465},
    "NYC": {"recall": 0.1702, "ndcg": 0.0917},
    "Singapore": {"recall": 0.4430, "ndcg": 0.2351},
    "Tokyo": {"recall": 0.1323, "ndcg": 0.0781},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Scheme-6 adaptive TermBridge-FGWRec under official protocol.")
    parser.add_argument("--data-root", default="data/official_split")
    parser.add_argument("--feature-root", default="data/official_adaptive_terms")
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--target", default="all")
    parser.add_argument("--brand-topq", type=int, default=10)
    parser.add_argument("--brand-tau", type=float, default=0.1)
    parser.add_argument("--fgw-alpha", type=float, default=0.35)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--region-solver", choices=["signature", "pot"], default="signature")
    parser.add_argument("--beta-max", type=float, default=0.35)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--sparsity-power", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", default="outputs/adaptive_official_adaptive")
    args = parser.parse_args()

    cities = available_cities(args.data_root)
    targets = cities if args.target == "all" else [c.strip() for c in args.target.split(",")]
    city_data = {city: load_city(args.data_root, city) for city in cities}
    scores = {city: np.load(Path(args.score_root) / city / "scores.npy") for city in cities}
    brand_terms = {city: np.load(Path(args.feature_root) / city / "brand_terms.npy") for city in cities}
    region_terms = {city: np.load(Path(args.feature_root) / city / "region_terms.npy") for city in cities}
    test_lists = {
        city: load_test_lists(Path(args.data_root) / city / "official_test_lists.json")
        for city in cities
    }

    region_cache: dict[tuple[str, str], object] = {}
    results = {}
    for target in targets:
        base_scores = scores[target]
        base_metrics = official_metrics(base_scores, test_lists[target], k=args.k)
        z_base = zscore_rows(base_scores)
        train_counts = np.asarray([len(city_data[target].train_pos.get(i, set())) for i in range(city_data[target].num_brands)])
        beta = sparsity_beta(train_counts, beta_min=args.beta_min, beta_max=args.beta_max, power=args.sparsity_power)

        transfer_parts = []
        reliabilities = []
        source_details = []
        for source in cities:
            if source == target:
                continue
            brand_bridge = termbridge_weights(
                target_terms=brand_terms[target],
                source_terms=brand_terms[source],
                topq=args.brand_topq,
                tau=args.brand_tau,
            )
            region_tr = get_region_transport(
                region_cache,
                source=source,
                target=target,
                source_terms=region_terms[source],
                target_terms=region_terms[target],
                alpha=args.fgw_alpha,
                reg=args.reg,
                solver=args.region_solver,
            )
            region_mapping = row_stochastic(region_tr.plan)
            transferred = brand_bridge @ scores[source] @ region_mapping
            transfer_parts.append(zscore_rows(transferred))

            brand_rel = topq_term_similarity(brand_terms[target], brand_terms[source], topq=args.brand_topq)
            region_rel = float(np.exp(-max(region_tr.distance, 0.0)))
            reliability = brand_rel * region_rel
            reliabilities.append(reliability)
            source_details.append(
                {
                    "source": source,
                    "region_solver": region_tr.solver,
                    "region_distance": float(region_tr.distance),
                    "region_confidence": float(region_tr.confidence),
                    "region_reliability": region_rel,
                    "brand_reliability_mean": float(np.mean(brand_rel)),
                    "brand_reliability_std": float(np.std(brand_rel)),
                }
            )

        transfer_mix = weighted_transfer_mix(transfer_parts, reliabilities)
        final_scores = z_base + beta[:, None] * transfer_mix
        final_metrics = official_metrics(final_scores, test_lists[target], k=args.k)
        np.save(Path(args.out) / target / "final_scores.npy", ensure_dir(Path(args.out) / target, final_scores))

        paper = PAPER_MF.get(target, {"recall": np.nan, "ndcg": np.nan})
        results[target] = {
            "method": "Adaptive transfer-Adaptive-TermBridge-FGWRec",
            "target": target,
            "base": base_metrics,
            "adaptive": final_metrics,
            "paper_mf": paper,
            "delta_vs_base": {
                "recall": final_metrics["recall"] - base_metrics["recall"],
                "ndcg": final_metrics["ndcg"] - base_metrics["ndcg"],
            },
            "delta_vs_paper_mf": {
                "recall": final_metrics["recall"] - paper["recall"],
                "ndcg": final_metrics["ndcg"] - paper["ndcg"],
            },
            "beta": {
                "min": float(np.min(beta)),
                "max": float(np.max(beta)),
                "mean": float(np.mean(beta)),
                "std": float(np.std(beta)),
            },
            "sources": source_details,
            "config": vars(args),
        }
        print(
            f"{target}: base {base_metrics['recall']:.4f}/{base_metrics['ndcg']:.4f} | "
            f"adaptive {final_metrics['recall']:.4f}/{final_metrics['ndcg']:.4f} | "
            f"delta {final_metrics['recall'] - base_metrics['recall']:+.4f}/"
            f"{final_metrics['ndcg'] - base_metrics['ndcg']:+.4f}",
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adaptive_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, results)


def ensure_dir(path: Path, arr: np.ndarray) -> np.ndarray:
    path.mkdir(parents=True, exist_ok=True)
    return arr


def load_test_lists(path: Path) -> dict[int, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


def official_metrics(scores: np.ndarray, test_lists: dict[int, list[int]], k: int = 20) -> dict[str, float | int]:
    top_k = min(k, scores.shape[1])
    recalls = []
    ndcgs = []
    for brand, positives in test_lists.items():
        if not positives:
            continue
        row = np.asarray(scores[brand], dtype=np.float64)
        if top_k == row.shape[0]:
            ranked = np.argsort(-row)
        else:
            candidate = np.argpartition(-row, top_k - 1)[:top_k]
            ranked = candidate[np.argsort(-row[candidate])]
        hits = np.asarray([1.0 if int(region) in positives else 0.0 for region in ranked[:top_k]], dtype=np.float64)
        recalls.append(float(np.sum(hits) / max(len(positives), 1)))
        ideal_len = min(len(positives), top_k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_len + 2)))) if ideal_len > 0 else 1.0
        dcg = float(np.sum(hits / np.log2(np.arange(2, top_k + 2))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return {"recall": float(np.mean(recalls)), "ndcg": float(np.mean(ndcgs)), "num_brands": len(recalls)}


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, keepdims=True)
    return (arr - mean) / np.maximum(std, 1e-8)


def sparsity_beta(counts: np.ndarray, beta_min: float, beta_max: float, power: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    max_count = float(np.max(counts)) if counts.size and np.max(counts) > 0 else 1.0
    dense = np.log1p(counts) / np.log1p(max_count)
    sparse = np.power(np.clip(1.0 - dense, 0.0, 1.0), float(power))
    return float(beta_min) + (float(beta_max) - float(beta_min)) * sparse


def topq_term_similarity(target_terms: np.ndarray, source_terms: np.ndarray, topq: int) -> np.ndarray:
    target = normalize_rows(np.asarray(target_terms, dtype=np.float64))
    source = normalize_rows(np.asarray(source_terms, dtype=np.float64))
    sim = target @ source.T
    q = min(max(1, int(topq)), source.shape[0])
    idx = np.argpartition(-sim, q - 1, axis=1)[:, :q]
    vals = np.take_along_axis(sim, idx, axis=1)
    return np.clip(np.mean(vals, axis=1), 0.0, 1.0)


def weighted_transfer_mix(parts: list[np.ndarray], reliabilities: list[np.ndarray]) -> np.ndarray:
    weights = np.stack(reliabilities, axis=0)
    weights = weights / np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-12)
    mix = np.zeros_like(parts[0], dtype=np.float64)
    for idx, part in enumerate(parts):
        mix += weights[idx, :, None] * part
    return mix


def get_region_transport(
    cache: dict[tuple[str, str], object],
    source: str,
    target: str,
    source_terms: np.ndarray,
    target_terms: np.ndarray,
    alpha: float,
    reg: float,
    solver: str,
):
    key = (source, target)
    if key in cache:
        return cache[key]
    reverse = (target, source)
    if reverse in cache:
        old = cache[reverse]
        copied = type(old)(
            plan=old.plan.T.copy(),
            distance=old.distance,
            confidence=old.confidence,
            solver=f"{old.solver}-transpose-cache",
        )
        cache[key] = copied
        return copied
    result = compute_region_fgw(
        source_terms,
        target_terms,
        alpha=alpha,
        reg=reg,
        solver=solver,
        max_iter=200,
    )
    cache[key] = result
    return result


def write_summary(out_dir: Path, results: dict[str, dict]) -> None:
    rows = []
    for city, row in results.items():
        rows.append(
            {
                "city": city,
                "base_recall": row["base"]["recall"],
                "base_ndcg": row["base"]["ndcg"],
                "scheme_recall": row["adaptive"]["recall"],
                "scheme_ndcg": row["adaptive"]["ndcg"],
                "paper_recall": row["paper_mf"]["recall"],
                "paper_ndcg": row["paper_mf"]["ndcg"],
            }
        )
    rows.append(
        {
            "city": "Average",
            "base_recall": float(np.mean([r["base_recall"] for r in rows])),
            "base_ndcg": float(np.mean([r["base_ndcg"] for r in rows])),
            "scheme_recall": float(np.mean([r["scheme_recall"] for r in rows])),
            "scheme_ndcg": float(np.mean([r["scheme_ndcg"] for r in rows])),
            "paper_recall": float(np.mean([r["paper_recall"] for r in rows])),
            "paper_ndcg": float(np.mean([r["paper_ndcg"] for r in rows])),
        }
    )
    with (out_dir / "adaptive_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Adaptive transfer Official Adaptive Results",
        "",
        "| City | MF Base R@20 | Adaptive transfer R@20 | Delta R | Paper MF R@20 | MF Base nDCG@20 | Adaptive transfer nDCG@20 | Delta nDCG | Paper MF nDCG@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['city']} | {r['base_recall']:.4f} | {r['scheme_recall']:.4f} | "
            f"{r['scheme_recall'] - r['base_recall']:+.4f} | {r['paper_recall']:.4f} | "
            f"{r['base_ndcg']:.4f} | {r['scheme_ndcg']:.4f} | "
            f"{r['scheme_ndcg'] - r['base_ndcg']:+.4f} | {r['paper_ndcg']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
