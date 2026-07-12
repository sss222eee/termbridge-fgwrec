from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.semantic import normalize_rows  # noqa: E402
from adaptive_transfer import (  # noqa: E402
    PAPER_MF,
    get_region_transport,
    load_test_lists,
    official_metrics,
    topq_term_similarity,
    zscore_rows,
)
from tail_source_rrf import (  # noqa: E402
    make_pseudo_valid_lists,
    reciprocal_rank_scores,
    source_proxy_gain,
    sparsify_rows,
    weighted_mix,
)
from target_adaptive import apply_boundary_rerank, target_adaptive_source_weights  # noqa: E402
from run_termbridge_fgw import row_stochastic, termbridge_weights  # noqa: E402


@dataclass
class Components:
    local: np.ndarray
    semantic: np.ndarray
    transfer: np.ndarray
    raw_base: np.ndarray
    raw_semantic: np.ndarray
    raw_transfer: np.ndarray
    gamma: np.ndarray
    transfer_conf: np.ndarray
    source_details: list[dict]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Adaptive fusion pseudo-valid-driven fusion rerank.")
    parser.add_argument("--data-root", default="data/official_split")
    parser.add_argument("--feature-root", default="data/term_features")
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--target", default="all")
    parser.add_argument("--brand-topq", type=int, default=10)
    parser.add_argument("--brand-tau", type=float, default=0.1)
    parser.add_argument("--brand-rel-power", type=float, default=1.0)
    parser.add_argument("--fgw-alpha", type=float, default=0.35)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--region-solver", choices=["signature", "pot"], default="signature")
    parser.add_argument("--region-tau", type=float, default=0.2)
    parser.add_argument("--region-rel-power", type=float, default=1.0)
    parser.add_argument("--region-topk", type=int, default=10)
    parser.add_argument("--source-mode", choices=["none", "soft", "prune", "positive"], default="prune")
    parser.add_argument("--source-prune-eps", type=float, default=0.0005)
    parser.add_argument("--source-gain-scale", type=float, default=10.0)
    parser.add_argument("--source-gain-floor", type=float, default=0.05)
    parser.add_argument("--no-fallback-top-source", action="store_true")
    parser.add_argument("--inner-valid-ratio", type=float, default=0.2)
    parser.add_argument("--inner-valid-seed", type=int, default=2026)
    parser.add_argument("--inner-valid-min-items", type=int, default=4)
    parser.add_argument("--probe-lambda", type=float, default=0.05)
    parser.add_argument("--alpha-candidates", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--lambda-candidates", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--protect-tops", default="0,5,10")
    parser.add_argument("--top-l", type=int, default=200)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--valid-tie-tol", type=float, default=1e-12)
    parser.add_argument("--out", default="outputs/fusion_fusion")
    args = parser.parse_args()

    cities = available_cities(args.data_root)
    targets = cities if args.target == "all" else [c.strip() for c in args.target.split(",")]
    alpha_candidates = parse_float_list(args.alpha_candidates)
    lambda_candidates = parse_float_list(args.lambda_candidates)
    protect_tops = [int(x) for x in parse_float_list(args.protect_tops)]

    city_data = {city: load_city(args.data_root, city) for city in cities}
    scores = {city: np.load(Path(args.score_root) / city / "scores.npy") for city in cities}
    brand_terms = {city: np.load(Path(args.feature_root) / city / "brand_terms.npy") for city in cities}
    region_terms = {city: np.load(Path(args.feature_root) / city / "region_terms.npy") for city in cities}
    test_lists = {city: load_test_lists(Path(args.data_root) / city / "official_test_lists.json") for city in cities}

    region_cache: dict[tuple[str, str], object] = {}
    results: dict[str, dict] = {}
    for target in targets:
        pseudo_valid = make_pseudo_valid_lists(
            city_data[target].train_edges,
            ratio=args.inner_valid_ratio,
            seed=args.inner_valid_seed,
            min_items=args.inner_valid_min_items,
        )
        components = build_components(
            args=args,
            target=target,
            cities=cities,
            scores=scores,
            brand_terms=brand_terms,
            region_terms=region_terms,
            city_data=city_data,
            pseudo_valid=pseudo_valid,
            region_cache=region_cache,
        )

        target_result = {
            "method": "Adaptive fusion-Pseudo-valid-City-Tail-Adaptive-Fusion",
            "target": target,
            "paper_mf": PAPER_MF.get(target, {"recall": np.nan, "ndcg": np.nan}),
            "base": official_metrics(scores[target], test_lists[target], k=args.k),
            "component_only": component_metrics(components, test_lists[target], args.k),
            "protect_top_runs": {},
            "sources": components.source_details,
            "config": vars(args),
        }
        for protect_top in protect_tops:
            city_run = select_city_fusion(
                components=components,
                pseudo_valid=pseudo_valid,
                test_lists=test_lists[target],
                alpha_candidates=alpha_candidates,
                lambda_candidates=lambda_candidates,
                top_l=args.top_l,
                protect_top=protect_top,
                k=args.k,
                tie_tol=args.valid_tie_tol,
            )
            group_run = select_group_fusion(
                components=components,
                city=city_data[target],
                pseudo_valid=pseudo_valid,
                test_lists=test_lists[target],
                alpha_candidates=alpha_candidates,
                lambda_candidates=lambda_candidates,
                top_l=args.top_l,
                protect_top=protect_top,
                k=args.k,
                tie_tol=args.valid_tie_tol,
            )
            target_result["protect_top_runs"][str(protect_top)] = {
                "city_specific": city_run,
                "city_tail_group": group_run,
            }
            out_city = Path(args.out) / f"protect{protect_top}" / target
            out_city.mkdir(parents=True, exist_ok=True)
            np.save(out_city / "city_specific_scores.npy", city_run["final_scores"])
            np.save(out_city / "city_tail_group_scores.npy", group_run["final_scores"])
            print(
                f"{target} protect={protect_top}: city {city_run['metrics']['recall']:.4f}/"
                f"{city_run['metrics']['ndcg']:.4f} a={city_run['selected']['alpha']} "
                f"l={city_run['selected']['lambda']} | group {group_run['metrics']['recall']:.4f}/"
                f"{group_run['metrics']['ndcg']:.4f}",
                flush=True,
            )

        strip_arrays(target_result)
        results[target] = target_result

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fusion_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, results, protect_tops)
    print(f"Saved Adaptive fusion results to {out_dir}", flush=True)


def build_components(
    args,
    target: str,
    cities: list[str],
    scores: dict[str, np.ndarray],
    brand_terms: dict[str, np.ndarray],
    region_terms: dict[str, np.ndarray],
    city_data: dict,
    pseudo_valid: dict[int, list[int]],
    region_cache: dict[tuple[str, str], object],
) -> Components:
    base_scores = scores[target]
    sem_scores = semantic_scores(brand_terms[target], region_terms[target])
    transferred_parts: list[np.ndarray] = []
    brand_rel_parts: list[np.ndarray] = []
    structural_rel_parts: list[np.ndarray] = []
    source_gains: list[float] = []
    source_details: list[dict] = []

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
        region_mapping = sparsify_rows(row_stochastic(region_tr.plan), topk=args.region_topk)
        transferred = brand_bridge @ scores[source] @ region_mapping
        brand_rel = topq_term_similarity(brand_terms[target], brand_terms[source], topq=args.brand_topq)
        region_rel = float(np.exp(-max(float(region_tr.distance), 0.0) / max(args.region_tau, 1e-8)))
        structural_rel = np.power(np.clip(brand_rel, 0.0, 1.0), args.brand_rel_power) * (
            region_rel ** args.region_rel_power
        )
        gain = source_proxy_gain(
            base_scores=base_scores,
            transferred_scores=transferred,
            pseudo_valid=pseudo_valid,
            k=args.k,
            probe_lambda=args.probe_lambda,
            rrf_k=args.rrf_k,
        )
        transferred_parts.append(transferred)
        brand_rel_parts.append(brand_rel)
        structural_rel_parts.append(structural_rel)
        source_gains.append(gain)
        source_details.append(
            {
                "source": source,
                "region_solver": region_tr.solver,
                "region_distance": float(region_tr.distance),
                "region_confidence": float(region_tr.confidence),
                "region_reliability": float(region_rel),
                "brand_reliability_mean": float(np.mean(brand_rel)),
                "brand_reliability_std": float(np.std(brand_rel)),
                "train_only_proxy_gain": float(gain),
            }
        )

    gamma, selected_mask = target_adaptive_source_weights(
        structural_rel_parts=structural_rel_parts,
        source_gains=source_gains,
        mode=args.source_mode,
        prune_eps=args.source_prune_eps,
        gain_scale=args.source_gain_scale,
        gain_floor=args.source_gain_floor,
        fallback_top_source=not args.no_fallback_top_source,
    )
    for idx, selected in enumerate(selected_mask):
        source_details[idx]["selected"] = bool(selected)

    transfer_components = [reciprocal_rank_scores(part, k_const=args.rrf_k) for part in transferred_parts]
    transfer_rrf = weighted_mix(transfer_components, gamma)
    transfer_raw = weighted_mix(transferred_parts, gamma)
    transfer_conf = np.sum(gamma * np.stack(brand_rel_parts, axis=0), axis=0)
    return Components(
        local=reciprocal_rank_scores(base_scores, k_const=args.rrf_k),
        semantic=reciprocal_rank_scores(sem_scores, k_const=args.rrf_k),
        transfer=transfer_rrf,
        raw_base=base_scores,
        raw_semantic=sem_scores,
        raw_transfer=transfer_raw,
        gamma=gamma,
        transfer_conf=transfer_conf,
        source_details=source_details,
    )


def component_metrics(components: Components, test_lists: dict[int, list[int]], k: int) -> dict[str, dict]:
    out = {
        "local_mf": official_metrics(components.raw_base, test_lists, k=k),
        "rrf_local": official_metrics(components.local, test_lists, k=k),
        "semantic_only": official_metrics(components.raw_semantic, test_lists, k=k),
        "rrf_semantic_only": official_metrics(components.semantic, test_lists, k=k),
        "transfer_only": official_metrics(components.raw_transfer, test_lists, k=k),
        "rrf_transfer_only": official_metrics(components.transfer, test_lists, k=k),
    }
    return out


def select_city_fusion(
    components: Components,
    pseudo_valid: dict[int, list[int]],
    test_lists: dict[int, list[int]],
    alpha_candidates: list[float],
    lambda_candidates: list[float],
    top_l: int,
    protect_top: int,
    k: int,
    tie_tol: float,
) -> dict:
    best = None
    for alpha, lam in product(alpha_candidates, lambda_candidates):
        final = fuse_with_vectors(
            components,
            alpha_vec=np.full(components.local.shape[0], float(alpha), dtype=np.float64),
            lambda_vec=np.full(components.local.shape[0], float(lam), dtype=np.float64),
            top_l=top_l,
            protect_top=protect_top,
        )
        valid = official_metrics(final, pseudo_valid, k=k) if pseudo_valid else {"recall": 0.0, "ndcg": 0.0}
        valid_score = float(valid["recall"] + valid["ndcg"])
        complexity = float(alpha + lam)
        if best is None or valid_score > best["valid_score"] + tie_tol:
            best = {
                "final_scores": final,
                "valid": valid,
                "valid_score": valid_score,
                "complexity": complexity,
                "selected": {"alpha": float(alpha), "lambda": float(lam)},
            }
        elif abs(valid_score - best["valid_score"]) <= tie_tol and complexity < best["complexity"]:
            best.update(
                {
                    "final_scores": final,
                    "valid": valid,
                    "valid_score": valid_score,
                    "complexity": complexity,
                    "selected": {"alpha": float(alpha), "lambda": float(lam)},
                }
            )
    assert best is not None
    best["metrics"] = official_metrics(best["final_scores"], test_lists, k=k)
    return best


def select_group_fusion(
    components: Components,
    city,
    pseudo_valid: dict[int, list[int]],
    test_lists: dict[int, list[int]],
    alpha_candidates: list[float],
    lambda_candidates: list[float],
    top_l: int,
    protect_top: int,
    k: int,
    tie_tol: float,
) -> dict:
    counts = np.asarray([len(city.train_pos.get(i, set())) for i in range(city.num_brands)], dtype=np.int64)
    groups = assign_groups(counts)
    alpha_vec = np.zeros(city.num_brands, dtype=np.float64)
    lambda_vec = np.zeros(city.num_brands, dtype=np.float64)
    group_meta = {}

    for group_name in ["tail", "mid", "head"]:
        brands = np.where(groups == group_name)[0]
        group_valid = {int(b): pseudo_valid[int(b)] for b in brands if int(b) in pseudo_valid}
        if not group_valid:
            group_meta[group_name] = {"alpha": 0.0, "lambda": 0.0, "valid": {}, "num_brands": int(len(brands))}
            continue
        best = None
        for alpha, lam in product(alpha_candidates, lambda_candidates):
            local_alpha = np.zeros(city.num_brands, dtype=np.float64)
            local_lambda = np.zeros(city.num_brands, dtype=np.float64)
            local_alpha[brands] = float(alpha)
            local_lambda[brands] = float(lam)
            final = fuse_with_vectors(
                components,
                alpha_vec=local_alpha,
                lambda_vec=local_lambda,
                top_l=top_l,
                protect_top=protect_top,
            )
            valid = official_metrics(final, group_valid, k=k)
            valid_score = float(valid["recall"] + valid["ndcg"])
            complexity = float(alpha + lam)
            if best is None or valid_score > best["valid_score"] + tie_tol:
                best = {"alpha": float(alpha), "lambda": float(lam), "valid": valid, "valid_score": valid_score, "complexity": complexity}
            elif abs(valid_score - best["valid_score"]) <= tie_tol and complexity < best["complexity"]:
                best.update({"alpha": float(alpha), "lambda": float(lam), "valid": valid, "valid_score": valid_score, "complexity": complexity})
        assert best is not None
        alpha_vec[brands] = best["alpha"]
        lambda_vec[brands] = best["lambda"]
        group_meta[group_name] = {
            "alpha": best["alpha"],
            "lambda": best["lambda"],
            "valid": best["valid"],
            "valid_score": best["valid_score"],
            "num_brands": int(len(brands)),
            "num_valid_brands": int(len(group_valid)),
        }

    final_scores = fuse_with_vectors(
        components,
        alpha_vec=alpha_vec,
        lambda_vec=lambda_vec,
        top_l=top_l,
        protect_top=protect_top,
    )
    return {
        "final_scores": final_scores,
        "metrics": official_metrics(final_scores, test_lists, k=k),
        "valid": official_metrics(final_scores, pseudo_valid, k=k) if pseudo_valid else {"recall": 0.0, "ndcg": 0.0},
        "selected": group_meta,
        "alpha": stats(alpha_vec),
        "lambda": stats(lambda_vec),
        "group_counts": {g: int(np.sum(groups == g)) for g in ["tail", "mid", "head"]},
    }


def fuse_with_vectors(
    components: Components,
    alpha_vec: np.ndarray,
    lambda_vec: np.ndarray,
    top_l: int,
    protect_top: int,
) -> np.ndarray:
    fused = (
        components.local
        + np.asarray(alpha_vec, dtype=np.float64)[:, None] * components.semantic
        + np.asarray(lambda_vec, dtype=np.float64)[:, None] * components.transfer
    )
    return apply_boundary_rerank(components.raw_base, fused, top_l=top_l, protect_top=protect_top)


def assign_groups(counts: np.ndarray) -> np.ndarray:
    groups = np.empty(len(counts), dtype=object)
    groups[counts <= 5] = "tail"
    groups[(counts > 5) & (counts <= 20)] = "mid"
    groups[counts > 20] = "head"
    return groups


def semantic_scores(brand: np.ndarray, region: np.ndarray) -> np.ndarray:
    return normalize_rows(np.asarray(brand, dtype=np.float64)) @ normalize_rows(np.asarray(region, dtype=np.float64)).T


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def strip_arrays(obj) -> None:
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if isinstance(obj[key], np.ndarray):
                del obj[key]
            else:
                strip_arrays(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            strip_arrays(item)


def write_summary(out_dir: Path, results: dict[str, dict], protect_tops: list[int]) -> None:
    rows = []
    for city, row in results.items():
        for protect_top in protect_tops:
            runs = row["protect_top_runs"][str(protect_top)]
            rows.append(
                {
                    "city": city,
                    "protect_top": protect_top,
                    "base_recall": row["base"]["recall"],
                    "base_ndcg": row["base"]["ndcg"],
                    "city_recall": runs["city_specific"]["metrics"]["recall"],
                    "city_ndcg": runs["city_specific"]["metrics"]["ndcg"],
                    "group_recall": runs["city_tail_group"]["metrics"]["recall"],
                    "group_ndcg": runs["city_tail_group"]["metrics"]["ndcg"],
                    "city_alpha": runs["city_specific"]["selected"]["alpha"],
                    "city_lambda": runs["city_specific"]["selected"]["lambda"],
                }
            )
    avg_rows = []
    for protect_top in protect_tops:
        parts = [r for r in rows if int(r["protect_top"]) == int(protect_top)]
        avg_rows.append(
            {
                "city": "Average",
                "protect_top": protect_top,
                "base_recall": float(np.mean([r["base_recall"] for r in parts])),
                "base_ndcg": float(np.mean([r["base_ndcg"] for r in parts])),
                "city_recall": float(np.mean([r["city_recall"] for r in parts])),
                "city_ndcg": float(np.mean([r["city_ndcg"] for r in parts])),
                "group_recall": float(np.mean([r["group_recall"] for r in parts])),
                "group_ndcg": float(np.mean([r["group_ndcg"] for r in parts])),
                "city_alpha": "",
                "city_lambda": "",
            }
        )
    all_rows = rows + avg_rows
    with (out_dir / "fusion_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    component_rows = []
    for city, row in results.items():
        for name, metrics in row["component_only"].items():
            component_rows.append({"city": city, "component": name, "recall": metrics["recall"], "ndcg": metrics["ndcg"]})
    for name in sorted({r["component"] for r in component_rows}):
        vals = [r for r in component_rows if r["component"] == name]
        component_rows.append(
            {
                "city": "Average",
                "component": name,
                "recall": float(np.mean([r["recall"] for r in vals])),
                "ndcg": float(np.mean([r["ndcg"] for r in vals])),
            }
        )
    with (out_dir / "fusion_component_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        writer.writerows(component_rows)

    lines = [
        "# Adaptive fusion Fusion Results",
        "",
        "## Protect Top Comparison",
        "",
        "| Protect Top | City Fusion R@20 | City Fusion nDCG@20 | City+Tail R@20 | City+Tail nDCG@20 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in avg_rows:
        lines.append(
            f"| {r['protect_top']} | {r['city_recall']:.4f} | {r['city_ndcg']:.4f} | "
            f"{r['group_recall']:.4f} | {r['group_ndcg']:.4f} |"
        )

    lines.extend(["", "## Component-Only Average", "", "| Component | Avg R@20 | Avg nDCG@20 |", "|---|---:|---:|"])
    for r in [x for x in component_rows if x["city"] == "Average"]:
        lines.append(f"| {r['component']} | {r['recall']:.4f} | {r['ndcg']:.4f} |")

    best_protect = max(avg_rows, key=lambda r: (r["group_recall"] + r["group_ndcg"], r["group_recall"]))
    lines.extend(
        [
            "",
            "## Best City+Tail Detail",
            "",
            f"Best protect_top = {best_protect['protect_top']}",
            "",
            "| City | Base R/nDCG | City Fusion R/nDCG | City+Tail R/nDCG | City alpha/lambda | Tail alpha/lambda | Mid alpha/lambda | Head alpha/lambda |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    best_key = str(best_protect["protect_top"])
    for city, row in results.items():
        runs = row["protect_top_runs"][best_key]
        city_sel = runs["city_specific"]["selected"]
        groups = runs["city_tail_group"]["selected"]
        lines.append(
            f"| {city} | {row['base']['recall']:.4f}/{row['base']['ndcg']:.4f} | "
            f"{runs['city_specific']['metrics']['recall']:.4f}/{runs['city_specific']['metrics']['ndcg']:.4f} | "
            f"{runs['city_tail_group']['metrics']['recall']:.4f}/{runs['city_tail_group']['metrics']['ndcg']:.4f} | "
            f"{city_sel['alpha']}/{city_sel['lambda']} | "
            f"{groups['tail']['alpha']}/{groups['tail']['lambda']} | "
            f"{groups['mid']['alpha']}/{groups['mid']['lambda']} | "
            f"{groups['head']['alpha']}/{groups['head']['lambda']} |"
        )
    (out_dir / "NUMERIC_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
