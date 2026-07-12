from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from otc.data import available_cities, load_city  # noqa: E402
from adaptive_transfer import load_test_lists, official_metrics  # noqa: E402
from tail_source_rrf import make_pseudo_valid_lists  # noqa: E402
from adaptive_fusion import (  # noqa: E402
    Components,
    build_components,
    component_metrics,
    fuse_with_vectors,
    parse_float_list,
)


GROUP_ORDER = ["tail", "mid", "head"]
ALL_GROUP_KEYS = ["tail", "low_mid", "mid", "high_mid", "head"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final TermBridge-FGWRec experiment.")
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
    parser.add_argument("--rrf-ks", default="20,30,40,60")
    parser.add_argument("--protect-tops", default="3,5,7")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--top-l", type=int, default=200)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--valid-tie-tol", type=float, default=1e-12)
    parser.add_argument("--out", default="outputs/final_small_tuning")
    args = parser.parse_args()

    cities = available_cities(args.data_root)
    targets = cities if args.target == "all" else [c.strip() for c in args.target.split(",")]
    rrf_ks = parse_float_list(args.rrf_ks)
    protect_tops = [int(x) for x in parse_float_list(args.protect_tops)]

    city_data = {city: load_city(args.data_root, city) for city in cities}
    scores = {city: np.load(Path(args.score_root) / city / "scores.npy") for city in cities}
    brand_terms = {city: np.load(Path(args.feature_root) / city / "brand_terms.npy") for city in cities}
    region_terms = {city: np.load(Path(args.feature_root) / city / "region_terms.npy") for city in cities}
    test_lists = {city: load_test_lists(Path(args.data_root) / city / "official_test_lists.json") for city in cities}

    variants = build_variants()
    if args.variants != "all":
        wanted = {x.strip() for x in args.variants.split(",") if x.strip()}
        variants = [v for v in variants if v["name"] in wanted]
        if not variants:
            raise SystemExit(f"No variants selected from {sorted(wanted)}")
    rows: list[dict] = []
    selected_rows: list[dict] = []
    component_rows: list[dict] = []
    best_scores: dict[tuple[float, int, str], dict[str, np.ndarray]] = {}

    for rrf_k in rrf_ks:
        local_args = argparse.Namespace(**vars(args))
        local_args.rrf_k = float(rrf_k)
        region_cache: dict[tuple[str, str], object] = {}
        for target in targets:
            pseudo_valid = make_pseudo_valid_lists(
                city_data[target].train_edges,
                ratio=args.inner_valid_ratio,
                seed=args.inner_valid_seed,
                min_items=args.inner_valid_min_items,
            )
            components = build_components(
                args=local_args,
                target=target,
                cities=cities,
                scores=scores,
                brand_terms=brand_terms,
                region_terms=region_terms,
                city_data=city_data,
                pseudo_valid=pseudo_valid,
                region_cache=region_cache,
            )
            if float(rrf_k) == 30.0:
                for comp_name, metrics in component_metrics(components, test_lists[target], args.k).items():
                    component_rows.append(
                        {"city": target, "component": comp_name, "recall": metrics["recall"], "ndcg": metrics["ndcg"]}
                    )
            for protect_top in protect_tops:
                for variant in variants:
                    run = select_group_fusion_variant(
                        components=components,
                        city=city_data[target],
                        pseudo_valid=pseudo_valid,
                        test_lists=test_lists[target],
                        alpha_candidates=variant["alpha"],
                        lambda_candidates=variant["lambda"],
                        top_l=args.top_l,
                        protect_top=protect_top,
                        k=args.k,
                        tie_tol=args.valid_tie_tol,
                        group_mode=variant["group_mode"],
                        objective=variant["objective"],
                        head_transfer=variant["head_transfer"],
                    )
                    metrics = run["metrics"]
                    rows.append(
                        {
                            "rrf_k": float(rrf_k),
                            "protect_top": int(protect_top),
                            "city": target,
                            "variant": variant["name"],
                            "recall": metrics["recall"],
                            "ndcg": metrics["ndcg"],
                            "valid_recall": run["valid"]["recall"],
                            "valid_ndcg": run["valid"]["ndcg"],
                        }
                    )
                    selected_rows.append(
                        {
                            "rrf_k": float(rrf_k),
                            "protect_top": int(protect_top),
                            "city": target,
                            "variant": variant["name"],
                            **flatten_selected(run["selected"]),
                        }
                    )
                    key = (float(rrf_k), int(protect_top), variant["name"])
                    best_scores.setdefault(key, {})[target] = run["final_scores"]
                print(f"k={rrf_k:g} {target} p={protect_top}: {len(variants)} variants", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    write_csv(out_dir / "final_all_runs.csv", rows)
    write_csv(out_dir / "final_aggregate_runs.csv", aggregate)
    write_csv(out_dir / "final_selected_weights.csv", selected_rows)
    write_csv(out_dir / "final_component_summary.csv", with_component_average(component_rows))

    best = max(aggregate, key=lambda x: (x["avg_recall"] + x["avg_ndcg"], x["avg_recall"]))
    best_key = (float(best["rrf_k"]), int(best["protect_top"]), best["variant"])
    best_dir = out_dir / "best_scores"
    for city, final_scores in best_scores[best_key].items():
        city_dir = best_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        np.save(city_dir / "final_scores.npy", final_scores)
    write_final_summary(out_dir, aggregate, rows, component_rows, city_data, test_lists, scores, best_scores[best_key], best)
    print(f"Saved Final experiment tuning to {out_dir}", flush=True)


def build_variants() -> list[dict]:
    alpha_base = [0.0, 0.02, 0.05, 0.1, 0.2]
    lambda_base = [0.0, 0.02, 0.05, 0.1, 0.2]
    alpha_wide = [0.0, 0.05, 0.1, 0.2, 0.4]
    return [
        {
            "name": "q30_sum",
            "group_mode": "q30",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "q20_sum",
            "group_mode": "q20",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "q25_sum",
            "group_mode": "q25",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "q30_50_20_sum",
            "group_mode": "q30_50_20",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "quartile4_sum",
            "group_mode": "quartile4",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "q30_head_no_transfer",
            "group_mode": "q30",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": False,
        },
        {
            "name": "q30_alpha_wide",
            "group_mode": "q30",
            "alpha": alpha_wide,
            "lambda": lambda_base,
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "q25_alpha_transfer_tuned",
            "group_mode": "q25",
            "alpha": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
            "lambda": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15],
            "objective": "sum",
            "head_transfer": True,
        },
        {
            "name": "q25_valid_r_plus_2n",
            "group_mode": "q25",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "recall_plus_2ndcg",
            "head_transfer": True,
        },
        {
            "name": "q30_valid_recall",
            "group_mode": "q30",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "recall",
            "head_transfer": True,
        },
        {
            "name": "q30_valid_ndcg",
            "group_mode": "q30",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "ndcg",
            "head_transfer": True,
        },
        {
            "name": "q30_valid_r_plus_2n",
            "group_mode": "q30",
            "alpha": alpha_base,
            "lambda": lambda_base,
            "objective": "recall_plus_2ndcg",
            "head_transfer": True,
        },
    ]


def select_group_fusion_variant(
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
    group_mode: str,
    objective: str,
    head_transfer: bool,
) -> dict:
    counts = np.asarray([len(city.train_pos.get(i, set())) for i in range(city.num_brands)], dtype=np.int64)
    groups = assign_groups(counts, group_mode)
    alpha_vec = np.zeros(city.num_brands, dtype=np.float64)
    lambda_vec = np.zeros(city.num_brands, dtype=np.float64)
    selected = {}
    group_order = groups_for_mode(group_mode)
    for group in group_order:
        brands = np.where(groups == group)[0]
        group_valid = {int(b): pseudo_valid[int(b)] for b in brands if int(b) in pseudo_valid}
        if not group_valid:
            selected[group] = {"alpha": 0.0, "lambda": 0.0, "num_brands": int(len(brands)), "num_valid_brands": 0}
            continue
        local_lambdas = [0.0] if group == "head" and not head_transfer else lambda_candidates
        best = None
        for alpha, lam in product(alpha_candidates, local_lambdas):
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
            score = valid_objective(valid, objective)
            complexity = float(alpha + lam)
            if best is None or score > best["score"] + tie_tol:
                best = {"alpha": float(alpha), "lambda": float(lam), "valid": valid, "score": score, "complexity": complexity}
            elif abs(score - best["score"]) <= tie_tol and complexity < best["complexity"]:
                best.update(
                    {"alpha": float(alpha), "lambda": float(lam), "valid": valid, "score": score, "complexity": complexity}
                )
        assert best is not None
        alpha_vec[brands] = best["alpha"]
        lambda_vec[brands] = best["lambda"]
        selected[group] = {
            "alpha": best["alpha"],
            "lambda": best["lambda"],
            "valid": best["valid"],
            "score": best["score"],
            "num_brands": int(len(brands)),
            "num_valid_brands": int(len(group_valid)),
        }
    final_scores = fuse_with_vectors(components, alpha_vec=alpha_vec, lambda_vec=lambda_vec, top_l=top_l, protect_top=protect_top)
    return {
        "final_scores": final_scores,
        "metrics": official_metrics(final_scores, test_lists, k=k),
        "valid": official_metrics(final_scores, pseudo_valid, k=k) if pseudo_valid else {"recall": 0.0, "ndcg": 0.0},
        "selected": selected,
    }


def assign_groups(counts: np.ndarray, mode: str) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    groups = np.empty(len(counts), dtype=object)
    if mode == "q20":
        q1, q3 = np.quantile(counts, [0.20, 0.80])
        groups[counts <= q1] = "tail"
        groups[(counts > q1) & (counts <= q3)] = "mid"
        groups[counts > q3] = "head"
    elif mode == "q25":
        q1, q3 = np.quantile(counts, [0.25, 0.75])
        groups[counts <= q1] = "tail"
        groups[(counts > q1) & (counts <= q3)] = "mid"
        groups[counts > q3] = "head"
    elif mode == "q30_50_20":
        q1, q3 = np.quantile(counts, [0.30, 0.80])
        groups[counts <= q1] = "tail"
        groups[(counts > q1) & (counts <= q3)] = "mid"
        groups[counts > q3] = "head"
    elif mode == "quartile4":
        q1, q2, q3 = np.quantile(counts, [0.25, 0.50, 0.75])
        groups[counts <= q1] = "tail"
        groups[(counts > q1) & (counts <= q2)] = "low_mid"
        groups[(counts > q2) & (counts <= q3)] = "high_mid"
        groups[counts > q3] = "head"
    else:
        q1, q3 = np.quantile(counts, [0.30, 0.70])
        groups[counts <= q1] = "tail"
        groups[(counts > q1) & (counts <= q3)] = "mid"
        groups[counts > q3] = "head"
    return groups


def groups_for_mode(mode: str) -> list[str]:
    if mode == "quartile4":
        return ["tail", "low_mid", "high_mid", "head"]
    return GROUP_ORDER


def valid_objective(valid: dict, objective: str) -> float:
    recall = float(valid["recall"])
    ndcg = float(valid["ndcg"])
    if objective == "recall":
        return recall
    if objective == "ndcg":
        return ndcg
    if objective == "recall_plus_2ndcg":
        return recall + 2.0 * ndcg
    return recall + ndcg


def flatten_selected(selected: dict) -> dict:
    out = {}
    for group in ALL_GROUP_KEYS:
        row = selected.get(group, {})
        out[f"{group}_alpha"] = row.get("alpha", "")
        out[f"{group}_lambda"] = row.get("lambda", "")
        out[f"{group}_valid_recall"] = row.get("valid", {}).get("recall", "")
        out[f"{group}_valid_ndcg"] = row.get("valid", {}).get("ndcg", "")
        out[f"{group}_brands"] = row.get("num_brands", "")
    return out


def aggregate_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["rrf_k"], r["protect_top"], r["variant"]) for r in rows})
    out = []
    for rrf_k, protect_top, variant in keys:
        part = [r for r in rows if r["rrf_k"] == rrf_k and r["protect_top"] == protect_top and r["variant"] == variant]
        out.append(
            {
                "rrf_k": rrf_k,
                "protect_top": protect_top,
                "variant": variant,
                "avg_recall": float(np.mean([r["recall"] for r in part])),
                "avg_ndcg": float(np.mean([r["ndcg"] for r in part])),
                "avg_valid_recall": float(np.mean([r["valid_recall"] for r in part])),
                "avg_valid_ndcg": float(np.mean([r["valid_ndcg"] for r in part])),
            }
        )
    return out


def city_group_rows(city_data, test_lists, base_scores, final_scores, k: int) -> tuple[list[dict], list[dict]]:
    city_rows = []
    group_rows = []
    for city, scores in final_scores.items():
        base = official_metrics(base_scores[city], test_lists[city], k=k)
        final = official_metrics(scores, test_lists[city], k=k)
        city_rows.append(
            {
                "city": city,
                "base_recall": base["recall"],
                "final_recall": final["recall"],
                "delta_recall": final["recall"] - base["recall"],
                "base_ndcg": base["ndcg"],
                "final_ndcg": final["ndcg"],
                "delta_ndcg": final["ndcg"] - base["ndcg"],
            }
        )
        counts = np.asarray([len(city_data[city].train_pos.get(i, set())) for i in range(city_data[city].num_brands)], dtype=np.int64)
        groups = assign_groups(counts, "q30")
        for group in GROUP_ORDER:
            group_tests = {int(b): regs for b, regs in test_lists[city].items() if b < len(groups) and groups[int(b)] == group}
            if not group_tests:
                continue
            base_g = official_metrics(base_scores[city], group_tests, k=k)
            final_g = official_metrics(scores, group_tests, k=k)
            group_rows.append(
                {
                    "city": city,
                    "group": group,
                    "num_test_brands": base_g["num_brands"],
                    "base_recall": base_g["recall"],
                    "final_recall": final_g["recall"],
                    "delta_recall": final_g["recall"] - base_g["recall"],
                    "base_ndcg": base_g["ndcg"],
                    "final_ndcg": final_g["ndcg"],
                    "delta_ndcg": final_g["ndcg"] - base_g["ndcg"],
                }
            )
    return city_rows, group_rows


def average_group_rows(rows: list[dict]) -> list[dict]:
    out = []
    for group in GROUP_ORDER:
        part = [r for r in rows if r["group"] == group]
        if not part:
            continue
        out.append(
            {
                "city": "Average",
                "group": group,
                "num_test_brands": int(np.sum([r["num_test_brands"] for r in part])),
                "base_recall": float(np.mean([r["base_recall"] for r in part])),
                "final_recall": float(np.mean([r["final_recall"] for r in part])),
                "delta_recall": float(np.mean([r["delta_recall"] for r in part])),
                "base_ndcg": float(np.mean([r["base_ndcg"] for r in part])),
                "final_ndcg": float(np.mean([r["final_ndcg"] for r in part])),
                "delta_ndcg": float(np.mean([r["delta_ndcg"] for r in part])),
            }
        )
    return out


def with_component_average(rows: list[dict]) -> list[dict]:
    out = list(rows)
    for comp in sorted({r["component"] for r in rows}):
        part = [r for r in rows if r["component"] == comp]
        out.append(
            {
                "city": "Average",
                "component": comp,
                "recall": float(np.mean([r["recall"] for r in part])),
                "ndcg": float(np.mean([r["ndcg"] for r in part])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_final_summary(
    out_dir: Path,
    aggregate: list[dict],
    rows: list[dict],
    component_rows: list[dict],
    city_data,
    test_lists,
    base_scores,
    best_scores,
    best: dict,
) -> None:
    city_rows, group_rows = city_group_rows(city_data, test_lists, base_scores, best_scores, k=20)
    write_csv(out_dir / "final_best_city_summary.csv", city_rows)
    write_csv(out_dir / "final_best_group_summary.csv", group_rows + average_group_rows(group_rows))
    component_avg = [r for r in with_component_average(component_rows) if r["city"] == "Average"]
    top = sorted(aggregate, key=lambda x: (x["avg_recall"] + x["avg_ndcg"], x["avg_recall"]), reverse=True)[:20]

    lines = [
        "# Final experiment Small Tuning Results",
        "",
        "## Best Run",
        "",
        "| Variant | RRF K | protect_top | Avg R@20 | Avg nDCG@20 |",
        "|---|---:|---:|---:|---:|",
        f"| {best['variant']} | {best['rrf_k']:.0f} | {best['protect_top']} | {best['avg_recall']:.4f} | {best['avg_ndcg']:.4f} |",
        "",
        "## Top Candidates",
        "",
        "| Variant | K | protect | Avg R@20 | Avg nDCG@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(f"| {row['variant']} | {row['rrf_k']:.0f} | {row['protect_top']} | {row['avg_recall']:.4f} | {row['avg_ndcg']:.4f} |")

    lines.extend(["", "## Component-Only", "", "| Component | Avg R@20 | Avg nDCG@20 |", "|---|---:|---:|"])
    for row in component_avg:
        lines.append(f"| {row['component']} | {row['recall']:.4f} | {row['ndcg']:.4f} |")

    lines.extend(["", "## Best City Detail", "", "| City | MF R@20 | Final experiment R@20 | Delta R | MF nDCG | Final experiment nDCG | Delta nDCG |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in city_rows:
        lines.append(
            f"| {row['city']} | {row['base_recall']:.4f} | {row['final_recall']:.4f} | {row['delta_recall']:+.4f} | "
            f"{row['base_ndcg']:.4f} | {row['final_ndcg']:.4f} | {row['delta_ndcg']:+.4f} |"
        )
    lines.append(
        f"| Average | {np.mean([r['base_recall'] for r in city_rows]):.4f} | "
        f"{np.mean([r['final_recall'] for r in city_rows]):.4f} | "
        f"{np.mean([r['delta_recall'] for r in city_rows]):+.4f} | "
        f"{np.mean([r['base_ndcg'] for r in city_rows]):.4f} | "
        f"{np.mean([r['final_ndcg'] for r in city_rows]):.4f} | "
        f"{np.mean([r['delta_ndcg'] for r in city_rows]):+.4f} |"
    )

    lines.extend(["", "## Best Group Detail", "", "| Group | MF R@20 | Final experiment R@20 | Delta R | MF nDCG | Final experiment nDCG | Delta nDCG |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in average_group_rows(group_rows):
        lines.append(
            f"| {row['group']} | {row['base_recall']:.4f} | {row['final_recall']:.4f} | {row['delta_recall']:+.4f} | "
            f"{row['base_ndcg']:.4f} | {row['final_ndcg']:.4f} | {row['delta_ndcg']:+.4f} |"
        )
    (out_dir / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
