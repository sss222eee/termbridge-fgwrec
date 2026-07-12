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
    tail_strength,
    weighted_mix,
)
from run_termbridge_fgw import row_stochastic, termbridge_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Target adaptive Target-adaptive Tail-aware Rerank on saved official MF scores."
    )
    parser.add_argument("--data-root", default="data/official_split")
    parser.add_argument("--feature-root", default="data/official_adaptive_terms")
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
    parser.add_argument("--tail-mode", choices=["tau", "normalized"], default="normalized")
    parser.add_argument("--tail-tau", type=float, default=1.0)
    parser.add_argument("--tail-power", type=float, default=3.0)
    parser.add_argument("--confidence-power", type=float, default=0.0)
    parser.add_argument("--lambda-candidates", default="0,0.02,0.05,0.08,0.1")
    parser.add_argument("--alpha-candidates", default="0,0.02,0.05,0.08,0.1")
    parser.add_argument("--source-mode", choices=["none", "soft", "prune", "positive"], default="prune")
    parser.add_argument("--source-prune-eps", type=float, default=0.0005)
    parser.add_argument("--source-gain-scale", type=float, default=10.0)
    parser.add_argument("--source-gain-floor", type=float, default=0.05)
    parser.add_argument("--no-fallback-top-source", action="store_true")
    parser.add_argument("--inner-valid-ratio", type=float, default=0.2)
    parser.add_argument("--inner-valid-seed", type=int, default=2026)
    parser.add_argument("--inner-valid-min-items", type=int, default=4)
    parser.add_argument("--probe-lambda", type=float, default=0.05)
    parser.add_argument("--fusion", choices=["rrf", "zscore"], default="rrf")
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--top-l", type=int, default=200)
    parser.add_argument("--protect-top", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--valid-tie-tol", type=float, default=1e-12)
    parser.add_argument("--adaptive-json", default="outputs/adaptive_official_adaptive_b005_p3/adaptive_results.json")
    parser.add_argument("--tail_source-json", default="outputs/tail_source_final_tail_source_sparse/tail_source_results.json")
    parser.add_argument("--out", default="outputs/target_adaptive_target_adaptive")
    args = parser.parse_args()

    cities = available_cities(args.data_root)
    targets = cities if args.target == "all" else [c.strip() for c in args.target.split(",")]
    lambda_candidates = parse_float_list(args.lambda_candidates)
    alpha_candidates = parse_float_list(args.alpha_candidates)

    city_data = {city: load_city(args.data_root, city) for city in cities}
    scores = {city: np.load(Path(args.score_root) / city / "scores.npy") for city in cities}
    brand_terms = {city: np.load(Path(args.feature_root) / city / "brand_terms.npy") for city in cities}
    region_terms = {city: np.load(Path(args.feature_root) / city / "region_terms.npy") for city in cities}
    test_lists = {city: load_test_lists(Path(args.data_root) / city / "official_test_lists.json") for city in cities}
    adaptive = load_json(args.adaptive_json)
    tail_source = load_json(args.tail_source_json)

    region_cache: dict[tuple[str, str], object] = {}
    results: dict[str, dict] = {}
    for target in targets:
        base_scores = scores[target]
        base_metrics = official_metrics(base_scores, test_lists[target], k=args.k)
        pseudo_valid = make_pseudo_valid_lists(
            city_data[target].train_edges,
            ratio=args.inner_valid_ratio,
            seed=args.inner_valid_seed,
            min_items=args.inner_valid_min_items,
        )
        train_counts = np.asarray(
            [len(city_data[target].train_pos.get(i, set())) for i in range(city_data[target].num_brands)],
            dtype=np.float64,
        )
        tail = tail_strength(train_counts, tau=args.tail_tau, power=args.tail_power, mode=args.tail_mode)
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
                    "region_reliability": region_rel,
                    "brand_reliability_mean": float(np.mean(brand_rel)),
                    "brand_reliability_std": float(np.std(brand_rel)),
                    "train_only_proxy_gain": gain,
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

        transfer_conf = np.sum(gamma * np.stack(brand_rel_parts, axis=0), axis=0)
        selection = select_target_params(
            base_scores=base_scores,
            sem_scores=sem_scores,
            transferred_parts=transferred_parts,
            gamma=gamma,
            tail=tail,
            transfer_conf=transfer_conf,
            pseudo_valid=pseudo_valid,
            lambda_candidates=lambda_candidates,
            alpha_candidates=alpha_candidates,
            confidence_power=args.confidence_power,
            fusion=args.fusion,
            rrf_k=args.rrf_k,
            top_l=args.top_l,
            protect_top=args.protect_top,
            k=args.k,
            tie_tol=args.valid_tie_tol,
        )
        final_scores = selection["final_scores"]
        final_metrics = official_metrics(final_scores, test_lists[target], k=args.k)

        out_city = Path(args.out) / target
        out_city.mkdir(parents=True, exist_ok=True)
        np.save(out_city / "final_scores.npy", final_scores)
        np.save(out_city / "source_gamma.npy", gamma)
        np.save(out_city / "semantic_scores.npy", sem_scores)
        np.save(out_city / "lambda_bt.npy", selection["lambda_bt"])
        np.save(out_city / "alpha_bt.npy", selection["alpha_bt"])

        paper = PAPER_MF.get(target, {"recall": np.nan, "ndcg": np.nan})
        adaptive_city = adaptive.get(target, {})
        tail_source_city = tail_source.get(target, {})
        results[target] = {
            "method": "Target adaptive-Target-adaptive-Tail-aware-Rerank",
            "target": target,
            "base": base_metrics,
            "adaptive": adaptive_city.get("adaptive", {}),
            "tail_source": tail_source_city.get("tail_source", {}),
            "target_adaptive": final_metrics,
            "paper_mf": paper,
            "selected": {
                "lambda_max": selection["lambda_max"],
                "alpha_max": selection["alpha_max"],
                "valid": selection["valid"],
                "valid_score": selection["valid_score"],
            },
            "delta_vs_base": {
                "recall": final_metrics["recall"] - base_metrics["recall"],
                "ndcg": final_metrics["ndcg"] - base_metrics["ndcg"],
            },
            "delta_vs_tail_source": delta_metrics(final_metrics, tail_source_city.get("tail_source", {})),
            "delta_vs_adaptive": delta_metrics(final_metrics, adaptive_city.get("adaptive", {})),
            "lambda": stats(selection["lambda_bt"]),
            "alpha": stats(selection["alpha_bt"]),
            "tail": stats(tail),
            "transfer_confidence": stats(transfer_conf),
            "pseudo_valid_brands": len(pseudo_valid),
            "sources": source_details,
            "config": vars(args),
        }
        print(
            f"{target}: base {base_metrics['recall']:.4f}/{base_metrics['ndcg']:.4f} | "
            f"target_adaptive {final_metrics['recall']:.4f}/{final_metrics['ndcg']:.4f} | "
            f"lambda={selection['lambda_max']:.3g} alpha={selection['alpha_max']:.3g} | "
            f"delta {final_metrics['recall'] - base_metrics['recall']:+.4f}/"
            f"{final_metrics['ndcg'] - base_metrics['ndcg']:+.4f}",
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "target_adaptive_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, results)
    print(f"Saved Target adaptive results to {out_dir}", flush=True)


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_json(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def semantic_scores(brand: np.ndarray, region: np.ndarray) -> np.ndarray:
    return normalize_rows(np.asarray(brand, dtype=np.float64)) @ normalize_rows(np.asarray(region, dtype=np.float64)).T


def target_adaptive_source_weights(
    structural_rel_parts: list[np.ndarray],
    source_gains: list[float],
    mode: str,
    prune_eps: float,
    gain_scale: float,
    gain_floor: float,
    fallback_top_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    structural = np.stack(structural_rel_parts, axis=0)
    gains = np.asarray(source_gains, dtype=np.float64)
    if mode == "none":
        selected = np.ones(structural.shape[0], dtype=bool)
    elif mode == "positive":
        selected = gains > 0.0
    elif mode == "prune":
        selected = gains > -float(prune_eps)
    else:
        selected = np.ones(structural.shape[0], dtype=bool)

    if fallback_top_source and not np.any(selected):
        selected[int(np.argmax(np.mean(structural, axis=1)))] = True

    gain_factor = np.clip(1.0 + float(gain_scale) * gains, float(gain_floor), None)
    if mode == "none":
        gain_factor = np.ones_like(gains)
    rel = structural * gain_factor[:, None]
    rel[~selected, :] = 0.0
    col_sum = np.sum(rel, axis=0, keepdims=True)
    gamma = rel / np.maximum(col_sum, 1e-12)
    gamma[:, col_sum[0] <= 1e-12] = 0.0
    return gamma, selected


def select_target_params(
    base_scores: np.ndarray,
    sem_scores: np.ndarray,
    transferred_parts: list[np.ndarray],
    gamma: np.ndarray,
    tail: np.ndarray,
    transfer_conf: np.ndarray,
    pseudo_valid: dict[int, list[int]],
    lambda_candidates: list[float],
    alpha_candidates: list[float],
    confidence_power: float,
    fusion: str,
    rrf_k: float,
    top_l: int,
    protect_top: int,
    k: int,
    tie_tol: float,
) -> dict:
    best: dict | None = None
    for lambda_max, alpha_max in product(lambda_candidates, alpha_candidates):
        final_scores, lambda_bt, alpha_bt = fuse_target_adaptive(
            base_scores=base_scores,
            sem_scores=sem_scores,
            transferred_parts=transferred_parts,
            gamma=gamma,
            tail=tail,
            transfer_conf=transfer_conf,
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            confidence_power=confidence_power,
            fusion=fusion,
            rrf_k=rrf_k,
            top_l=top_l,
            protect_top=protect_top,
        )
        valid = official_metrics(final_scores, pseudo_valid, k=k) if pseudo_valid else {"recall": 0.0, "ndcg": 0.0}
        valid_score = float(valid["recall"] + valid["ndcg"])
        complexity = float(lambda_max + alpha_max)
        if best is None or valid_score > best["valid_score"] + tie_tol:
            best = {
                "valid_score": valid_score,
                "complexity": complexity,
                "lambda_max": float(lambda_max),
                "alpha_max": float(alpha_max),
                "valid": valid,
                "final_scores": final_scores,
                "lambda_bt": lambda_bt,
                "alpha_bt": alpha_bt,
            }
        elif abs(valid_score - best["valid_score"]) <= tie_tol and complexity < best["complexity"]:
            best.update(
                {
                    "complexity": complexity,
                    "lambda_max": float(lambda_max),
                    "alpha_max": float(alpha_max),
                    "valid": valid,
                    "final_scores": final_scores,
                    "lambda_bt": lambda_bt,
                    "alpha_bt": alpha_bt,
                }
            )
    assert best is not None
    return best


def fuse_target_adaptive(
    base_scores: np.ndarray,
    sem_scores: np.ndarray,
    transferred_parts: list[np.ndarray],
    gamma: np.ndarray,
    tail: np.ndarray,
    transfer_conf: np.ndarray,
    lambda_max: float,
    alpha_max: float,
    confidence_power: float,
    fusion: str,
    rrf_k: float,
    top_l: int,
    protect_top: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambda_bt = np.clip(
        float(lambda_max) * tail * np.power(np.clip(transfer_conf, 0.0, 1.0), float(confidence_power)),
        0.0,
        float(lambda_max),
    )
    alpha_bt = np.clip(float(alpha_max) * tail, 0.0, float(alpha_max))

    if fusion == "rrf":
        local_component = reciprocal_rank_scores(base_scores, k_const=rrf_k)
        sem_component = reciprocal_rank_scores(sem_scores, k_const=rrf_k)
        transfer_components = [reciprocal_rank_scores(part, k_const=rrf_k) for part in transferred_parts]
    else:
        local_component = zscore_rows(base_scores)
        sem_component = zscore_rows(sem_scores)
        transfer_components = [zscore_rows(part) for part in transferred_parts]

    transfer_mix = weighted_mix(transfer_components, gamma)
    fused = local_component + alpha_bt[:, None] * sem_component + lambda_bt[:, None] * transfer_mix
    return apply_boundary_rerank(base_scores, fused, top_l=top_l, protect_top=protect_top), lambda_bt, alpha_bt


def apply_boundary_rerank(base_scores: np.ndarray, fused_scores: np.ndarray, top_l: int, protect_top: int) -> np.ndarray:
    num_regions = base_scores.shape[1]
    if top_l <= 0 and protect_top <= 0:
        return fused_scores
    local_order = np.argsort(-base_scores, axis=1)
    limit = num_regions if top_l <= 0 else min(max(20, int(top_l)), num_regions)
    protect = min(max(0, int(protect_top)), limit)
    final = np.full_like(fused_scores, -np.inf, dtype=np.float64)
    for brand in range(base_scores.shape[0]):
        protected = local_order[brand, :protect]
        candidates = local_order[brand, protect:limit]
        final[brand, candidates] = fused_scores[brand, candidates]
        if protect > 0:
            final[brand, protected] = 1e6 - np.arange(protect, dtype=np.float64)
    return final


def delta_metrics(current: dict, previous: dict) -> dict[str, float]:
    if not previous:
        return {"recall": np.nan, "ndcg": np.nan}
    return {
        "recall": float(current["recall"] - previous.get("recall", np.nan)),
        "ndcg": float(current["ndcg"] - previous.get("ndcg", np.nan)),
    }


def stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def write_summary(out_dir: Path, results: dict[str, dict]) -> None:
    rows = []
    for city, row in results.items():
        rows.append(
            {
                "city": city,
                "base_recall": row["base"]["recall"],
                "base_ndcg": row["base"]["ndcg"],
                "adaptive_recall": float(row.get("adaptive", {}).get("recall", np.nan)),
                "adaptive_ndcg": float(row.get("adaptive", {}).get("ndcg", np.nan)),
                "tail_source_recall": float(row.get("tail_source", {}).get("recall", np.nan)),
                "tail_source_ndcg": float(row.get("tail_source", {}).get("ndcg", np.nan)),
                "target_adaptive_recall": row["target_adaptive"]["recall"],
                "target_adaptive_ndcg": row["target_adaptive"]["ndcg"],
                "paper_recall": row["paper_mf"]["recall"],
                "paper_ndcg": row["paper_mf"]["ndcg"],
                "lambda_max": row["selected"]["lambda_max"],
                "alpha_max": row["selected"]["alpha_max"],
                "lambda_mean": row["lambda"]["mean"],
                "alpha_mean": row["alpha"]["mean"],
            }
        )
    rows.append(
        {
            "city": "Average",
            "base_recall": float(np.mean([r["base_recall"] for r in rows])),
            "base_ndcg": float(np.mean([r["base_ndcg"] for r in rows])),
            "adaptive_recall": float(np.nanmean([r["adaptive_recall"] for r in rows])),
            "adaptive_ndcg": float(np.nanmean([r["adaptive_ndcg"] for r in rows])),
            "tail_source_recall": float(np.nanmean([r["tail_source_recall"] for r in rows])),
            "tail_source_ndcg": float(np.nanmean([r["tail_source_ndcg"] for r in rows])),
            "target_adaptive_recall": float(np.mean([r["target_adaptive_recall"] for r in rows])),
            "target_adaptive_ndcg": float(np.mean([r["target_adaptive_ndcg"] for r in rows])),
            "paper_recall": float(np.mean([r["paper_recall"] for r in rows])),
            "paper_ndcg": float(np.mean([r["paper_ndcg"] for r in rows])),
            "lambda_max": "",
            "alpha_max": "",
            "lambda_mean": float(np.mean([r["lambda_mean"] for r in rows])),
            "alpha_mean": float(np.mean([r["alpha_mean"] for r in rows])),
        }
    )

    with (out_dir / "target_adaptive_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = write_svg_plot(out_dir, rows)

    lines = [
        "# Target adaptive Target-adaptive Tail-aware Rerank Results",
        "",
        "| City | MF Base R@20 | Adaptive transfer R@20 | Tail-source RRF R@20 | Target adaptive R@20 | Delta vs Base R | Delta vs Tail-source RRF R | Paper MF R@20 | MF Base nDCG@20 | Adaptive transfer nDCG@20 | Tail-source RRF nDCG@20 | Target adaptive nDCG@20 | Delta vs Base nDCG | Delta vs Tail-source RRF nDCG | Paper MF nDCG@20 | lambda_max | alpha_max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['city']} | {fmt(r['base_recall'])} | {fmt(r['adaptive_recall'])} | "
            f"{fmt(r['tail_source_recall'])} | {fmt(r['target_adaptive_recall'])} | "
            f"{fmt(r['target_adaptive_recall'] - r['base_recall'])} | "
            f"{fmt(r['target_adaptive_recall'] - r['tail_source_recall'])} | {fmt(r['paper_recall'])} | "
            f"{fmt(r['base_ndcg'])} | {fmt(r['adaptive_ndcg'])} | {fmt(r['tail_source_ndcg'])} | "
            f"{fmt(r['target_adaptive_ndcg'])} | {fmt(r['target_adaptive_ndcg'] - r['base_ndcg'])} | "
            f"{fmt(r['target_adaptive_ndcg'] - r['tail_source_ndcg'])} | {fmt(r['paper_ndcg'])} | "
            f"{r['lambda_max']} | {r['alpha_max']} |"
        )
    lines.extend(["", f"![Target adaptive comparison]({plot_path.name})", "", "## Source Selection", ""])
    lines.append("| Target | Source | Selected | Train-only proxy gain | Region rel | Brand rel mean |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for target, row in results.items():
        for src in row["sources"]:
            lines.append(
                f"| {target} | {src['source']} | {src['selected']} | "
                f"{src['train_only_proxy_gain']:.6f} | {src['region_reliability']:.4f} | "
                f"{src['brand_reliability_mean']:.4f} |"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_svg_plot(out_dir: Path, rows: list[dict]) -> Path:
    labels = [row["city"] for row in rows]
    series = [
        ("Paper MF", "paper", "#7a869a"),
        ("Reproduced MF", "base", "#2f6fed"),
        ("Tail-source RRF", "tail_source", "#23a455"),
        ("Target adaptive", "target_adaptive", "#e06a2c"),
    ]
    width, height = 1220, 440
    panel_w, panel_h = 540, 300
    panel_y = 70
    panel_xs = [55, 650]
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#222}.title{font-size:18px;font-weight:700}.axis{stroke:#c9d1dc;stroke-width:1}.grid{stroke:#e8edf3;stroke-width:1}.legend{font-size:12px}</style>',
    ]
    for panel_x, metric, title in zip(panel_xs, ["recall", "ndcg"], ["Recall@20", "nDCG@20"]):
        values = []
        for _, key, _ in series:
            values.extend(float(row[f"{key}_{metric}"]) for row in rows if np.isfinite(row[f"{key}_{metric}"]))
        ymax = max(values) * 1.14 if values else 1.0
        svg.append(f'<text class="title" x="{panel_x}" y="34">{title}</text>')
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = panel_y + panel_h - frac * panel_h
            val = ymax * frac
            svg.append(f'<line class="grid" x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_w}" y2="{y:.1f}"/>')
            svg.append(f'<text x="{panel_x - 8}" y="{y + 4:.1f}" text-anchor="end">{val:.2f}</text>')
        svg.append(f'<line class="axis" x1="{panel_x}" y1="{panel_y + panel_h}" x2="{panel_x + panel_w}" y2="{panel_y + panel_h}"/>')
        group_w = panel_w / len(labels)
        bar_w = group_w / 6.2
        for i, label in enumerate(labels):
            center = panel_x + group_w * (i + 0.5)
            for j, (_, key, color) in enumerate(series):
                val = float(rows[i][f"{key}_{metric}"])
                if not np.isfinite(val):
                    continue
                h = max(0.0, val / ymax * panel_h)
                x = center + (j - 1.5) * bar_w
                y = panel_y + panel_h - h
                svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.88:.1f}" height="{h:.1f}" fill="{color}"/>')
            svg.append(
                f'<text x="{center:.1f}" y="{panel_y + panel_h + 25}" text-anchor="middle" transform="rotate(25 {center:.1f} {panel_y + panel_h + 25})">{label}</text>'
            )
    legend_x, legend_y = 375, 415
    for i, (name, _, color) in enumerate(series):
        x = legend_x + i * 150
        svg.append(f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text class="legend" x="{x + 20}" y="{legend_y}">{name}</text>')
    svg.append("</svg>")
    path = out_dir / "target_adaptive_comparison.svg"
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
