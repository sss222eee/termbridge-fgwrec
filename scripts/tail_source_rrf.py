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
from adaptive_transfer import (  # noqa: E402
    PAPER_MF,
    get_region_transport,
    load_test_lists,
    official_metrics,
    topq_term_similarity,
    zscore_rows,
)
from run_termbridge_fgw import row_stochastic, termbridge_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Scheme-7 Tail-aware Source-selective TermBridge-FGWRec on saved official MF scores."
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
    parser.add_argument("--lambda-max", type=float, default=0.30)
    parser.add_argument("--tail-mode", choices=["tau", "normalized"], default="tau")
    parser.add_argument("--tail-tau", type=float, default=1.0)
    parser.add_argument("--tail-power", type=float, default=1.0)
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument("--source-gain-mode", choices=["hard", "soft", "none"], default="hard")
    parser.add_argument("--source-gain-scale", type=float, default=10.0)
    parser.add_argument("--min-source-gain", type=float, default=0.0)
    parser.add_argument("--source-gain-floor", type=float, default=1e-4)
    parser.add_argument("--disable-train-gain", action="store_true")
    parser.add_argument("--no-fallback-top-source", action="store_true")
    parser.add_argument("--inner-valid-ratio", type=float, default=0.2)
    parser.add_argument("--inner-valid-seed", type=int, default=2026)
    parser.add_argument("--inner-valid-min-items", type=int, default=4)
    parser.add_argument("--probe-lambda", type=float, default=0.05)
    parser.add_argument("--fusion", choices=["rrf", "zscore"], default="rrf")
    parser.add_argument("--zscore-style", choices=["convex", "additive"], default="convex")
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--top-l", type=int, default=200)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--adaptive-json", default="outputs/adaptive_official_adaptive_b005_p3/adaptive_results.json")
    parser.add_argument("--out", default="outputs/tail_source_tail_source_rrf")
    args = parser.parse_args()
    if args.disable_train_gain:
        args.source_gain_mode = "none"

    cities = available_cities(args.data_root)
    targets = cities if args.target == "all" else [c.strip() for c in args.target.split(",")]
    city_data = {city: load_city(args.data_root, city) for city in cities}
    scores = {city: np.load(Path(args.score_root) / city / "scores.npy") for city in cities}
    brand_terms = {city: np.load(Path(args.feature_root) / city / "brand_terms.npy") for city in cities}
    region_terms = {city: np.load(Path(args.feature_root) / city / "region_terms.npy") for city in cities}
    test_lists = {city: load_test_lists(Path(args.data_root) / city / "official_test_lists.json") for city in cities}
    adaptive = load_adaptive(args.adaptive_json)

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
            raw_gain = 0.0 if args.source_gain_mode == "none" else source_proxy_gain(
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
            source_gains.append(raw_gain)
            source_details.append(
                {
                    "source": source,
                    "region_solver": region_tr.solver,
                    "region_distance": float(region_tr.distance),
                    "region_confidence": float(region_tr.confidence),
                    "region_reliability": region_rel,
                    "brand_reliability_mean": float(np.mean(brand_rel)),
                    "brand_reliability_std": float(np.std(brand_rel)),
                    "train_only_proxy_gain": raw_gain,
                    "selected_by_gain": bool(args.source_gain_mode != "hard" or raw_gain > args.min_source_gain),
                }
            )

        gamma, selected_mask = source_weights(
            structural_rel_parts=structural_rel_parts,
            source_gains=source_gains,
            min_source_gain=args.min_source_gain,
            gain_floor=args.source_gain_floor,
            source_gain_mode=args.source_gain_mode,
            source_gain_scale=args.source_gain_scale,
            fallback_top_source=not args.no_fallback_top_source,
        )
        for idx, selected in enumerate(selected_mask):
            source_details[idx]["selected_after_fallback"] = bool(selected)

        brand_rel_stack = np.stack(brand_rel_parts, axis=0)
        transfer_conf = np.sum(gamma * brand_rel_stack, axis=0)
        lambda_bt = np.clip(
            args.lambda_max * tail * np.power(np.clip(transfer_conf, 0.0, 1.0), args.confidence_power),
            0.0,
            args.lambda_max,
        )

        final_scores, transfer_mix = fuse_scores(
            base_scores=base_scores,
            transferred_parts=transferred_parts,
            gamma=gamma,
            lambda_bt=lambda_bt,
            fusion=args.fusion,
            zscore_style=args.zscore_style,
            rrf_k=args.rrf_k,
            top_l=args.top_l,
        )
        final_metrics = official_metrics(final_scores, test_lists[target], k=args.k)

        out_city = Path(args.out) / target
        out_city.mkdir(parents=True, exist_ok=True)
        np.save(out_city / "final_scores.npy", final_scores)
        np.save(out_city / "lambda_bt.npy", lambda_bt)
        np.save(out_city / "source_gamma.npy", gamma)
        np.save(out_city / "transfer_mix.npy", transfer_mix)

        paper = PAPER_MF.get(target, {"recall": np.nan, "ndcg": np.nan})
        adaptive_city = adaptive.get(target, {})
        results[target] = {
            "method": "Tail-source RRF-Tail-aware-Source-selective-TermBridge-FGWRec",
            "target": target,
            "base": base_metrics,
            "tail_source": final_metrics,
            "adaptive": adaptive_city.get("adaptive", {}),
            "paper_mf": paper,
            "delta_vs_base": {
                "recall": final_metrics["recall"] - base_metrics["recall"],
                "ndcg": final_metrics["ndcg"] - base_metrics["ndcg"],
            },
            "delta_vs_adaptive": delta_vs_adaptive(final_metrics, adaptive_city.get("adaptive", {})),
            "delta_vs_paper_mf": {
                "recall": final_metrics["recall"] - paper["recall"],
                "ndcg": final_metrics["ndcg"] - paper["ndcg"],
            },
            "lambda": {
                "min": float(np.min(lambda_bt)),
                "max": float(np.max(lambda_bt)),
                "mean": float(np.mean(lambda_bt)),
                "std": float(np.std(lambda_bt)),
            },
            "tail": {
                "min": float(np.min(tail)),
                "max": float(np.max(tail)),
                "mean": float(np.mean(tail)),
                "std": float(np.std(tail)),
            },
            "transfer_confidence": {
                "min": float(np.min(transfer_conf)),
                "max": float(np.max(transfer_conf)),
                "mean": float(np.mean(transfer_conf)),
                "std": float(np.std(transfer_conf)),
            },
            "pseudo_valid_brands": len(pseudo_valid),
            "sources": source_details,
            "config": vars(args),
        }
        print(
            f"{target}: base {base_metrics['recall']:.4f}/{base_metrics['ndcg']:.4f} | "
            f"tail_source {final_metrics['recall']:.4f}/{final_metrics['ndcg']:.4f} | "
            f"delta {final_metrics['recall'] - base_metrics['recall']:+.4f}/"
            f"{final_metrics['ndcg'] - base_metrics['ndcg']:+.4f}",
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tail_source_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, results)
    print(f"Saved Tail-source RRF results to {out_dir}", flush=True)


def load_adaptive(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def make_pseudo_valid_lists(
    train_edges: np.ndarray,
    ratio: float,
    seed: int,
    min_items: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    by_brand: dict[int, list[int]] = {}
    for brand, region in np.asarray(train_edges, dtype=np.int64).reshape(-1, 2):
        by_brand.setdefault(int(brand), []).append(int(region))

    pseudo: dict[int, list[int]] = {}
    for brand, regions in by_brand.items():
        if len(regions) < max(2, int(min_items)):
            continue
        idx = np.arange(len(regions))
        rng.shuffle(idx)
        n_valid = int(np.floor(len(regions) * float(ratio)))
        n_valid = min(max(1, n_valid), len(regions) - 1)
        pseudo[int(brand)] = [int(regions[i]) for i in idx[:n_valid]]
    return pseudo


def source_proxy_gain(
    base_scores: np.ndarray,
    transferred_scores: np.ndarray,
    pseudo_valid: dict[int, list[int]],
    k: int,
    probe_lambda: float,
    rrf_k: float,
) -> float:
    if not pseudo_valid:
        return 0.0
    base = official_metrics(base_scores, pseudo_valid, k=k)
    probe_scores = reciprocal_rank_scores(base_scores, k_const=rrf_k) + float(probe_lambda) * reciprocal_rank_scores(
        transferred_scores, k_const=rrf_k
    )
    probe = official_metrics(probe_scores, pseudo_valid, k=k)
    return float((probe["recall"] + probe["ndcg"]) - (base["recall"] + base["ndcg"]))


def source_weights(
    structural_rel_parts: list[np.ndarray],
    source_gains: list[float],
    min_source_gain: float,
    gain_floor: float,
    source_gain_mode: str,
    source_gain_scale: float,
    fallback_top_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    structural = np.stack(structural_rel_parts, axis=0)
    gains = np.asarray(source_gains, dtype=np.float64)
    if source_gain_mode == "none":
        selected = np.ones(structural.shape[0], dtype=bool)
        gain_factor = np.ones_like(gains)
    elif source_gain_mode == "soft":
        selected = np.ones(structural.shape[0], dtype=bool)
        gain_factor = 1.0 + float(source_gain_scale) * np.maximum(gains, 0.0)
    else:
        selected = gains > float(min_source_gain)
        gain_factor = np.maximum(gains, float(gain_floor))

    rel = structural * gain_factor[:, None]
    rel[~selected, :] = 0.0

    if fallback_top_source and not np.any(selected):
        fallback = int(np.argmax(np.mean(structural, axis=1)))
        selected[fallback] = True
        rel[fallback, :] = structural[fallback, :] * max(float(gain_floor), 1e-8)

    col_sum = np.sum(rel, axis=0, keepdims=True)
    gamma = rel / np.maximum(col_sum, 1e-12)
    gamma[:, col_sum[0] <= 1e-12] = 0.0
    return gamma, selected


def tail_strength(counts: np.ndarray, tau: float, power: float, mode: str) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if mode == "normalized":
        max_count = float(np.max(counts)) if counts.size and np.max(counts) > 0 else 1.0
        dense = np.log1p(np.maximum(counts, 0.0)) / np.log1p(max_count)
        base = 1.0 - dense
    else:
        base = float(tau) / (np.log1p(np.maximum(counts, 0.0)) + float(tau))
    return np.power(np.clip(base, 0.0, 1.0), float(power))


def sparsify_rows(matrix: np.ndarray, topk: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if topk <= 0 or topk >= arr.shape[1]:
        return arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    k = max(1, int(topk))
    out = np.zeros_like(arr)
    idx = np.argpartition(-arr, k - 1, axis=1)[:, :k]
    rows = np.arange(arr.shape[0])[:, None]
    out[rows, idx] = arr[rows, idx]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def reciprocal_rank_scores(scores: np.ndarray, k_const: float = 60.0) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rank_values = np.arange(1, arr.shape[1] + 1, dtype=np.float64)
    rows = np.arange(arr.shape[0])[:, None]
    ranks[rows, order] = rank_values[None, :]
    return 1.0 / (float(k_const) + ranks)


def fuse_scores(
    base_scores: np.ndarray,
    transferred_parts: list[np.ndarray],
    gamma: np.ndarray,
    lambda_bt: np.ndarray,
    fusion: str,
    zscore_style: str,
    rrf_k: float,
    top_l: int,
) -> tuple[np.ndarray, np.ndarray]:
    if fusion == "rrf":
        base_component = reciprocal_rank_scores(base_scores, k_const=rrf_k)
        transfer_components = [reciprocal_rank_scores(part, k_const=rrf_k) for part in transferred_parts]
        transfer_mix = weighted_mix(transfer_components, gamma)
        fused = base_component + lambda_bt[:, None] * transfer_mix
    else:
        base_component = zscore_rows(base_scores)
        transfer_components = [zscore_rows(part) for part in transferred_parts]
        transfer_mix = weighted_mix(transfer_components, gamma)
        if zscore_style == "additive":
            fused = base_component + lambda_bt[:, None] * transfer_mix
        else:
            fused = (1.0 - lambda_bt[:, None]) * base_component + lambda_bt[:, None] * transfer_mix
    return apply_top_l_rerank(base_scores, fused, top_l=top_l), transfer_mix


def weighted_mix(parts: list[np.ndarray], gamma: np.ndarray) -> np.ndarray:
    out = np.zeros_like(parts[0], dtype=np.float64)
    for idx, part in enumerate(parts):
        out += gamma[idx, :, None] * part
    return out


def apply_top_l_rerank(base_scores: np.ndarray, fused_scores: np.ndarray, top_l: int) -> np.ndarray:
    if top_l <= 0 or top_l >= base_scores.shape[1]:
        return fused_scores
    l_value = max(20, min(int(top_l), base_scores.shape[1]))
    final = np.full_like(fused_scores, -np.inf, dtype=np.float64)
    idx = np.argpartition(-base_scores, l_value - 1, axis=1)[:, :l_value]
    rows = np.arange(base_scores.shape[0])[:, None]
    final[rows, idx] = fused_scores[rows, idx]
    return final


def delta_vs_adaptive(final_metrics: dict, adaptive_metrics: dict) -> dict[str, float]:
    if not adaptive_metrics:
        return {"recall": np.nan, "ndcg": np.nan}
    return {
        "recall": float(final_metrics["recall"] - adaptive_metrics.get("recall", np.nan)),
        "ndcg": float(final_metrics["ndcg"] - adaptive_metrics.get("ndcg", np.nan)),
    }


def write_summary(out_dir: Path, results: dict[str, dict]) -> None:
    rows = []
    for city, row in results.items():
        adaptive = row.get("adaptive", {})
        rows.append(
            {
                "city": city,
                "base_recall": row["base"]["recall"],
                "base_ndcg": row["base"]["ndcg"],
                "adaptive_recall": float(adaptive.get("recall", np.nan)),
                "adaptive_ndcg": float(adaptive.get("ndcg", np.nan)),
                "tail_source_recall": row["tail_source"]["recall"],
                "tail_source_ndcg": row["tail_source"]["ndcg"],
                "paper_recall": row["paper_mf"]["recall"],
                "paper_ndcg": row["paper_mf"]["ndcg"],
                "lambda_mean": row["lambda"]["mean"],
                "lambda_max": row["lambda"]["max"],
            }
        )
    rows.append(
        {
            "city": "Average",
            "base_recall": float(np.mean([r["base_recall"] for r in rows])),
            "base_ndcg": float(np.mean([r["base_ndcg"] for r in rows])),
            "adaptive_recall": float(np.nanmean([r["adaptive_recall"] for r in rows])),
            "adaptive_ndcg": float(np.nanmean([r["adaptive_ndcg"] for r in rows])),
            "tail_source_recall": float(np.mean([r["tail_source_recall"] for r in rows])),
            "tail_source_ndcg": float(np.mean([r["tail_source_ndcg"] for r in rows])),
            "paper_recall": float(np.mean([r["paper_recall"] for r in rows])),
            "paper_ndcg": float(np.mean([r["paper_ndcg"] for r in rows])),
            "lambda_mean": float(np.mean([r["lambda_mean"] for r in rows])),
            "lambda_max": float(np.max([r["lambda_max"] for r in rows])),
        }
    )

    with (out_dir / "tail_source_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = write_plot(out_dir, rows)

    lines = [
        "# Tail-source RRF Tail-aware Source-selective Results",
        "",
        "| City | MF Base R@20 | Adaptive transfer R@20 | Tail-source RRF R@20 | Delta vs Base R | Delta vs Adaptive transfer R | Paper MF R@20 | MF Base nDCG@20 | Adaptive transfer nDCG@20 | Tail-source RRF nDCG@20 | Delta vs Base nDCG | Delta vs Adaptive transfer nDCG | Paper MF nDCG@20 | Mean lambda |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['city']} | {fmt(r['base_recall'])} | {fmt(r['adaptive_recall'])} | "
            f"{fmt(r['tail_source_recall'])} | {fmt(r['tail_source_recall'] - r['base_recall'])} | "
            f"{fmt(r['tail_source_recall'] - r['adaptive_recall'])} | {fmt(r['paper_recall'])} | "
            f"{fmt(r['base_ndcg'])} | {fmt(r['adaptive_ndcg'])} | {fmt(r['tail_source_ndcg'])} | "
            f"{fmt(r['tail_source_ndcg'] - r['base_ndcg'])} | {fmt(r['tail_source_ndcg'] - r['adaptive_ndcg'])} | "
            f"{fmt(r['paper_ndcg'])} | {fmt(r['lambda_mean'])} |"
        )
    lines.extend(["", f"![Tail-source RRF comparison]({plot_path.name})", "", "## Source Selection", ""])
    lines.append("| Target | Source | Selected | Train-only proxy gain | Region rel | Brand rel mean |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for target, row in results.items():
        for src in row["sources"]:
            lines.append(
                f"| {target} | {src['source']} | {src['selected_after_fallback']} | "
                f"{src['train_only_proxy_gain']:.6f} | {src['region_reliability']:.4f} | "
                f"{src['brand_reliability_mean']:.4f} |"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_plot(out_dir: Path, rows: list[dict]) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return write_svg_plot(out_dir, rows)

    labels = [row["city"] for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=160)
    for ax, metric, title in [
        (axes[0], "recall", "Recall@20"),
        (axes[1], "ndcg", "nDCG@20"),
    ]:
        base = [row[f"base_{metric}"] for row in rows]
        adaptive = [row[f"adaptive_{metric}"] for row in rows]
        tail_source = [row[f"tail_source_{metric}"] for row in rows]
        paper = [row[f"paper_{metric}"] for row in rows]
        ax.bar(x - width * 1.5, paper, width, label="Paper MF")
        ax.bar(x - width * 0.5, base, width, label="Reproduced MF")
        ax.bar(x + width * 0.5, adaptive, width, label="Adaptive transfer")
        ax.bar(x + width * 1.5, tail_source, width, label="Tail-source RRF")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        finite_vals = [v for v in paper + base + adaptive + tail_source if np.isfinite(v)]
        ax.set_ylim(0, max(finite_vals) * 1.16 if finite_vals else 1)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out_dir / "tail_source_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_svg_plot(out_dir: Path, rows: list[dict]) -> Path:
    labels = [row["city"] for row in rows]
    series = [
        ("Paper MF", "paper", "#7a869a"),
        ("Reproduced MF", "base", "#2f6fed"),
        ("Adaptive transfer", "adaptive", "#23a455"),
        ("Tail-source RRF", "tail_source", "#e06a2c"),
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
    path = out_dir / "tail_source_comparison.svg"
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
