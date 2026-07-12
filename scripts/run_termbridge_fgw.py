from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.evaluate import all_ranking_metrics, grouped_metrics  # noqa: E402
from otc.otc import (  # noqa: E402
    TransportResult,
    balance_transport_plan,
    normalize_cost,
    pairwise_euclidean,
    sinkhorn_uniform,
    transport_confidence,
    transport_marginal_error,
)
from otc.semantic import normalize_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Scheme-4 TermBridge-FGWRec inference.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--run-dir", required=True, help="Directory containing trained TermStruct city artifacts.")
    parser.add_argument("--target", default="all")
    parser.add_argument("--baseline-json", default="outputs/otc_mf_doc_adjusted_seed2024_default_emd/otc_results.json")
    parser.add_argument("--gamma-candidates", default="0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--per-source-gamma", action="store_true")
    parser.add_argument("--brand-topq", type=int, default=10)
    parser.add_argument("--brand-tau", type=float, default=0.1)
    parser.add_argument("--fgw-alpha", type=float, default=0.5)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--region-solver", choices=["pot", "signature"], default="pot")
    parser.add_argument("--ot-max-iter", type=int, default=200)
    parser.add_argument("--ot-emd-max-iter", type=int, default=100_000)
    parser.add_argument("--ot-tol", type=float, default=1e-9)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--no-mask-train", action="store_true")
    parser.add_argument("--drop-eval-train-overlap", action="store_true")
    parser.add_argument("--standardize-transfer", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--out", default="outputs/termbridge_fgw")
    args = parser.parse_args()

    city_names = available_cities(args.data_root)
    targets = city_names if args.target == "all" else [c.strip() for c in args.target.split(",")]
    gammas = [float(x) for x in args.gamma_candidates.split(",") if x.strip()]

    city_data = {name: load_city(args.data_root, name) for name in city_names}
    scores = {name: load_scores(args.run_dir, name) for name in city_names}
    brand_terms = {name: np.load(Path(args.feature_root) / name / "brand_terms.npy") for name in city_names}
    region_terms = {name: np.load(Path(args.feature_root) / name / "region_terms.npy") for name in city_names}
    region_cache: dict[tuple[str, str], TransportResult] = {}

    results: dict[str, dict] = {}
    for target in targets:
        target_data = city_data[target]
        base_scores = scores[target]
        eval_train_pos = {} if args.no_mask_train else target_data.train_pos
        base_valid = all_ranking_metrics(
            base_scores,
            eval_train_pos,
            target_data.valid_pos,
            k=args.k,
            drop_train_overlap=args.drop_eval_train_overlap,
        )
        base_test = all_ranking_metrics(
            base_scores,
            eval_train_pos,
            target_data.test_pos,
            k=args.k,
            drop_train_overlap=args.drop_eval_train_overlap,
        )

        transferred_scores: list[np.ndarray] = []
        details: list[dict] = []
        for source in city_names:
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
                max_iter=args.ot_max_iter,
                emd_max_iter=args.ot_emd_max_iter,
                tol=args.ot_tol,
            )
            region_mapping = row_stochastic(region_tr.plan)
            transferred = brand_bridge @ scores[source] @ region_mapping
            if args.standardize_transfer:
                transferred = match_score_scale(transferred, base_scores)
            transferred_scores.append(transferred)

            detail = {
                "source": source,
                "region_solver": region_tr.solver,
                "region_distance": region_tr.distance,
                "region_confidence": region_tr.confidence,
                "brand_topq": args.brand_topq,
                "brand_tau": args.brand_tau,
            }
            if args.diagnostics:
                detail["brand_bridge_stats"] = bridge_stats(brand_bridge)
                detail["region_plan_stats"] = plan_stats(region_tr.plan)
                detail["transfer_score_stats"] = score_stats(transferred)
            details.append(detail)

        fusion = search_fusion(
            base_scores=base_scores,
            source_scores=transferred_scores,
            gammas=gammas,
            per_source_gamma=args.per_source_gamma,
            eval_train_pos=eval_train_pos,
            target_data=target_data,
            k=args.k,
            drop_eval_train_overlap=args.drop_eval_train_overlap,
        )
        best_gamma = fusion["best_gamma"]
        final_scores = fusion["final_scores"]
        groups = grouped_metrics(
            final_scores,
            eval_train_pos,
            target_data.test_pos,
            target_data.train_counts,
            k=args.k,
            drop_train_overlap=args.drop_eval_train_overlap,
        )

        for idx, detail in enumerate(details):
            if args.per_source_gamma and isinstance(best_gamma, list):
                detail["gamma"] = float(best_gamma[idx])
            elif not args.per_source_gamma:
                detail["gamma"] = float(best_gamma)

        results[target] = {
            "method": "TermBridge-FGWRec",
            "data_root": args.data_root,
            "feature_root": args.feature_root,
            "run_dir": args.run_dir,
            "per_source_gamma": args.per_source_gamma,
            "gamma_candidates": gammas,
            "brand_topq": args.brand_topq,
            "brand_tau": args.brand_tau,
            "fgw_alpha": args.fgw_alpha,
            "reg": args.reg,
            "region_solver": args.region_solver,
            "ot_max_iter": args.ot_max_iter,
            "ot_emd_max_iter": args.ot_emd_max_iter,
            "ot_tol": args.ot_tol,
            "standardize_transfer": args.standardize_transfer,
            "no_mask_train": args.no_mask_train,
            "drop_eval_train_overlap": args.drop_eval_train_overlap,
            "best_gamma": best_gamma,
            "base_valid": base_valid.as_dict(),
            "base_test": base_test.as_dict(),
            "termbridge_valid": fusion["valid"].as_dict(),
            "termbridge_test": fusion["test"].as_dict(),
            "valid_selection_score": fusion["valid_selection_score"],
            "groups": groups,
            "sources": details,
        }
        if args.diagnostics:
            results[target]["score_stats"] = {
                "base": score_stats(base_scores),
                "final": score_stats(final_scores),
            }

        gamma_label = ",".join(f"{g:g}" for g in best_gamma) if isinstance(best_gamma, list) else f"{best_gamma:g}"
        print(
            f"{target}: TermStruct base R@{args.k}={base_test.recall:.4f} nDCG@{args.k}={base_test.ndcg:.4f} | "
            f"TermBridge-FGW gamma={gamma_label} R@{args.k}={fusion['test'].recall:.4f} "
            f"nDCG@{args.k}={fusion['test'].ndcg:.4f}",
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "termbridge_fgw_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, results, args)
    print(f"Saved results to {result_path}", flush=True)


def load_scores(run_dir: str | Path, city: str) -> np.ndarray:
    city_dir = Path(run_dir) / city
    return np.load(city_dir / "scores.npy")


def termbridge_weights(
    target_terms: np.ndarray,
    source_terms: np.ndarray,
    topq: int = 10,
    tau: float = 0.1,
) -> np.ndarray:
    target = normalize_rows(np.asarray(target_terms, dtype=np.float64))
    source = normalize_rows(np.asarray(source_terms, dtype=np.float64))
    sim = target @ source.T
    q = min(max(1, int(topq)), source.shape[0])
    weights = np.zeros_like(sim, dtype=np.float64)
    for i in range(sim.shape[0]):
        idx = np.argpartition(-sim[i], q - 1)[:q]
        logits = sim[i, idx] / max(float(tau), 1e-8)
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        weights[i, idx] = exp / np.maximum(float(np.sum(exp)), 1e-12)
    return weights


def get_region_transport(
    cache: dict[tuple[str, str], TransportResult],
    source: str,
    target: str,
    source_terms: np.ndarray,
    target_terms: np.ndarray,
    alpha: float,
    reg: float,
    solver: str,
    max_iter: int,
    emd_max_iter: int,
    tol: float,
) -> TransportResult:
    key = (source, target)
    if key in cache:
        return cache[key]
    reverse_key = (target, source)
    if reverse_key in cache:
        reverse = cache[reverse_key]
        result = TransportResult(
            plan=reverse.plan.T.copy(),
            distance=reverse.distance,
            confidence=reverse.confidence,
            solver=f"{reverse.solver}-transpose-cache",
        )
        cache[key] = result
        return result

    result = compute_region_fgw(
        source_terms,
        target_terms,
        alpha=alpha,
        reg=reg,
        solver=solver,
        max_iter=max_iter,
        emd_max_iter=emd_max_iter,
        tol=tol,
    )
    cache[key] = result
    return result


def compute_region_fgw(
    source_terms: np.ndarray,
    target_terms: np.ndarray,
    alpha: float = 0.5,
    reg: float = 0.05,
    solver: str = "pot",
    max_iter: int = 200,
    emd_max_iter: int = 100_000,
    tol: float = 1e-9,
) -> TransportResult:
    source_structure = term_cosine_cost(source_terms, source_terms, diagonal_zero=True)
    target_structure = term_cosine_cost(target_terms, target_terms, diagonal_zero=True)
    feature_cost = term_cosine_cost(source_terms, target_terms, diagonal_zero=False)
    p = np.full(source_structure.shape[0], 1.0 / source_structure.shape[0], dtype=np.float64)
    q = np.full(target_structure.shape[0], 1.0 / target_structure.shape[0], dtype=np.float64)

    if solver == "pot":
        try:
            import ot  # type: ignore

            plan, log = ot.gromov.fused_gromov_wasserstein(
                feature_cost,
                source_structure,
                target_structure,
                p,
                q,
                loss_fun="square_loss",
                alpha=alpha,
                log=True,
                max_iter=max_iter,
                numItermaxEmd=emd_max_iter,
                tol_rel=tol,
                tol_abs=tol,
            )
            tolerance = max(tol * 10.0, 1e-6)
            plan = np.asarray(plan, dtype=np.float64)
            solver_name = "pot-fgw-term"
            if transport_marginal_error(plan, p, q) > tolerance:
                plan = balance_transport_plan(plan, p, q, max_iter=max(1000, max_iter), tol=tolerance)
                solver_name = "pot-fgw-term-balanced"
            distance = float(log.get("fgw_dist", np.sum(plan * feature_cost)))
            return TransportResult(
                plan=np.maximum(plan, 0.0),
                distance=distance,
                confidence=transport_confidence(plan),
                solver=solver_name,
            )
        except Exception as exc:
            fallback = signature_fgw_transport(source_structure, target_structure, feature_cost, alpha, reg, max_iter)
            fallback.solver = f"{fallback.solver}-fallback:{type(exc).__name__}"
            return fallback

    return signature_fgw_transport(source_structure, target_structure, feature_cost, alpha, reg, max_iter)


def term_cosine_cost(x: np.ndarray, y: np.ndarray, diagonal_zero: bool) -> np.ndarray:
    left = normalize_rows(np.asarray(x, dtype=np.float64))
    right = normalize_rows(np.asarray(y, dtype=np.float64))
    cost = normalize_cost(1.0 - left @ right.T)
    if diagonal_zero and cost.shape[0] == cost.shape[1]:
        np.fill_diagonal(cost, 0.0)
    return cost


def signature_fgw_transport(
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    feature_cost: np.ndarray,
    alpha: float,
    reg: float,
    max_iter: int,
) -> TransportResult:
    source_sig = structure_signature_from_cost(source_structure)
    target_sig = structure_signature_from_cost(target_structure)
    structure_cost = normalize_cost(pairwise_euclidean(source_sig, target_sig))
    cost = float(alpha) * structure_cost + (1.0 - float(alpha)) * feature_cost
    plan = sinkhorn_uniform(cost, reg=reg, max_iter=max_iter)
    return TransportResult(
        plan=plan,
        distance=float(np.sum(plan * cost)),
        confidence=transport_confidence(plan),
        solver="signature-fgw-sinkhorn",
    )


def structure_signature_from_cost(cost: np.ndarray, bins: int = 16) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 2)[1:-1]
    return np.quantile(cost, qs, axis=1).T


def row_stochastic(plan: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float64)
    return plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-12)


def match_score_scale(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    src_std = float(np.std(src))
    ref_std = float(np.std(ref))
    if src_std < 1e-12 or ref_std < 1e-12:
        return src
    return (src - float(np.mean(src))) / src_std * ref_std + float(np.mean(ref))


def search_fusion(
    base_scores: np.ndarray,
    source_scores: list[np.ndarray],
    gammas: list[float],
    per_source_gamma: bool,
    eval_train_pos: dict[int, set[int]],
    target_data,
    k: int,
    drop_eval_train_overlap: bool,
) -> dict:
    best_gamma: float | list[float] = gammas[0]
    best_valid_score = -1.0
    best_valid = None
    best_final_scores = base_scores.copy()

    gamma_iter = itertools.product(gammas, repeat=len(source_scores)) if per_source_gamma else ((g,) for g in gammas)
    for gamma_combo in gamma_iter:
        final_scores = base_scores.copy()
        if per_source_gamma:
            for gamma, score in zip(gamma_combo, source_scores):
                final_scores += float(gamma) * score
        else:
            transfer = np.zeros_like(base_scores, dtype=np.float64)
            for score in source_scores:
                transfer += score
            final_scores += float(gamma_combo[0]) * transfer

        valid = all_ranking_metrics(
            final_scores,
            eval_train_pos,
            target_data.valid_pos,
            k=k,
            drop_train_overlap=drop_eval_train_overlap,
        )
        valid_score = valid.recall + valid.ndcg
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_gamma = [float(x) for x in gamma_combo] if per_source_gamma else float(gamma_combo[0])
            best_valid = valid
            best_final_scores = final_scores

    test = all_ranking_metrics(
        best_final_scores,
        eval_train_pos,
        target_data.test_pos,
        k=k,
        drop_train_overlap=drop_eval_train_overlap,
    )
    return {
        "best_gamma": best_gamma,
        "valid": best_valid,
        "valid_selection_score": float(best_valid_score),
        "test": test,
        "final_scores": best_final_scores,
    }


def write_summary(out_dir: Path, results: dict[str, dict], args: argparse.Namespace) -> None:
    baseline = load_baseline(args.baseline_json)
    rows = []
    for city, row in results.items():
        otc = baseline.get(city, {})
        otc_test = otc.get("otc_test", {})
        rows.append(
            {
                "city": city,
                "otc_recall": float(otc_test.get("recall", np.nan)),
                "otc_ndcg": float(otc_test.get("ndcg", np.nan)),
                "base_recall": float(row["base_test"]["recall"]),
                "base_ndcg": float(row["base_test"]["ndcg"]),
                "final_recall": float(row["termbridge_test"]["recall"]),
                "final_ndcg": float(row["termbridge_test"]["ndcg"]),
                "gamma": row["best_gamma"],
            }
        )

    avg = {
        "city": "Average",
        "otc_recall": float(np.nanmean([r["otc_recall"] for r in rows])),
        "otc_ndcg": float(np.nanmean([r["otc_ndcg"] for r in rows])),
        "base_recall": float(np.mean([r["base_recall"] for r in rows])),
        "base_ndcg": float(np.mean([r["base_ndcg"] for r in rows])),
        "final_recall": float(np.mean([r["final_recall"] for r in rows])),
        "final_ndcg": float(np.mean([r["final_ndcg"] for r in rows])),
        "gamma": "",
    }
    rows_with_avg = rows + [avg]
    csv_path = out_dir / "termbridge_vs_otc.csv"
    write_csv(csv_path, rows_with_avg)
    plot_path = write_plot(out_dir, rows_with_avg)

    lines = [
        "# Scheme-4 TermBridge-FGWRec Results",
        "",
        "## Protocol",
        "",
        f"- Data: `{args.data_root}`",
        f"- Feature root: `{args.feature_root}`",
        f"- Backbone run: `{args.run_dir}`",
        "- Offline terms: deterministic 23-term proxy; no online LLM call during training or inference.",
        f"- Brand bridge: Top-{args.brand_topq} softmax, tau={args.brand_tau:g}",
        f"- Region transfer: FGW alpha={args.fgw_alpha:g}, solver={args.region_solver}",
        f"- Fusion: {'per-source gamma' if args.per_source_gamma else 'shared gamma'} selected on validation, candidates `{args.gamma_candidates}`",
        "- Evaluation: train positives masked, eval/train overlap dropped from denominator"
        if args.drop_eval_train_overlap
        else "- Evaluation: train positives masked unless `--no-mask-train` is used",
        "",
        "## Comparison With Reproduced OTC-MF",
        "",
        "| City | OTC-MF R@20 | TermStruct local R@20 | TermBridge-FGWRec R@20 | Delta vs OTC R | OTC-MF nDCG@20 | TermStruct local nDCG@20 | TermBridge-FGWRec nDCG@20 | Delta vs OTC nDCG | Gamma |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows_with_avg:
        lines.append(
            "| {city} | {otc_r} | {base_r} | {final_r} | {delta_r} | {otc_n} | {base_n} | {final_n} | {delta_n} | {gamma} |".format(
                city=r["city"],
                otc_r=fmt(r["otc_recall"]),
                base_r=fmt(r["base_recall"]),
                final_r=fmt(r["final_recall"]),
                delta_r=fmt(r["final_recall"] - r["otc_recall"]),
                otc_n=fmt(r["otc_ndcg"]),
                base_n=fmt(r["base_ndcg"]),
                final_n=fmt(r["final_ndcg"]),
                delta_n=fmt(r["final_ndcg"] - r["otc_ndcg"]),
                gamma=gamma_label(r["gamma"]),
            )
        )
    lines.extend(
        [
            "",
            "## Source Details",
            "",
            "| Target | Source | Region Solver | Region Distance | Gamma |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for target, row in results.items():
        for src in row["sources"]:
            lines.append(
                f"| {target} | {src['source']} | {src['region_solver']} | "
                f"{src['region_distance']:.4f} | {src.get('gamma', '')} |"
            )
    lines.extend(["", f"![Comparison]({plot_path.name})", ""])
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def load_baseline(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = [
        "city",
        "otc_recall",
        "base_recall",
        "final_recall",
        "delta_recall_vs_otc",
        "otc_ndcg",
        "base_ndcg",
        "final_ndcg",
        "delta_ndcg_vs_otc",
        "gamma",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "city": row["city"],
                    "otc_recall": row["otc_recall"],
                    "base_recall": row["base_recall"],
                    "final_recall": row["final_recall"],
                    "delta_recall_vs_otc": row["final_recall"] - row["otc_recall"],
                    "otc_ndcg": row["otc_ndcg"],
                    "base_ndcg": row["base_ndcg"],
                    "final_ndcg": row["final_ndcg"],
                    "delta_ndcg_vs_otc": row["final_ndcg"] - row["otc_ndcg"],
                    "gamma": json.dumps(row["gamma"], ensure_ascii=False),
                }
            )


def write_plot(out_dir: Path, rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["city"] for row in rows]
    x = np.arange(len(labels))
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for ax, metric, title in [
        (axes[0], "recall", "Recall@20"),
        (axes[1], "ndcg", "nDCG@20"),
    ]:
        otc = [row[f"otc_{metric}"] for row in rows]
        base = [row[f"base_{metric}"] for row in rows]
        final = [row[f"final_{metric}"] for row in rows]
        ax.bar(x - width, otc, width, label="OTC-MF")
        ax.bar(x, base, width, label="TermStruct local")
        ax.bar(x + width, final, width, label="TermBridge-FGWRec")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(max(otc), max(base), max(final)) * 1.18)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out_dir / "termbridge_vs_otc.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def gamma_label(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(x):g}" for x in value) + "]"
    if value == "":
        return ""
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def score_stats(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bridge_stats(weights: np.ndarray) -> dict[str, float]:
    nonzero = np.count_nonzero(weights, axis=1)
    return {
        "row_sum_min": float(np.min(weights.sum(axis=1))),
        "row_sum_max": float(np.max(weights.sum(axis=1))),
        "nonzero_min": float(np.min(nonzero)),
        "nonzero_max": float(np.max(nonzero)),
        "max_weight_mean": float(np.mean(np.max(weights, axis=1))),
    }


def plan_stats(plan: np.ndarray) -> dict[str, float | list[int]]:
    arr = np.asarray(plan, dtype=np.float64)
    row_sum = arr.sum(axis=1)
    col_sum = arr.sum(axis=0)
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "sum": float(np.sum(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "row_sum_mean": float(np.mean(row_sum)),
        "row_sum_std": float(np.std(row_sum)),
        "col_sum_mean": float(np.mean(col_sum)),
        "col_sum_std": float(np.std(col_sum)),
    }


if __name__ == "__main__":
    main()
