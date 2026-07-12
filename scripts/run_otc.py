from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city, load_optional_npy  # noqa: E402
from otc.evaluate import all_ranking_metrics, grouped_metrics  # noqa: E402
from otc.otc import TransportResult, compute_transport, softmax_negative_distance, source_to_target_scores  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", required=True, help="Directory produced by train_mf.py")
    parser.add_argument("--target", default="all")
    parser.add_argument("--method", choices=["gw", "fgw"], default="gw")
    parser.add_argument("--gamma-candidates", default="0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument("--per-source-gamma", action="store_true")
    parser.add_argument("--normalize-projection", action="store_true", default=True)
    parser.add_argument("--no-normalize-projection", action="store_false", dest="normalize_projection")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--fgw-alpha", type=float, default=0.5)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--ot-max-iter", type=int, default=10000)
    parser.add_argument("--ot-emd-max-iter", type=int, default=1_000_000)
    parser.add_argument("--ot-tol", type=float, default=1e-9)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--no-mask-train",
        action="store_true",
        help="Do not mask training positives during validation/test ranking.",
    )
    parser.add_argument(
        "--drop-eval-train-overlap",
        action="store_true",
        help="When train positives are masked, remove overlapping eval positives from metric denominators.",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Save score and transport-plan diagnostics.")
    parser.add_argument("--ablate-sources", action="store_true", help="Evaluate each source city separately.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    city_names = available_cities(args.data_root)
    targets = city_names if args.target == "all" else [c.strip() for c in args.target.split(",")]
    gammas = [float(x) for x in args.gamma_candidates.split(",") if x.strip()]

    city_data = {name: load_city(args.data_root, name) for name in city_names}
    embeddings = {name: load_embeddings(args.run_dir, name) for name in city_names}
    semantics = {
        name: (
            load_optional_npy(args.data_root, name, "brand_semantic.npy"),
            load_optional_npy(args.data_root, name, "region_semantic.npy"),
        )
        for name in city_names
    }
    transport_cache: dict[tuple[str, str, str], TransportResult] = {}

    results: dict[str, dict] = {}
    for target in targets:
        target_data = city_data[target]
        target_brand, target_region = embeddings[target]
        base_scores = target_brand @ target_region.T
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
        source_scores = []
        distances = []
        confidences = []
        details = []

        for source in city_names:
            if source == target:
                continue
            source_brand, source_region = embeddings[source]
            source_brand_sem, source_region_sem = semantics[source]
            target_brand_sem, target_region_sem = semantics[target]
            if args.method == "gw":
                source_brand_sem = source_region_sem = target_brand_sem = target_region_sem = None

            brand_tr = get_cached_transport(
                transport_cache,
                "brand",
                source,
                target,
                source_brand,
                target_brand,
                source_brand_sem,
                target_brand_sem,
                method=args.method,
                fgw_alpha=args.fgw_alpha,
                reg=args.reg,
                max_iter=args.ot_max_iter,
                emd_max_iter=args.ot_emd_max_iter,
                tol=args.ot_tol,
            )
            region_tr = get_cached_transport(
                transport_cache,
                "region",
                source,
                target,
                source_region,
                target_region,
                source_region_sem,
                target_region_sem,
                method=args.method,
                fgw_alpha=args.fgw_alpha,
                reg=args.reg,
                max_iter=args.ot_max_iter,
                emd_max_iter=args.ot_emd_max_iter,
                tol=args.ot_tol,
            )
            transferred = source_to_target_scores(
                source_brand,
                source_region,
                brand_tr.plan,
                region_tr.plan,
                normalize_projection=args.normalize_projection,
            )
            distance = brand_tr.distance + region_tr.distance
            confidence = float(np.sqrt(brand_tr.confidence * region_tr.confidence))
            source_scores.append(transferred)
            distances.append(distance)
            confidences.append(confidence)
            details.append(
                {
                    "source": source,
                    "distance": distance,
                    "confidence": confidence,
                    "brand_solver": brand_tr.solver,
                    "region_solver": region_tr.solver,
                }
            )
            if args.diagnostics:
                details[-1]["brand_plan_stats"] = plan_stats(brand_tr.plan)
                details[-1]["region_plan_stats"] = plan_stats(region_tr.plan)
                details[-1]["transfer_score_stats"] = score_stats(transferred)

        if args.adaptive:
            weights = softmax_negative_distance(distances, tau=args.tau)
        else:
            weights = np.ones(len(source_scores), dtype=np.float64)
        if args.confidence:
            weights = weights * np.asarray(confidences, dtype=np.float64)

        weighted_source_scores = [float(weight) * score for weight, score in zip(weights, source_scores)]
        weighted_transfer = np.zeros_like(base_scores, dtype=np.float64)
        for score in weighted_source_scores:
            weighted_transfer += score

        fusion = search_fusion(
            base_scores=base_scores,
            source_scores=weighted_source_scores,
            gammas=gammas,
            per_source_gamma=args.per_source_gamma,
            eval_train_pos=eval_train_pos,
            target_data=target_data,
            k=args.k,
            drop_eval_train_overlap=args.drop_eval_train_overlap,
        )
        best_gamma = fusion["best_gamma"]
        best_valid = fusion["valid"]
        test = fusion["test"]
        final_scores = fusion["final_scores"]
        selected_source_scores = apply_selected_gammas(
            weighted_source_scores,
            best_gamma,
            per_source_gamma=args.per_source_gamma,
        )
        selected_transfer = np.zeros_like(base_scores, dtype=np.float64)
        for score in selected_source_scores:
            selected_transfer += score
        groups = grouped_metrics(
            final_scores,
            eval_train_pos,
            target_data.test_pos,
            target_data.train_counts,
            k=args.k,
            drop_train_overlap=args.drop_eval_train_overlap,
        )

        for idx, weight in enumerate(weights):
            details[idx]["weight"] = float(weight)
            if args.diagnostics:
                details[idx]["weighted_transfer_score_stats"] = score_stats(weighted_source_scores[idx])
                details[idx]["selected_transfer_score_stats"] = score_stats(selected_source_scores[idx])
            if args.per_source_gamma and isinstance(best_gamma, list):
                details[idx]["gamma"] = float(best_gamma[idx])
            elif not args.per_source_gamma:
                details[idx]["gamma"] = float(best_gamma)

        ablations = {}
        if args.ablate_sources:
            for idx, source_detail in enumerate(details):
                source_fusion = search_fusion(
                    base_scores=base_scores,
                    source_scores=[weighted_source_scores[idx]],
                    gammas=gammas,
                    per_source_gamma=False,
                    eval_train_pos=eval_train_pos,
                    target_data=target_data,
                    k=args.k,
                    drop_eval_train_overlap=args.drop_eval_train_overlap,
                )
                ablations[source_detail["source"]] = {
                    "best_gamma": source_fusion["best_gamma"],
                    "otc_valid": source_fusion["valid"].as_dict(),
                    "otc_test": source_fusion["test"].as_dict(),
                    "delta_test_recall": source_fusion["test"].recall - base_test.recall,
                    "delta_test_ndcg": source_fusion["test"].ndcg - base_test.ndcg,
                    "final_score_stats": score_stats(source_fusion["final_scores"]) if args.diagnostics else {},
                }

        results[target] = {
            "method": args.method,
            "adaptive": args.adaptive,
            "confidence": args.confidence,
            "per_source_gamma": args.per_source_gamma,
            "normalize_projection": args.normalize_projection,
            "no_mask_train": args.no_mask_train,
            "drop_eval_train_overlap": args.drop_eval_train_overlap,
            "tau": args.tau,
            "fgw_alpha": args.fgw_alpha,
            "ot_max_iter": args.ot_max_iter,
            "ot_emd_max_iter": args.ot_emd_max_iter,
            "ot_tol": args.ot_tol,
            "best_gamma": best_gamma,
            "base_valid": base_valid.as_dict(),
            "base_test": base_test.as_dict(),
            "otc_valid": best_valid.as_dict() if best_valid else {},
            "otc_test": test.as_dict(),
            "valid_selection_score": fusion["valid_selection_score"],
            "groups": groups,
            "sources": details,
            "source_ablations": ablations,
        }
        if args.diagnostics:
            results[target]["score_stats"] = {
                "base": score_stats(base_scores),
                "weighted_transfer_sum_before_gamma": score_stats(weighted_transfer),
                "selected_transfer_after_gamma": score_stats(selected_transfer),
                "final": score_stats(final_scores),
            }
        gamma_label = (
            ",".join(f"{g:g}" for g in best_gamma)
            if isinstance(best_gamma, list)
            else f"{best_gamma:g}"
        )
        print(
            f"{target}: base test R@{args.k}={base_test.recall:.4f} nDCG@{args.k}={base_test.ndcg:.4f} | "
            f"OTC gamma={gamma_label} test R@{args.k}={test.recall:.4f} nDCG@{args.k}={test.ndcg:.4f}",
            flush=True,
        )

    out_path = Path(args.out) if args.out else Path(args.run_dir) / f"otc_{args.method}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}", flush=True)


def load_embeddings(run_dir: str | Path, city: str) -> tuple[np.ndarray, np.ndarray]:
    city_dir = Path(run_dir) / city
    return np.load(city_dir / "brand_embeddings.npy"), np.load(city_dir / "region_embeddings.npy")


def get_cached_transport(
    cache: dict[tuple[str, str, str], TransportResult],
    entity: str,
    source: str,
    target: str,
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    source_semantic: np.ndarray | None,
    target_semantic: np.ndarray | None,
    method: str,
    fgw_alpha: float,
    reg: float,
    max_iter: int,
    emd_max_iter: int,
    tol: float,
) -> TransportResult:
    key = (entity, source, target)
    if key in cache:
        return cache[key]

    reverse_key = (entity, target, source)
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

    result = compute_transport(
        source_embedding,
        target_embedding,
        source_semantic,
        target_semantic,
        method=method,
        fgw_alpha=fgw_alpha,
        reg=reg,
        max_iter=max_iter,
        emd_max_iter=emd_max_iter,
        tol=tol,
    )
    cache[key] = result
    return result


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

    if per_source_gamma:
        gamma_iter = itertools.product(gammas, repeat=len(source_scores))
    else:
        gamma_iter = ((gamma,) for gamma in gammas)

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


def apply_selected_gammas(
    source_scores: list[np.ndarray],
    best_gamma: float | list[float],
    per_source_gamma: bool,
) -> list[np.ndarray]:
    if per_source_gamma:
        if not isinstance(best_gamma, list):
            raise ValueError("per-source gamma search must return a list of gammas")
        return [float(gamma) * score for gamma, score in zip(best_gamma, source_scores)]
    return [float(best_gamma) * score for score in source_scores]


def score_stats(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def plan_stats(plan: np.ndarray) -> dict[str, float | list[int]]:
    arr = np.asarray(plan, dtype=np.float64)
    row_sum = arr.sum(axis=1)
    col_sum = arr.sum(axis=0)
    expected_row_sum = 1.0 / arr.shape[0]
    expected_col_sum = 1.0 / arr.shape[1]
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "sum": float(np.sum(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "expected_row_sum": float(expected_row_sum),
        "expected_col_sum": float(expected_col_sum),
        "row_sum_mean": float(np.mean(row_sum)),
        "row_sum_std": float(np.std(row_sum)),
        "row_sum_min": float(np.min(row_sum)),
        "row_sum_max": float(np.max(row_sum)),
        "row_sum_max_error": float(np.max(np.abs(row_sum - expected_row_sum))),
        "col_sum_mean": float(np.mean(col_sum)),
        "col_sum_std": float(np.std(col_sum)),
        "col_sum_min": float(np.min(col_sum)),
        "col_sum_max": float(np.max(col_sum)),
        "col_sum_max_error": float(np.max(np.abs(col_sum - expected_col_sum))),
    }


if __name__ == "__main__":
    main()
