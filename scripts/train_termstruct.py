from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.train import TrainConfig, save_training_artifacts, train_mf  # noqa: E402


VARIANTS = {
    "term": {"use_terms": True, "use_structure": False},
    "struct": {"use_terms": False, "use_structure": True},
    "termstruct": {"use_terms": True, "use_structure": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Term-MF, Struct-MF, and TermStruct-MF variants.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--variant", default="all", help="'all' or comma-separated: term,struct,termstruct")
    parser.add_argument("--city", default="all")
    parser.add_argument("--out", default="outputs/termstruct")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--normalize-semantic", action="store_true")
    parser.add_argument("--structure-weight", type=float, default=0.01)
    parser.add_argument("--structure-batch-size", type=int, default=1024)
    parser.add_argument("--no-mask-train", action="store_true")
    parser.add_argument("--drop-eval-train-overlap", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cities = available_cities(args.data_root) if args.city == "all" else [c.strip() for c in args.city.split(",")]
    variants = list(VARIANTS) if args.variant == "all" else [v.strip() for v in args.variant.split(",")]
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")

    summary: dict[str, dict[str, dict]] = {}
    for variant in variants:
        variant_cfg = VARIANTS[variant]
        summary[variant] = {}
        for city_name in cities:
            city = load_city(args.data_root, city_name)
            brand_terms, region_terms, brand_structure, region_structure = load_features(args.feature_root, city_name)
            config = TrainConfig(
                embed_dim=args.embed_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                eval_every=args.eval_every,
                k=args.k,
                seed=args.seed,
                device=device,
                use_semantics=variant_cfg["use_terms"],
                semantic_weight=args.semantic_weight,
                normalize_semantic=args.normalize_semantic,
                structure_weight=args.structure_weight if variant_cfg["use_structure"] else 0.0,
                structure_batch_size=args.structure_batch_size,
                mask_train_eval=not args.no_mask_train,
                drop_eval_train_overlap=args.drop_eval_train_overlap,
            )
            model, result = train_mf(
                city,
                config,
                brand_semantic=brand_terms if variant_cfg["use_terms"] else None,
                region_semantic=region_terms if variant_cfg["use_terms"] else None,
                brand_structure=brand_structure if variant_cfg["use_structure"] else None,
                region_structure=region_structure if variant_cfg["use_structure"] else None,
            )
            save_training_artifacts(Path(args.out) / variant, city_name, model, result)
            summary[variant][city_name] = {
                "best_epoch": result["best_epoch"],
                "valid": result["valid"],
                "test": result["test"],
                "groups": result["groups"],
            }
            print(
                f"{variant}/{city_name}: best_epoch={result['best_epoch']} "
                f"valid R@{args.k}={result['valid']['recall']:.4f} "
                f"valid nDCG@{args.k}={result['valid']['ndcg']:.4f} "
                f"test R@{args.k}={result['test']['recall']:.4f} "
                f"test nDCG@{args.k}={result['test']['ndcg']:.4f}",
                flush=True,
            )

    out_path = Path(args.out) / "termstruct_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}", flush=True)


def load_features(feature_root: str | Path, city: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    city_dir = Path(feature_root) / city
    return (
        np.load(city_dir / "brand_terms.npy"),
        np.load(city_dir / "region_terms.npy"),
        np.load(city_dir / "brand_structure.npy"),
        np.load(city_dir / "region_structure.npy"),
    )


if __name__ == "__main__":
    main()
