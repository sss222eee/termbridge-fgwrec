from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city, load_optional_npy  # noqa: E402
from otc.train import TrainConfig, save_training_artifacts, train_mf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--city", default="all", help="'all' or comma-separated city names")
    parser.add_argument("--out", default="outputs/mf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-semantics", action="store_true")
    parser.add_argument(
        "--no-mask-train",
        action="store_true",
        help="Do not mask training positives during validation/test ranking. Useful for matching the official baseline code.",
    )
    parser.add_argument(
        "--drop-eval-train-overlap",
        action="store_true",
        help="When train positives are masked, remove overlapping eval positives from the metric denominator.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cities = available_cities(args.data_root) if args.city == "all" else [c.strip() for c in args.city.split(",")]
    if not cities:
        raise SystemExit(f"No processed cities found under {args.data_root}")

    for city_name in cities:
        city = load_city(args.data_root, city_name)
        brand_sem = load_optional_npy(args.data_root, city_name, "brand_semantic.npy")
        region_sem = load_optional_npy(args.data_root, city_name, "region_semantic.npy")
        if not args.use_semantics:
            brand_sem = None
            region_sem = None
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
            use_semantics=args.use_semantics,
            mask_train_eval=not args.no_mask_train,
            drop_eval_train_overlap=args.drop_eval_train_overlap,
        )
        model, result = train_mf(city, config, brand_semantic=brand_sem, region_semantic=region_sem)
        save_training_artifacts(args.out, city_name, model, result)
        print(
            f"{city_name}: best_epoch={result['best_epoch']} "
            f"valid R@{args.k}={result['valid']['recall']:.4f} "
            f"valid nDCG@{args.k}={result['valid']['ndcg']:.4f} "
            f"test R@{args.k}={result['test']['recall']:.4f} "
            f"test nDCG@{args.k}={result['test']['ndcg']:.4f}"
        )


if __name__ == "__main__":
    main()
