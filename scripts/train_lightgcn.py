from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city
from otc.lightgcn import LightGCNConfig, save_lightgcn_artifacts, train_lightgcn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGCN baseline on OpenSiteRec-style city splits.")
    parser.add_argument("--data-root", default="data/processed_raw_poi")
    parser.add_argument("--city", default="all", help="City name or 'all'.")
    parser.add_argument("--out", default="outputs/lightgcn_seed2024_paper_strict")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument("--graph-weighting", default="binary", choices=["binary", "count", "sqrt_count"])
    parser.add_argument("--no-mask-train", action="store_true")
    parser.add_argument("--drop-eval-train-overlap", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cities = available_cities(data_root) if args.city == "all" else [args.city]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for city_name in cities:
        city = load_city(data_root, city_name)
        config = LightGCNConfig(
            embed_dim=args.embed_dim,
            num_layers=args.layers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            patience=args.patience,
            eval_every=args.eval_every,
            k=args.k,
            seed=args.seed,
            device=args.device,
            mask_train_eval=not args.no_mask_train,
            drop_eval_train_overlap=args.drop_eval_train_overlap,
            init_std=args.init_std,
            graph_weighting=args.graph_weighting,
        )
        model, result = train_lightgcn(city, config)
        save_lightgcn_artifacts(out_dir, city_name, model, result)
        row = {
            "city": city_name,
            "best_epoch": result["best_epoch"],
            "valid": result["valid"],
            "test": result["test"],
            "config": result["config"],
        }
        rows.append(row)
        print(
            f"{city_name}: best_epoch={result['best_epoch']} "
            f"valid R={result['valid']['recall']:.4f} nDCG={result['valid']['ndcg']:.4f} "
            f"test R={result['test']['recall']:.4f} nDCG={result['test']['ndcg']:.4f}"
        )

    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
