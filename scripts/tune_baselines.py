from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city
from otc.lightgcn import LightGCNConfig, save_lightgcn_artifacts, train_lightgcn
from otc.train import TrainConfig, save_training_artifacts, train_mf


PAPER_TARGETS = {
    "mf": {
        "Chicago": {"recall": 0.2494, "ndcg": 0.1465},
        "NYC": {"recall": 0.1702, "ndcg": 0.0917},
        "Singapore": {"recall": 0.4430, "ndcg": 0.2351},
        "Tokyo": {"recall": 0.1323, "ndcg": 0.0781},
    },
    "lightgcn": {
        "Chicago": {"recall": 0.2875, "ndcg": 0.1902},
        "NYC": {"recall": 0.2087, "ndcg": 0.1088},
        "Singapore": {"recall": 0.5013, "ndcg": 0.2745},
        "Tokyo": {"recall": 0.1751, "ndcg": 0.1068},
    },
}


MF_GRID = [
    {"name": "dim100_bs512_lr005_wd1e4", "embed_dim": 100, "batch_size": 512, "lr": 5e-3, "weight_decay": 1e-4},
    {"name": "dim64_bs512_lr001_wd1e4", "embed_dim": 64, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-4},
    {"name": "dim128_bs256_lr005_wd1e5", "embed_dim": 128, "batch_size": 256, "lr": 5e-3, "weight_decay": 1e-5},
    {"name": "dim64_bs256_lr005_wd0", "embed_dim": 64, "batch_size": 256, "lr": 5e-3, "weight_decay": 0.0},
    {"name": "dim128_bs128_lr003_wd1e5", "embed_dim": 128, "batch_size": 128, "lr": 3e-3, "weight_decay": 1e-5},
    {"name": "dim200_bs256_lr003_wd1e5", "embed_dim": 200, "batch_size": 256, "lr": 3e-3, "weight_decay": 1e-5},
    {"name": "paper_like_dim100_bs128_lr001_wd1e4", "embed_dim": 100, "batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4},
    {"name": "dim100_bs128_lr003_wd1e4", "embed_dim": 100, "batch_size": 128, "lr": 3e-3, "weight_decay": 1e-4},
    {"name": "dim100_bs64_lr003_wd1e4", "embed_dim": 100, "batch_size": 64, "lr": 3e-3, "weight_decay": 1e-4},
]


LIGHTGCN_GRID = [
    {"name": "dim64_l1_lr001_wd1e4_binary", "embed_dim": 64, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "binary"},
    {"name": "dim64_l2_lr001_wd1e4_binary", "embed_dim": 64, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "binary"},
    {"name": "dim64_l3_lr001_wd1e4_binary", "embed_dim": 64, "num_layers": 3, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "binary"},
    {"name": "dim100_l2_lr001_wd1e4_binary", "embed_dim": 100, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "binary"},
    {"name": "dim64_l1_lr001_wd1e4_count", "embed_dim": 64, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "count"},
    {"name": "dim64_l2_lr001_wd1e4_count", "embed_dim": 64, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "count"},
    {"name": "dim64_l1_lr001_wd1e4_sqrt", "embed_dim": 64, "num_layers": 1, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "sqrt_count"},
    {"name": "dim64_l2_lr001_wd1e4_sqrt", "embed_dim": 64, "num_layers": 2, "lr": 1e-3, "weight_decay": 1e-4, "graph_weighting": "sqrt_count"},
    {"name": "dim100_l2_lr003_wd1e5_sqrt", "embed_dim": 100, "num_layers": 2, "lr": 3e-3, "weight_decay": 1e-5, "graph_weighting": "sqrt_count"},
    {"name": "dim64_l2_lr005_wd1e5_sqrt", "embed_dim": 64, "num_layers": 2, "lr": 5e-3, "weight_decay": 1e-5, "graph_weighting": "sqrt_count"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune MF and LightGCN baselines under the same strict split protocol.")
    parser.add_argument("--data-root", default="data/processed_raw_poi")
    parser.add_argument("--out", default="outputs/baseline_tuning_seed2024")
    parser.add_argument("--models", nargs="+", default=["mf", "lightgcn"], choices=["mf", "lightgcn"])
    parser.add_argument("--cities", nargs="+", default=["all"])
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size-lightgcn", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-mf-configs", type=int, default=None)
    parser.add_argument("--max-lightgcn-configs", type=int, default=None)
    parser.add_argument("--lightgcn-config-start", type=int, default=0)
    return parser.parse_args()


def score(row: dict) -> float:
    return float(row["valid"]["recall"]) + float(row["valid"]["ndcg"])


def gap(model: str, city: str, test: dict) -> dict[str, float]:
    target = PAPER_TARGETS[model][city]
    return {
        "recall": float(test["recall"]) - target["recall"],
        "ndcg": float(test["ndcg"]) - target["ndcg"],
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cities = available_cities(data_root) if args.cities == ["all"] else args.cities
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    best_rows: dict[str, dict[str, dict]] = {model: {} for model in args.models}

    for city_name in cities:
        city = load_city(data_root, city_name)
        if "mf" in args.models:
            grid = MF_GRID[: args.max_mf_configs] if args.max_mf_configs else MF_GRID
            for cfg in grid:
                config = TrainConfig(
                    embed_dim=cfg["embed_dim"],
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    batch_size=cfg["batch_size"],
                    epochs=args.epochs,
                    patience=args.patience,
                    eval_every=args.eval_every,
                    k=20,
                    seed=args.seed,
                    device=args.device,
                    mask_train_eval=True,
                    drop_eval_train_overlap=True,
                )
                run_dir = out / "artifacts" / "mf" / cfg["name"]
                model, result = train_mf(city, config)
                save_training_artifacts(run_dir, city_name, model, result)
                row = {
                    "model": "mf",
                    "city": city_name,
                    "config_name": cfg["name"],
                    "best_epoch": result["best_epoch"],
                    "valid": result["valid"],
                    "test": result["test"],
                    "paper_gap": gap("mf", city_name, result["test"]),
                    "config": asdict(config),
                }
                all_rows.append(row)
                if city_name not in best_rows["mf"] or score(row) > score(best_rows["mf"][city_name]):
                    best_rows["mf"][city_name] = row
                print(
                    f"mf {city_name} {cfg['name']}: valid {row['valid']['recall']:.4f}/{row['valid']['ndcg']:.4f} "
                    f"test {row['test']['recall']:.4f}/{row['test']['ndcg']:.4f}",
                    flush=True,
                )

        if "lightgcn" in args.models:
            grid = LIGHTGCN_GRID[args.lightgcn_config_start :]
            grid = grid[: args.max_lightgcn_configs] if args.max_lightgcn_configs else grid
            for cfg in grid:
                config = LightGCNConfig(
                    embed_dim=cfg["embed_dim"],
                    num_layers=cfg["num_layers"],
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    batch_size=args.batch_size_lightgcn,
                    epochs=args.epochs,
                    patience=args.patience,
                    eval_every=args.eval_every,
                    k=20,
                    seed=args.seed,
                    device=args.device,
                    mask_train_eval=True,
                    drop_eval_train_overlap=True,
                    graph_weighting=cfg["graph_weighting"],
                )
                run_dir = out / "artifacts" / "lightgcn" / cfg["name"]
                model, result = train_lightgcn(city, config)
                save_lightgcn_artifacts(run_dir, city_name, model, result)
                row = {
                    "model": "lightgcn",
                    "city": city_name,
                    "config_name": cfg["name"],
                    "best_epoch": result["best_epoch"],
                    "valid": result["valid"],
                    "test": result["test"],
                    "paper_gap": gap("lightgcn", city_name, result["test"]),
                    "config": asdict(config),
                }
                all_rows.append(row)
                if city_name not in best_rows["lightgcn"] or score(row) > score(best_rows["lightgcn"][city_name]):
                    best_rows["lightgcn"][city_name] = row
                print(
                    f"lightgcn {city_name} {cfg['name']}: valid {row['valid']['recall']:.4f}/{row['valid']['ndcg']:.4f} "
                    f"test {row['test']['recall']:.4f}/{row['test']['ndcg']:.4f}",
                    flush=True,
                )

    best_list = [row for rows_by_city in best_rows.values() for row in rows_by_city.values()]
    (out / "all_results.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "best_by_city.json").write_text(json.dumps(best_list, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out / "summary.md", all_rows, best_list)


def write_summary(path: Path, all_rows: list[dict], best_rows: list[dict]) -> None:
    lines = ["# Baseline Tuning Summary", ""]
    lines.append("Selection uses validation Recall@20 + nDCG@20 under train-positive masking and eval/train-overlap dropping.")
    lines.append("")
    lines.append("## Best by City")
    lines.append("")
    lines.append("| Model | City | Config | Valid R | Valid nDCG | Test R | Test nDCG | Paper Gap R | Paper Gap nDCG |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(best_rows, key=lambda x: (x["model"], x["city"])):
        lines.append(
            f"| {row['model']} | {row['city']} | {row['config_name']} | "
            f"{row['valid']['recall']:.4f} | {row['valid']['ndcg']:.4f} | "
            f"{row['test']['recall']:.4f} | {row['test']['ndcg']:.4f} | "
            f"{row['paper_gap']['recall']:+.4f} | {row['paper_gap']['ndcg']:+.4f} |"
        )
    lines.append("")
    lines.append("## All Runs")
    lines.append("")
    lines.append("| Model | City | Config | Valid R | Valid nDCG | Test R | Test nDCG |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in all_rows:
        lines.append(
            f"| {row['model']} | {row['city']} | {row['config_name']} | "
            f"{row['valid']['recall']:.4f} | {row['valid']['ndcg']:.4f} | "
            f"{row['test']['recall']:.4f} | {row['test']['ndcg']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
