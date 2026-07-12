from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.train import TrainConfig, train_mf  # noqa: E402


PAPER_MF = {
    "Chicago": {"recall": 0.2494, "ndcg": 0.1465},
    "NYC": {"recall": 0.1702, "ndcg": 0.0917},
    "Singapore": {"recall": 0.4430, "ndcg": 0.2351},
    "Tokyo": {"recall": 0.1323, "ndcg": 0.0781},
}


GRID_CONFIGS = [
    {"name": "current_bpr", "embed_dim": 64, "batch_size": 1024, "lr": 1e-3, "weight_decay": 1e-5},
    {"name": "paper_default_like", "embed_dim": 100, "batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4},
    {"name": "dim100_bs512_wd1e-4", "embed_dim": 100, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-4},
    {"name": "dim64_bs512_wd1e-4", "embed_dim": 64, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-4},
    {"name": "dim100_bs512_lr5e-4", "embed_dim": 100, "batch_size": 512, "lr": 5e-4, "weight_decay": 1e-4},
    {"name": "dim100_bs512_lr5e-3", "embed_dim": 100, "batch_size": 512, "lr": 5e-3, "weight_decay": 1e-4},
]


RAW_FILES = {
    "Chicago": ROOT / "data" / "raw" / "OpenSiteRec" / "Chicago" / "Chicago_KG_plus.csv",
    "NYC": ROOT / "data" / "raw" / "OpenSiteRec" / "NYC" / "NYC_KG_plus.csv",
    "Singapore": ROOT / "data" / "raw" / "OpenSiteRec" / "Singapore" / "Singapore_KG_plus.csv",
    "Tokyo": ROOT / "data" / "raw" / "OpenSiteRec" / "Tokyo" / "Tokyo_KG_plus.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/mf_alignment")
    parser.add_argument("--processed-root", default="data/alignment_processed")
    parser.add_argument("--grid-seed", type=int, default=2024)
    parser.add_argument("--seeds", default="2020,2021,2022,2023,2024")
    parser.add_argument("--cities", default="Chicago,NYC,Singapore,Tokyo")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_root = ROOT / args.out_root
    processed_root = ROOT / args.processed_root
    out_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    cities = [x.strip() for x in args.cities.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    grid_data_root = ensure_processed(processed_root, args.grid_seed, cities)
    grid_rows = run_grid(grid_data_root, cities, args)
    write_rows(out_root / "grid_results.csv", grid_rows)
    (out_root / "grid_results.json").write_text(
        json.dumps(grid_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_by_city = select_best_by_city(grid_rows)
    seed_rows = []
    for seed in seeds:
        data_root = ensure_processed(processed_root, seed, cities)
        for city in cities:
            cfg = best_by_city[city]
            row = train_one(data_root, city, cfg, args, seed=seed, tag=f"seed{seed}_{city}_{cfg['name']}")
            seed_rows.append(row)
            print_result("seed", row)
    write_rows(out_root / "seed_results.csv", seed_rows)
    (out_root / "seed_results.json").write_text(
        json.dumps(seed_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(out_root, grid_rows, seed_rows, best_by_city)


def ensure_processed(base: Path, seed: int, cities: list[str]) -> Path:
    out = base / f"raw_poi_seed{seed}"
    expected = [out / city / "train.txt" for city in cities]
    if all(path.exists() for path in expected):
        return out
    for city in cities:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "preprocess_edges.py"),
                "--input",
                str(RAW_FILES[city]),
                "--out",
                str(out),
                "--city-name",
                city,
                "--city-col",
                "city",
                "--brand-col",
                "Brand",
                "--region-col",
                "Region_ID",
                "--seed",
                str(seed),
                "--min-brand-degree",
                "5",
                "--preserve-region-ids",
                "--split-mode",
                "raw_poi",
            ],
            check=True,
            cwd=ROOT,
        )
    return out


def run_grid(data_root: Path, cities: list[str], args: argparse.Namespace) -> list[dict]:
    rows = []
    for cfg in GRID_CONFIGS:
        for city in cities:
            row = train_one(data_root, city, cfg, args, seed=args.grid_seed, tag=f"grid_{city}_{cfg['name']}")
            rows.append(row)
            print_result("grid", row)
    return rows


def train_one(data_root: Path, city_name: str, cfg: dict, args: argparse.Namespace, seed: int, tag: str) -> dict:
    city = load_city(data_root, city_name)
    config = TrainConfig(
        embed_dim=int(cfg["embed_dim"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        batch_size=int(cfg["batch_size"]),
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        k=args.k,
        seed=seed,
        device=args.device,
        mask_train_eval=True,
        drop_eval_train_overlap=True,
    )
    _, result = train_mf(city, config)
    paper = PAPER_MF[city_name]
    return {
        "tag": tag,
        "city": city_name,
        "seed": seed,
        "config_name": cfg["name"],
        **{k: cfg[k] for k in ["embed_dim", "batch_size", "lr", "weight_decay"]},
        "best_epoch": result["best_epoch"],
        "valid_recall": result["valid"]["recall"],
        "valid_ndcg": result["valid"]["ndcg"],
        "valid_score": result["valid"]["recall"] + result["valid"]["ndcg"],
        "test_recall": result["test"]["recall"],
        "test_ndcg": result["test"]["ndcg"],
        "paper_recall": paper["recall"],
        "paper_ndcg": paper["ndcg"],
        "abs_gap": abs(result["test"]["recall"] - paper["recall"])
        + abs(result["test"]["ndcg"] - paper["ndcg"]),
        "eval_brands": result["test"]["num_brands"],
        "train_config": asdict(config),
    }


def select_best_by_city(rows: list[dict]) -> dict[str, dict]:
    best = {}
    for row in rows:
        city = row["city"]
        if city not in best or row["valid_score"] > best[city]["valid_score"]:
            best[city] = row
    return {
        city: {
            "name": row["config_name"],
            "embed_dim": row["embed_dim"],
            "batch_size": row["batch_size"],
            "lr": row["lr"],
            "weight_decay": row["weight_decay"],
        }
        for city, row in best.items()
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [k for k in rows[0] if k != "train_config"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_summary(out_root: Path, grid_rows: list[dict], seed_rows: list[dict], best_by_city: dict[str, dict]) -> None:
    lines = ["# MF Alignment Experiments", ""]
    lines += ["## Best Grid Config By City", ""]
    lines += ["| City | Config | Valid R@20 | Valid nDCG@20 | Test R@20 | Test nDCG@20 | Paper R@20 | Paper nDCG@20 |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|"]
    for city in ["Chicago", "NYC", "Singapore", "Tokyo"]:
        rows = [r for r in grid_rows if r["city"] == city and r["config_name"] == best_by_city[city]["name"]]
        row = max(rows, key=lambda r: r["valid_score"])
        lines.append(
            f"| {city} | {row['config_name']} | {row['valid_recall']:.4f} | {row['valid_ndcg']:.4f} | "
            f"{row['test_recall']:.4f} | {row['test_ndcg']:.4f} | {row['paper_recall']:.4f} | {row['paper_ndcg']:.4f} |"
        )
    lines += ["", "## Multi-Seed Results With Best Grid Config", ""]
    lines += ["| City | Mean R@20 | Std R@20 | Mean nDCG@20 | Std nDCG@20 | Paper R@20 | Paper nDCG@20 |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|"]
    for city in ["Chicago", "NYC", "Singapore", "Tokyo"]:
        rows = [r for r in seed_rows if r["city"] == city]
        recalls = [float(r["test_recall"]) for r in rows]
        ndcgs = [float(r["test_ndcg"]) for r in rows]
        paper = PAPER_MF[city]
        lines.append(
            f"| {city} | {mean(recalls):.4f} | {std(recalls):.4f} | "
            f"{mean(ndcgs):.4f} | {std(ndcgs):.4f} | {paper['recall']:.4f} | {paper['ndcg']:.4f} |"
        )
    lines += [
        "",
        "Artifacts:",
        "- `grid_results.csv` / `grid_results.json`",
        "- `seed_results.csv` / `seed_results.json`",
    ]
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return (sum((x - mu) ** 2 for x in values) / len(values)) ** 0.5


def print_result(stage: str, row: dict) -> None:
    print(
        f"[{stage}] {row['city']} {row['config_name']} seed={row['seed']} "
        f"valid={row['valid_recall']:.4f}/{row['valid_ndcg']:.4f} "
        f"test={row['test_recall']:.4f}/{row['test_ndcg']:.4f} "
        f"epoch={row['best_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
