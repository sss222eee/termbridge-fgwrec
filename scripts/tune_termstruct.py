from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.train import TrainConfig, train_mf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Term/Struct/TermStruct hyperparameters.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--out", default="outputs/termstruct_tuning")
    parser.add_argument("--city", default="all")
    parser.add_argument("--term-alphas", default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--struct-lambdas", default="0.0001,0.001,0.005,0.01")
    parser.add_argument("--termstruct-alphas", default="0.05,0.1")
    parser.add_argument("--termstruct-lambdas", default="0.00005,0.0005,0.0025,0.005")
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
    parser.add_argument("--structure-batch-size", type=int, default=1024)
    parser.add_argument("--no-mask-train", action="store_true")
    parser.add_argument("--drop-eval-train-overlap", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cities = available_cities(args.data_root) if args.city == "all" else [c.strip() for c in args.city.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args)
    rows: list[dict] = []
    detailed: dict[str, dict] = {}

    for exp in experiments:
        exp_rows = []
        detailed[exp["name"]] = {"config": exp, "cities": {}}
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
                use_semantics=exp["use_terms"],
                semantic_weight=exp["alpha"],
                normalize_semantic=exp["use_terms"],
                structure_weight=exp["lambda_struct"],
                structure_batch_size=args.structure_batch_size,
                mask_train_eval=not args.no_mask_train,
                drop_eval_train_overlap=args.drop_eval_train_overlap,
            )
            _, result = train_mf(
                city,
                config,
                brand_semantic=brand_terms if exp["use_terms"] else None,
                region_semantic=region_terms if exp["use_terms"] else None,
                brand_structure=brand_structure if exp["use_structure"] else None,
                region_structure=region_structure if exp["use_structure"] else None,
            )
            row = {
                "experiment": exp["name"],
                "family": exp["family"],
                "city": city_name,
                "alpha": exp["alpha"],
                "lambda_struct": exp["lambda_struct"],
                "best_epoch": result["best_epoch"],
                "valid_recall": result["valid"]["recall"],
                "valid_ndcg": result["valid"]["ndcg"],
                "test_recall": result["test"]["recall"],
                "test_ndcg": result["test"]["ndcg"],
            }
            rows.append(row)
            exp_rows.append(row)
            detailed[exp["name"]]["cities"][city_name] = {
                "best_epoch": result["best_epoch"],
                "valid": result["valid"],
                "test": result["test"],
                "groups": result["groups"],
            }
            print(
                f"{exp['name']}/{city_name}: "
                f"valid R={result['valid']['recall']:.4f} N={result['valid']['ndcg']:.4f} "
                f"test R={result['test']['recall']:.4f} N={result['test']['ndcg']:.4f}",
                flush=True,
            )

        avg = aggregate_rows(exp_rows)
        detailed[exp["name"]]["aggregate"] = avg
        print(
            f"{exp['name']}: avg valid={avg['avg_valid_selection']:.4f} "
            f"avg test R={avg['avg_test_recall']:.4f} N={avg['avg_test_ndcg']:.4f}",
            flush=True,
        )

    write_rows(out_dir / "tuning_city_results.csv", rows)
    aggregates = aggregate_experiments(detailed)
    write_rows(out_dir / "tuning_aggregate_results.csv", aggregates)
    (out_dir / "tuning_results.json").write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, aggregates)
    plot_results(out_dir, aggregates)
    print(f"Saved tuning outputs to {out_dir}", flush=True)


def build_experiments(args: argparse.Namespace) -> list[dict]:
    experiments: list[dict] = []
    for alpha in parse_floats(args.term_alphas):
        experiments.append(
            {
                "name": f"Term-alpha{alpha:g}",
                "family": "Term-MF",
                "use_terms": True,
                "use_structure": False,
                "alpha": alpha,
                "lambda_struct": 0.0,
            }
        )
    for lambda_struct in parse_floats(args.struct_lambdas):
        experiments.append(
            {
                "name": f"Struct-lambda{lambda_struct:g}",
                "family": "Struct-MF",
                "use_terms": False,
                "use_structure": True,
                "alpha": 0.0,
                "lambda_struct": lambda_struct,
            }
        )
    for alpha in parse_floats(args.termstruct_alphas):
        for lambda_struct in parse_floats(args.termstruct_lambdas):
            experiments.append(
                {
                    "name": f"TermStruct-alpha{alpha:g}-lambda{lambda_struct:g}",
                    "family": "TermStruct-MF",
                    "use_terms": True,
                    "use_structure": True,
                    "alpha": alpha,
                    "lambda_struct": lambda_struct,
                }
            )
    return experiments


def parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def load_features(feature_root: str | Path, city: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    city_dir = Path(feature_root) / city
    return (
        np.load(city_dir / "brand_terms.npy"),
        np.load(city_dir / "region_terms.npy"),
        np.load(city_dir / "brand_structure.npy"),
        np.load(city_dir / "region_structure.npy"),
    )


def aggregate_rows(rows: list[dict]) -> dict[str, float]:
    valid_recall = float(np.mean([row["valid_recall"] for row in rows]))
    valid_ndcg = float(np.mean([row["valid_ndcg"] for row in rows]))
    test_recall = float(np.mean([row["test_recall"] for row in rows]))
    test_ndcg = float(np.mean([row["test_ndcg"] for row in rows]))
    return {
        "avg_valid_recall": valid_recall,
        "avg_valid_ndcg": valid_ndcg,
        "avg_valid_selection": valid_recall + valid_ndcg,
        "avg_test_recall": test_recall,
        "avg_test_ndcg": test_ndcg,
    }


def aggregate_experiments(detailed: dict[str, dict]) -> list[dict]:
    rows = []
    for name, record in detailed.items():
        cfg = record["config"]
        avg = record["aggregate"]
        rows.append(
            {
                "experiment": name,
                "family": cfg["family"],
                "alpha": cfg["alpha"],
                "lambda_struct": cfg["lambda_struct"],
                **avg,
            }
        )
    rows.sort(key=lambda row: (row["family"], row["alpha"], row["lambda_struct"]))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_by_family(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        family = row["family"]
        if family not in best or row["avg_valid_selection"] > best[family]["avg_valid_selection"]:
            best[family] = row
    return best


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    best = best_by_family(rows)
    lines = [
        "# TermStruct Tuning Summary",
        "",
        "All hyperparameters are selected by average validation Recall@20 + nDCG@20. Test metrics are reported only after selection under the same split and evaluation protocol.",
        "",
        "## Best By Family",
        "",
        "| Family | Selected Experiment | Avg Valid Score | Avg Test Recall@20 | Avg Test nDCG@20 |",
        "|---|---|---:|---:|---:|",
    ]
    for family in ["Term-MF", "Struct-MF", "TermStruct-MF"]:
        row = best.get(family)
        if row is None:
            continue
        lines.append(
            f"| {family} | {row['experiment']} | {row['avg_valid_selection']:.4f} | "
            f"{row['avg_test_recall']:.4f} | {row['avg_test_ndcg']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tuning_city_results.csv`: city-level validation/test metrics for every run.",
            "- `tuning_aggregate_results.csv`: average validation/test metrics for every hyperparameter setting.",
            "- `tuning_results.json`: full per-city metrics and configs.",
            "- `tuning_curves.png`: validation/test tuning curves.",
            "- `best_family_comparison.png`: best tuned variants compared by test metrics.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(out_dir: Path, rows: list[dict]) -> None:
    family_styles = {
        "Term-MF": "o-",
        "Struct-MF": "s-",
        "TermStruct-MF": "^-",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    for family, style in family_styles.items():
        sub = [row for row in rows if row["family"] == family]
        if not sub:
            continue
        x = np.arange(len(sub))
        labels = [short_label(row) for row in sub]
        axes[0].plot(x, [row["avg_valid_selection"] for row in sub], style, label=family)
        axes[1].plot(x, [row["avg_test_recall"] for row in sub], style, label=f"{family} Recall")
        axes[1].plot(x, [row["avg_test_ndcg"] for row in sub], style.replace("-", "--"), label=f"{family} nDCG")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    axes[0].set_title("Validation Selection Score")
    axes[1].set_title("Average Test Metrics")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "tuning_curves.png", bbox_inches="tight")
    plt.close(fig)

    best = best_by_family(rows)
    labels = list(best)
    x = np.arange(len(labels))
    recall = [best[label]["avg_test_recall"] for label in labels]
    ndcg = [best[label]["avg_test_ndcg"] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    width = 0.35
    ax.bar(x - width / 2, recall, width, label="Avg Recall@20", color="#4C78A8")
    ax.bar(x + width / 2, ndcg, width, label="Avg nDCG@20", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Best Tuned Variant By Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "best_family_comparison.png", bbox_inches="tight")
    plt.close(fig)


def short_label(row: dict) -> str:
    if row["family"] == "Term-MF":
        return f"a={row['alpha']:g}"
    if row["family"] == "Struct-MF":
        return f"l={row['lambda_struct']:g}"
    return f"a={row['alpha']:g}\nl={row['lambda_struct']:g}"


if __name__ == "__main__":
    main()
