from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.evaluate import all_ranking_metrics  # noqa: E402


@dataclass
class BCEConfig:
    embed_dim: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 300
    patience: int = 30
    eval_every: int = 5
    k: int = 20
    seed: int = 2024
    device: str = "cpu"
    label_smoothing: bool = True
    select_metric: str = "paper_strict"


class VanillaMFBCE(nn.Module):
    def __init__(self, num_brands: int, num_regions: int, embed_dim: int) -> None:
        super().__init__()
        self.brand_embedding = nn.Embedding(num_brands, embed_dim)
        self.region_embedding = nn.Embedding(num_regions, embed_dim)
        nn.init.xavier_normal_(self.brand_embedding.weight)
        nn.init.xavier_normal_(self.region_embedding.weight)

    def forward(self, brand_ids: torch.Tensor) -> torch.Tensor:
        brand = self.brand_embedding(brand_ids.long())
        return torch.sigmoid(brand @ self.region_embedding.weight.T)

    def score_matrix(self) -> torch.Tensor:
        return torch.sigmoid(self.brand_embedding.weight @ self.region_embedding.weight.T)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--city", default="all")
    parser.add_argument("--out", default="outputs/mf_bce")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-label-smoothing", action="store_true")
    parser.add_argument(
        "--select-metric",
        choices=["paper_strict", "official_like"],
        default="paper_strict",
        help="Validation mode used for early stopping.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cities = available_cities(args.data_root) if args.city == "all" else [c.strip() for c in args.city.split(",")]
    if not cities:
        raise SystemExit(f"No processed cities found under {args.data_root}")

    config = BCEConfig(
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
        label_smoothing=not args.no_label_smoothing,
        select_metric=args.select_metric,
    )
    for city_name in cities:
        city = load_city(args.data_root, city_name)
        result = train_city(city, config)
        save_result(args.out, city_name, result)
        paper = result["test"]["paper_strict"]
        official = result["test"]["official_like"]
        print(
            f"{city_name}: best_epoch={result['best_epoch']} "
            f"paper_strict R@{args.k}={paper['recall']:.4f} nDCG@{args.k}={paper['ndcg']:.4f} | "
            f"official_like R@{args.k}={official['recall']:.4f} nDCG@{args.k}={official['ndcg']:.4f}"
        )


def train_city(city, config: BCEConfig) -> dict:
    set_seed(config.seed)
    device = torch.device(config.device)
    model = VanillaMFBCE(city.num_brands, city.num_regions, config.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCELoss()
    labels = build_labels(city, config.label_smoothing).to(device)
    rng = np.random.default_rng(config.seed)
    brand_ids = np.arange(city.num_brands, dtype=np.int64)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        rng.shuffle(brand_ids)
        losses = []
        for start in range(0, city.num_brands, config.batch_size):
            batch = brand_ids[start : start + config.batch_size]
            batch_t = torch.as_tensor(batch, dtype=torch.long, device=device)
            pred = model(batch_t)
            loss = criterion(pred, labels[batch_t])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            scores = predict_scores(model)
            metrics = evaluate_all_modes(scores, city, config.k)
            row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
            history.append(row)
            valid = metrics["valid"][config.select_metric]
            score = valid["recall"] + valid["ndcg"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += config.eval_every
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict_scores(model)
    final_metrics = evaluate_all_modes(scores, city, config.k)
    return {
        "best_epoch": best_epoch,
        "history": history,
        **final_metrics,
        "config": asdict(config),
        "scores": scores,
        "brand_embeddings": model.brand_embedding.weight.detach().cpu().numpy(),
        "region_embeddings": model.region_embedding.weight.detach().cpu().numpy(),
    }


def build_labels(city, smooth: bool) -> torch.Tensor:
    labels = torch.zeros((city.num_brands, city.num_regions), dtype=torch.float32)
    for brand, regions in city.train_pos.items():
        for region in regions:
            if 0 <= region < city.num_regions:
                labels[brand, region] = 1.0
    if smooth:
        labels = 0.9 * labels + (1.0 / city.num_regions)
    return labels


def evaluate_all_modes(scores: np.ndarray, city, k: int) -> dict:
    return {
        "valid": {
            "paper_strict": all_ranking_metrics(
                scores,
                city.train_pos,
                city.valid_pos,
                k=k,
                drop_train_overlap=True,
            ).as_dict(),
            "official_like": all_ranking_metrics(scores, {}, city.valid_pos, k=k).as_dict(),
        },
        "test": {
            "paper_strict": all_ranking_metrics(
                scores,
                city.train_pos,
                city.test_pos,
                k=k,
                drop_train_overlap=True,
            ).as_dict(),
            "official_like": all_ranking_metrics(scores, {}, city.test_pos, k=k).as_dict(),
        },
    }


def predict_scores(model: VanillaMFBCE) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.score_matrix().detach().cpu().numpy()


def save_result(out_dir: str | Path, city_name: str, result: dict) -> None:
    city_dir = Path(out_dir) / city_name
    city_dir.mkdir(parents=True, exist_ok=True)
    np.save(city_dir / "scores.npy", result["scores"])
    np.save(city_dir / "brand_embeddings.npy", result["brand_embeddings"])
    np.save(city_dir / "region_embeddings.npy", result["region_embeddings"])
    serializable = {k: v for k, v in result.items() if k not in {"scores", "brand_embeddings", "region_embeddings"}}
    (city_dir / "metrics.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
