from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import CityData
from .evaluate import all_ranking_metrics, grouped_metrics
from .train import sample_bpr_batch, set_seed


@dataclass
class LightGCNConfig:
    embed_dim: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 500
    steps_per_epoch: int | None = None
    patience: int = 50
    eval_every: int = 5
    k: int = 20
    seed: int = 2024
    device: str = "cpu"
    mask_train_eval: bool = True
    drop_eval_train_overlap: bool = True
    init_std: float = 0.1
    graph_weighting: str = "binary"


class LightGCN(nn.Module):
    def __init__(
        self,
        num_brands: int,
        num_regions: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_brands = int(num_brands)
        self.num_regions = int(num_regions)
        self.num_layers = int(num_layers)
        self.brand_embedding = nn.Embedding(num_brands, embed_dim)
        self.region_embedding = nn.Embedding(num_regions, embed_dim)
        self.init_std = float(init_std)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.brand_embedding.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.region_embedding.weight, mean=0.0, std=self.init_std)

    def encode_all(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.cat([self.brand_embedding.weight, self.region_embedding.weight], dim=0)
        layer_outputs = [emb]
        for _ in range(self.num_layers):
            emb = torch.sparse.mm(norm_adj, emb)
            layer_outputs.append(emb)
        final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return final[: self.num_brands], final[self.num_brands :]


def build_lightgcn_adj(city: CityData, device: torch.device, weighting: str = "binary") -> torch.Tensor:
    num_nodes = city.num_brands + city.num_regions
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []

    if weighting == "binary":
        edge_iter = ((brand, region, 1.0) for brand, regions in city.train_pos.items() for region in regions)
    elif weighting in {"count", "sqrt_count"}:
        counts: dict[tuple[int, int], int] = {}
        for brand, region in city.train_edges:
            key = (int(brand), int(region))
            counts[key] = counts.get(key, 0) + 1
        edge_iter = (
            (brand, region, float(np.sqrt(count) if weighting == "sqrt_count" else count))
            for (brand, region), count in counts.items()
        )
    else:
        raise ValueError(f"Unsupported LightGCN graph weighting: {weighting}")

    for brand, region, weight in edge_iter:
        brand_id = int(brand)
        region_id = city.num_brands + int(region)
        src.extend([brand_id, region_id])
        dst.extend([region_id, brand_id])
        weights.extend([weight, weight])

    if not src:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()

    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    weight_arr = np.asarray(weights, dtype=np.float32)
    degree = np.bincount(src_arr, weights=weight_arr, minlength=num_nodes).astype(np.float32)
    degree[degree == 0.0] = 1.0
    values_arr = weight_arr / np.sqrt(degree[src_arr] * degree[dst_arr])

    indices = torch.as_tensor(np.vstack([src_arr, dst_arr]), dtype=torch.long, device=device)
    values = torch.as_tensor(values_arr, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()


def train_lightgcn(city: CityData, config: LightGCNConfig) -> tuple[LightGCN, dict]:
    set_seed(config.seed)
    device = torch.device(config.device)
    norm_adj = build_lightgcn_adj(city, device, weighting=config.graph_weighting)
    model = LightGCN(
        city.num_brands,
        city.num_regions,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        init_std=config.init_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    steps = config.steps_per_epoch or max(1, int(np.ceil(len(city.train_edges) / config.batch_size)))

    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for _ in range(steps):
            brands, pos_regions, neg_regions = sample_bpr_batch(city, config.batch_size, rng)
            brands_t = torch.as_tensor(brands, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos_regions, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg_regions, dtype=torch.long, device=device)
            brand_emb, region_emb = model.encode_all(norm_adj)
            pos_score = (brand_emb[brands_t] * region_emb[pos_t]).sum(dim=-1)
            neg_score = (brand_emb[brands_t] * region_emb[neg_t]).sum(dim=-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        should_eval = epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs
        if should_eval:
            scores, _, _ = predict_all_lightgcn(model, norm_adj)
            eval_train_pos = city.train_pos if config.mask_train_eval else {}
            valid = all_ranking_metrics(
                scores,
                eval_train_pos,
                city.valid_pos,
                k=config.k,
                drop_train_overlap=config.drop_eval_train_overlap,
            )
            test = all_ranking_metrics(
                scores,
                eval_train_pos,
                city.test_pos,
                k=config.k,
                drop_train_overlap=config.drop_eval_train_overlap,
            )
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "valid": valid.as_dict(),
                "test": test.as_dict(),
            }
            history.append(row)
            score = valid.recall + valid.ndcg
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
    scores, brand_emb, region_emb = predict_all_lightgcn(model, norm_adj)
    eval_train_pos = city.train_pos if config.mask_train_eval else {}
    result = {
        "best_epoch": best_epoch,
        "history": history,
        "valid": all_ranking_metrics(
            scores,
            eval_train_pos,
            city.valid_pos,
            k=config.k,
            drop_train_overlap=config.drop_eval_train_overlap,
        ).as_dict(),
        "test": all_ranking_metrics(
            scores,
            eval_train_pos,
            city.test_pos,
            k=config.k,
            drop_train_overlap=config.drop_eval_train_overlap,
        ).as_dict(),
        "groups": grouped_metrics(
            scores,
            eval_train_pos,
            city.test_pos,
            city.train_counts,
            k=config.k,
            drop_train_overlap=config.drop_eval_train_overlap,
        ),
        "config": asdict(config),
        "scores": scores,
        "brand_embeddings": brand_emb,
        "region_embeddings": region_emb,
    }
    return model, result


def predict_all_lightgcn(
    model: LightGCN,
    norm_adj: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        brand_emb, region_emb = model.encode_all(norm_adj)
        scores = brand_emb @ region_emb.T
    return (
        scores.detach().cpu().numpy(),
        brand_emb.detach().cpu().numpy(),
        region_emb.detach().cpu().numpy(),
    )


def save_lightgcn_artifacts(out_dir: str | Path, city_name: str, model: LightGCN, result: dict) -> None:
    city_dir = Path(out_dir) / city_name
    city_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), city_dir / "lightgcn.pt")
    np.save(city_dir / "scores.npy", result["scores"])
    np.save(city_dir / "brand_embeddings.npy", result["brand_embeddings"])
    np.save(city_dir / "region_embeddings.npy", result["region_embeddings"])
    serializable = {k: v for k, v in result.items() if k not in {"scores", "brand_embeddings", "region_embeddings"}}
    (city_dir / "metrics.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
