from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import CityData
from .evaluate import all_ranking_metrics, grouped_metrics
from .models import MFBPR


@dataclass
class TrainConfig:
    embed_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 500
    steps_per_epoch: int | None = None
    patience: int = 30
    eval_every: int = 5
    k: int = 20
    seed: int = 2024
    device: str = "cpu"
    use_semantics: bool = False
    semantic_weight: float = 1.0
    normalize_semantic: bool = False
    structure_weight: float = 0.0
    structure_batch_size: int = 1024
    mask_train_eval: bool = True
    drop_eval_train_overlap: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mf(
    city: CityData,
    config: TrainConfig,
    brand_semantic: np.ndarray | None = None,
    region_semantic: np.ndarray | None = None,
    brand_structure: np.ndarray | None = None,
    region_structure: np.ndarray | None = None,
) -> tuple[MFBPR, dict]:
    set_seed(config.seed)
    device = torch.device(config.device)

    use_sem = config.use_semantics and brand_semantic is not None and region_semantic is not None
    model = MFBPR(
        city.num_brands,
        city.num_regions,
        embed_dim=config.embed_dim,
        brand_semantic_dim=brand_semantic.shape[1] if use_sem else None,
        region_semantic_dim=region_semantic.shape[1] if use_sem else None,
        semantic_weight=config.semantic_weight,
        normalize_semantic=config.normalize_semantic,
    ).to(device)
    brand_sem_tensor = _to_tensor(brand_semantic, device) if use_sem else None
    region_sem_tensor = _to_tensor(region_semantic, device) if use_sem else None
    brand_structure_pairs = _structure_pairs(brand_structure, device)
    region_structure_pairs = _structure_pairs(region_structure, device)

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
            pos_score = model.score_pairs(brands_t, pos_t, brand_sem_tensor, region_sem_tensor)
            neg_score = model.score_pairs(brands_t, neg_t, brand_sem_tensor, region_sem_tensor)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            if config.structure_weight > 0:
                brand_emb, region_emb = model.encode_all(brand_sem_tensor, region_sem_tensor)
                struct_loss = _sample_structure_loss(
                    brand_emb,
                    brand_structure_pairs,
                    config.structure_batch_size,
                    rng,
                    device,
                )
                struct_loss = struct_loss + _sample_structure_loss(
                    region_emb,
                    region_structure_pairs,
                    config.structure_batch_size,
                    rng,
                    device,
                )
                loss = loss + config.structure_weight * struct_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        should_eval = epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs
        if should_eval:
            scores, brand_emb, region_emb = predict_all(model, brand_sem_tensor, region_sem_tensor)
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
            score = valid.ndcg + valid.recall
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_arrays = (scores, brand_emb, region_emb)
            else:
                stale += config.eval_every
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores, brand_emb, region_emb = predict_all(model, brand_sem_tensor, region_sem_tensor)
    result = {
        "best_epoch": best_epoch,
        "history": history,
        "valid": all_ranking_metrics(
            scores,
            city.train_pos if config.mask_train_eval else {},
            city.valid_pos,
            k=config.k,
            drop_train_overlap=config.drop_eval_train_overlap,
        ).as_dict(),
        "test": all_ranking_metrics(
            scores,
            city.train_pos if config.mask_train_eval else {},
            city.test_pos,
            k=config.k,
            drop_train_overlap=config.drop_eval_train_overlap,
        ).as_dict(),
        "groups": grouped_metrics(
            scores,
            city.train_pos if config.mask_train_eval else {},
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


def sample_bpr_batch(city: CityData, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(city.train_edges), size=batch_size)
    batch = city.train_edges[idx]
    brands = batch[:, 0].astype(np.int64)
    pos_regions = batch[:, 1].astype(np.int64)
    neg_regions = np.empty(batch_size, dtype=np.int64)

    for i, brand in enumerate(brands):
        positives = city.train_pos.get(int(brand), set())
        if len(positives) >= city.num_regions:
            neg_regions[i] = int(rng.integers(0, city.num_regions))
            continue
        neg = int(rng.integers(0, city.num_regions))
        while neg in positives:
            neg = int(rng.integers(0, city.num_regions))
        neg_regions[i] = neg
    return brands, pos_regions, neg_regions


def predict_all(
    model: MFBPR,
    brand_semantic: torch.Tensor | None = None,
    region_semantic: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        brand_emb, region_emb = model.encode_all(brand_semantic, region_semantic)
        scores = brand_emb @ region_emb.T
    return (
        scores.detach().cpu().numpy(),
        brand_emb.detach().cpu().numpy(),
        region_emb.detach().cpu().numpy(),
    )


def save_training_artifacts(
    out_dir: str | Path,
    city_name: str,
    model: MFBPR,
    result: dict,
) -> None:
    city_dir = Path(out_dir) / city_name
    city_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), city_dir / "mf.pt")
    np.save(city_dir / "scores.npy", result["scores"])
    np.save(city_dir / "brand_embeddings.npy", result["brand_embeddings"])
    np.save(city_dir / "region_embeddings.npy", result["region_embeddings"])
    serializable = {k: v for k, v in result.items() if k not in {"scores", "brand_embeddings", "region_embeddings"}}
    (city_dir / "metrics.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_tensor(x: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _structure_pairs(
    structure: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if structure is None:
        return None
    arr = np.asarray(structure, dtype=np.float32)
    src, dst = np.nonzero(arr > 0)
    if src.size == 0:
        return None
    weights = arr[src, dst]
    return (
        torch.as_tensor(src, dtype=torch.long, device=device),
        torch.as_tensor(dst, dtype=torch.long, device=device),
        torch.as_tensor(weights, dtype=torch.float32, device=device),
    )


def _sample_structure_loss(
    embeddings: torch.Tensor,
    pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    if pairs is None:
        return embeddings.sum() * 0.0
    src, dst, weights = pairs
    num_pairs = int(weights.shape[0])
    sample_size = max(1, min(batch_size, num_pairs))
    idx_np = rng.integers(0, num_pairs, size=sample_size)
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
    left = F.normalize(embeddings[src[idx]], dim=-1)
    right = F.normalize(embeddings[dst[idx]], dim=-1)
    loss = 1.0 - (left * right).sum(dim=-1)
    return (loss * weights[idx]).mean()
