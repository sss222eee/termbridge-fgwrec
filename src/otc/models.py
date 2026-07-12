from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MFBPR(nn.Module):
    def __init__(
        self,
        num_brands: int,
        num_regions: int,
        embed_dim: int = 64,
        brand_semantic_dim: int | None = None,
        region_semantic_dim: int | None = None,
        semantic_weight: float = 1.0,
        normalize_semantic: bool = False,
    ) -> None:
        super().__init__()
        self.brand_embedding = nn.Embedding(num_brands, embed_dim)
        self.region_embedding = nn.Embedding(num_regions, embed_dim)
        self.semantic_weight = float(semantic_weight)
        self.normalize_semantic = bool(normalize_semantic)
        self.brand_semantic = (
            nn.Linear(brand_semantic_dim, embed_dim, bias=False) if brand_semantic_dim else None
        )
        self.region_semantic = (
            nn.Linear(region_semantic_dim, embed_dim, bias=False) if region_semantic_dim else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.brand_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.region_embedding.weight, mean=0.0, std=0.02)
        if self.brand_semantic is not None:
            nn.init.xavier_uniform_(self.brand_semantic.weight)
        if self.region_semantic is not None:
            nn.init.xavier_uniform_(self.region_semantic.weight)

    def encode_all(
        self,
        brand_semantic: torch.Tensor | None = None,
        region_semantic: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        brand = self.brand_embedding.weight
        region = self.region_embedding.weight
        if self.brand_semantic is not None and brand_semantic is not None:
            brand_term = self.brand_semantic(brand_semantic)
            brand = fuse_semantic(brand, brand_term, self.semantic_weight, self.normalize_semantic)
        if self.region_semantic is not None and region_semantic is not None:
            region_term = self.region_semantic(region_semantic)
            region = fuse_semantic(region, region_term, self.semantic_weight, self.normalize_semantic)
        return brand, region

    def score_pairs(
        self,
        brand_ids: torch.Tensor,
        region_ids: torch.Tensor,
        brand_semantic: torch.Tensor | None = None,
        region_semantic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        brand, region = self.encode_all(brand_semantic, region_semantic)
        return (brand[brand_ids] * region[region_ids]).sum(dim=-1)

    def score_matrix(
        self,
        brand_semantic: torch.Tensor | None = None,
        region_semantic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        brand, region = self.encode_all(brand_semantic, region_semantic)
        return brand @ region.T


def fuse_semantic(
    base_embedding: torch.Tensor,
    semantic_embedding: torch.Tensor,
    semantic_weight: float,
    normalize_semantic: bool,
) -> torch.Tensor:
    if normalize_semantic:
        base_embedding = F.normalize(base_embedding, dim=-1)
        semantic_embedding = F.normalize(semantic_embedding, dim=-1)
    return base_embedding + semantic_weight * semantic_embedding
