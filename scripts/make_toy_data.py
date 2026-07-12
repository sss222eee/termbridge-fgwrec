from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import CityData, split_edges_by_brand, write_city  # noqa: E402
from otc.semantic import cosine_topk_structure, normalize_rows  # noqa: E402


CITY_SPECS = {
    "Chicago": (72, 64),
    "NYC": (96, 82),
    "Singapore": (84, 76),
    "Tokyo": (108, 90),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--semantic-dim", type=int, default=12)
    parser.add_argument("--min-pos", type=int, default=6)
    parser.add_argument("--max-pos", type=int, default=12)
    args = parser.parse_args()

    out = Path(args.out) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for city_idx, (city, (num_brands, num_regions)) in enumerate(CITY_SPECS.items()):
        city_shift = rng.normal(scale=0.25, size=(args.latent_dim,))
        brand_latent = rng.normal(size=(num_brands, args.latent_dim)) + city_shift
        region_latent = rng.normal(size=(num_regions, args.latent_dim)) + 0.5 * city_shift
        score = brand_latent @ region_latent.T / np.sqrt(args.latent_dim)
        score += rng.normal(scale=0.2, size=score.shape)

        edges: list[tuple[int, int]] = []
        for brand in range(num_brands):
            count = int(rng.integers(args.min_pos, args.max_pos + 1))
            top = np.argsort(-score[brand])[: max(args.max_pos * 2, count)]
            chosen = rng.choice(top, size=count, replace=False)
            edges.extend((brand, int(region)) for region in chosen)

        train, valid, test = split_edges_by_brand(np.asarray(edges, dtype=np.int64), seed=args.seed + city_idx)
        city_data = CityData(
            name=city,
            num_brands=num_brands,
            num_regions=num_regions,
            train_edges=train,
            valid_edges=valid,
            test_edges=test,
            brand2id={f"{city}_brand_{i}": i for i in range(num_brands)},
            region2id={f"{city}_region_{i}": i for i in range(num_regions)},
        )
        write_city(out, city_data)

        brand_sem = make_semantic(brand_latent, args.semantic_dim, rng)
        region_sem = make_semantic(region_latent, args.semantic_dim, rng)
        city_dir = out / city
        np.save(city_dir / "brand_semantic.npy", brand_sem)
        np.save(city_dir / "region_semantic.npy", region_sem)
        np.save(city_dir / "brand_structure.npy", cosine_topk_structure(brand_sem, topk=10))
        np.save(city_dir / "region_structure.npy", cosine_topk_structure(region_sem, topk=10))
        print(f"{city}: train={len(train)} valid={len(valid)} test={len(test)}")

    print(f"Toy data written to {out}")


def make_semantic(latent: np.ndarray, dim: int, rng: np.random.Generator) -> np.ndarray:
    projection = rng.normal(size=(latent.shape[1], dim))
    semantic = latent @ projection + rng.normal(scale=0.1, size=(latent.shape[0], dim))
    return normalize_rows(semantic).astype(np.float32)


if __name__ == "__main__":
    main()

