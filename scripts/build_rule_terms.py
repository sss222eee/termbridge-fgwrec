from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import available_cities, load_city  # noqa: E402
from otc.semantic import cosine_topk_structure, normalize_rows  # noqa: E402


TERM_VOCAB = ["high_traffic", "daily_need", "shopping", "office", "residential"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule-based term features for TermStruct-MF.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--raw-root", default="data/raw/OpenSiteRec")
    parser.add_argument("--out", default="data/term_features_rule")
    parser.add_argument("--city", default="all")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    cities = available_cities(args.data_root) if args.city == "all" else [c.strip() for c in args.city.split(",")]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for city_name in cities:
        city = load_city(args.data_root, city_name)
        raw_path = Path(args.raw_root) / city_name / f"{city_name}_KG_plus.csv"
        if not raw_path.exists():
            raise SystemExit(f"Missing raw CSV for {city_name}: {raw_path}")
        raw = pd.read_csv(raw_path)
        brand_terms = build_brand_terms(city.brand2id, raw)
        region_terms = build_region_terms(city.train_edges, brand_terms, city.num_regions)
        brand_structure = cosine_topk_structure(brand_terms, topk=args.topk, symmetric=True)
        region_structure = cosine_topk_structure(region_terms, topk=args.topk, symmetric=True)

        city_dir = out_root / city_name
        city_dir.mkdir(parents=True, exist_ok=True)
        np.save(city_dir / "brand_terms.npy", brand_terms)
        np.save(city_dir / "region_terms.npy", region_terms)
        np.save(city_dir / "brand_structure.npy", brand_structure)
        np.save(city_dir / "region_structure.npy", region_structure)
        (city_dir / "term_vocab.json").write_text(json.dumps(TERM_VOCAB, indent=2), encoding="utf-8")
        print(
            f"{city_name}: brand_terms={brand_terms.shape} region_terms={region_terms.shape} "
            f"brand_edges={int(np.count_nonzero(brand_structure))} "
            f"region_edges={int(np.count_nonzero(region_structure))}"
        )


def build_brand_terms(brand2id: dict[str, int], raw: pd.DataFrame) -> np.ndarray:
    terms = np.zeros((len(brand2id), len(TERM_VOCAB)), dtype=np.float32)
    counts = np.zeros(len(brand2id), dtype=np.float32)
    for _, row in raw.iterrows():
        brand = str(row.get("Brand", ""))
        brand_id = brand2id.get(brand)
        if brand_id is None:
            continue
        terms[brand_id] += row_term_vector(row)
        counts[brand_id] += 1.0
    terms = terms / np.maximum(counts[:, None], 1.0)
    zero_rows = np.where(np.linalg.norm(terms, axis=1) == 0)[0]
    if zero_rows.size:
        terms[zero_rows] = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    return normalize_rows(terms).astype(np.float32)


def build_region_terms(train_edges: np.ndarray, brand_terms: np.ndarray, num_regions: int) -> np.ndarray:
    terms = np.zeros((num_regions, brand_terms.shape[1]), dtype=np.float32)
    counts = np.zeros(num_regions, dtype=np.float32)
    for brand, region in train_edges:
        region = int(region)
        if 0 <= region < num_regions:
            terms[region] += brand_terms[int(brand)]
            counts[region] += 1.0
    terms = terms / np.maximum(counts[:, None], 1.0)
    nonzero = counts > 0
    if np.any(nonzero):
        fallback = terms[nonzero].mean(axis=0)
    else:
        fallback = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    terms[~nonzero] = fallback
    return normalize_rows(terms).astype(np.float32)


def row_term_vector(row: pd.Series) -> np.ndarray:
    text = " ".join(
        str(row.get(col, "")).lower()
        for col in ["Name", "Brand", "cate_1", "cate_2", "cate_3"]
        if not pd.isna(row.get(col, ""))
    )
    vec = np.zeros(len(TERM_VOCAB), dtype=np.float32)

    if any_word(text, ["coffee", "cafe", "tea"]):
        vec = np.maximum(vec, [1.0, 0.5, 0.4, 1.0, 0.2])
    if any_word(text, ["fast_food", "restaurant", "cuisine", "burger", "pizza", "chicken", "food"]):
        vec = np.maximum(vec, [1.0, 1.0, 0.4, 0.6, 0.5])
    if any_word(text, ["convenience", "general_store", "supermarket", "grocery", "pharmacy", "market"]):
        vec = np.maximum(vec, [0.8, 1.0, 0.3, 0.5, 0.9])
    if any_word(text, ["fashion", "clothes", "clothing", "shoes", "footwear", "beauty", "cosmetics"]):
        vec = np.maximum(vec, [0.9, 0.2, 1.0, 0.2, 0.2])
    if any_word(text, ["shopping", "mall", "retail", "store"]):
        vec = np.maximum(vec, [0.8, 0.4, 0.9, 0.3, 0.4])
    if any_word(text, ["bank", "finance", "postal", "office", "telecom", "mobile"]):
        vec = np.maximum(vec, [0.6, 0.7, 0.2, 0.9, 0.5])
    if any_word(text, ["transport", "station", "parking", "gas", "fuel"]):
        vec = np.maximum(vec, [0.8, 0.5, 0.2, 0.5, 0.4])
    if any_word(text, ["hotel", "fitness", "gym", "health", "school", "library"]):
        vec = np.maximum(vec, [0.7, 0.5, 0.3, 0.5, 0.6])

    if not np.any(vec):
        vec[:] = [0.5, 0.5, 0.5, 0.5, 0.5]
    return vec


def any_word(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


if __name__ == "__main__":
    main()
