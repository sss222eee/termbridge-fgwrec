from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import CityData, split_edges_by_brand, split_poi_edges_by_brand, write_city  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a city/brand/region POI table into processed OTC edge splits."
    )
    parser.add_argument("--input", required=True, help="CSV/TSV file containing POI or brand-region rows")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--city-col", default="city")
    parser.add_argument("--brand-col", default="brand")
    parser.add_argument("--region-col", default="region")
    parser.add_argument("--city-name", default=None, help="Use when the input has no city column")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--min-brand-degree", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-mode",
        choices=["unique_pairs", "raw_poi"],
        default="unique_pairs",
        help="unique_pairs deduplicates brand-region before split; raw_poi splits original POI rows.",
    )
    parser.add_argument(
        "--preserve-region-ids",
        action="store_true",
        help="Keep integer region IDs in their original ID space instead of compacting observed regions.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=None, engine="python")
    if args.city_name is not None:
        df[args.city_col] = args.city_name
    required = [args.city_col, args.brand_col, args.region_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    df = df[required].dropna()
    if args.split_mode == "unique_pairs":
        df = df.drop_duplicates()
    out = Path(args.out)
    for city_idx, (city_name, city_df) in enumerate(df.groupby(args.city_col)):
        brands = sorted(str(x) for x in city_df[args.brand_col].unique())
        raw_brand2id = {brand: idx for idx, brand in enumerate(brands)}
        if args.preserve_region_ids:
            region_values = {
                str(raw): _as_non_negative_int(raw, args.region_col) for raw in city_df[args.region_col].unique()
            }
            regions = sorted(region_values, key=lambda x: region_values[x])
            raw_region2id = region_values
        else:
            regions = sorted(str(x) for x in city_df[args.region_col].unique())
            raw_region2id = {region: idx for idx, region in enumerate(regions)}
        edges = np.asarray(
            [
                (raw_brand2id[str(row[args.brand_col])], raw_region2id[str(row[args.region_col])])
                for _, row in city_df.iterrows()
            ],
            dtype=np.int64,
        )
        split_fn = split_poi_edges_by_brand if args.split_mode == "raw_poi" else split_edges_by_brand
        train, valid, test = split_fn(
            edges,
            seed=args.seed + city_idx,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            min_brand_degree=args.min_brand_degree,
        )
        non_empty_splits = [part for part in [train, valid, test] if part.size > 0]
        all_split = np.vstack(non_empty_splits) if non_empty_splits else np.empty((0, 2), dtype=np.int64)
        kept_brand_ids = sorted(set(all_split[:, 0].tolist())) if all_split.size else []
        brand_id_map = {old: new for new, old in enumerate(kept_brand_ids)}
        if args.preserve_region_ids:
            region_id_map = {region_id: region_id for region_id in raw_region2id.values()}
        else:
            kept_region_ids = sorted(set(all_split[:, 1].tolist())) if all_split.size else []
            region_id_map = {old: new for new, old in enumerate(kept_region_ids)}

        def remap(split: np.ndarray) -> np.ndarray:
            if split.size == 0:
                return split
            return np.asarray(
                [(brand_id_map[int(b)], region_id_map[int(r)]) for b, r in split],
                dtype=np.int64,
            )

        train = remap(train)
        valid = remap(valid)
        test = remap(test)
        id2brand = {idx: brand for brand, idx in raw_brand2id.items()}
        id2region = {idx: region for region, idx in raw_region2id.items()}
        brand2id = {id2brand[old]: new for old, new in brand_id_map.items()}
        region2id = {id2region[old]: new for old, new in region_id_map.items()}
        num_regions = max(region_id_map.values()) + 1 if region_id_map else 0
        city = CityData(
            name=str(city_name).replace(" ", "_"),
            num_brands=len(brand2id),
            num_regions=num_regions,
            train_edges=train,
            valid_edges=valid,
            test_edges=test,
            brand2id=brand2id,
            region2id=region2id,
        )
        write_city(out, city)
        print(
            f"{city.name}: split_mode={args.split_mode} raw_edges={len(edges)} "
            f"raw_brands={len(brands)} raw_regions={len(regions)} "
            f"kept_brands={len(brand2id)} kept_regions={len(region2id)} num_regions={num_regions} "
            f"train={len(train)} valid={len(valid)} test={len(test)}"
        )

def _as_non_negative_int(value: object, column_name: str) -> int:
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--preserve-region-ids requires numeric {column_name}; got {value!r}") from exc
    as_int = int(as_float)
    if as_int < 0 or abs(as_float - as_int) > 1e-9:
        raise SystemExit(f"--preserve-region-ids requires non-negative integer {column_name}; got {value!r}")
    return as_int


if __name__ == "__main__":
    main()
