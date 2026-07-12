from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otc.data import CityData, write_city  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the official OpenSiteRec split to otc_baseline format.")
    parser.add_argument("--official-root", default="../official_OpenSiteRec")
    parser.add_argument("--out", default="data/official_split")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Official train_test_split random_state.")
    parser.add_argument("--cities", nargs="+", default=["Chicago", "NYC", "Singapore", "Tokyo"])
    args = parser.parse_args()

    official_root = Path(args.official_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for city_name in args.cities:
        mapped, brand2id, category_maps = build_official_mapping(official_root, city_name, args.threshold)
        split_dir = official_root / city_name / "split"
        train_path = split_dir / "train.pkl"
        test_path = split_dir / "test.pkl"
        if train_path.exists() and test_path.exists():
            train_df = pd.read_pickle(train_path)
            test_df = pd.read_pickle(test_path)
        else:
            split_dir.mkdir(parents=True, exist_ok=True)
            train_df, test_df = split_like_official(mapped, args.seed)
            train_df.to_pickle(train_path)
            test_df.to_pickle(test_path)

        train_edges = train_df[["Brand_ID", "Region_ID"]].to_numpy(dtype=np.int64)
        test_edges = test_df[["Brand_ID", "Region_ID"]].to_numpy(dtype=np.int64)
        num_brands = int(max(train_df["Brand_ID"].max(), test_df["Brand_ID"].max()) + 1)
        num_regions = int(max(train_df["Region_ID"].max(), test_df["Region_ID"].max()) + 1)
        region2id = {str(i): i for i in range(num_regions)}
        city = CityData(
            name=city_name,
            num_brands=num_brands,
            num_regions=num_regions,
            train_edges=train_edges,
            valid_edges=np.empty((0, 2), dtype=np.int64),
            test_edges=test_edges,
            brand2id=brand2id,
            region2id=region2id,
        )
        write_city(out_root, city)

        city_out = out_root / city_name
        train_df.to_csv(city_out / "official_train.csv", index=False)
        test_df.to_csv(city_out / "official_test.csv", index=False)
        (city_out / "category_maps.json").write_text(
            json.dumps(category_maps, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (city_out / "official_test_lists.json").write_text(
            json.dumps(build_test_lists(test_df), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{city_name}: brands={num_brands} regions={num_regions} "
            f"train_rows={len(train_edges)} test_rows={len(test_edges)}",
            flush=True,
        )


def build_official_mapping(
    official_root: Path,
    city_name: str,
    threshold: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, dict[str, int]]]:
    df = pd.read_pickle(official_root / city_name / f"{city_name}_KG_plus.pkl")
    bvc = df["Brand"].value_counts() >= threshold
    kept = bvc[bvc > 0].index
    df = df[df["Brand"].isin(kept)].reset_index(drop=True)

    brand2id: dict[str, int] = {}
    cate12id: dict[str, int] = {}
    cate22id: dict[str, int] = {}
    cate32id: dict[str, int] = {}
    for _, row in df.iterrows():
        brand = str(row["Brand"])
        cate_1 = str(row["cate_1"])
        cate_2 = str(row["cate_2"])
        cate_3 = str(row["cate_3"])
        if brand not in brand2id:
            brand2id[brand] = len(brand2id)
        if cate_1 not in cate12id:
            cate12id[cate_1] = len(cate12id)
        if cate_2 not in cate22id:
            cate22id[cate_2] = len(cate22id)
        if cate_3 not in cate32id:
            cate32id[cate_3] = len(cate32id)

    mapped = df.copy()
    mapped["Brand_ID"] = mapped["Brand"].astype(str).map(brand2id)
    mapped["Cate1_ID"] = mapped["cate_1"].astype(str).map(cate12id)
    mapped["Cate2_ID"] = mapped["cate_2"].astype(str).map(cate22id)
    mapped["Cate3_ID"] = mapped["cate_3"].astype(str).map(cate32id)
    mapped = mapped[["ID", "Name", "Brand", "cate_1", "cate_2", "cate_3", "Brand_ID", "Cate1_ID", "Cate2_ID", "Cate3_ID", "Region_ID"]]
    category_maps = {"cate_1": cate12id, "cate_2": cate22id, "cate_3": cate32id}
    return mapped, brand2id, category_maps


def split_like_official(mapped: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for brand_id in range(int(mapped["Brand_ID"].max()) + 1):
        data = mapped[mapped["Brand_ID"] == brand_id]
        x_train, x_test, y_train, y_test = train_test_split(
            data[["Brand_ID", "Cate1_ID", "Cate2_ID", "Cate3_ID"]],
            data["Region_ID"],
            test_size=0.2,
            random_state=seed,
        )
        x_train["Region_ID"] = y_train
        x_test["Region_ID"] = y_test
        train_parts.append(x_train)
        test_parts.append(x_test)
    return pd.concat(train_parts, axis=0), pd.concat(test_parts, axis=0)


def build_test_lists(test_df: pd.DataFrame) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for _, row in test_df.iterrows():
        brand = str(int(row["Brand_ID"]))
        out.setdefault(brand, []).append(int(row["Region_ID"]))
    return out


if __name__ == "__main__":
    main()
