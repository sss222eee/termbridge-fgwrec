from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


EDGE_SPLIT_RE = re.compile(r"[\s,\t]+")


@dataclass
class CityData:
    name: str
    num_brands: int
    num_regions: int
    train_edges: np.ndarray
    valid_edges: np.ndarray
    test_edges: np.ndarray
    brand2id: dict[str, int] = field(default_factory=dict)
    region2id: dict[str, int] = field(default_factory=dict)
    train_pos: dict[int, set[int]] = field(init=False)
    valid_pos: dict[int, set[int]] = field(init=False)
    test_pos: dict[int, set[int]] = field(init=False)
    all_pos: dict[int, set[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.train_edges = _ensure_edges(self.train_edges)
        self.valid_edges = _ensure_edges(self.valid_edges)
        self.test_edges = _ensure_edges(self.test_edges)
        self.train_pos = edges_to_pos(self.train_edges)
        self.valid_pos = edges_to_pos(self.valid_edges)
        self.test_pos = edges_to_pos(self.test_edges)
        self.all_pos = merge_pos(self.train_pos, self.valid_pos, self.test_pos)

    @property
    def train_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_brands, dtype=np.int64)
        for brand, regions in self.train_pos.items():
            counts[brand] = len(regions)
        return counts


def _ensure_edges(edges: np.ndarray | Iterable[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return arr.reshape(-1, 2)


def edges_to_pos(edges: np.ndarray) -> dict[int, set[int]]:
    pos: dict[int, set[int]] = {}
    for brand, region in _ensure_edges(edges):
        pos.setdefault(int(brand), set()).add(int(region))
    return pos


def merge_pos(*parts: dict[int, set[int]]) -> dict[int, set[int]]:
    merged: dict[int, set[int]] = {}
    for part in parts:
        for brand, regions in part.items():
            merged.setdefault(brand, set()).update(regions)
    return merged


def available_cities(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "train.txt").exists())


def read_edge_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        return np.empty((0, 2), dtype=np.int64)

    edges: list[tuple[int, int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in EDGE_SPLIT_RE.split(line) if p]
        if len(parts) < 2:
            continue
        if parts[0].lower() in {"brand", "brand_id", "user", "user_id"}:
            continue
        edges.append((int(parts[0]), int(parts[1])))
    return _ensure_edges(edges)


def write_edge_file(path: str | Path, edges: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _ensure_edges(edges)
    lines = [f"{int(b)} {int(r)}" for b, r in arr]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_city(data_root: str | Path, city: str) -> CityData:
    city_dir = Path(data_root) / city
    train = read_edge_file(city_dir / "train.txt")
    valid = read_edge_file(city_dir / "valid.txt")
    test = read_edge_file(city_dir / "test.txt")

    brand2id = _read_json(city_dir / "brand2id.json")
    region2id = _read_json(city_dir / "region2id.json")
    meta = _read_json(city_dir / "meta.json")

    all_edges = np.vstack([x for x in [train, valid, test] if x.size > 0]) if any(
        x.size > 0 for x in [train, valid, test]
    ) else np.empty((0, 2), dtype=np.int64)
    inferred_brands = int(all_edges[:, 0].max() + 1) if all_edges.size else 0
    inferred_regions = int(all_edges[:, 1].max() + 1) if all_edges.size else 0
    num_brands = int(meta.get("num_brands") or len(brand2id) or inferred_brands)
    num_regions = int(meta.get("num_regions") or len(region2id) or inferred_regions)

    return CityData(
        name=city,
        num_brands=num_brands,
        num_regions=num_regions,
        train_edges=train,
        valid_edges=valid,
        test_edges=test,
        brand2id=brand2id,
        region2id=region2id,
    )


def load_cities(data_root: str | Path, cities: Iterable[str] | None = None) -> dict[str, CityData]:
    names = list(cities) if cities is not None else available_cities(data_root)
    return {city: load_city(data_root, city) for city in names}


def write_city(data_root: str | Path, city: CityData) -> None:
    city_dir = Path(data_root) / city.name
    city_dir.mkdir(parents=True, exist_ok=True)
    write_edge_file(city_dir / "train.txt", city.train_edges)
    write_edge_file(city_dir / "valid.txt", city.valid_edges)
    write_edge_file(city_dir / "test.txt", city.test_edges)
    (city_dir / "brand2id.json").write_text(
        json.dumps(city.brand2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (city_dir / "region2id.json").write_text(
        json.dumps(city.region2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (city_dir / "meta.json").write_text(
        json.dumps(
            {"num_brands": city.num_brands, "num_regions": city.num_regions},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def split_edges_by_brand(
    edges: np.ndarray,
    seed: int = 2024,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    min_brand_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_brand = edges_to_pos(edges)
    train: list[tuple[int, int]] = []
    valid: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []

    for brand, region_set in sorted(by_brand.items()):
        regions = np.array(sorted(region_set), dtype=np.int64)
        if len(regions) < min_brand_degree:
            continue
        rng.shuffle(regions)
        n = len(regions)
        n_train = max(1, int(math.floor(n * train_ratio)))
        n_valid = max(1, int(math.floor(n * valid_ratio)))
        if n_train + n_valid >= n:
            n_train = max(1, n - 2)
            n_valid = 1
        train.extend((brand, int(r)) for r in regions[:n_train])
        valid.extend((brand, int(r)) for r in regions[n_train : n_train + n_valid])
        test.extend((brand, int(r)) for r in regions[n_train + n_valid :])

    return _ensure_edges(train), _ensure_edges(valid), _ensure_edges(test)


def split_poi_edges_by_brand(
    edges: np.ndarray,
    seed: int = 2024,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    min_brand_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw POI rows by brand, preserving repeated brand-region pairs."""
    rng = np.random.default_rng(seed)
    by_brand: dict[int, list[int]] = {}
    for brand, region in _ensure_edges(edges):
        by_brand.setdefault(int(brand), []).append(int(region))

    train: list[tuple[int, int]] = []
    valid: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []

    for brand, region_list in sorted(by_brand.items()):
        regions = np.asarray(region_list, dtype=np.int64)
        if len(regions) < min_brand_degree:
            continue
        rng.shuffle(regions)
        n = len(regions)
        n_train = max(1, int(math.floor(n * train_ratio)))
        n_valid = max(1, int(math.floor(n * valid_ratio)))
        if n_train + n_valid >= n:
            n_train = max(1, n - 2)
            n_valid = 1
        train.extend((brand, int(r)) for r in regions[:n_train])
        valid.extend((brand, int(r)) for r in regions[n_train : n_train + n_valid])
        test.extend((brand, int(r)) for r in regions[n_train + n_valid :])

    return _ensure_edges(train), _ensure_edges(valid), _ensure_edges(test)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_npy(data_root: str | Path, city: str, name: str) -> np.ndarray | None:
    path = Path(data_root) / city / name
    if not path.exists():
        return None
    return np.load(path)
