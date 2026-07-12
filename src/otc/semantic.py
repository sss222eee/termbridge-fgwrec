from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_llm_profiles_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def term_matrix_from_profiles(
    profiles: list[dict],
    id_key: str = "id",
    terms_key: str = "terms",
    vocab: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if vocab is None:
        vocab = sorted({term for row in profiles for term in row.get(terms_key, {}).keys()})
    term2id = {term: idx for idx, term in enumerate(vocab)}
    max_id = max(int(row[id_key]) for row in profiles) if profiles else -1
    matrix = np.zeros((max_id + 1, len(vocab)), dtype=np.float32)
    for row in profiles:
        entity_id = int(row[id_key])
        for term, value in row.get(terms_key, {}).items():
            if term in term2id:
                matrix[entity_id, term2id[term]] = float(value)
    return matrix, vocab


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_topk_structure(x: np.ndarray, topk: int = 20, symmetric: bool = True) -> np.ndarray:
    z = normalize_rows(np.asarray(x, dtype=np.float32))
    sim = z @ z.T
    np.fill_diagonal(sim, 0.0)
    out = np.zeros_like(sim, dtype=np.float32)
    k = min(topk, max(1, sim.shape[1] - 1))
    for i in range(sim.shape[0]):
        idx = np.argpartition(-sim[i], k - 1)[:k]
        out[i, idx] = sim[i, idx]
    if symmetric:
        out = np.maximum(out, out.T)
    return out

