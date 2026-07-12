from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransportResult:
    plan: np.ndarray
    distance: float
    confidence: float
    solver: str


def pairwise_euclidean(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    dist2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.sqrt(dist2 + 1e-12)


def normalize_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return np.zeros_like(cost)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo < 1e-12:
        return np.zeros_like(cost)
    return (cost - lo) / (hi - lo)


def structural_signature(x: np.ndarray, bins: int = 16) -> np.ndarray:
    dist = pairwise_euclidean(x)
    qs = np.linspace(0.0, 1.0, bins + 2)[1:-1]
    return np.quantile(dist, qs, axis=1).T


def sinkhorn_uniform(cost: np.ndarray, reg: float = 0.05, max_iter: int = 200) -> np.ndarray:
    cost = normalize_cost(cost)
    n, m = cost.shape
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)
    kernel = np.exp(-cost / max(reg, 1e-6))
    kernel = np.maximum(kernel, 1e-300)
    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)
    for _ in range(max_iter):
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
    plan = (u[:, None] * kernel) * v[None, :]
    return plan / np.maximum(plan.sum(), 1e-300)


def transport_marginal_error(
    plan: np.ndarray,
    row_marginal: np.ndarray,
    col_marginal: np.ndarray,
) -> float:
    plan = np.asarray(plan, dtype=np.float64)
    if plan.shape != (row_marginal.size, col_marginal.size):
        return float("inf")
    if not np.all(np.isfinite(plan)) or np.min(plan) < -1e-12:
        return float("inf")
    plan = np.maximum(plan, 0.0)
    row_error = np.max(np.abs(plan.sum(axis=1) - row_marginal))
    col_error = np.max(np.abs(plan.sum(axis=0) - col_marginal))
    mass_error = abs(float(plan.sum()) - 1.0)
    return float(max(row_error, col_error, mass_error))


def balance_transport_plan(
    plan: np.ndarray,
    row_marginal: np.ndarray,
    col_marginal: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float64)
    if plan.shape != (row_marginal.size, col_marginal.size):
        raise ValueError(
            f"transport plan shape {plan.shape} does not match "
            f"marginals {(row_marginal.size, col_marginal.size)}"
        )

    balanced = np.where(np.isfinite(plan) & (plan > 0.0), plan, 0.0)
    support_floor = max(float(np.max(balanced)) * 1e-12, 1e-12 / balanced.size, 1e-300)
    balanced = np.maximum(balanced, support_floor)
    for _ in range(max(1, max_iter)):
        balanced *= (row_marginal / np.maximum(balanced.sum(axis=1), 1e-300))[:, None]
        balanced *= (col_marginal / np.maximum(balanced.sum(axis=0), 1e-300))[None, :]
        if transport_marginal_error(balanced, row_marginal, col_marginal) <= tol:
            break
    return balanced / np.maximum(float(balanced.sum()), 1e-300)


def transport_confidence(plan: np.ndarray) -> float:
    flat = np.asarray(plan, dtype=np.float64).ravel()
    flat = flat[flat > 0]
    if flat.size <= 1:
        return 1.0
    entropy = -float(np.sum(flat * np.log(flat)))
    max_entropy = np.log(plan.size)
    return float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))


def compute_transport(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    source_semantic: np.ndarray | None = None,
    target_semantic: np.ndarray | None = None,
    method: str = "gw",
    fgw_alpha: float = 0.5,
    reg: float = 0.05,
    max_iter: int = 200,
    emd_max_iter: int = 1_000_000,
    tol: float = 1e-9,
) -> TransportResult:
    method = method.lower()
    source_embedding = np.asarray(source_embedding, dtype=np.float64)
    target_embedding = np.asarray(target_embedding, dtype=np.float64)
    source_structure = normalize_cost(pairwise_euclidean(source_embedding))
    target_structure = normalize_cost(pairwise_euclidean(target_embedding))

    pot_result = _try_pot_transport(
        source_structure,
        target_structure,
        source_semantic,
        target_semantic,
        method=method,
        fgw_alpha=fgw_alpha,
        max_iter=max_iter,
        emd_max_iter=emd_max_iter,
        tol=tol,
    )
    if pot_result is not None:
        return pot_result

    sig_source = structural_signature(source_embedding)
    sig_target = structural_signature(target_embedding)
    structure_cost = normalize_cost(pairwise_euclidean(sig_source, sig_target))
    if method == "fgw" and source_semantic is not None and target_semantic is not None:
        semantic_cost = normalize_cost(pairwise_euclidean(source_semantic, target_semantic))
        cost = fgw_alpha * structure_cost + (1.0 - fgw_alpha) * semantic_cost
        solver = "fallback-signature-fgw-sinkhorn"
    else:
        cost = structure_cost
        solver = "fallback-signature-gw-sinkhorn"

    plan = sinkhorn_uniform(cost, reg=reg, max_iter=max_iter)
    distance = float(np.sum(plan * cost))
    return TransportResult(plan=plan, distance=distance, confidence=transport_confidence(plan), solver=solver)


def _try_pot_transport(
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    source_semantic: np.ndarray | None,
    target_semantic: np.ndarray | None,
    method: str,
    fgw_alpha: float,
    max_iter: int,
    emd_max_iter: int,
    tol: float,
) -> TransportResult | None:
    try:
        import ot  # type: ignore
    except Exception:
        return None

    p = np.full(source_structure.shape[0], 1.0 / source_structure.shape[0], dtype=np.float64)
    q = np.full(target_structure.shape[0], 1.0 / target_structure.shape[0], dtype=np.float64)
    try:
        if method == "fgw" and source_semantic is not None and target_semantic is not None:
            feature_cost = normalize_cost(pairwise_euclidean(source_semantic, target_semantic))
            plan, log = ot.gromov.fused_gromov_wasserstein(
                feature_cost,
                source_structure,
                target_structure,
                p,
                q,
                loss_fun="square_loss",
                alpha=fgw_alpha,
                log=True,
                max_iter=max_iter,
                numItermaxEmd=emd_max_iter,
                tol_rel=tol,
                tol_abs=tol,
            )
            distance = float(log.get("fgw_dist", np.sum(plan * feature_cost)))
            solver = "pot-fgw"
        else:
            plan, log = ot.gromov.gromov_wasserstein(
                source_structure,
                target_structure,
                p,
                q,
                loss_fun="square_loss",
                log=True,
                max_iter=max_iter,
                numItermaxEmd=emd_max_iter,
                tol_rel=tol,
                tol_abs=tol,
            )
            distance = float(log.get("gw_dist", 0.0))
            solver = "pot-gw"
        plan, solver = _ensure_valid_plan(plan, p, q, solver, max_iter=max_iter, tol=tol)
    except Exception:
        return None
    return TransportResult(plan=plan, distance=distance, confidence=transport_confidence(plan), solver=solver)


def _ensure_valid_plan(
    plan: np.ndarray,
    row_marginal: np.ndarray,
    col_marginal: np.ndarray,
    solver: str,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, str]:
    plan = np.asarray(plan, dtype=np.float64)
    tolerance = max(tol * 10.0, 1e-6)
    if transport_marginal_error(plan, row_marginal, col_marginal) <= tolerance:
        return np.maximum(plan, 0.0), solver

    balanced = balance_transport_plan(
        plan,
        row_marginal,
        col_marginal,
        max_iter=max(1000, max_iter),
        tol=tolerance,
    )
    if transport_marginal_error(balanced, row_marginal, col_marginal) > tolerance:
        raise RuntimeError(f"{solver} returned an invalid transport plan")
    return balanced, f"{solver}-balanced"


def project_embeddings(
    plan_source_to_target: np.ndarray,
    source_embedding: np.ndarray,
    normalize_projection: bool = False,
) -> np.ndarray:
    projected = plan_source_to_target.T @ source_embedding
    if normalize_projection:
        col_mass = np.maximum(plan_source_to_target.sum(axis=0), 1e-12)
        projected = projected / col_mass[:, None]
    return projected


def source_to_target_scores(
    source_brand_embedding: np.ndarray,
    source_region_embedding: np.ndarray,
    brand_plan: np.ndarray,
    region_plan: np.ndarray,
    normalize_projection: bool = False,
) -> np.ndarray:
    projected_brand = project_embeddings(brand_plan, source_brand_embedding, normalize_projection)
    projected_region = project_embeddings(region_plan, source_region_embedding, normalize_projection)
    return projected_brand @ projected_region.T


def softmax_negative_distance(distances: list[float], tau: float = 1.0) -> np.ndarray:
    x = -np.asarray(distances, dtype=np.float64) / max(tau, 1e-8)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.maximum(np.sum(exp), 1e-12)
