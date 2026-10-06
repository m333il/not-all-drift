from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), eps))


def orthonormal_basis(vectors: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [directions, hidden]")
    _, singular, vh = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(singular > tolerance * (singular[0] if len(singular) else 1)))
    return vh[:rank].T


def project(vector: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parallel = basis @ (basis.T @ vector) if basis.size else np.zeros_like(vector)
    return parallel, vector - parallel


def projection_energy(vector: np.ndarray, basis: np.ndarray) -> float:
    parallel, _ = project(vector, basis)
    denominator = float(np.dot(vector, vector))
    return 0.0 if denominator == 0 else float(np.dot(parallel, parallel) / denominator)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / denom) if denom else 0.0


def rbf_cka(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    def kernel(values: np.ndarray) -> np.ndarray:
        distances = np.maximum(
            np.sum(values * values, axis=1)[:, None]
            + np.sum(values * values, axis=1)[None, :]
            - 2 * values @ values.T,
            0,
        )
        nonzero = distances[distances > 0]
        width = sigma or (float(np.sqrt(np.median(nonzero))) if len(nonzero) else 1.0)
        return np.exp(-distances / (2 * width * width))

    kernel_x, kernel_y = kernel(x), kernel(y)
    center = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
    kc, lc = center @ kernel_x @ center, center @ kernel_y @ center
    denom = np.linalg.norm(kc, "fro") * np.linalg.norm(lc, "fro")
    return float(np.sum(kc * lc) / denom) if denom else 0.0


def procrustes_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("Procrustes inputs need the same examples")
    x, y = x - x.mean(0), y - y.mean(0)
    x /= max(np.linalg.norm(x), 1e-12)
    y /= max(np.linalg.norm(y), 1e-12)
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    aligned = x @ (u @ vt)
    return float(np.linalg.norm(aligned - y))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def projection_null_test(
    vector: np.ndarray,
    basis: np.ndarray,
    *,
    draws: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    observed = projection_energy(vector, basis)
    rng = np.random.default_rng(seed)
    controls = rng.normal(size=(draws, len(vector)))
    null = np.asarray([projection_energy(control, basis) for control in controls])
    p_value = (1 + int(np.sum(null >= observed))) / (draws + 1)
    return observed, float(null.mean()), p_value


def solution_consensus_correlation(
    scores: np.ndarray, vectors: np.ndarray, *, top_k: int
) -> tuple[float, np.ndarray]:
    if scores.ndim != 1 or vectors.ndim != 2 or len(scores) != len(vectors):
        raise ValueError("scores and solution vectors must align")
    if top_k <= 0 or top_k > len(scores):
        raise ValueError("top_k must select at least one available solution")
    strongest = np.argsort(-scores, kind="stable")[:top_k]
    consensus = vectors[strongest].mean(axis=0)
    alignments = np.asarray([cosine(vector, consensus) for vector in vectors])
    correlation = spearmanr(scores, alignments).statistic
    return float(correlation), alignments
