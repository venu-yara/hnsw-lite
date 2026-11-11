"""Exact (brute-force) k-NN: the ground truth that HNSW is measured against."""

from __future__ import annotations

import numpy as np


def exact_knn(
    data: np.ndarray, queries: np.ndarray, k: int, metric: str = "l2", batch: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, distances)`` of shape ``(n_queries, k)``, closest first.

    Uses the same distance definitions as :class:`hnswlite.HNSWIndex`:
    squared L2, or 1 - cosine similarity.
    """
    X = np.asarray(data, dtype=np.float32)
    Q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    k = min(k, len(X))
    if metric == "cosine":
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    elif metric != "l2":
        raise ValueError(f"unknown metric {metric!r}")

    x_sq = np.einsum("ij,ij->i", X, X)
    out_i = np.empty((len(Q), k), dtype=np.int64)
    out_d = np.empty((len(Q), k), dtype=np.float32)
    for start in range(0, len(Q), batch):
        q = Q[start : start + batch]
        if metric == "l2":
            d = x_sq[None, :] - 2.0 * (q @ X.T) + np.einsum("ij,ij->i", q, q)[:, None]
        else:
            d = 1.0 - q @ X.T
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part, axis=1)
        order = np.argsort(part_d, axis=1)
        out_i[start : start + len(q)] = np.take_along_axis(part, order, axis=1)
        out_d[start : start + len(q)] = np.take_along_axis(part_d, order, axis=1)
    return out_i, out_d


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of the true k nearest neighbors that were returned."""
    hits = sum(len(set(f.tolist()) & set(t.tolist())) for f, t in zip(found, truth))
    return hits / truth.size
