"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index.

Implements the algorithm from Malkov & Yashunin, "Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs"
(arXiv:1603.09320), in pure Python + NumPy.

Design notes
------------
* Vectors live in one contiguous float32 array that grows by doubling, so
  neighbor distances are computed with a single vectorized NumPy call per
  expanded node instead of a Python loop per neighbor.
* Graph links are plain Python lists: ``_links[node][layer] -> [neighbor ids]``.
* Neighbor selection uses the paper's heuristic (Algorithm 4), which keeps a
  candidate only if it is closer to the new element than to any neighbor already
  chosen. This preserves "long-range" edges between clusters and is what makes
  HNSW robust on clustered data.
* Deletes are tombstones: deleted nodes still route searches (so the graph stays
  connected) but are never returned.
* Persistence uses ``np.savez`` with ``allow_pickle=False`` -- loading an index
  file can never execute code.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

METRICS = ("l2", "cosine")
_FORMAT_VERSION = 1


class HNSWIndex:
    """Approximate k-NN index over fixed-dimension float vectors.

    Parameters
    ----------
    dim:
        Vector dimensionality.
    metric:
        ``"l2"`` (squared Euclidean) or ``"cosine"`` (1 - cosine similarity).
    M:
        Max neighbors per node on layers >= 1. Layer 0 allows ``2 * M``.
        Higher M -> better recall, more memory, slower inserts.
    ef_construction:
        Beam width used while inserting. Higher -> better graph, slower build.
    ef_search:
        Default beam width for queries (can be overridden per call).
    seed:
        Seed for the level generator; makes builds reproducible.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if M < 2:
            raise ValueError("M must be >= 2")

        self.dim = dim
        self.metric = metric
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = ef_search
        self._ml = 1.0 / math.log(M)  # level multiplier from the paper
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._size = 0
        self._levels: list[int] = []
        self._links: list[list[list[int]]] = []
        self._labels: list[int] = []
        self._label_to_id: dict[int, int] = {}
        self._deleted: set[int] = set()
        self._entry = -1
        self._max_level = -1

        #: Cumulative number of vector-to-vector distance evaluations performed
        #: by searches and inserts. Reset it to measure the cost of a workload.
        self.distance_count = 0

    # ------------------------------------------------------------------ public

    def __len__(self) -> int:
        return self._size - len(self._deleted)

    def __contains__(self, label: int) -> bool:
        return label in self._label_to_id

    def add(self, vectors: np.ndarray, labels: Iterable[int] | None = None) -> None:
        """Insert one vector (shape ``(dim,)``) or many (shape ``(n, dim)``)."""
        X = self._prepare(np.atleast_2d(np.asarray(vectors)))
        n = len(X)
        if labels is None:
            start = max(self._labels, default=-1) + 1
            labels = list(range(start, start + n))
        else:
            labels = [int(x) for x in labels]
            if len(labels) != n:
                raise ValueError(f"got {n} vectors but {len(labels)} labels")
            if len(set(labels)) != n:
                raise ValueError("labels must be unique")
            taken = [lab for lab in labels if lab in self._label_to_id]
            if taken:
                raise ValueError(f"labels already in index: {taken[:5]}")

        self._reserve(self._size + n)
        for x, label in zip(X, labels):
            self._insert(x, label)

    def search(
        self, query: np.ndarray, k: int = 10, ef: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, distances)`` of the ~k nearest live vectors, closest first."""
        if k <= 0:
            raise ValueError("k must be positive")
        q = self._prepare(np.asarray(query).reshape(1, -1))[0]
        if len(self) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        ef = max(ef if ef is not None else self.ef_search, k)
        ep = self._entry
        ep_d = float(self._distances(q, [ep])[0])
        for layer in range(self._max_level, 0, -1):
            ep, ep_d = self._greedy_closest(q, ep, ep_d, layer)
        found = self._search_layer(q, [(ep_d, ep)], ef, 0, live_only=True)[:k]

        labels = np.fromiter((self._labels[i] for _, i in found), dtype=np.int64, count=len(found))
        dists = np.fromiter((d for d, _ in found), dtype=np.float32, count=len(found))
        return labels, dists

    def search_batch(
        self, queries: np.ndarray, k: int = 10, ef: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search many queries. Returns ``(n, k)`` arrays padded with label -1 / dist inf."""
        Q = np.atleast_2d(np.asarray(queries))
        out_labels = np.full((len(Q), k), -1, dtype=np.int64)
        out_dists = np.full((len(Q), k), np.inf, dtype=np.float32)
        for row, q in enumerate(Q):
            labels, dists = self.search(q, k, ef)
            out_labels[row, : len(labels)] = labels
            out_dists[row, : len(dists)] = dists
        return out_labels, out_dists

    def delete(self, label: int) -> None:
        """Remove a vector from results. The label may be re-used by a later ``add``."""
        try:
            node = self._label_to_id.pop(label)
        except KeyError:
            raise KeyError(f"label {label} not in index") from None
        self._deleted.add(node)

    def save(self, path: str | Path) -> None:
        """Write the index to a single ``.npz`` file (no pickle)."""
        counts, flat = [], []
        for node_links in self._links:
            for layer_links in node_links:
                counts.append(len(layer_links))
                flat.extend(layer_links)
        meta = {
            "format_version": _FORMAT_VERSION,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "entry": self._entry,
            "max_level": self._max_level,
        }
        np.savez(
            path,
            meta=np.array(json.dumps(meta)),
            vectors=self._vectors[: self._size],
            levels=np.asarray(self._levels, dtype=np.int32),
            labels=np.asarray(self._labels, dtype=np.int64),
            deleted=np.asarray(sorted(self._deleted), dtype=np.int64),
            link_counts=np.asarray(counts, dtype=np.int32),
            link_data=np.asarray(flat, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "HNSWIndex":
        """Load an index written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("format_version") != _FORMAT_VERSION:
                raise ValueError(f"unsupported index format: {meta.get('format_version')}")
            idx = cls(
                meta["dim"],
                meta["metric"],
                meta["M"],
                meta["ef_construction"],
                meta["ef_search"],
                seed=seed,
            )
            vectors = f["vectors"].astype(np.float32)
            levels = f["levels"].tolist()
            labels = f["labels"].tolist()
            deleted = set(f["deleted"].tolist())
            counts = f["link_counts"].tolist()
            flat = f["link_data"].tolist()

        idx._vectors = vectors.copy()
        idx._size = len(vectors)
        idx._levels = levels
        idx._labels = labels
        idx._deleted = deleted
        idx._label_to_id = {lab: i for i, lab in enumerate(labels) if i not in deleted}
        pos = c = 0
        for level in levels:
            node_links = []
            for _ in range(level + 1):
                node_links.append(flat[pos : pos + counts[c]])
                pos += counts[c]
                c += 1
            idx._links.append(node_links)
        idx._entry = meta["entry"]
        idx._max_level = meta["max_level"]
        return idx

    # ---------------------------------------------------------------- internals

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got shape {X.shape}")
        X = X.astype(np.float32, copy=True)
        if not np.isfinite(X).all():
            raise ValueError("vectors must be finite (no NaN/inf)")
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("zero vector is undefined under cosine metric")
            X /= norms
        return X

    def _reserve(self, capacity: int) -> None:
        if capacity <= len(self._vectors):
            return
        new_cap = max(capacity, 2 * len(self._vectors), 64)
        grown = np.empty((new_cap, self.dim), dtype=np.float32)
        grown[: self._size] = self._vectors[: self._size]
        self._vectors = grown

    def _distances(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        """Distances from q to each node in ids -- one vectorized call."""
        self.distance_count += len(ids)
        V = self._vectors[ids]
        if self.metric == "l2":
            diff = V - q
            return np.einsum("ij,ij->i", diff, diff)
        return 1.0 - V @ q

    def _pairwise(self, ids: list[int]) -> np.ndarray:
        V = self._vectors[ids]
        gram = V @ V.T
        if self.metric == "l2":
            sq = np.diag(gram)
            return sq[:, None] + sq[None, :] - 2.0 * gram
        return 1.0 - gram

    def _random_level(self) -> int:
        # -ln(U) * mL with U in (0, 1]: exponentially decaying layer assignment.
        return int(-math.log(1.0 - self._rng.random()) * self._ml)

    def _insert(self, x: np.ndarray, label: int) -> None:
        node = self._size
        self._vectors[node] = x
        self._size += 1
        level = self._random_level()
        self._levels.append(level)
        self._links.append([[] for _ in range(level + 1)])
        self._labels.append(label)
        self._label_to_id[label] = node

        if self._entry == -1:
            self._entry, self._max_level = node, level
            return

        # Phase 1: greedy descent through layers above the new node's level.
        ep = self._entry
        ep_d = float(self._distances(x, [ep])[0])
        for layer in range(self._max_level, level, -1):
            ep, ep_d = self._greedy_closest(x, ep, ep_d, layer)

        # Phase 2: on each layer the node lives in, find candidates and link.
        entry_points = [(ep_d, ep)]
        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(x, entry_points, self.ef_construction, layer)
            neighbors = self._select_neighbors(candidates, self.M)
            self._links[node][layer] = [n for _, n in neighbors]

            cap = self.M0 if layer == 0 else self.M
            for _, n in neighbors:
                n_links = self._links[n][layer]
                n_links.append(node)
                if len(n_links) > cap:  # shrink neighbor's list with the same heuristic
                    d = self._distances(self._vectors[n], n_links).tolist()
                    kept = self._select_neighbors(sorted(zip(d, n_links)), cap)
                    self._links[n][layer] = [i for _, i in kept]
            entry_points = candidates

        if level > self._max_level:
            self._entry, self._max_level = node, level

    def _greedy_closest(
        self, q: np.ndarray, ep: int, ep_d: float, layer: int
    ) -> tuple[int, float]:
        """Hill-climb to a local minimum on one layer (search with ef=1)."""
        while True:
            nbrs = self._links[ep][layer]
            if not nbrs:
                return ep, ep_d
            d = self._distances(q, nbrs)
            j = int(np.argmin(d))
            if d[j] >= ep_d:
                return ep, ep_d
            ep, ep_d = nbrs[j], float(d[j])

    def _search_layer(
        self,
        q: np.ndarray,
        entry_points: list[tuple[float, int]],
        ef: int,
        layer: int,
        live_only: bool = False,
    ) -> list[tuple[float, int]]:
        """Beam search on one layer (Algorithm 2). Returns (dist, id) sorted ascending.

        With ``live_only``, tombstoned nodes are still traversed but never enter
        the result set, so deletes don't disconnect the graph.
        """
        skip = self._deleted if live_only else ()
        visited = set()
        candidates: list[tuple[float, int]] = []  # min-heap by distance
        results: list[tuple[float, int]] = []  # max-heap via negated distance
        for d, i in entry_points:
            visited.add(i)
            heapq.heappush(candidates, (d, i))
            if i not in skip:
                heapq.heappush(results, (-d, i))
        while len(results) > ef:
            heapq.heappop(results)

        links = self._links
        while candidates:
            d, c = heapq.heappop(candidates)
            if len(results) >= ef and d > -results[0][0]:
                break  # closest unexplored candidate is worse than our worst result
            nbrs = [n for n in links[c][layer] if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for dn, n in zip(self._distances(q, nbrs).tolist(), nbrs):
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    if n not in skip:
                        heapq.heappush(results, (-dn, n))
                        if len(results) > ef:
                            heapq.heappop(results)

        return sorted((-nd, i) for nd, i in results)

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[tuple[float, int]]:
        """Neighbor-selection heuristic (Algorithm 4). ``candidates`` must be sorted.

        A candidate is kept only if it is closer to the base element than to every
        neighbor already kept. Otherwise an existing neighbor already "covers" that
        direction, and the edge would be redundant.
        """
        if len(candidates) < m:
            return candidates
        pair = self._pairwise([i for _, i in candidates])
        chosen: list[int] = []
        for j, (d, _) in enumerate(candidates):
            row = pair[j]
            if all(row[s] > d for s in chosen):
                chosen.append(j)
                if len(chosen) == m:
                    break
        return [candidates[j] for j in chosen]
