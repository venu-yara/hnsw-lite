"""Benchmark hnsw-lite against exact search.

Measures two things:
  1. Recall/cost trade-off: sweep ef_search on the full index and record
     recall@k, distance evaluations per query, and latency.
  2. Scaling: build the index incrementally and, at each size checkpoint, record
     how many distance evaluations a query needs. Exact search needs exactly n;
     HNSW should grow roughly logarithmically.

Usage:
    python bench/run_benchmark.py                       # synthetic clustered data
    python bench/run_benchmark.py --data base.fvecs --queries query.fvecs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnswlite import HNSWIndex, exact_knn, recall_at_k  # noqa: E402

OUT = Path(__file__).resolve().parent


def read_vectors(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False).astype(np.float32)
    if path.endswith(".fvecs"):  # TEXMEX format (SIFT/GIST): int32 dim, then dim float32s
        raw = np.fromfile(path, dtype=np.int32)
        dim = raw[0]
        return raw.reshape(-1, dim + 1)[:, 1:].view(np.float32).copy()
    raise ValueError("supported formats: .npy, .fvecs")


def synthetic(
    kind: str, n: int, n_queries: int, dim: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``clustered``: overlapping Gaussian clusters of varying size and spread, a rough
    stand-in for embedding data. ``uniform``: i.i.d. uniform noise, the worst case for
    any graph/tree index because there is no low-dimensional structure to exploit."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return (rng.random((n, dim), dtype=np.float32), rng.random((n_queries, dim), dtype=np.float32))
    n_clusters = 100
    # Center spacing (~sqrt(2*dim)) is comparable to cluster radius (~sqrt(dim)),
    # so clusters overlap and neighbors frequently cross cluster boundaries.
    centers = rng.normal(size=(n_clusters, dim))
    spread = rng.uniform(0.5, 1.5, size=n_clusters)
    weights = rng.dirichlet(np.ones(n_clusters))

    def draw(m):
        c = rng.choice(n_clusters, size=m, p=weights)
        return (centers[c] + rng.normal(size=(m, dim)) * spread[c, None]).astype(np.float32)

    return draw(n), draw(n_queries)


def measure(idx: HNSWIndex, Q: np.ndarray, truth: np.ndarray, k: int, ef: int) -> dict:
    idx.distance_count = 0
    t0 = time.perf_counter()
    found, _ = idx.search_batch(Q, k=k, ef=ef)
    elapsed = time.perf_counter() - t0
    return {
        "ef": ef,
        "recall": recall_at_k(found, truth),
        "dist_evals": idx.distance_count / len(Q),
        "latency_ms": 1000 * elapsed / len(Q),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help=".npy or .fvecs base vectors (default: synthetic)")
    p.add_argument("--queries", help=".npy or .fvecs query vectors (with --data)")
    p.add_argument("--synthetic", default="clustered", choices=["clustered", "uniform"])
    p.add_argument("--out", default="results", help="output file stem inside bench/")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--n-queries", type=int, default=300)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=100)
    p.add_argument("--metric", default="l2", choices=["l2", "cosine"])
    p.add_argument("--scaling-ef", type=int, default=64)
    args = p.parse_args()

    if args.data:
        X = read_vectors(args.data)[: args.n]
        Q = read_vectors(args.queries)[: args.n_queries]
        source = Path(args.data).name
    else:
        X, Q = synthetic(args.synthetic, args.n, args.n_queries, args.dim)
        source = {
            "clustered": "synthetic overlapping Gaussian clusters (100 clusters)",
            "uniform": "synthetic uniform random (worst case)",
        }[args.synthetic]
    n, dim = X.shape
    print(f"data: {source}  n={n}  dim={dim}  queries={len(Q)}  k={args.k}  metric={args.metric}")

    idx = HNSWIndex(dim, args.metric, M=args.M, ef_construction=args.ef_construction, seed=0)
    checkpoints = sorted({c for c in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000) if c < n} | {n})

    scaling, built, build_s = [], 0, 0.0
    for size in checkpoints:
        t0 = time.perf_counter()
        idx.add(X[built:size])
        build_s += time.perf_counter() - t0
        built = size
        truth, _ = exact_knn(X[:size], Q, args.k, args.metric)
        r = measure(idx, Q, truth, args.k, args.scaling_ef)
        r["n"] = size
        scaling.append(r)
        print(f"  n={size:>7}  recall={r['recall']:.3f}  dist_evals/query={r['dist_evals']:8.0f}"
              f"  ({100 * r['dist_evals'] / size:5.1f}% of n)  build so far {build_s:6.1f}s")

    # Exact-search latency, one query at a time, for an honest wall-clock baseline.
    t0 = time.perf_counter()
    for q in Q:
        exact_knn(X, q[None, :], args.k, args.metric)
    exact_ms = 1000 * (time.perf_counter() - t0) / len(Q)

    sweep = []
    for ef in (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        if ef < args.k:
            continue
        r = measure(idx, Q, truth, args.k, ef)
        sweep.append(r)
        print(f"  ef={ef:>4}  recall={r['recall']:.3f}  dist_evals={r['dist_evals']:7.0f}"
              f"  latency={r['latency_ms']:.2f}ms")

    results = {
        "config": {"source": source, "n": n, "dim": dim, "queries": len(Q), "k": args.k,
                   "metric": args.metric, "M": args.M, "ef_construction": args.ef_construction},
        "build_seconds": build_s,
        "exact_latency_ms": exact_ms,
        "sweep": sweep,
        "scaling": scaling,
    }
    (OUT / f"{args.out}.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, OUT / f"{args.out}.md")
    plot(results, OUT / f"{args.out}.png")
    print(f"wrote {OUT / args.out}.{{json,md,png}}")


def write_markdown(r: dict, path: Path) -> None:
    c = r["config"]
    lines = [
        f"**Data:** {c['source']}, n={c['n']:,}, dim={c['dim']}, {c['queries']} queries, "
        f"k={c['k']}, metric={c['metric']}, M={c['M']}, ef_construction={c['ef_construction']}. "
        f"Build time: {r['build_seconds']:.0f}s (pure Python).",
        "",
        "| ef_search | recall@k | distance evals / query | % of exact | latency (ms) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in r["sweep"]:
        lines.append(f"| {s['ef']} | {s['recall']:.3f} | {s['dist_evals']:,.0f} | "
                     f"{100 * s['dist_evals'] / c['n']:.2f}% | {s['latency_ms']:.2f} |")
    lines.append(f"| exact | 1.000 | {c['n']:,} | 100% | {r['exact_latency_ms']:.2f} |")
    lines += ["", "| n | recall@k | distance evals / query | % of exact |", "|---:|---:|---:|---:|"]
    for s in r["scaling"]:
        lines.append(f"| {s['n']:,} | {s['recall']:.3f} | {s['dist_evals']:,.0f} | "
                     f"{100 * s['dist_evals'] / s['n']:.2f}% |")
    path.write_text("\n".join(lines) + "\n")


def plot(r: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    sw = r["sweep"]
    a.plot([s["dist_evals"] for s in sw], [s["recall"] for s in sw], "o-")
    for s in sw:
        a.annotate(f"ef={s['ef']}", (s["dist_evals"], s["recall"]), fontsize=7,
                   xytext=(4, -10), textcoords="offset points")
    a.set_xlabel("distance evaluations per query")
    a.set_ylabel(f"recall@{r['config']['k']}")
    a.set_title(f"Recall vs. cost (n={r['config']['n']:,}; exact = {r['config']['n']:,} evals)")
    a.grid(alpha=0.3)

    sc = r["scaling"]
    ns = [s["n"] for s in sc]
    b.loglog(ns, ns, "--", color="gray", label="exact search (n)")
    b.loglog(ns, [s["dist_evals"] for s in sc], "o-", label="HNSW")
    b.set_xlabel("index size n")
    b.set_ylabel("distance evaluations per query")
    b.set_title("Query cost scaling")
    b.legend()
    b.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
