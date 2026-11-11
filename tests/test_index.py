import numpy as np
import pytest

from hnswlite import HNSWIndex, exact_knn, recall_at_k


def clustered(n, dim, n_clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 4
    assign = rng.integers(n_clusters, size=n)
    return (centers[assign] + rng.normal(size=(n, dim))).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    X = clustered(3000, 32)
    rng = np.random.default_rng(1)
    Q = X[rng.choice(len(X), 100, replace=False)] + rng.normal(size=(100, 32)).astype(np.float32) * 0.5
    return X, Q


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_recall_vs_exact(data, metric):
    X, Q = data
    idx = HNSWIndex(dim=32, metric=metric, M=12, ef_construction=100, seed=42)
    idx.add(X)
    found, _ = idx.search_batch(Q, k=10, ef=64)
    truth, _ = exact_knn(X, Q, k=10, metric=metric)
    assert recall_at_k(found, truth) >= 0.95


def test_distances_match_exact(data):
    X, Q = data
    idx = HNSWIndex(dim=32, seed=0)
    idx.add(X[:500])
    labels, dists = idx.search(Q[0], k=5, ef=500)
    truth_i, truth_d = exact_knn(X[:500], Q[:1], k=5)
    np.testing.assert_array_equal(labels, truth_i[0])
    np.testing.assert_allclose(dists, truth_d[0], rtol=1e-4, atol=1e-3)


def test_results_sorted_and_unique(data):
    X, Q = data
    idx = HNSWIndex(dim=32, seed=0)
    idx.add(X[:1000])
    labels, dists = idx.search(Q[0], k=20)
    assert np.all(np.diff(dists) >= 0)
    assert len(set(labels.tolist())) == len(labels)


def test_graph_invariants(data):
    X, _ = data
    idx = HNSWIndex(dim=32, M=8, ef_construction=50, seed=3)
    idx.add(X[:1500])
    for node, node_links in enumerate(idx._links):
        assert len(node_links) == idx._levels[node] + 1
        for layer, nbrs in enumerate(node_links):
            cap = idx.M0 if layer == 0 else idx.M
            assert len(nbrs) <= cap
            assert node not in nbrs, "self-loop"
            assert len(set(nbrs)) == len(nbrs), "duplicate edge"
            assert all(idx._levels[n] >= layer for n in nbrs), "edge to node absent from layer"
    assert idx._levels[idx._entry] == idx._max_level


def test_layer0_graph_is_connected(data):
    X, _ = data
    idx = HNSWIndex(dim=32, M=8, ef_construction=50, seed=3)
    idx.add(X[:1500])
    seen, stack = {idx._entry}, [idx._entry]
    while stack:
        for n in idx._links[stack.pop()][0]:
            if n not in seen:
                seen.add(n)
                stack.append(n)
    assert len(seen) == 1500


def test_level_distribution_is_geometric():
    idx = HNSWIndex(dim=2, M=16, seed=0)
    levels = np.array([idx._random_level() for _ in range(100_000)])
    # P(level >= 1) = 1/M under mL = 1/ln(M)
    assert abs((levels >= 1).mean() - 1 / 16) < 0.005


def test_empty_and_single():
    idx = HNSWIndex(dim=4)
    labels, dists = idx.search(np.ones(4), k=3)
    assert labels.size == 0 and dists.size == 0
    idx.add(np.ones(4), labels=[7])
    labels, _ = idx.search(np.zeros(4), k=3)
    assert labels.tolist() == [7]


def test_k_larger_than_index():
    idx = HNSWIndex(dim=3, seed=0)
    idx.add(np.eye(3))
    labels, _ = idx.search(np.ones(3), k=10)
    assert sorted(labels.tolist()) == [0, 1, 2]
    batch, bd = idx.search_batch(np.ones((2, 3)), k=5)
    assert batch.shape == (2, 5) and (batch[:, 3:] == -1).all() and np.isinf(bd[:, 3:]).all()


def test_custom_labels_and_autolabel_continuation():
    idx = HNSWIndex(dim=2, seed=0)
    idx.add([[0, 0], [10, 10]], labels=[100, 200])
    idx.add([[5, 5]])  # auto label continues after the max existing label
    assert 201 in idx and len(idx) == 3
    labels, _ = idx.search([10, 10], k=1)
    assert labels.tolist() == [200]


def test_duplicate_labels_rejected():
    idx = HNSWIndex(dim=2)
    idx.add([[0, 0]], labels=[1])
    with pytest.raises(ValueError):
        idx.add([[1, 1]], labels=[1])
    with pytest.raises(ValueError):
        idx.add([[1, 1], [2, 2]], labels=[5, 5])


def test_delete_excludes_and_keeps_k(data):
    X, Q = data
    idx = HNSWIndex(dim=32, seed=0)
    idx.add(X[:1000])
    before, _ = idx.search(Q[0], k=10)
    for lab in before[:5]:
        idx.delete(int(lab))
    after, _ = idx.search(Q[0], k=10)
    assert len(after) == 10
    assert not set(before[:5].tolist()) & set(after.tolist())
    assert len(idx) == 995
    with pytest.raises(KeyError):
        idx.delete(int(before[0]))


def test_delete_majority_still_finds_survivors(data):
    X, Q = data
    idx = HNSWIndex(dim=32, seed=0)
    idx.add(X[:1000])
    for lab in range(1000):  # delete 90%, keep every 10th
        if lab % 10:
            idx.delete(lab)
    found, _ = idx.search_batch(Q, k=5, ef=100)
    survivors = np.arange(0, 1000, 10)
    truth, _ = exact_knn(X[survivors], Q, k=5)
    assert recall_at_k(found, survivors[truth]) >= 0.9


def test_readd_label_after_delete():
    idx = HNSWIndex(dim=2, seed=0)
    idx.add([[0, 0], [1, 1]], labels=[1, 2])
    idx.delete(1)
    idx.add([[9, 9]], labels=[1])
    labels, _ = idx.search([9, 9], k=1)
    assert labels.tolist() == [1]


def test_save_load_roundtrip(tmp_path, data):
    X, Q = data
    idx = HNSWIndex(dim=32, metric="cosine", M=8, seed=0)
    idx.add(X[:800])
    idx.delete(3)
    path = tmp_path / "index.npz"
    idx.save(path)
    loaded = HNSWIndex.load(path)
    a = idx.search_batch(Q[:20], k=10)
    b = loaded.search_batch(Q[:20], k=10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert len(loaded) == len(idx) and 3 not in loaded
    loaded.add(X[800:900])  # loaded index stays writable
    assert len(loaded) == 899


def test_seed_makes_builds_deterministic(data):
    X, Q = data
    runs = []
    for _ in range(2):
        idx = HNSWIndex(dim=32, seed=123)
        idx.add(X[:500])
        runs.append(idx.search_batch(Q[:10], k=5)[0])
    np.testing.assert_array_equal(*runs)


@pytest.mark.parametrize(
    "kwargs", [dict(dim=0), dict(dim=4, metric="dot"), dict(dim=4, M=1)]
)
def test_bad_constructor_args(kwargs):
    with pytest.raises(ValueError):
        HNSWIndex(**kwargs)


def test_bad_inputs():
    idx = HNSWIndex(dim=3, metric="cosine")
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 4)))
    with pytest.raises(ValueError):
        idx.add(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        idx.add(np.array([[1, np.nan, 0]]))
    idx.add(np.ones(3))
    with pytest.raises(ValueError):
        idx.search(np.ones(3), k=0)
