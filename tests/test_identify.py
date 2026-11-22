import numpy as np

from zebrasight.identify import bootstrap_ci, rank_zebras, topk_hits


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_rank_uses_best_photo_per_zebra():
    g = unit([[1, 0], [0, 1], [0.6, 0.8]])
    g_ids = np.array(["A", "B", "B"])
    q = unit([[0.5, 0.9]])
    ids, s = rank_zebras(q, g, g_ids)
    assert list(ids) == ["A", "B"]
    assert s[0, 1] == max(float(q[0] @ g[1]), float(q[0] @ g[2]))
    assert np.argmax(s[0]) == 1


def test_topk_hits():
    ids = np.array(["A", "B", "C"])
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.3, 0.2]])
    truth = np.array(["A", "C"])
    assert topk_hits(scores, ids, truth, 1).tolist() == [True, False]
    assert topk_hits(scores, ids, truth, 2).tolist() == [True, True]


def test_bootstrap_ci_brackets_mean():
    hits = np.array([1, 1, 0, 1, 0, 1, 1, 0], bool)
    groups = np.array(list("aabbccdd"))
    lo, hi = bootstrap_ci(hits, groups, n_boot=200)
    assert 0 <= lo <= hits.mean() <= hi <= 1
