import numpy as np
import pytest

from drills.data import (
    cluster_sample_weights,
    confident_learning_flags,
    flag_label_issues,
    flag_similar_to_train,
    group_split,
    maxmin_pick,
    nearest_neighbour_similarity,
    out_of_fold_probs,
    similarity_components,
    similarity_split,
    tanimoto_matrix,
    temperature_sampling_probs,
    temporal_split,
)

# Six tiny fingerprints used in several tests.
FPS = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0],
    ],
    dtype=bool,
)


def _same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])


# ---------------------------------------------------------------- Drill #64


def test_drill64_group_split_no_group_on_both_sides():
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 12, size=200)
    tr, te = group_split(groups, test_frac=0.25, seed=3)
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(200))
    assert np.array_equal(tr, np.sort(tr)) and np.array_equal(te, np.sort(te))
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert len(te) >= 0.25 * 200
    # stops as soon as the target is reached: dropping any test group goes below it
    for g in np.unique(groups[te]):
        assert len(te) - (groups[te] == g).sum() < 0.25 * 200


def test_drill64_group_split_string_ids_and_zero_frac():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    tr, te = group_split(groups, test_frac=0.0, seed=0)
    assert len(te) == 0 and np.array_equal(tr, np.arange(6))
    tr, te = group_split(groups, test_frac=0.5, seed=0)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert len(te) >= 3


def test_drill64_group_split_deterministic():
    groups = np.repeat(np.arange(10), 3)
    a = group_split(groups, 0.3, seed=7)
    b = group_split(groups, 0.3, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_drill64_temporal_split_cutoff_goes_to_test():
    dates = np.array(
        ["2021-05-01", "2023-01-01", "2020-01-01", "2022-12-31", "2023-06-01"],
        dtype="datetime64[D]",
    )
    tr, te = temporal_split(dates, np.datetime64("2023-01-01"))
    assert tr.tolist() == [0, 2, 3]
    assert te.tolist() == [1, 4]


def test_drill64_temporal_split_numeric():
    tr, te = temporal_split(np.array([3.0, 1.0, 2.0]), 0.5)
    assert tr.tolist() == [] and te.tolist() == [0, 1, 2]


def test_drill64_flag_similar_to_train_threshold_is_strict():
    sim = np.array([[0.10, 0.45], [0.46, 0.00], [0.30, 0.20]])
    assert flag_similar_to_train(sim, 0.45).tolist() == [False, True, False]


def test_drill64_flag_similar_to_train_empty_train():
    out = flag_similar_to_train(np.zeros((4, 0)), 0.45)
    assert out.dtype == bool and out.tolist() == [False] * 4


def test_drill64_similarity_components_transitive_chain():
    # 0~1 and 1~3 (chain); 2 alone; 4~5.
    sim = np.eye(6)
    for i, j, s in [(0, 1, 0.9), (1, 3, 0.6), (4, 5, 0.8), (2, 3, 0.5), (0, 2, 0.2)]:
        sim[i, j] = sim[j, i] = s
    lab = similarity_components(sim, threshold=0.5)
    assert lab.tolist() == [0, 0, 1, 0, 2, 2]
    assert lab.dtype == np.int64


def test_drill64_similarity_components_matches_scipy():
    from scipy.sparse.csgraph import connected_components

    rng = np.random.default_rng(2)
    x = rng.random((40, 40))
    sim = (x + x.T) / 2
    adj = sim > 0.93
    np.fill_diagonal(adj, False)
    _, ref = connected_components(adj.astype(int), directed=False)
    assert _same_partition(similarity_components(sim, 0.93), ref)


def test_drill64_similarity_split_no_similar_pair_crosses():
    rng = np.random.default_rng(4)
    x = rng.random((60, 60))
    sim = (x + x.T) / 2
    tr, te = similarity_split(sim, 0.95, test_frac=0.2, seed=0)
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(60))
    assert not (sim[np.ix_(tr, te)] > 0.95).any()
    assert len(te) >= 12


# ---------------------------------------------------------------- Drill #65


def test_drill65_tanimoto_hand_values():
    t = tanimoto_matrix(FPS, FPS)
    assert t.shape == (5, 5) and t.dtype == np.float64
    np.testing.assert_allclose(np.diag(t), 1.0)
    np.testing.assert_allclose(t[0, 1], 2 / 3)
    np.testing.assert_allclose(t[0, 4], 1 / 3)
    np.testing.assert_allclose(t[1, 4], 1 / 4)
    np.testing.assert_allclose(t[0, 2], 0.0)
    np.testing.assert_allclose(t, t.T)


def test_drill65_tanimoto_matches_scipy_jaccard_and_handles_zeros():
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(5)
    a = rng.random((7, 32)) < 0.3
    b = rng.random((4, 32)) < 0.3
    b[0] = False  # an all-zero fingerprint
    a[0] = False
    t = tanimoto_matrix(a, b)
    assert t.shape == (7, 4)
    assert np.isfinite(t).all()
    assert t[0, 0] == 0.0
    np.testing.assert_allclose(t[1:, 1:], 1 - cdist(a[1:], b[1:], "jaccard"), atol=1e-12)


def test_drill65_tanimoto_accepts_int_fingerprints():
    t = tanimoto_matrix(FPS.astype(np.int64), FPS[:2].astype(np.int64))
    np.testing.assert_allclose(t[1, 0], 2 / 3)


def test_drill65_maxmin_pick_hand_case_with_tie():
    # After 0, items 2 and 3 tie at distance 1; the lower index wins.
    assert maxmin_pick(FPS, 3, first=0).tolist() == [0, 2, 4]
    assert maxmin_pick(FPS, 5, first=0).tolist() == [0, 2, 4, 1, 3]


def test_drill65_maxmin_pick_edge_cases():
    out = maxmin_pick(FPS, 0)
    assert len(out) == 0
    full = maxmin_pick(FPS, 5, first=3)
    assert full[0] == 3 and sorted(full.tolist()) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        maxmin_pick(FPS, 6)


def test_drill65_nearest_neighbour_similarity():
    nn = nearest_neighbour_similarity(FPS[[1, 3, 4]], FPS[[0, 2]])
    np.testing.assert_allclose(nn, [2 / 3, 2 / 3, 1 / 3])
    out = nearest_neighbour_similarity(FPS, FPS[:0])
    np.testing.assert_allclose(out, np.zeros(5))


# ---------------------------------------------------------------- Drill #66


def test_drill66_temperature_probs_hand_case():
    p = temperature_sampling_probs(np.array([100, 400]), temperature=2.0)
    np.testing.assert_allclose(p, [1 / 3, 2 / 3], rtol=1e-12)
    np.testing.assert_allclose(
        temperature_sampling_probs(np.array([1, 2, 7]), 1.0), [0.1, 0.2, 0.7]
    )


def test_drill66_temperature_probs_limits_and_scale_invariance():
    sizes = np.array([10, 1000, 100000])
    p_hot = temperature_sampling_probs(sizes, 1e6)
    np.testing.assert_allclose(p_hot, np.full(3, 1 / 3), atol=1e-4)
    p = temperature_sampling_probs(sizes, 3.0)
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, temperature_sampling_probs(sizes * 17, 3.0))
    assert np.all(np.diff(p) > 0)


def test_drill66_cluster_weights_literal():
    w = cluster_sample_weights(
        np.array([0, 0, 1, 2, 2, 2]),
        np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [2, 0, 0]]),
        np.array([3.0, 3.0, 1.0]),
        beta=0.5,
    )
    np.testing.assert_allclose(
        w,
        [0.15254237, 0.20338983, 0.30508475, 0.10169492, 0.03389831, 0.20338983],
        atol=1e-7,
    )


def test_drill66_cluster_weights_each_cluster_total_independent_of_size():
    # Same composition everywhere: every cluster gets equal total mass.
    ids = np.array([5, 5, 5, 5, 9, 2, 2])
    w = cluster_sample_weights(ids, np.ones((7, 1)), np.array([2.0]), beta=3.0)
    np.testing.assert_allclose(w.sum(), 1.0)
    for c in (5, 9, 2):
        np.testing.assert_allclose(w[ids == c].sum(), 1 / 3)


# ---------------------------------------------------------------- Drill #67


def _memoriser(x_tr, y_tr, x_te):
    """Returns the true label one-hot for rows it has seen, uniform otherwise."""
    k = 3
    out = np.full((len(x_te), k), 1 / k)
    seen = {tuple(r): int(l) for r, l in zip(x_tr, y_tr)}
    for i, r in enumerate(x_te):
        if tuple(r) in seen:
            out[i] = np.eye(k)[seen[tuple(r)]]
    return out


def _centroid(x_tr, y_tr, x_te):
    k = 2
    cent = np.stack([x_tr[y_tr == c].mean(0) for c in range(k)])
    d = ((x_te[:, None, :] - cent[None]) ** 2).sum(-1)
    e = np.exp(-(d - d.min(1, keepdims=True)))
    return e / e.sum(1, keepdims=True)


def test_drill67_oof_never_sees_its_own_row():
    x = np.arange(12, dtype=float).reshape(12, 1)
    y = np.arange(12) % 3
    p = out_of_fold_probs(x, y, n_folds=4, fit_predict=_memoriser)
    assert p.shape == (12, 3)
    np.testing.assert_allclose(p, np.full((12, 3), 1 / 3))


def test_drill67_oof_fold_assignment_is_i_mod_k():
    calls = []

    def spy(x_tr, y_tr, x_te):
        calls.append(sorted(x_te[:, 0].astype(int).tolist()))
        return np.tile([1.0, 0.0], (len(x_te), 1))

    x = np.arange(7, dtype=float).reshape(7, 1)
    out_of_fold_probs(x, np.zeros(7, dtype=int), n_folds=3, fit_predict=spy)
    assert sorted(calls) == [[0, 3, 6], [1, 4], [2, 5]]


def test_drill67_oof_finds_flipped_labels():
    rng = np.random.default_rng(0)
    y_true = np.repeat([0, 1], 40)
    x = rng.normal(size=(80, 2)) + 4.0 * y_true[:, None]
    noisy = y_true.copy()
    flipped = np.array([3, 17, 45, 70])
    noisy[flipped] = 1 - noisy[flipped]
    p = out_of_fold_probs(x, noisy, n_folds=5, fit_predict=_centroid)
    flags = flag_label_issues(p, noisy, threshold=0.2)
    assert set(flipped.tolist()) <= set(np.flatnonzero(flags).tolist())
    assert flags.sum() <= 6


def test_drill67_flag_label_issues_strict_threshold():
    p = np.array([[0.8, 0.2], [0.1, 0.9], [0.19, 0.81]])
    assert flag_label_issues(p, np.array([1, 1, 0]), 0.2).tolist() == [False, False, True]


def test_drill67_confident_learning_literal():
    p = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.5, 0.4, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.7, 0.2],
            [0.3, 0.3, 0.4],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 0, 2])
    flags = confident_learning_flags(p, y)
    assert flags.tolist() == [False, False, False, True, False, True, False]
    # the global threshold misses item 3
    assert flag_label_issues(p, y, 0.2).tolist() == [False] * 5 + [True, False]


def test_drill67_confident_learning_empty_class_never_predicted():
    # Class 2 has no items: its threshold is +inf, so it is never a candidate.
    p = np.array([[0.1, 0.1, 0.8], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    y = np.array([0, 0, 1])
    assert confident_learning_flags(p, y).tolist() == [False, False, False]
