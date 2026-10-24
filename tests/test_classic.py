"""Tests for Section 7: classic ML from scratch (Drills #59-#63)."""

import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F

from drills.classic import (
    fit_logistic_regression,
    kmeans,
    kmeans_plusplus_init,
    knn_predict,
    linear_regression_gd,
    logistic_loss_and_grad,
    nn1_as_linear,
    pca,
    ridge_loo_residuals,
    ridge_normal_equation,
)


def _blobs(seed=0, n_per=30, centers=((0.0, 0.0), (8.0, 0.0), (0.0, 8.0))):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(c, 0.5, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(len(centers)), n_per)
    return X, y


# ---------------------------------------------------------------------------
# Drill #59: k-means
# ---------------------------------------------------------------------------


def test_drill59_kmeans_recovers_blobs():
    X, y = _blobs()
    C, labels = kmeans(X, 3, seed=0)
    assert C.shape == (3, 2)
    assert labels.shape == (90,)
    assert np.issubdtype(labels.dtype, np.integer)
    # Same partition as the truth, up to a permutation of cluster ids.
    for t in range(3):
        assert len(np.unique(labels[y == t])) == 1
    assert len(np.unique(labels)) == 3
    true_means = np.stack([X[y == t].mean(0) for t in range(3)])
    d = np.linalg.norm(C[:, None, :] - true_means[None], axis=-1).min(1)
    np.testing.assert_allclose(d, 0.0, atol=1e-8)


def test_drill59_kmeans_fixed_point_labels_match_centroids():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    C, labels = kmeans(X, 4, max_iters=200, seed=2)
    nearest = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    np.testing.assert_array_equal(labels, nearest)
    # Each centroid is the mean of its members (Lloyd fixed point).
    for j in range(4):
        np.testing.assert_allclose(C[j], X[labels == j].mean(0), atol=1e-8)


def test_drill59_kmeans_duplicates_no_empty_cluster():
    # Only 3 distinct points, heavily duplicated: naive init repeats points
    # and leaves clusters empty.
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    X = np.repeat(base, [50, 2, 2], axis=0)
    for seed in range(5):
        C, labels = kmeans(X, 3, seed=seed)
        assert np.all(np.bincount(labels, minlength=3) > 0)
        assert np.isfinite(C).all()
        assert len(np.unique(C.round(8), axis=0)) == 3


def test_drill59_kmeans_deterministic_given_seed():
    X, _ = _blobs(seed=4)
    C1, l1 = kmeans(X, 3, seed=7)
    C2, l2 = kmeans(X, 3, seed=7)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(C1, C2)


def test_drill59_kmeans_plusplus_distinct_rows_of_X():
    base = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    X = np.repeat(base, [40, 1, 1, 1], axis=0)
    for seed in range(5):
        C = kmeans_plusplus_init(X, 4, seed=seed)
        assert C.shape == (4, 2)
        # Every centroid is a row of X, and all are distinct.
        assert all((np.abs(X - c).sum(1) == 0).any() for c in C)
        assert len(np.unique(C, axis=0)) == 4


# ---------------------------------------------------------------------------
# Drill #60: kNN and 1-NN as a linear layer
# ---------------------------------------------------------------------------


def test_drill60_knn_k1_matches_cdist_argmin():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(40, 5))
    ytr = rng.integers(0, 4, size=40)
    Xte = rng.normal(size=(15, 5))
    ref_idx = torch.cdist(torch.from_numpy(Xte), torch.from_numpy(Xtr)).argmin(1).numpy()
    pred = knn_predict(Xtr, ytr, Xte, 1)
    assert pred.shape == (15,)
    np.testing.assert_array_equal(pred, ytr[ref_idx])


def test_drill60_knn_majority_vote_and_tie_break():
    Xtr = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    ytr = np.array([2, 2, 1, 0, 0])
    Xte = np.array([[0.5], [10.5], [1.2]])
    # k=3: neighbours of 0.5 -> labels {2,2,1}; of 10.5 -> {0,0,1}; of 1.2 -> {2,2,1}
    np.testing.assert_array_equal(knn_predict(Xtr, ytr, Xte, 3), [2, 0, 2])
    # k=2 at 1.6: neighbours 1.0 (label 2) and 2.0 (label 1): tie -> smallest label.
    np.testing.assert_array_equal(knn_predict(Xtr, ytr, np.array([[1.6]]), 2), [1])
    # k = N: global majority (2 vs 2 vs 1 -> tie between 0 and 2 -> 0).
    np.testing.assert_array_equal(knn_predict(Xtr, ytr, np.array([[0.0]]), 5), [0])


def test_drill60_nn1_linear_layer_matches_argmin():
    rng = np.random.default_rng(2)
    Xtr = rng.normal(size=(30, 4)) * np.array([1.0, 3.0, 0.2, 5.0])
    Xte = rng.normal(size=(25, 4)) * 2
    W, b = nn1_as_linear(Xtr)
    assert W.shape == (30, 4) and b.shape == (30,)
    pred = (Xte @ W.T + b).argmax(1)
    ref = torch.cdist(torch.from_numpy(Xte), torch.from_numpy(Xtr)).argmin(1).numpy()
    np.testing.assert_array_equal(pred, ref)


def test_drill60_nn1_linear_layer_is_affine_in_distance():
    # score_j(x) = -0.5 ||x - x_j||^2 + (term independent of j)
    rng = np.random.default_rng(3)
    Xtr = rng.normal(size=(10, 3))
    x = rng.normal(size=(1, 3))
    W, b = nn1_as_linear(Xtr)
    s = (x @ W.T + b)[0]
    d2 = ((x - Xtr) ** 2).sum(1)
    diff = s + 0.5 * d2
    np.testing.assert_allclose(diff, diff[0], atol=1e-10)


# ---------------------------------------------------------------------------
# Drill #61: logistic regression
# ---------------------------------------------------------------------------


def _logreg_data(seed=0, n=80, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5])[:d]
    p = 1 / (1 + np.exp(-(X @ w_true + 0.3)))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_drill61_loss_matches_torch_bce_with_logits():
    X, y = _logreg_data()
    rng = np.random.default_rng(5)
    w = rng.normal(size=3)
    b = 0.7
    l2 = 0.1
    loss, gw, gb = logistic_loss_and_grad(w, b, X, y, l2)
    wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    z = torch.from_numpy(X) @ wt + bt
    ref = F.binary_cross_entropy_with_logits(z, torch.from_numpy(y)) + 0.5 * l2 * (wt * wt).sum()
    ref.backward()
    np.testing.assert_allclose(loss, ref.item(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gw, wt.grad.numpy(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(gb, bt.grad.item(), rtol=1e-8, atol=1e-12)


def test_drill61_gradient_finite_differences():
    X, y = _logreg_data(seed=1)
    w = np.array([0.3, -0.1, 0.2])
    b = -0.4
    _, gw, gb = logistic_loss_and_grad(w, b, X, y, 0.05)
    eps = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        fd = (logistic_loss_and_grad(w + e, b, X, y, 0.05)[0]
              - logistic_loss_and_grad(w - e, b, X, y, 0.05)[0]) / (2 * eps)
        np.testing.assert_allclose(gw[i], fd, rtol=1e-5, atol=1e-8)
    fd_b = (logistic_loss_and_grad(w, b + eps, X, y, 0.05)[0]
            - logistic_loss_and_grad(w, b - eps, X, y, 0.05)[0]) / (2 * eps)
    np.testing.assert_allclose(gb, fd_b, rtol=1e-5, atol=1e-8)


def test_drill61_large_logits_are_finite():
    X = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    w = np.array([1e4])
    loss, gw, gb = logistic_loss_and_grad(w, 0.0, X, y)
    assert np.isfinite(loss) and np.isfinite(gw).all() and np.isfinite(gb)
    # Two examples are wrong by a margin of 1e4 each: mean loss = 1e4 / 2.
    np.testing.assert_allclose(loss, 5e3, rtol=1e-10)
    # dL/dz = (p - y)/N: [0, 1, -1, 0]/4 -> grad_w = (0 + 1 + 1 + 0)/4
    np.testing.assert_allclose(gw, [0.5], atol=1e-12)
    np.testing.assert_allclose(gb, 0.0, atol=1e-12)


def test_drill61_fit_reaches_regularised_optimum():
    X, y = _logreg_data(seed=2)
    l2 = 0.1
    w, b = fit_logistic_regression(X, y, lr=0.5, n_iters=3000, l2=l2)
    assert w.shape == (3,)

    def obj(theta):
        z = X @ theta[:3] + theta[3]
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * l2 * theta[:3] @ theta[:3]

    opt = scipy.optimize.minimize(obj, np.zeros(4), method="BFGS", options={"gtol": 1e-10})
    np.testing.assert_allclose(np.r_[w, b], opt.x, atol=1e-4)


# ---------------------------------------------------------------------------
# Drill #62: linear regression, ridge, LOO
# ---------------------------------------------------------------------------


def _reg_data(seed=3, n=8, d=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=n)


def test_drill62_ols_matches_lstsq():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 1.5 + 0.1 * rng.normal(size=30)
    w, b = ridge_normal_equation(X, y, 0.0)
    A = np.hstack([X, np.ones((30, 1))])
    ref = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose(np.r_[w, b], ref, atol=1e-10)


def test_drill62_ridge_stationarity_and_unpenalised_intercept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)) + 5.0  # offset: a penalised intercept would fail
    y = rng.normal(size=20) + 10.0
    l2 = 2.5
    w, b = ridge_normal_equation(X, y, l2)
    r = X @ w + b - y
    # Gradient of sum-of-squares + l2 ||w||^2 is zero; intercept makes residuals sum to 0.
    np.testing.assert_allclose(2 * X.T @ r + 2 * l2 * w, 0.0, atol=1e-9)
    np.testing.assert_allclose(r.sum(), 0.0, atol=1e-9)


def test_drill62_gd_matches_ols():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    y = X @ np.array([2.0, -1.0]) - 0.5 + 0.05 * rng.normal(size=50)
    w, b = linear_regression_gd(X, y, lr=0.1, n_iters=5000)
    A = np.hstack([X, np.ones((50, 1))])
    ref = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose(np.r_[w, b], ref, atol=1e-6)


def test_drill62_loo_residuals_literal():
    X, y = _reg_data()
    expected = np.array([1.75336819, -0.12123206, 1.41905696, 1.31807707,
                         -1.30385201, 0.57215628, -1.22793124, -0.24789248])
    np.testing.assert_allclose(ridge_loo_residuals(X, y, 0.7), expected, atol=1e-7)


def test_drill62_loo_residuals_match_explicit_refit_loop():
    X, y = _reg_data(seed=4, n=12, d=4)
    l2 = 0.3
    fast = ridge_loo_residuals(X, y, l2)
    assert fast.shape == (12,)
    slow = np.empty(12)
    for i in range(12):
        keep = np.arange(12) != i
        w, b = ridge_normal_equation(X[keep], y[keep], l2)
        slow[i] = y[i] - (X[i] @ w + b)
    np.testing.assert_allclose(fast, slow, atol=1e-10)


# ---------------------------------------------------------------------------
# Drill #63: PCA via SVD
# ---------------------------------------------------------------------------


def _pca_data(seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(100, 4)) * np.array([5.0, 2.0, 1.0, 0.3])
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    return Z @ Q.T + np.array([10.0, -3.0, 7.0, 1.0])


def test_drill63_variance_and_components_match_covariance_eig():
    X = _pca_data()
    comps, ev, scores = pca(X, 3)
    assert comps.shape == (3, 4) and ev.shape == (3,) and scores.shape == (100, 3)
    evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))  # ddof = 1, ascending
    np.testing.assert_allclose(ev, evals[::-1][:3], rtol=1e-10)
    # Same directions up to sign.
    np.testing.assert_allclose(np.abs(comps @ evecs[:, ::-1][:, :3]), np.eye(3), atol=1e-8)
    np.testing.assert_allclose(comps @ comps.T, np.eye(3), atol=1e-10)


def test_drill63_sign_convention_and_scores():
    X = _pca_data(seed=1)
    comps, ev, scores = pca(X, 4)
    idx = np.abs(comps).argmax(1)
    assert np.all(comps[np.arange(4), idx] > 0)
    Xc = X - X.mean(0)
    np.testing.assert_allclose(scores, Xc @ comps.T, atol=1e-10)
    # Full rank: scores reconstruct the centred data exactly.
    np.testing.assert_allclose(scores @ comps, Xc, atol=1e-10)
    np.testing.assert_allclose(scores.var(0, ddof=1), ev, rtol=1e-10)


def test_drill63_translation_invariant():
    X = _pca_data(seed=2)
    c1, e1, s1 = pca(X, 2)
    c2, e2, s2 = pca(X + 1000.0, 2)
    np.testing.assert_allclose(c1, c2, atol=1e-8)
    np.testing.assert_allclose(e1, e2, rtol=1e-8)
    np.testing.assert_allclose(s1, s2, atol=1e-6)
