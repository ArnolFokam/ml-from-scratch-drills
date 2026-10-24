"""Tests for Section 1: losses and metrics (Drills #1-#10)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats

from drills import losses as L


# ---------------------------------------------------------------- Drill #1


def test_drill01_softmax_matches_torch_any_axis():
    x = np.random.randn(3, 4, 5)
    for axis in (0, 1, -1):
        got = L.softmax(x, axis=axis)
        ref = torch.softmax(torch.from_numpy(x), dim=axis).numpy()
        assert got.shape == x.shape
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_drill01_log_softmax_matches_torch():
    x = np.random.randn(4, 7)
    got = L.log_softmax(x, axis=1)
    ref = torch.log_softmax(torch.from_numpy(x), dim=1).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_drill01_large_values_stay_finite():
    x = np.array([[1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -999.0]])
    p = L.softmax(x)
    lp = L.log_softmax(x)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(lp))
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(lp, torch.log_softmax(torch.from_numpy(x), -1).numpy(), atol=1e-9)
    assert lp[0, 2] == pytest.approx(-2000.0)


def test_drill01_fp16_and_input_not_modified():
    x = (np.random.randn(2, 6) * 8).astype(np.float16)
    x_copy = x.copy()
    p = L.softmax(x)
    lp = L.log_softmax(x)
    np.testing.assert_array_equal(x, x_copy)
    assert p.dtype == np.float16 and lp.dtype == np.float16
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(lp))
    np.testing.assert_allclose(p.astype(np.float32).sum(-1), 1.0, atol=5e-3)


# ---------------------------------------------------------------- Drill #2


def _torch_ce_and_grad(logits, targets):
    t = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    loss = F.cross_entropy(t, torch.tensor(targets))
    loss.backward()
    return loss.item(), t.grad.numpy()


def test_drill02_forward_and_backward_match_torch():
    logits = np.random.randn(6, 5)
    targets = np.array([0, 4, 2, 2, 1, 3])
    loss, grad = L.softmax_cross_entropy(logits, targets)
    ref_loss, ref_grad = _torch_ce_and_grad(logits, targets)
    assert isinstance(loss, float)
    assert loss == pytest.approx(ref_loss, rel=1e-9)
    assert grad.shape == logits.shape
    np.testing.assert_allclose(grad, ref_grad, rtol=1e-8, atol=1e-12)


def test_drill02_large_logits_and_initial_loss():
    logits = np.array([[1000.0, -1000.0, 0.0], [0.0, 1000.0, 999.0]])
    targets = np.array([1, 2])
    loss, grad = L.softmax_cross_entropy(logits, targets)
    ref_loss, ref_grad = _torch_ce_and_grad(logits, targets)
    assert math.isfinite(loss) and np.all(np.isfinite(grad))
    assert loss == pytest.approx(ref_loss, rel=1e-9)
    np.testing.assert_allclose(grad, ref_grad, atol=1e-12)
    # tiny logits -> loss close to ln C
    small = np.random.randn(2000, 10) * 1e-3
    l0, _ = L.softmax_cross_entropy(small, np.random.randint(0, 10, 2000))
    assert l0 == pytest.approx(math.log(10), abs=1e-3)


def test_drill02_gradient_rows_sum_to_zero():
    logits = np.random.randn(4, 3)
    _, grad = L.softmax_cross_entropy(logits, np.array([0, 1, 2, 0]))
    np.testing.assert_allclose(grad.sum(1), 0.0, atol=1e-12)


# ---------------------------------------------------------------- Drill #3


def test_drill03_matches_torch_with_ignore_index():
    logits = torch.randn(2, 5, 4, dtype=torch.float64)
    targets = torch.tensor([[0, 3, 1, -100, -100], [2, 2, -100, 1, 0]])
    got = L.masked_cross_entropy(logits, targets)
    ref = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)


def test_drill03_mask_smoothing_temperature_match_torch():
    logits = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 5, (3, 4))
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=torch.bool)
    got = L.masked_cross_entropy(logits, targets, mask=mask, label_smoothing=0.1, temperature=2.0)
    t_ref = torch.where(mask, targets, torch.full_like(targets, -100))
    ref = F.cross_entropy(
        (logits / 2.0).reshape(-1, 5), t_ref.reshape(-1), ignore_index=-100, label_smoothing=0.1
    )
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)
    got.backward()
    assert logits.grad is not None
    # masked positions get exactly zero gradient
    assert torch.all(logits.grad[~mask] == 0)


def test_drill03_padding_invariance():
    logits = torch.randn(1, 3, 6, dtype=torch.float64)
    targets = torch.tensor([[1, 4, 0]])
    base = L.masked_cross_entropy(logits, targets)
    pad_logits = torch.cat([logits, 50 * torch.randn(1, 4, 6, dtype=torch.float64)], dim=1)
    pad_targets = torch.cat([targets, torch.full((1, 4), -100)], dim=1)
    padded = L.masked_cross_entropy(pad_logits, pad_targets, label_smoothing=0.0)
    torch.testing.assert_close(base, padded, rtol=1e-10, atol=1e-12)


def test_drill03_empty_mask_returns_zero_not_nan():
    logits = torch.randn(2, 3, 4)
    targets = torch.randint(0, 4, (2, 3))
    mask = torch.zeros(2, 3, dtype=torch.bool)
    out = L.masked_cross_entropy(logits, targets, mask=mask, label_smoothing=0.1)
    assert torch.isfinite(out)
    assert out.item() == 0.0


# ---------------------------------------------------------------- Drill #4


def test_drill04_matches_torch_loss_and_grad():
    z = np.random.randn(8) * 3
    y = (np.random.rand(8) > 0.5).astype(np.float64)
    loss, dz = L.bce_with_logits(z, y)
    zt = torch.tensor(z, requires_grad=True)
    ref = F.binary_cross_entropy_with_logits(zt, torch.tensor(y))
    ref.backward()
    assert loss == pytest.approx(ref.item(), rel=1e-10)
    np.testing.assert_allclose(dz, zt.grad.numpy(), rtol=1e-9, atol=1e-14)


def test_drill04_extreme_logits():
    z = np.array([1000.0, -1000.0, 1000.0, -1000.0])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    loss, dz = L.bce_with_logits(z, y)
    # two perfect predictions (0 loss) and two wrong by 1000 each
    assert loss == pytest.approx(500.0, rel=1e-12)
    np.testing.assert_allclose(dz, [0.0, 0.0, 0.25, -0.25], atol=1e-12)


def test_drill04_finite_difference_gradient():
    z = np.random.randn(5)
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    _, dz = L.bce_with_logits(z, y)
    eps = 1e-6
    for i in range(5):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd = (L.bce_with_logits(zp, y)[0] - L.bce_with_logits(zm, y)[0]) / (2 * eps)
        assert dz[i] == pytest.approx(fd, abs=1e-7)


# ---------------------------------------------------------------- Drill #5

_FZ = torch.tensor([-3.0, -0.5, 0.0, 0.7, 2.5, 8.0])
_FY = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])


def test_drill05_gamma_zero_is_bce():
    z = torch.randn(10, dtype=torch.float64) * 4
    y = (torch.rand(10) > 0.5).double()
    got = L.binary_focal_loss(z, y, alpha=None, gamma=0.0, reduction="mean")
    ref = F.binary_cross_entropy_with_logits(z, y)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "alpha,gamma,expected",
    [
        (None, 2.0, [0.00010928329, 0.37741157, 0.17328680, 0.49254480, 0.00045396807, 7.9949713]),
        (0.25, 2.0, [8.1962469e-05, 0.094352894, 0.043321699, 0.36940861, 0.00011349202, 5.9962282]),
        (None, 1.0, [0.0023042972, 0.60632324, 0.34657359, 0.73713535, 0.0059844316, 7.9976530]),
        (0.25, 1.0, [0.0017282229, 0.15158081, 0.086643398, 0.55285150, 0.0014961079, 5.9982395]),
    ],
)
def test_drill05_known_values(alpha, gamma, expected):
    got = L.binary_focal_loss(_FZ, _FY, alpha=alpha, gamma=gamma, reduction="none")
    assert got.shape == (6,)
    torch.testing.assert_close(got, torch.tensor(expected), rtol=1e-5, atol=1e-7)


def test_drill05_reductions_and_large_logits():
    z = torch.tensor([1000.0, -1000.0, 1000.0])
    y = torch.tensor([1.0, 0.0, 0.0])
    none = L.binary_focal_loss(z, y, alpha=0.25, gamma=2.0, reduction="none")
    assert torch.all(torch.isfinite(none))
    torch.testing.assert_close(none, torch.tensor([0.0, 0.0, 750.0]))
    torch.testing.assert_close(L.binary_focal_loss(z, y, 0.25, 2.0, "sum"), torch.tensor(750.0))
    torch.testing.assert_close(L.binary_focal_loss(z, y, 0.25, 2.0, "mean"), torch.tensor(250.0))


# ---------------------------------------------------------------- Drill #6


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def test_drill06_matches_symmetric_ce_with_b_ne_d():
    B, D = 5, 3  # B != D so a missing transpose fails
    img = _norm(torch.randn(B, D, dtype=torch.float64))
    txt = _norm(torch.randn(B, D, dtype=torch.float64))
    ls = torch.tensor(math.log(10.0), dtype=torch.float64)
    got = L.clip_loss(img, txt, ls)
    logits = 10.0 * img @ txt.T
    lab = torch.arange(B)
    ref = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
    assert got.shape == ()
    torch.testing.assert_close(got.double(), ref, rtol=1e-6, atol=1e-8)


def test_drill06_asymmetric_case_needs_both_directions():
    # Hand-built: the image->text and text->image losses differ here.
    img = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    txt = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ls = torch.tensor(0.0)
    logits = img @ txt.T
    lab = torch.arange(3)
    one_dir = F.cross_entropy(logits, lab)
    got = L.clip_loss(img, txt, ls)
    ref = 0.5 * (one_dir + F.cross_entropy(logits.T, lab))
    assert not torch.isclose(one_dir, ref)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_drill06_gradient_flows_to_log_scale_and_fp16():
    img = _norm(torch.randn(4, 6))
    txt = _norm(torch.randn(4, 6))
    ls = torch.tensor(math.log(1 / 0.07), requires_grad=True)
    out = L.clip_loss(img.half(), txt.half(), ls)
    assert out.dtype == torch.float32
    assert torch.isfinite(out)
    out.backward()
    assert ls.grad is not None and torch.isfinite(ls.grad)
    # perfectly aligned pairs at a large scale -> loss near 0
    eye = torch.eye(4)
    assert L.clip_loss(eye, eye, torch.tensor(5.0)).item() < 1e-3


# ---------------------------------------------------------------- Drill #7

_P = torch.tensor([5.0, 6.2, 4.1, 7.0, 5.5])
_Y = torch.tensor([5.3, 6.0, 5.0, 6.0, 6.5])
_G = torch.tensor([False, False, True, True, False])


def test_drill07_censored_huber_uncensored_matches_torch():
    pred = torch.randn(6, dtype=torch.float64)
    y = torch.randn(6, dtype=torch.float64)
    got = L.censored_huber_loss(pred, y, torch.zeros(6, dtype=torch.bool), delta=0.5)
    torch.testing.assert_close(got, F.huber_loss(pred, y, delta=0.5))


def test_drill07_censored_known_values_and_zero_grad():
    torch.testing.assert_close(L.censored_huber_loss(_P, _Y, _G, delta=0.5), torch.tensor(0.153), atol=1e-6, rtol=0)
    torch.testing.assert_close(L.censored_huber_loss(_P, _Y, _G, delta=1.0), torch.tensor(0.194), atol=1e-6, rtol=0)
    p = _P.clone().requires_grad_(True)
    L.censored_huber_loss(p, _Y, _G).backward()
    # row 3 is a satisfied '>' label (pred 7.0 > bound 6.0): no gradient
    assert p.grad[3].item() == 0.0
    assert p.grad[2].item() < 0.0  # unsatisfied '>' row pushes pred up


def test_drill07_pairwise_known_values_and_offset_invariance():
    torch.testing.assert_close(L.pairwise_difference_loss(_P, _Y, delta=0.5), torch.tensor(0.393), atol=1e-6, rtol=0)
    torch.testing.assert_close(L.pairwise_difference_loss(_P, _Y, delta=1.0), torch.tensor(0.5875), atol=1e-6, rtol=0)
    pred = torch.randn(5, dtype=torch.float64)
    y = torch.randn(5, dtype=torch.float64)
    base = L.pairwise_difference_loss(pred, y)
    torch.testing.assert_close(L.pairwise_difference_loss(pred + 3.7, y), base)
    torch.testing.assert_close(L.pairwise_difference_loss(pred, y - 2.0), base)
    # perfect up to an offset -> zero
    assert L.pairwise_difference_loss(y + 1.0, y).item() == pytest.approx(0.0, abs=1e-12)


def test_drill07_total_loss():
    torch.testing.assert_close(L.affinity_loss(_P, _Y, _G), torch.tensor(0.369), atol=1e-6, rtol=0)
    extra = torch.tensor(0.25)
    torch.testing.assert_close(
        L.affinity_loss(_P, _Y, _G, binary_loss=extra, w_dif=0.5, w_abs=0.5),
        torch.tensor(0.5 * 0.393 + 0.5 * 0.153 + 0.25),
        atol=1e-6,
        rtol=0,
    )


# ---------------------------------------------------------------- Drill #8


def test_drill08_kl_discrete_matches_torch_distributions():
    p = torch.softmax(torch.randn(4, 6, dtype=torch.float64), -1)
    q = torch.softmax(torch.randn(4, 6, dtype=torch.float64), -1)
    ref = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=p), torch.distributions.Categorical(probs=q)
    )
    got = L.kl_discrete(p, q, dim=-1)
    assert got.shape == (4,)
    torch.testing.assert_close(got, ref)
    # dim argument
    torch.testing.assert_close(L.kl_discrete(p.T, q.T, dim=0), ref)


def test_drill08_kl_discrete_zero_probabilities():
    p = torch.tensor([0.5, 0.5, 0.0])
    q = torch.tensor([0.25, 0.25, 0.5])
    got = L.kl_discrete(p, q)
    assert torch.isfinite(got)
    assert got.item() == pytest.approx(math.log(2.0), rel=1e-6)
    assert L.kl_discrete(q, q).item() == pytest.approx(0.0, abs=1e-7)


def test_drill08_gaussian_kl_matches_torch():
    mu = torch.randn(3, 4, dtype=torch.float64)
    logvar = torch.randn(3, 4, dtype=torch.float64)
    Normal = torch.distributions.Normal
    ref = torch.distributions.kl_divergence(
        Normal(mu, torch.exp(0.5 * logvar)), Normal(torch.zeros_like(mu), torch.ones_like(mu))
    ).sum(-1)
    got = L.gaussian_kl(mu, logvar)
    assert got.shape == (3,)
    torch.testing.assert_close(got, ref)
    assert torch.all(L.gaussian_kl(torch.zeros(2, 5), torch.zeros(2, 5)) == 0)


def test_drill08_vae_loss_sum_then_mean():
    B, Fdim, Z = 4, 7, 3
    logits = torch.randn(B, Fdim, dtype=torch.float64)
    x = torch.rand(B, Fdim, dtype=torch.float64)
    mu = torch.randn(B, Z, dtype=torch.float64)
    logvar = torch.randn(B, Z, dtype=torch.float64)
    Normal = torch.distributions.Normal
    kl = torch.distributions.kl_divergence(
        Normal(mu, torch.exp(0.5 * logvar)), Normal(torch.zeros_like(mu), torch.ones_like(mu))
    ).sum()
    ref = (F.binary_cross_entropy_with_logits(logits, x, reduction="sum") + kl) / B
    torch.testing.assert_close(L.vae_loss(logits, x, mu, logvar), ref)


# ---------------------------------------------------------------- Drill #9


def test_drill09_prf_binary_hand_computed():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    p, r, f = L.precision_recall_f1(y_true, y_pred)  # TP=3 FP=2 FN=1
    assert (p, r) == (pytest.approx(0.6), pytest.approx(0.75))
    assert f == pytest.approx(2 * 0.6 * 0.75 / 1.35)


def test_drill09_prf_macro_micro_and_zero_division():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    # class 0: TP1 FP2 FN1 -> P1/3 R1/2 F0.4; class 1: TP2 FP1 FN0 -> P2/3 R1 F0.8; class 2: P0(0/0) R0 F0
    p, r, f = L.precision_recall_f1(y_true, y_pred, average="macro")
    assert p == pytest.approx((1 / 3 + 2 / 3 + 0) / 3)
    assert r == pytest.approx((0.5 + 1 + 0) / 3)
    assert f == pytest.approx((0.4 + 0.8 + 0) / 3)
    pm, rm, fm = L.precision_recall_f1(y_true, y_pred, average="micro")
    assert pm == rm == fm == pytest.approx(0.5)
    # no predicted positives -> precision is zero_division
    p0, r0, f0 = L.precision_recall_f1(np.array([1, 0]), np.array([0, 0]), zero_division=1.0)
    assert p0 == 1.0 and r0 == 0.0
    p2, _, _ = L.precision_recall_f1(np.array([1, 0]), np.array([0, 0]), zero_division=0.0)
    assert p2 == 0.0


def test_drill09_rankdata_with_ties_matches_scipy():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0])
    np.testing.assert_allclose(L.rankdata_average(x), stats.rankdata(x, method="average"))
    np.testing.assert_allclose(L.rankdata_average(np.array([10, 20, 20, 30])), [1, 2.5, 2.5, 4])


def test_drill09_auc_with_ties():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert L.roc_auc(y, s) == pytest.approx(0.75)
    # every score tied -> 0.5
    assert L.roc_auc(np.array([0, 1, 0, 1, 1]), np.ones(5)) == pytest.approx(0.5)
    # random with heavy ties vs Mann-Whitney U
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.integers(0, 5, 200).astype(float)
    u = stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic
    assert L.roc_auc(y, s) == pytest.approx(u / ((y == 1).sum() * (y == 0).sum()))


def test_drill09_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 6, 50).astype(float)
    y = x + rng.normal(size=50)
    assert L.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, rel=1e-10)
    assert L.spearman(np.arange(5.0), np.arange(5.0) ** 3) == pytest.approx(1.0)
    assert L.spearman(np.arange(5.0), -np.arange(5.0)) == pytest.approx(-1.0)


# ---------------------------------------------------------------- Drill #10


def test_drill10_enrichment_factor_basic():
    scores = np.arange(10, 0, -1).astype(float)  # item 0 ranks first
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # top 20% = 2 items, 1 active; base rate 2/10 -> EF = (1/2)/(0.2) = 2.5
    assert L.enrichment_factor(scores, labels, 0.2) == pytest.approx(2.5)
    # top 30% -> 3 items, both actives -> (2/3)/0.2
    assert L.enrichment_factor(scores, labels, 0.3) == pytest.approx((2 / 3) / 0.2)
    assert L.enrichment_factor(scores, labels, 1.0) == pytest.approx(1.0)
    # 15% of 10 -> ceil -> 2 items
    assert L.enrichment_factor(scores, labels, 0.15) == pytest.approx(2.5)


def test_drill10_ties_at_cutoff_are_order_independent():
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
    labels = np.array([0, 1, 0, 0, 1, 0])
    # top 2: item 0 + one of three tied items (1 active of 3) -> 1/3 expected actives
    assert L.top_k_recall(scores, labels, 2) == pytest.approx((1 / 3) / 2)
    perm = np.array([3, 1, 5, 0, 4, 2])
    assert L.top_k_recall(scores[perm], labels[perm], 2) == pytest.approx((1 / 3) / 2)
    # all tied -> EF exactly 1
    assert L.enrichment_factor(np.zeros(10), np.array([1, 1] + [0] * 8), 0.1) == pytest.approx(1.0)


def test_drill10_top_k_recall_edges():
    scores = np.array([0.3, 0.9, 0.1, 0.8])
    labels = np.array([1, 1, 0, 0])
    assert L.top_k_recall(scores, labels, 1) == pytest.approx(0.5)
    assert L.top_k_recall(scores, labels, 4) == pytest.approx(1.0)
    assert np.isnan(L.top_k_recall(scores, np.zeros(4), 2))


def test_drill10_bootstrap_resamples_groups_not_rows():
    # group A: 1 row, label 1; group B: 9 rows, label 0. metric = mean label.
    # Group resampling can only give {A,A}=1, {A,B}=0.1, {B,B}=0.
    labels = np.array([1] + [0] * 9, dtype=float)
    scores = np.zeros(10)
    groups = np.array(["A"] + ["B"] * 9)
    lo, hi = L.grouped_bootstrap_ci(scores, labels, groups, lambda s, l: float(np.mean(l)), n_boot=2000, seed=0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_drill10_bootstrap_degenerate_and_ordered():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=60)
    labels = (rng.random(60) < 0.3).astype(float)
    groups = np.repeat(np.arange(6), 10)
    metric = lambda s, l: float(np.mean(s[l == 1])) if l.sum() else 0.0  # noqa: E731
    lo, hi = L.grouped_bootstrap_ci(scores, labels, groups, metric, n_boot=300, ci=0.9, seed=1)
    assert lo <= hi
    # one group only -> every replicate equals the full-data metric
    one = np.zeros(10, dtype=int)
    lo1, hi1 = L.grouped_bootstrap_ci(scores[:10], labels[:10], one, metric, n_boot=50)
    assert lo1 == pytest.approx(metric(scores[:10], labels[:10]))
    assert hi1 == pytest.approx(lo1)
    # deterministic for a fixed seed
    again = L.grouped_bootstrap_ci(scores, labels, groups, metric, n_boot=300, ci=0.9, seed=1)
    assert again == (lo, hi)
