"""Tests for src/drills/structure_losses.py (Drills #36-#41, #45)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from drills.structure_losses import (
    aligned_weighted_mse,
    bin_values,
    center_random_augmentation,
    compute_ptm,
    distogram_loss,
    distogram_target,
    edm_denoise,
    edm_diffusion_loss,
    edm_loss_weight,
    edm_preconditioning,
    expected_value_from_bins,
    karras_sample,
    karras_schedule,
    lddt,
    pde_loss,
    plddt_loss,
    quaternion_to_matrix,
    random_quaternions,
    sample_training_sigma,
    smooth_lddt_loss,
    weighted_minimum_rmsd,
    weighted_rigid_align,
    weighted_rmsd,
)


def _rot(seed: int, dtype=torch.float32) -> torch.Tensor:
    """A fixed random proper rotation matrix (column-vector convention)."""
    return torch.tensor(Rotation.random(random_state=seed).as_matrix(), dtype=dtype)


def _signed_volume(x: torch.Tensor) -> torch.Tensor:
    """Signed volume of the tetrahedron on the first four points of (N, 3)."""
    return torch.det(torch.stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]]))


# ---------------------------------------------------------------------------
# Drill #36
# ---------------------------------------------------------------------------


def test_drill36_target_bins_hand_case():
    xs = [0.0, 1.5, 2.0, 2.1, 5.0, 30.0]
    coords = torch.tensor([[[x, 0.0, 0.0] for x in xs]])
    t = distogram_target(coords)
    assert t.shape == (1, 6, 6, 64)
    assert t.dtype == torch.float32
    torch.testing.assert_close(t.sum(-1), torch.ones(1, 6, 6))
    bins = t.argmax(-1)[0, 0]  # distances from the atom at x = 0
    # d = 0, 1.5 and exactly 2.0 -> bin 0; 2.1 -> 1; 5.0 -> 10; 30 -> 63
    assert bins.tolist() == [0, 0, 0, 1, 10, 63]


def test_drill36_target_is_rigid_invariant():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 7, 3, generator=g) * 6
    y = x @ _rot(1).T + torch.tensor([3.0, -2.0, 10.0])
    torch.testing.assert_close(distogram_target(x), distogram_target(y))


def test_drill36_loss_matches_cross_entropy_one_hot():
    g = torch.Generator().manual_seed(0)
    B, L, K = 2, 5, 64
    coords = torch.randn(B, L, 3, generator=g) * 8
    logits = torch.randn(B, L, L, K, generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    target = distogram_target(coords)
    ce = F.cross_entropy(
        logits.reshape(-1, K), target.argmax(-1).reshape(-1), reduction="none"
    ).reshape(B, L, L)
    pm = mask[:, :, None] * mask[:, None, :] * (1 - torch.eye(L))
    expected = ((ce * pm).sum((1, 2)) / (1e-5 + pm.sum((1, 2)))).mean()
    torch.testing.assert_close(distogram_loss(logits, target, mask), expected, rtol=1e-5, atol=1e-6)


def test_drill36_soft_target_constant():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 4, 4, 6, generator=g)
    target = torch.softmax(torch.randn(2, 4, 4, 6, generator=g), -1)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
    out = distogram_loss(logits, target, mask)
    assert out.shape == ()
    assert out.item() == pytest.approx(2.197112560272217, rel=1e-5)


def test_drill36_large_logits_finite():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 4, 4, 8, generator=g) * 1e4
    target = F.one_hot(torch.randint(0, 8, (1, 4, 4), generator=g), 8).float()
    out = distogram_loss(logits, target, torch.ones(1, 4))
    assert torch.isfinite(out)
    assert out.item() > 0


def test_drill36_diagonal_and_padding_ignored():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 5, 5, 8, generator=g)
    target = F.one_hot(torch.randint(0, 8, (1, 5, 5), generator=g), 8).float()
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    base = distogram_loss(logits, target, mask)
    logits2 = logits.clone()
    logits2[0, range(5), range(5)] = 100 * torch.randn(5, 8, generator=g)  # diagonal
    logits2[0, 3:, :] = 50.0  # padded rows
    logits2[0, :, 3:] = -50.0  # padded columns
    torch.testing.assert_close(distogram_loss(logits2, target, mask), base)
    # same as the unpadded example
    small = distogram_loss(logits[:, :3, :3], target[:, :3, :3], torch.ones(1, 3))
    torch.testing.assert_close(small, base)


# ---------------------------------------------------------------------------
# Drill #37
# ---------------------------------------------------------------------------


def test_drill37_rigid_copy_gives_zero_rmsd():
    g = torch.Generator().manual_seed(0)
    true = torch.randn(2, 10, 3, generator=g) * 5
    R = torch.stack([_rot(3), _rot(4)])
    pred = true @ R.transpose(-1, -2) + torch.tensor([[[1.0, 2.0, 3.0]], [[-4.0, 0.0, 9.0]]])
    w = torch.ones(2, 10)
    m = torch.ones(2, 10)
    aligned = weighted_rigid_align(true, pred, w, m)
    assert aligned.shape == (2, 10, 3)
    torch.testing.assert_close(aligned, pred, atol=1e-4, rtol=0)
    assert weighted_rmsd(pred, true, w, m).max().item() < 1e-3
    assert aligned_weighted_mse(pred, true, w, m).max().item() < 1e-6


def test_drill37_matches_scipy_weighted_kabsch():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(8, 3)) * 4
    pred = true @ Rotation.random(random_state=5).as_matrix().T + rng.normal(size=(8, 3))
    w = rng.uniform(0.5, 2.0, size=8)
    mu_t = (w[:, None] * true).sum(0) / w.sum()
    mu_p = (w[:, None] * pred).sum(0) / w.sum()
    rot, rssd = Rotation.align_vectors(pred - mu_p, true - mu_t, weights=w)
    expected_aligned = rot.apply(true - mu_t) + mu_p
    expected_rmsd = rssd / math.sqrt(w.sum())

    t = torch.tensor(true[None], dtype=torch.float32)
    p = torch.tensor(pred[None], dtype=torch.float32)
    wt = torch.tensor(w[None], dtype=torch.float32)
    m = torch.ones(1, 8)
    aligned = weighted_rigid_align(t, p, wt, m)
    np.testing.assert_allclose(aligned[0].numpy(), expected_aligned, atol=1e-4)
    assert weighted_rmsd(p, t, wt, m).item() == pytest.approx(expected_rmsd, rel=1e-4)
    # factor 3 in the MSE, not in the RMSD
    mse = aligned_weighted_mse(p, t, wt, m).item()
    assert 3 * mse == pytest.approx(expected_rmsd**2, rel=1e-4)


def test_drill37_reflection_is_not_used():
    g = torch.Generator().manual_seed(1)
    true = torch.randn(1, 12, 3, generator=g) * 5
    pred = true * torch.tensor([-1.0, 1.0, 1.0])  # mirror image
    w, m = torch.ones(1, 12), torch.ones(1, 12)
    aligned = weighted_rigid_align(true, pred, w, m)
    # still a proper rigid copy of true: same distances, same handedness
    torch.testing.assert_close(torch.cdist(aligned, aligned), torch.cdist(true, true), atol=1e-4, rtol=0)
    assert torch.sign(_signed_volume(aligned[0])) == torch.sign(_signed_volume(true[0]))
    assert weighted_rmsd(pred, true, w, m).item() > 0.5


def test_drill37_equivariant_in_pred():
    g = torch.Generator().manual_seed(2)
    true = torch.randn(1, 9, 3, generator=g) * 3
    pred = torch.randn(1, 9, 3, generator=g) * 3
    w = torch.rand(1, 9, generator=g) + 0.5
    m = torch.ones(1, 9)
    Q, s = _rot(7), torch.tensor([5.0, -1.0, 2.0])
    a1 = weighted_rigid_align(true, pred, w, m)
    a2 = weighted_rigid_align(true, pred @ Q.T + s, w, m)
    torch.testing.assert_close(a2, a1 @ Q.T + s, atol=1e-4, rtol=0)
    r1 = weighted_rmsd(pred, true, w, m)
    r2 = weighted_rmsd(pred @ Q.T + s, true, w, m)
    torch.testing.assert_close(r1, r2, atol=1e-4, rtol=1e-4)


def test_drill37_masked_and_zero_weight_atoms_ignored():
    g = torch.Generator().manual_seed(3)
    true = torch.randn(1, 8, 3, generator=g) * 3
    pred = torch.randn(1, 8, 3, generator=g) * 3
    w = torch.ones(1, 8)
    w[0, 6] = 0.0
    m = torch.ones(1, 8)
    m[0, 7] = 0.0
    base = weighted_rmsd(pred, true, w, m)
    pred2 = pred.clone()
    pred2[0, 6:] += 100.0
    torch.testing.assert_close(weighted_rmsd(pred2, true, w, m), base, atol=1e-4, rtol=1e-4)


def test_drill37_target_detached_and_dtype_kept():
    g = torch.Generator().manual_seed(4)
    true = (torch.randn(2, 6, 3, generator=g, dtype=torch.float64) * 3).requires_grad_()
    pred = (torch.randn(2, 6, 3, generator=g, dtype=torch.float64) * 3).requires_grad_()
    w, m = torch.ones(2, 6, dtype=torch.float64), torch.ones(2, 6, dtype=torch.float64)
    aligned = weighted_rigid_align(true, pred, w, m)
    assert aligned.dtype == torch.float64
    assert not aligned.requires_grad
    mse = aligned_weighted_mse(pred, true, w, m)
    assert mse.shape == (2,)
    mse.sum().backward()
    assert true.grad is None
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_drill37_minimum_rmsd_picks_best_sample():
    g = torch.Generator().manual_seed(5)
    true = torch.randn(2, 10, 3, generator=g) * 4
    S = 4
    samples = true[:, None] + torch.randn(2, S, 10, 3, generator=g) * torch.tensor([2.0, 1.0, 0.5, 3.0])[None, :, None, None]
    samples[0, 2] = true[0] @ _rot(9).T + 7.0  # exact rigid copy
    w, m = torch.ones(2, 10), torch.ones(2, 10)
    rmsd, best = weighted_minimum_rmsd(samples, true, w, m)
    assert rmsd.shape == (2,) and best.shape == (2,)
    assert best.dtype == torch.int64
    assert best[0].item() == 2 and rmsd[0].item() < 1e-3
    assert best[1].item() == 2
    torch.testing.assert_close(rmsd[1], weighted_rmsd(samples[1:, 2], true[1:], w[1:], m[1:])[0])


# ---------------------------------------------------------------------------
# Drill #38
# ---------------------------------------------------------------------------


def _line(xs, dtype=torch.float32):
    return torch.tensor([[[x, 0.0, 0.0] for x in xs]], dtype=dtype)


def test_drill38_perfect_prediction():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 9, 3, generator=g) * 5
    nuc = torch.zeros(2, 9, dtype=torch.bool)
    m = torch.ones(2, 9)
    expected = 1.0 - torch.sigmoid(torch.tensor([0.5, 1.0, 2.0, 4.0])).mean().item()
    assert smooth_lddt_loss(x, x, nuc, m).item() == pytest.approx(expected, abs=1e-4)
    torch.testing.assert_close(lddt(x, x, nuc, m), torch.ones(2), atol=1e-4, rtol=0)


def test_drill38_hand_case():
    true = _line([0.0, 3.0, 6.0])
    pred = _line([0.0, 3.0, 7.5])
    nuc = torch.zeros(1, 3)
    m = torch.ones(1, 3)
    # pair errors 0, 1.5, 1.5 -> scores 1, 0.5, 0.5 over 6 ordered pairs
    assert lddt(pred, true, nuc, m).item() == pytest.approx(4 / 6, abs=1e-4)


def test_drill38_cutoff_uses_true_distances():
    true = _line([0.0, 3.0, 23.0])
    pred = _line([0.0, 3.0, 8.0])  # atom 2 is close in pred, far in true
    nuc = torch.zeros(1, 3)
    m = torch.ones(1, 3)
    assert lddt(pred, true, nuc, m).item() == pytest.approx(1.0, abs=1e-4)


def test_drill38_nucleotide_flag_is_on_the_row():
    true = _line([0.0, 3.0, 23.0])
    pred = _line([0.0, 3.0, 23.5])
    nuc = torch.tensor([[0.0, 0.0, 1.0]])
    m = torch.ones(1, 3)
    # pairs (0,1), (1,0) score 1; (2,0), (2,1) score 0.75; (0,2), (1,2) excluded
    assert lddt(pred, true, nuc, m).item() == pytest.approx(0.875, abs=1e-4)


def test_drill38_rigid_invariance_and_padding():
    g = torch.Generator().manual_seed(1)
    true = torch.randn(1, 10, 3, generator=g) * 5
    pred = true + torch.randn(1, 10, 3, generator=g) * 1.5
    nuc = (torch.rand(1, 10, generator=g) > 0.5).float()
    m = torch.ones(1, 10)
    m[0, 8:] = 0
    base = smooth_lddt_loss(pred, true, nuc, m)
    moved = pred @ _rot(11).T + torch.tensor([4.0, 4.0, -9.0])
    torch.testing.assert_close(smooth_lddt_loss(moved, true, nuc, m), base, atol=1e-5, rtol=1e-5)
    pred2 = pred.clone()
    pred2[0, 8:] = 1e3
    torch.testing.assert_close(smooth_lddt_loss(pred2, true, nuc, m), base, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        lddt(pred2, true, nuc, m), lddt(pred, true, nuc, m), atol=1e-6, rtol=1e-6
    )


def test_drill38_gradient_is_finite_and_matches_finite_differences():
    g = torch.Generator().manual_seed(2)
    true = torch.randn(1, 6, 3, generator=g, dtype=torch.float64) * 3
    pred = (true + torch.randn(1, 6, 3, generator=g, dtype=torch.float64)).requires_grad_()
    nuc = torch.zeros(1, 6, dtype=torch.float64)
    m = torch.ones(1, 6, dtype=torch.float64)
    loss = smooth_lddt_loss(pred, true, nuc, m)
    (grad,) = torch.autograd.grad(loss, pred)
    assert torch.isfinite(grad).all()
    assert torch.autograd.gradcheck(lambda p: smooth_lddt_loss(p, true, nuc, m), (pred,))


# ---------------------------------------------------------------------------
# Drill #39
# ---------------------------------------------------------------------------


def _edm_inputs():
    g = torch.Generator().manual_seed(2)
    coords = torch.randn(2, 5, 3, generator=g) * 4
    noise = torch.randn(2, 5, 3, generator=g)
    sigma = torch.tensor([3.0, 20.0])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    nuc = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]).bool()
    lig = torch.tensor([[0, 0, 0, 1, 0], [1, 1, 0, 0, 0]]).bool()
    return coords, mask, nuc, lig, sigma, noise


def _toy_net(x, c):
    return 0.5 * x + 0.1 * c[:, None, None] * torch.tensor([1.0, -2.0, 0.5])


def test_drill39_training_sigma_distribution():
    g = torch.Generator().manual_seed(0)
    s = sample_training_sigma(200_000, generator=g)
    assert s.shape == (200_000,) and s.dtype == torch.float32
    assert (s > 0).all()
    z = torch.log(s / 16.0)
    assert z.mean().item() == pytest.approx(-1.2, abs=0.02)
    assert z.std().item() == pytest.approx(1.5, abs=0.02)


def test_drill39_preconditioning_values():
    sigma = torch.tensor([16.0, 1e-4, 1e4])
    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, 16.0)
    assert c_skip[0].item() == pytest.approx(0.5, rel=1e-6)
    assert c_out[0].item() == pytest.approx(16 / math.sqrt(2), rel=1e-6)
    assert c_in[0].item() == pytest.approx(1 / (16 * math.sqrt(2)), rel=1e-6)
    assert c_noise[0].item() == pytest.approx(0.0, abs=1e-7)
    assert c_noise[1].item() == pytest.approx(0.25 * math.log(1e-4 / 16), rel=1e-5)
    # sigma -> 0: D(x) -> x
    assert c_skip[1].item() == pytest.approx(1.0, abs=1e-6)
    assert c_out[1].item() == pytest.approx(1e-4, rel=1e-3)
    # the loss weight is 1 / c_out^2
    torch.testing.assert_close(edm_loss_weight(sigma, 16.0) * c_out**2, torch.ones(3), rtol=1e-5, atol=0)


def test_drill39_denoise_calls_net_with_scaled_input():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 3, generator=g)
    sigma = torch.tensor([16.0, 1e-4])
    seen = {}

    def net(xs, c):
        seen["x"], seen["c"] = xs, c
        return torch.ones_like(xs)

    D = edm_denoise(net, x, sigma, 16.0)
    assert D.shape == (2, 4, 3)
    torch.testing.assert_close(seen["x"][0], x[0] / (16 * math.sqrt(2)))
    assert seen["c"].shape == (2,)
    torch.testing.assert_close(D[0], 0.5 * x[0] + 16 / math.sqrt(2))
    torch.testing.assert_close(D[1], x[1], atol=2e-4, rtol=0)  # tiny sigma: D(x) ~ x


def test_drill39_loss_constant():
    coords, mask, nuc, lig, sigma, noise = _edm_inputs()
    out = edm_diffusion_loss(_toy_net, coords, mask, nuc, lig, sigma, noise)
    assert out.shape == ()
    assert out.item() == pytest.approx(0.6004035472869873, rel=1e-4)
    out2 = edm_diffusion_loss(_toy_net, coords, mask, nuc, lig, sigma, noise, add_smooth_lddt=True)
    assert out2.item() == pytest.approx(1.5043939352035522, rel=1e-4)


def test_drill39_rotation_invariance_and_gradient():
    coords, mask, nuc, lig, sigma, noise = _edm_inputs()
    scale = torch.nn.Parameter(torch.tensor(0.7))
    net = lambda x, c: scale * x  # rotation-equivariant
    loss = edm_diffusion_loss(net, coords, mask, nuc, lig, sigma, noise)
    R = _rot(13)
    loss_r = edm_diffusion_loss(net, coords @ R.T, mask, nuc, lig, sigma, noise @ R.T)
    torch.testing.assert_close(loss_r, loss, rtol=1e-4, atol=1e-5)
    loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad) and scale.grad.abs() > 0


def test_drill39_padded_atoms_ignored():
    coords, mask, nuc, lig, sigma, noise = _edm_inputs()
    base = edm_diffusion_loss(_toy_net, coords, mask, nuc, lig, sigma, noise)
    coords2, noise2 = coords.clone(), noise.clone()
    coords2[0, 4] = 500.0
    noise2[0, 4] = -3.0
    out = edm_diffusion_loss(_toy_net, coords2, mask, nuc, lig, sigma, noise2)
    torch.testing.assert_close(out, base, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Drill #40
# ---------------------------------------------------------------------------


def test_drill40_schedule_endpoints():
    s = karras_schedule(200)
    assert s.shape == (201,)
    assert s[0].item() == pytest.approx(2560.0, rel=1e-5)
    assert s[-2].item() == pytest.approx(1e-4 * 16, rel=1e-4)
    assert s[-1].item() == 0.0
    assert (s[1:] < s[:-1]).all()
    s1 = karras_schedule(50, sigma_min=4e-4, rho=8.0)
    assert s1[-2].item() == pytest.approx(4e-4 * 16, rel=1e-4)


def test_drill40_schedule_values():
    s = karras_schedule(5)
    expected = torch.tensor([2560.0, 459.78927612, 47.030639648, 1.56473279, 1.6e-3, 0.0])
    torch.testing.assert_close(s, expected, rtol=1e-4, atol=1e-7)


def test_drill40_sampler_noise_levels_passed_to_denoiser():
    sigmas = torch.tensor([50.0, 10.0, 2.0, 0.5, 0.0])
    calls = []

    def D(x, t):
        calls.append(float(t))
        return torch.zeros_like(x)

    g = torch.Generator().manual_seed(0)
    out = karras_sample(D, (2, 5, 3), sigmas, gamma_0=0.8, gamma_min=1.0, generator=g)
    assert out.shape == (2, 5, 3)
    # churn only when the NEXT sigma is above gamma_min
    assert calls == pytest.approx([90.0, 18.0, 2.0, 0.5])


def test_drill40_sampler_converges_to_fixed_target():
    mu = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]])
    D = lambda x, t: mu.expand_as(x)
    sigmas = karras_schedule(20)
    g = torch.Generator().manual_seed(0)
    out = karras_sample(D, (3, 2, 3), sigmas, gamma_0=0.8, step_scale=1.0, generator=g)
    torch.testing.assert_close(out, mu.expand(3, 2, 3), atol=1e-3, rtol=0)


def test_drill40_sampler_reproducible_with_generator():
    sigmas = karras_schedule(10)
    D = lambda x, t: 0.9 * x
    a = karras_sample(D, (4, 3), sigmas, generator=torch.Generator().manual_seed(5))
    b = karras_sample(D, (4, 3), sigmas, generator=torch.Generator().manual_seed(5))
    c = karras_sample(D, (4, 3), sigmas, generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


# ---------------------------------------------------------------------------
# Drill #41
# ---------------------------------------------------------------------------


def test_drill41_quaternions_unit_and_positive():
    q = random_quaternions(1000, generator=torch.Generator().manual_seed(0))
    assert q.shape == (1000, 4) and q.dtype == torch.float32
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(1000))
    assert (q[:, 0] >= 0).all()


def test_drill41_quaternion_to_matrix_matches_scipy():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    expected = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()  # scipy is (x, y, z, w)
    R = quaternion_to_matrix(torch.tensor(q, dtype=torch.float64))
    assert R.shape == (20, 3, 3)
    np.testing.assert_allclose(R.numpy(), expected, atol=1e-10)
    # a non-unit quaternion gives the same rotation
    R3 = quaternion_to_matrix(3.0 * torch.tensor(q, dtype=torch.float64))
    np.testing.assert_allclose(R3.numpy(), expected, atol=1e-10)


def test_drill41_rotations_are_proper_and_uniform():
    q = random_quaternions(20_000, generator=torch.Generator().manual_seed(1))
    R = quaternion_to_matrix(q)
    eye = torch.eye(3).expand(20_000, 3, 3)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, atol=1e-5, rtol=0)
    torch.testing.assert_close(torch.det(R), torch.ones(20_000), atol=1e-5, rtol=0)
    # Haar-uniform rotations average to the zero matrix
    assert R.mean(0).abs().max().item() < 0.03


def test_drill41_augmentation_is_rigid_and_centred():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 8, 3, generator=g) * 5 + 20.0
    m = torch.ones(3, 8)
    out = center_random_augmentation(x, m, s_trans=0.0, generator=torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    torch.testing.assert_close(torch.cdist(out, out), torch.cdist(x, x), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.mean(1), torch.zeros(3, 3), atol=1e-4, rtol=0)
    for b in range(3):
        assert torch.sign(_signed_volume(out[b])) == torch.sign(_signed_volume(x[b]))
    centred = x - x.mean(1, keepdim=True)
    assert not torch.allclose(out, centred, atol=1e-2)  # it actually rotated


def test_drill41_padding_excluded_from_centroid():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 6, 3, generator=g)
    x[:, 4:] = 1e3  # padding
    m = torch.ones(2, 6)
    m[:, 4:] = 0
    out = center_random_augmentation(x, m, s_trans=0.0, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(out[:, :4].mean(1), torch.zeros(2, 3), atol=1e-4, rtol=0)
    a = center_random_augmentation(x, m, s_trans=1.0, generator=torch.Generator().manual_seed(4))
    b = center_random_augmentation(x, m, s_trans=1.0, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Drill #45
# ---------------------------------------------------------------------------


def test_drill45_bin_values():
    x = torch.tensor([0.0, 0.019, 0.03, 0.5, 0.999, 1.0, 1.2, -0.1])
    b = bin_values(x, 1.0, 50)
    assert b.dtype == torch.int64
    assert b.tolist() == [0, 0, 1, 25, 49, 49, 49, 0]
    e = torch.tensor([0.0, 0.49, 0.5, 31.9, 32.0, 100.0])
    assert bin_values(e, 32.0, 64).tolist() == [0, 0, 1, 63, 63, 63]


def test_drill45_expected_value_decode():
    assert expected_value_from_bins(torch.zeros(3, 50), 1.0).tolist() == pytest.approx([0.5] * 3)
    peaked = torch.full((2, 64), -1e4)
    peaked[0, 0] = 1e4
    peaked[1, 10] = 1e4
    out = expected_value_from_bins(peaked, 32.0)
    assert torch.isfinite(out).all()
    assert out.tolist() == pytest.approx([0.25, 5.25], abs=1e-5)  # bin width 0.5


def test_drill45_plddt_loss_matches_cross_entropy():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 50, generator=g)
    lddt_t = torch.tensor([[0.0, 0.31, 0.77, 1.0, 0.5], [0.99, 0.05, 0.2, 0.6, 0.45]])
    bins = torch.tensor([[0, 15, 38, 49, 25], [49, 2, 10, 30, 22]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    ce = F.cross_entropy(logits.reshape(-1, 50), bins.reshape(-1), reduction="none").reshape(2, 5)
    expected = ((ce * mask).sum(1) / (mask.sum(1) + 1e-5)).mean()
    torch.testing.assert_close(plddt_loss(logits, lddt_t, mask), expected, rtol=1e-5, atol=1e-6)


def test_drill45_plddt_loss_large_logits():
    logits = torch.full((1, 3, 50), -1e4)
    logits[0, :, 49] = 1e4
    target = torch.ones(1, 3)  # lDDT = 1.0 must land in the last bin
    out = plddt_loss(logits, target, torch.ones(1, 3))
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(0.0, abs=1e-4)


def test_drill45_pde_loss_constant_and_rigid_invariance():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 4, 4, 8, generator=g)
    pred = torch.randn(2, 4, 3, generator=g) * 5
    true = torch.randn(2, 4, 3, generator=g) * 5
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
    out = pde_loss(logits, pred, true, mask, max_dist=8.0)
    assert out.shape == ()
    assert out.item() == pytest.approx(2.406282663345337, rel=1e-5)
    moved = pred @ _rot(17).T + 3.0
    assert pde_loss(logits, moved, true, mask, max_dist=8.0).item() == pytest.approx(out.item(), rel=1e-4)


def test_drill45_ptm_hand_case_and_padding():
    N, K = 10, 64
    logits = torch.full((1, N + 2, N + 2, K), -1e4)
    logits[..., 0] = 1e4  # all mass on the first bin, centre 0.25 A
    mask = torch.ones(1, N + 2)
    mask[0, N:] = 0
    d0 = 1.24 * (19 - 15) ** (1 / 3) - 1.8  # n = 10 is clamped to 19
    expected = 1 / (1 + (0.25 / d0) ** 2)
    out = compute_ptm(logits, mask)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected, rel=1e-4)


def test_drill45_ptm_and_iptm_constants():
    g = torch.Generator().manual_seed(3)
    torch.randn(2, 4, 4, 8, generator=g)  # keep the same random stream as the reference
    torch.randn(2, 4, 3, generator=g)
    torch.randn(2, 4, 3, generator=g)
    pae = torch.randn(2, 4, 4, 64, generator=g) * 3
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
    asym = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
    torch.testing.assert_close(compute_ptm(pae, mask), torch.tensor([0.0038712309, 0.0111159533]), rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(compute_ptm(pae, mask, asym), torch.tensor([0.0014516592, 0.0147510683]), rtol=1e-4, atol=1e-8)
    single = compute_ptm(pae, mask, torch.zeros(2, 4, dtype=torch.long))
    torch.testing.assert_close(single, torch.zeros(2), atol=1e-6, rtol=0)
