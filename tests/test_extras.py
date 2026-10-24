"""Tests for Section 9 extras (Drills #68-#78)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from drills import extras as X


def _rot(n, seed=0, dtype=torch.float64):
    return torch.from_numpy(Rotation.random(n, random_state=seed).as_matrix()).to(dtype)


# ---------------------------------------------------------------------------
# Drill #68
# ---------------------------------------------------------------------------


def test_drill68_masked_huber_matches_builtin_on_valid_entries():
    pred = torch.randn(3, 7) * 3
    target = torch.randn(3, 7)
    mask = torch.rand(3, 7) > 0.4
    out = X.masked_huber_loss(pred, target, mask, delta=1.5)
    ref = F.huber_loss(pred[mask], target[mask], delta=1.5, reduction="mean")
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_drill68_masked_huber_padding_and_empty_mask():
    pred = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 5)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=torch.bool)
    base = X.masked_huber_loss(pred, target, mask)
    base.backward()
    assert torch.all(pred.grad[~mask] == 0)
    garbage = pred.detach().clone()
    garbage[~mask] = 1e6
    torch.testing.assert_close(X.masked_huber_loss(garbage, target, mask), base.detach(), rtol=1e-6, atol=1e-6)
    p2 = torch.randn(2, 5, requires_grad=True)
    empty = X.masked_huber_loss(p2, target, torch.zeros(2, 5, dtype=torch.bool))
    assert empty.item() == 0.0
    empty.backward()
    assert torch.isfinite(p2.grad).all()


def test_drill68_masked_huber_gradcheck():
    pred = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 6, dtype=torch.float64)
    mask = torch.rand(2, 6) > 0.3
    assert torch.autograd.gradcheck(lambda p: X.masked_huber_loss(p, target, mask, 0.7), (pred,))


def test_drill68_tiny_regressor_shape_params_and_independence():
    m = X.TinyRegressor(4, hidden=8)
    assert sum(p.numel() for p in m.parameters()) == 4 * 8 + 8 + 8 + 1
    x = torch.randn(2, 5, 4)
    y = m(x)
    assert y.shape == (2, 5)
    x2 = x.clone()
    x2[:, 3] += 10.0
    y2 = m(x2)
    torch.testing.assert_close(y2[:, :3], y[:, :3])
    assert m(torch.randn(3, 1, 4)).shape == (3, 1)


def test_drill68_overfit_one_batch_records_losses_before_update():
    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(3, 1)

        def forward(self, x):
            return self.l(x)[..., 0]

    model = Lin()
    x = torch.randn(2, 6, 3)
    y = x @ torch.tensor([1.0, -2.0, 0.5]) + 0.3
    mask = torch.ones(2, 6, dtype=torch.bool)

    def loss_fn(p, t, m):
        return ((p - t) ** 2)[m].mean()

    with torch.no_grad():
        init = loss_fn(model(x), y, mask).item()
    losses = X.overfit_one_batch(model, loss_fn, x, y, mask, steps=400, lr=5e-2)
    assert len(losses) == 400 and all(isinstance(v, float) for v in losses)
    assert losses[0] == pytest.approx(init, rel=1e-6)
    assert losses[-1] < 1e-3 * losses[0]


def test_drill68_integrated_tiny_regressor_overfits_with_masked_huber():
    torch.manual_seed(1)
    model = X.TinyRegressor(5, hidden=32)
    x = torch.randn(4, 8, 5)
    y = torch.randn(4, 8) * 2
    mask = torch.rand(4, 8) > 0.3
    y_bad = y.clone()
    y_bad[~mask] = 1e4  # padding targets must not matter
    losses = X.overfit_one_batch(model, X.masked_huber_loss, x, y_bad, mask, steps=400, lr=1e-2)
    assert losses[-1] < 0.05 * losses[0]


# ---------------------------------------------------------------------------
# Drill #69
# ---------------------------------------------------------------------------


def _lddt_case(far_true=16.0, far_pred=20.0):
    pred = torch.tensor([[[0.0, 0, 0], [1.5, 0, 0], [0, 2.5, 0], [0, 0, far_pred]]], dtype=torch.float64)
    true = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0], [0, 0, far_true]]], dtype=torch.float64)
    return pred, true


def test_drill69_smooth_lddt_hand_values():
    pred, true = _lddt_case()
    m = torch.ones(1, 4, dtype=torch.bool)
    nuc0 = torch.zeros(1, 4, dtype=torch.bool)
    nuc3 = torch.tensor([[False, False, False, True]])
    out = X.smooth_lddt_loss_from_pseudocode(pred, true, nuc0, m)
    assert out.item() == pytest.approx(0.2824521694476576, abs=1e-6)
    out = X.smooth_lddt_loss_from_pseudocode(pred, true, nuc3, m)
    assert out.item() == pytest.approx(0.4643039273599283, abs=1e-6)


def test_drill69_smooth_lddt_cutoff_uses_true_distances():
    pred, true = _lddt_case(far_true=14.0, far_pred=17.0)
    m = torch.ones(1, 4, dtype=torch.bool)
    nuc0 = torch.zeros(1, 4, dtype=torch.bool)
    out = X.smooth_lddt_loss_from_pseudocode(pred, true, nuc0, m)
    assert out.item() == pytest.approx(0.49351905518473504, abs=1e-6)


def test_drill69_smooth_lddt_perfect_prediction_is_not_zero():
    _, true = _lddt_case()
    m = torch.ones(1, 4, dtype=torch.bool)
    out = X.smooth_lddt_loss_from_pseudocode(true, true, torch.zeros(1, 4, dtype=torch.bool), m)
    expected = 1 - 0.25 * sum(1 / (1 + math.exp(-t)) for t in (0.5, 1, 2, 4))
    assert out.item() == pytest.approx(expected, abs=1e-5)


def test_drill69_smooth_lddt_rigid_and_padding_invariance():
    B, N = 2, 9
    true = torch.randn(B, N, 3, dtype=torch.float64) * 5
    pred = true + torch.randn(B, N, 3, dtype=torch.float64)
    nuc = torch.rand(B, N) > 0.5
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 6:] = False
    base = X.smooth_lddt_loss_from_pseudocode(pred, true, nuc, mask)
    R = _rot(B, seed=3)
    moved = pred @ R.transpose(-1, -2) + torch.randn(B, 1, 3, dtype=torch.float64) * 10
    torch.testing.assert_close(X.smooth_lddt_loss_from_pseudocode(moved, true, nuc, mask), base, rtol=0, atol=1e-10)
    pred2, true2 = pred.clone(), true.clone()
    pred2[1, 6:] = 1e3
    true2[1, 6:] = -1e3
    torch.testing.assert_close(X.smooth_lddt_loss_from_pseudocode(pred2, true2, nuc, mask), base, rtol=0, atol=1e-10)


def test_drill69_rigid_align_moves_truth_onto_pred():
    B, N = 2, 8
    true = torch.randn(B, N, 3, dtype=torch.float64)
    R = _rot(B, seed=1)
    t = torch.randn(B, 1, 3, dtype=torch.float64) * 4
    pred = true @ R.transpose(-1, -2) + t
    w = torch.rand(B, N, dtype=torch.float64) + 0.5
    mask = torch.ones(B, N, dtype=torch.bool)
    out = X.weighted_rigid_align_from_pseudocode(true, pred, w, mask)
    assert out.shape == (B, N, 3)
    torch.testing.assert_close(out, pred, rtol=0, atol=1e-8)


def test_drill69_rigid_align_detached_and_proper_on_mirror():
    N = 10
    true = torch.randn(1, N, 3, dtype=torch.float64, requires_grad=True)
    pred = (true.detach() * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)).requires_grad_(True)
    w = torch.ones(1, N, dtype=torch.float64)
    mask = torch.ones(1, N, dtype=torch.bool)
    out = X.weighted_rigid_align_from_pseudocode(true, pred, w, mask)
    assert not out.requires_grad
    # rigid: distances preserved
    torch.testing.assert_close(torch.cdist(out, out), torch.cdist(true, true).detach(), rtol=0, atol=1e-8)
    # proper: handedness (signed volume) preserved, so it cannot equal the mirror image
    def vol(x):
        a = x[0, 1:4] - x[0, 0]
        return torch.det(a)
    assert torch.sign(vol(out)) == torch.sign(vol(true.detach()))
    assert (out - pred.detach()).abs().max() > 1e-2


def test_drill69_rigid_align_ignores_masked_and_zero_weight_atoms():
    B, N = 1, 8
    true = torch.randn(B, N, 3, dtype=torch.float64)
    pred = true @ _rot(1, seed=5).transpose(-1, -2) + 2.0
    pred_noisy = pred.clone()
    pred_noisy[0, 6] += 50.0  # masked
    pred_noisy[0, 7] -= 80.0  # zero weight
    w = torch.ones(B, N, dtype=torch.float64)
    w[0, 7] = 0.0
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 6] = False
    out = X.weighted_rigid_align_from_pseudocode(true, pred_noisy, w, mask)
    torch.testing.assert_close(out[0, :6], pred[0, :6], rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------
# Drill #70
# ---------------------------------------------------------------------------


def test_drill70_outgoing_pair_product_matches_matmul():
    a = torch.randn(2, 5, 5, 3)
    b = torch.randn(2, 5, 5, 3)
    out = X.outgoing_pair_product(a, b)
    ref = (a.permute(0, 3, 1, 2) @ b.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
    assert out.shape == (2, 5, 5, 3)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_drill70_outgoing_pair_product_not_symmetric_in_args():
    a = torch.randn(1, 4, 4, 2)
    b = torch.randn(1, 4, 4, 2)
    out = X.outgoing_pair_product(a, b)
    torch.testing.assert_close(out.transpose(1, 2), X.outgoing_pair_product(b, a))
    assert out[0, 1, 2, 0].item() == pytest.approx((a[0, 1, :, 0] * b[0, 2, :, 0]).sum().item(), abs=1e-5)


def test_drill70_sequence_mask_and_triu():
    m = X.sequence_mask(torch.tensor([0, 2, 4]), 4)
    expected = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    assert m.dtype == torch.bool
    assert torch.equal(m, expected)
    for d in (-1, 0, 2):
        assert torch.equal(X.triu_mask(5, d), torch.triu(torch.ones(5, 5), d).bool())


def test_drill70_pooling_hand_case_and_empty_token():
    feats = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [100.0, 100.0]]])
    a2t = torch.tensor([[0, 0, 2, 2]])
    am = torch.tensor([[True, True, True, False]])
    expected = torch.tensor([[[2.0, 1.0], [0.0, 0.0], [5.0, 4.0]]])
    for fn in (X.pool_atoms_to_tokens_onehot, X.pool_atoms_to_tokens_scatter):
        out = fn(feats, a2t, am, 3)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, expected)


def test_drill70_pooling_two_ways_agree():
    B, A, T, D = 3, 20, 7, 4
    feats = torch.randn(B, A, D)
    a2t = torch.randint(0, T, (B, A))
    am = torch.rand(B, A) > 0.3
    o1 = X.pool_atoms_to_tokens_onehot(feats, a2t, am, T)
    o2 = X.pool_atoms_to_tokens_scatter(feats, a2t, am, T)
    torch.testing.assert_close(o1, o2, rtol=1e-5, atol=1e-6)


def test_drill70_gather_tokens_to_atoms():
    tok = torch.randn(2, 4, 3)
    idx = torch.tensor([[0, 0, 3, 1, 2], [2, 2, 2, 0, 1]])
    out = X.gather_tokens_to_atoms(tok, idx)
    assert out.shape == (2, 5, 3)
    torch.testing.assert_close(out, tok[torch.arange(2)[:, None], idx])


# ---------------------------------------------------------------------------
# Drill #71
# ---------------------------------------------------------------------------


class _Harmonic(nn.Module):
    """E = 0.5 * k * sum over valid atoms of |x|^2, so F = -k x on valid atoms."""

    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, pos, mask):
        return 0.5 * self.k * ((pos**2).sum(-1) * mask).sum(-1)


class _Morse(nn.Module):
    def forward(self, pos, mask):
        d = (pos[:, :, None] - pos[:, None]).pow(2).sum(-1).add(1e-9).sqrt()
        pm = (mask[:, :, None] & mask[:, None]).double() * (1 - torch.eye(pos.shape[1], dtype=pos.dtype))
        return ((1 - torch.exp(-(d - 1.5))) ** 2 * pm).sum((-1, -2)) / 2


def test_drill71_forces_harmonic_and_create_graph():
    model = _Harmonic()
    pos = torch.randn(2, 4, 3, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    e, f = X.energy_and_forces(model, pos, mask)
    assert not pos.requires_grad
    assert e.shape == (2,) and f.shape == (2, 4, 3)
    torch.testing.assert_close(f, -2.0 * pos * mask[..., None])
    f.pow(2).sum().backward()  # needs create_graph=True
    assert model.k.grad is not None and model.k.grad.abs() > 0


def test_drill71_forces_match_finite_differences():
    model = _Morse()
    pos = torch.randn(1, 4, 3, dtype=torch.float64) * 1.5
    mask = torch.ones(1, 4, dtype=torch.bool)
    _, f = X.energy_and_forces(model, pos, mask)
    h = 1e-6
    for n in range(4):
        for c in range(3):
            p1, p2 = pos.clone(), pos.clone()
            p1[0, n, c] += h
            p2[0, n, c] -= h
            fd = -(model(p1, mask) - model(p2, mask)) / (2 * h)
            assert f[0, n, c].item() == pytest.approx(fd.item(), abs=1e-6)


def test_drill71_pair_energy_model_invariances():
    torch.manual_seed(0)
    model = X.PairEnergyModel(hidden=8).double()
    B, N = 2, 6
    pos = torch.randn(B, N, 3, dtype=torch.float64) * 2
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 4:] = False
    e = model(pos, mask)
    assert e.shape == (B,)
    R = _rot(B, seed=2)
    moved = pos @ R.transpose(-1, -2) + 3.0
    torch.testing.assert_close(model(moved, mask), e, rtol=0, atol=1e-9)
    perm = torch.tensor([3, 1, 0, 2, 4, 5])
    torch.testing.assert_close(model(pos[:, perm], mask[:, perm]), e, rtol=0, atol=1e-9)
    garbage = pos.clone()
    garbage[1, 4:] = 1e3
    torch.testing.assert_close(model(garbage, mask), e, rtol=0, atol=1e-9)


def test_drill71_pair_energy_forces_finite_with_padding_and_overlap():
    model = X.PairEnergyModel(hidden=8).double()
    pos = torch.zeros(1, 4, 3, dtype=torch.float64)
    pos[0, 1, 0] = 1.0
    pos[0, 2, 1] = 1.0  # atom 3 is padding sitting on atom 0
    pos.requires_grad_(True)
    mask = torch.tensor([[True, True, True, False]])
    (g,) = torch.autograd.grad(model(pos, mask).sum(), pos)
    assert torch.isfinite(g).all()
    assert torch.all(g[0, 3] == 0)


def test_drill71_energy_force_loss_hand_case():
    e_pred = torch.tensor([1.0, 2.0])
    e_true = torch.tensor([0.0, 4.0])
    f_true = torch.zeros(2, 3, 3)
    f_pred = torch.zeros(2, 3, 3)
    f_pred[0, 0] = torch.tensor([3.0, 4.0, 0.0])
    f_pred[0, 1] = torch.tensor([0.0, 0.0, 1.0])
    f_pred[0, 2] = torch.tensor([100.0, 100.0, 100.0])  # padding
    f_pred[1, 0] = torch.tensor([0.0, 0.0, 2.0])
    f_pred[1, 1:] = 50.0  # padding
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    # energy 1.5, forces structure-wise (3 + 2) / 2 = 2.5 (atom-wise would be 8/3)
    out = X.energy_force_loss(e_pred, f_pred, e_true, f_true, mask)
    assert out.item() == pytest.approx(4.0, abs=1e-6)
    out = X.energy_force_loss(e_pred, f_pred, e_true, f_true, mask, energy_weight=2.0, force_weight=0.5)
    assert out.item() == pytest.approx(4.25, abs=1e-6)


def test_drill71_energy_force_loss_empty_structure():
    e = torch.zeros(2)
    f_pred = torch.zeros(2, 2, 3)
    f_pred[0, 0] = torch.tensor([3.0, 4.0, 0.0])
    f_pred[1] = 7.0
    mask = torch.tensor([[True, False], [False, False]])
    out = X.energy_force_loss(e, f_pred, e, torch.zeros(2, 2, 3), mask)
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(2.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Drill #72
# ---------------------------------------------------------------------------


def test_drill72_random_rotations_are_proper_and_seeded():
    g = torch.Generator().manual_seed(7)
    R = X.random_rotation_matrices(500, generator=g)
    assert R.shape == (500, 3, 3) and R.dtype == torch.float64
    eye = torch.eye(3, dtype=torch.float64).expand(500, 3, 3)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, rtol=0, atol=1e-10)
    torch.testing.assert_close(torch.det(R), torch.ones(500, dtype=torch.float64), rtol=0, atol=1e-10)
    R2 = X.random_rotation_matrices(500, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(R, R2)


def test_drill72_random_rotations_are_uniform():
    # Under the Haar measure E[R] = 0 and E[trace R] = 0.
    R = X.random_rotation_matrices(4000, generator=torch.Generator().manual_seed(1))
    assert R.mean(0).abs().max() < 0.06
    assert abs(torch.diagonal(R, dim1=-2, dim2=-1).sum(-1).mean().item()) < 0.1


def test_drill72_apply_rigid_motion_matches_scipy():
    rot = Rotation.random(2, random_state=4)
    R = torch.from_numpy(rot.as_matrix())
    t = torch.randn(2, 3, dtype=torch.float64)
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    out = X.apply_rigid_motion(x, R, t)
    ref = np.stack([rot[b].apply(x[b].numpy()) for b in range(2)]) + t[:, None].numpy()
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-12)


def test_drill72_invariance_and_equivariance_checks():
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    R = _rot(2, seed=8)
    t = torch.randn(2, 3, dtype=torch.float64) * 3

    def dist(y):
        return torch.cdist(y, y)

    def centroid(y):
        return y.mean(1, keepdim=True)

    def const(y):
        return torch.zeros(y.shape[0], 1, 3, dtype=y.dtype)

    assert X.is_invariant(dist, x, R, t)
    assert not X.is_invariant(centroid, x, R, t)
    assert X.is_equivariant(centroid, x, R, t)
    assert not X.is_equivariant(const, x, R, t)
    assert X.is_invariant(const, x, R, t)
    assert not X.is_equivariant(lambda y: y[..., [1, 0, 2]], x, R, t)


# ---------------------------------------------------------------------------
# Drill #73
# ---------------------------------------------------------------------------


def _edge_set(ei):
    return set(map(tuple, ei.t().tolist()))


def test_drill73_knn_graph_hand_case_with_offsets_and_mask():
    line = torch.tensor([0.0, 1.0, 3.0, 7.0])
    pos = torch.zeros(2, 4, 3)
    pos[:, :, 0] = line
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    ei = X.knn_graph(pos, mask, k=2)
    assert ei.shape[0] == 2 and ei.dtype == torch.int64
    expected = {
        (1, 0), (2, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 3), (1, 3),
        (5, 4), (6, 4), (4, 5), (6, 5), (5, 6), (4, 6),
    }
    assert _edge_set(ei) == expected
    assert ei.shape[1] == len(expected)


def test_drill73_knn_graph_fewer_valid_than_k():
    pos = torch.randn(1, 6, 3)
    mask = torch.tensor([[True, False, True, False, True, False]])
    ei = X.knn_graph(pos, mask, k=5)
    assert ei.shape[1] == 6
    nodes = set(ei.flatten().tolist())
    assert nodes == {0, 2, 4}
    assert all(s != t for s, t in ei.t().tolist())


def test_drill73_mean_aggregate():
    msg = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 0.0]])
    tgt = torch.tensor([0, 0, 2])
    out = X.mean_aggregate(msg, tgt, 4)
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0], [0.0, 0.0], [10.0, 0.0], [0.0, 0.0]]))


def test_drill73_message_passing_permutation_and_rigid():
    torch.manual_seed(0)
    layer = X.KNNMessagePassing(dim=4, k=3).double()
    B, N = 2, 7
    h = torch.randn(B, N, 4, dtype=torch.float64)
    pos = torch.randn(B, N, 3, dtype=torch.float64) * 3
    mask = torch.ones(B, N, dtype=torch.bool)
    out = layer(h, pos, mask)
    assert out.shape == (B, N, 4)
    perm = torch.randperm(N)
    torch.testing.assert_close(layer(h[:, perm], pos[:, perm], mask[:, perm]), out[:, perm], rtol=0, atol=1e-10)
    moved = pos @ _rot(B, seed=6).transpose(-1, -2) - 5.0
    torch.testing.assert_close(layer(h, moved, mask), out, rtol=0, atol=1e-10)


def test_drill73_message_passing_padding_invariance():
    torch.manual_seed(0)
    layer = X.KNNMessagePassing(dim=4, k=3).double()
    h = torch.randn(1, 5, 4, dtype=torch.float64)
    pos = torch.randn(1, 5, 3, dtype=torch.float64)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = layer(h, pos, mask)
    h2 = torch.cat([h, torch.randn(1, 3, 4, dtype=torch.float64) * 100], 1)
    pos2 = torch.cat([pos, pos[:, :3] + 1e-3], 1)  # padding placed right next to real atoms
    mask2 = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], 1)
    out2 = layer(h2, pos2, mask2)
    torch.testing.assert_close(out2[:, :5], out, rtol=0, atol=1e-10)
    assert torch.all(out2[:, 5:] == 0)


# ---------------------------------------------------------------------------
# Drill #74
# ---------------------------------------------------------------------------


def test_drill74_shared_dropout_mask_shapes_and_values():
    z = torch.zeros(4, 64, 64, 2)
    ones = X.shared_dropout_mask(z, 0.3, training=False)
    assert torch.all(ones == 1) and torch.broadcast_shapes(ones.shape, z.shape) == z.shape
    m = X.shared_dropout_mask(z, 0.3, training=True)
    assert m.shape == (4, 1, 64, 1)
    mc = X.shared_dropout_mask(z, 0.3, training=True, columnwise=True)
    assert mc.shape == (4, 64, 1, 1)
    u = m.unique()
    assert len(u) == 2
    torch.testing.assert_close(u, torch.tensor([0.0, 1 / 0.7]), rtol=1e-6, atol=1e-6)
    big = X.shared_dropout_mask(torch.zeros(8, 512, 512, 1), 0.3, training=True)
    assert abs(big.mean().item() - 1.0) < 0.1


class _Scale(nn.Module):
    def __init__(self, name, a, log):
        super().__init__()
        self.name, self.a, self.log = name, a, log

    def forward(self, z, pair_mask=None):
        self.log.append(self.name)
        return self.a * z


class _Att(nn.Module):
    def __init__(self, log):
        super().__init__()
        self.log = log
        self.seen = None

    def forward(self, s, z, mask):
        self.log.append("attention")
        self.seen = (s.dtype, z.detach().clone())
        return z.float().mean(dim=(2, 3))[..., None].expand_as(s)


def _layer(log, dropout=0.0, ckpt=False, zt=None):
    return X.PairformerLayer(
        _Scale("tri_mul_out", 0.1, log),
        _Scale("tri_mul_in", 0.2, log),
        _Scale("tri_att_start", 0.3, log),
        _Scale("tri_att_end", 0.4, log),
        zt if zt is not None else _Scale("z_transition", 0.5, log),
        _Att(log),
        _Scale("s_transition", 2.0, log),
        dropout=dropout,
        use_checkpointing=ckpt,
    )


def test_drill74_pairformer_order_residuals_and_s_track():
    log = []
    layer = _layer(log).eval()
    s = torch.randn(2, 5, 3)
    z = torch.randn(2, 5, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    s_out, z_out = layer(s, z, mask, mask[:, :, None] & mask[:, None])
    assert log == ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
                   "z_transition", "attention", "s_transition"]
    factor = 1.1 * 1.2 * 1.3 * 1.4 * 1.5
    torch.testing.assert_close(z_out, z * factor, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(layer.attention.seen[1], z_out, rtol=1e-6, atol=1e-6)
    att = z_out.mean(dim=(2, 3))[..., None]
    torch.testing.assert_close(s_out, 3.0 * (s + att), rtol=1e-5, atol=1e-5)


def test_drill74_pairformer_s_track_fp32():
    log = []
    layer = _layer(log).eval()
    s = torch.randn(1, 4, 2).bfloat16()
    z = torch.randn(1, 4, 4, 2).bfloat16()
    mask = torch.ones(1, 4, dtype=torch.bool)
    s_out, z_out = layer(s, z, mask, mask[:, :, None] & mask[:, None])
    assert s_out.dtype == torch.float32
    assert z_out.dtype == torch.bfloat16
    assert layer.attention.seen[0] == torch.float32


class _OnlyOnes(nn.Module):
    def forward(self, z, pair_mask=None):
        return torch.ones_like(z)


class _Zeros(nn.Module):
    def forward(self, *args):
        return torch.zeros_like(args[0]) if len(args) != 3 else torch.zeros_like(args[0])


def _dropout_layer(which):
    blocks = {n: _Zeros() for n in ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end"]}
    blocks[which] = _OnlyOnes()
    return X.PairformerLayer(blocks["tri_mul_out"], blocks["tri_mul_in"], blocks["tri_att_start"],
                             blocks["tri_att_end"], _Zeros(), _Zeros(), _Zeros(), dropout=0.5)


@pytest.mark.parametrize("which,shared_dim", [("tri_mul_out", 1), ("tri_att_start", 1), ("tri_att_end", 2)])
def test_drill74_pairformer_dropout_is_shared_along_the_right_axis(which, shared_dim):
    layer = _dropout_layer(which).train()
    z = torch.zeros(2, 16, 16, 3)
    mask = torch.ones(2, 16, dtype=torch.bool)
    _, z_out = layer(torch.zeros(2, 16, 3), z, mask, mask[:, :, None] & mask[:, None])
    first = z_out.select(shared_dim, 0).unsqueeze(shared_dim)
    assert torch.equal(z_out, first.expand_as(z_out))
    assert torch.equal(z_out, z_out[..., :1].expand_as(z_out))
    vals = set(z_out.unique().tolist())
    assert vals == {0.0, 2.0}
    layer.eval()
    _, z_eval = layer(torch.zeros(2, 16, 3), z, mask, mask[:, :, None] & mask[:, None])
    assert torch.all(z_eval == 1.0)


class _CountingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))
        self.calls = 0

    def forward(self, z):
        self.calls += 1
        return self.w * z


@pytest.mark.parametrize("training,ckpt,expected_calls", [(True, True, 2), (True, False, 1), (False, True, 1)])
def test_drill74_pairformer_checkpointing_only_in_training(training, ckpt, expected_calls):
    zt = _CountingLinear()
    layer = _layer([], dropout=0.0, ckpt=ckpt, zt=zt)
    layer.train(training)
    s = torch.randn(1, 4, 2)
    z = torch.randn(1, 4, 4, 2, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    s_out, z_out = layer(s, z, mask, mask[:, :, None] & mask[:, None])
    (s_out.sum() + z_out.sum()).backward()
    assert zt.calls == expected_calls
    assert zt.w.grad is not None


# ---------------------------------------------------------------------------
# Drill #75
# ---------------------------------------------------------------------------


def _scipy_rmsd(pred, true):
    p = pred - pred.mean(0)
    t = true - true.mean(0)
    _, rssd = Rotation.align_vectors(p, t)
    return rssd / math.sqrt(len(p))


def test_drill75_min_rmsd_finds_the_symmetric_swap():
    true = torch.randn(5, 3, dtype=torch.float64) * 2
    perms = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [4, 3, 2, 1, 0]])
    R = _rot(1, seed=9)[0]
    pred = torch.stack([
        true[perms[1]] @ R.T + 1.0,
        true @ R.T - 2.0,
    ])
    v, idx = X.min_rmsd_over_permutations(pred, true, perms)
    assert v.shape == (2,) and idx.dtype == torch.int64
    assert idx.tolist() == [1, 0]
    torch.testing.assert_close(v, torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-7)


def test_drill75_rmsd_values_match_scipy_and_reject_mirror():
    true = torch.randn(6, 3, dtype=torch.float64)
    perms = torch.arange(6)[None]
    noisy = true + 0.3 * torch.randn(6, 3, dtype=torch.float64)
    mirror = true * torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    v, _ = X.min_rmsd_over_permutations(torch.stack([noisy, mirror]), true, perms)
    assert v[0].item() == pytest.approx(_scipy_rmsd(noisy.numpy(), true.numpy()), abs=1e-8)
    assert v[1].item() == pytest.approx(_scipy_rmsd(mirror.numpy(), true.numpy()), abs=1e-8)
    assert v[1].item() > 0.05


def test_drill75_min_over_multiplicity_layout():
    vals = torch.tensor([3.0, 1.0, 2.0, 0.5, 9.0, 4.0])  # B=2, M=3, repeat_interleave layout
    v, i = X.min_over_multiplicity(vals, 3)
    torch.testing.assert_close(v, torch.tensor([1.0, 0.5]))
    assert i.tolist() == [1, 0]
    v2, i2 = X.min_over_multiplicity(vals, 2)
    torch.testing.assert_close(v2, torch.tensor([1.0, 0.5, 4.0]))
    assert i2.tolist() == [1, 1, 1]


# ---------------------------------------------------------------------------
# Drill #76
# ---------------------------------------------------------------------------


def test_drill76_iterator_epochs_cover_data():
    data = torch.arange(10)
    it = X.ResumableBatchIterator(data, batch_size=3, seed=0)
    batches = [next(it) for _ in range(8)]
    assert [len(b) for b in batches] == [3, 3, 3, 1, 3, 3, 3, 1]
    assert sorted(torch.cat(batches[:4]).tolist()) == list(range(10))
    assert sorted(torch.cat(batches[4:]).tolist()) == list(range(10))
    assert not torch.equal(torch.cat(batches[:4]), torch.cat(batches[4:]))


def test_drill76_iterator_resume_matches_uninterrupted():
    data = torch.arange(10)
    ref_it = X.ResumableBatchIterator(data, 3, seed=5)
    ref = [next(ref_it) for _ in range(9)]
    it = X.ResumableBatchIterator(data, 3, seed=5)
    for _ in range(5):
        next(it)
    state = it.state_dict()
    for _ in range(3):
        next(it)  # keep going: the saved state must not change
    new = X.ResumableBatchIterator(data, 3, seed=123)
    new.load_state_dict(state)
    for k in range(5, 9):
        assert torch.equal(next(new), ref[k])


def _make_run(init_seed):
    torch.manual_seed(init_seed)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    data = torch.randn(13, 5, generator=torch.Generator().manual_seed(42))
    it = X.ResumableBatchIterator(data, 4, seed=1)
    return model, opt, sched, it


def _train(model, opt, sched, it, steps):
    losses = []
    for _ in range(steps):
        batch = next(it)
        noise = torch.from_numpy(np.random.randn(*batch[:, :4].shape)).float() * 0.01
        pred = model(batch[:, :4] + noise)[:, 0]
        loss = ((pred - batch[:, 4]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def test_drill76_full_resume_reproduces_losses(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    full = _train(*_make_run(0), steps=8)

    torch.manual_seed(0)
    np.random.seed(0)
    run = _make_run(0)
    first = _train(*run, steps=4)
    path = str(tmp_path / "ckpt.pt")
    X.save_training_state(path, 4, *run)

    torch.manual_seed(999)  # different init and RNG state
    np.random.seed(999)
    run2 = _make_run(999)
    torch.rand(17)
    np.random.rand(3)
    step = X.load_training_state(path, *run2)
    assert step == 4
    second = _train(*run2, steps=4)
    assert first + second == pytest.approx(full, rel=1e-6, abs=1e-7)


def test_drill76_load_restores_global_rng(tmp_path):
    run = _make_run(0)
    torch.manual_seed(3)
    np.random.seed(3)
    path = str(tmp_path / "c.pt")
    X.save_training_state(path, 0, *run)
    expected_t, expected_n = torch.rand(4), np.random.rand(4)
    torch.manual_seed(77)
    np.random.seed(77)
    X.load_training_state(path, *_make_run(1))
    torch.testing.assert_close(torch.rand(4), expected_t)
    np.testing.assert_array_equal(np.random.rand(4), expected_n)


# ---------------------------------------------------------------------------
# Drill #77
# ---------------------------------------------------------------------------


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.5))
        self.head = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 1))

    def forward(self, x):
        return self.head(self.trunk(x))


def test_drill77_freeze_and_modes():
    net = _Net()
    X.freeze_module(net.trunk)
    assert all(not p.requires_grad for p in net.trunk.parameters())
    assert all(p.requires_grad for p in net.head.parameters())
    net.eval()
    X.set_finetune_mode(net, "trunk")
    assert all(not m.training for m in net.trunk.modules())
    assert all(m.training for m in net.head.modules())
    assert net.training


def test_drill77_head_only_training_leaves_trunk_unchanged():
    net = _Net()
    X.freeze_module(net.trunk)
    X.set_finetune_mode(net, "trunk")
    groups = X.head_param_groups(net, weight_decay=0.5)
    opt = torch.optim.AdamW(groups, lr=1e-2)
    trunk0 = {k: v.clone() for k, v in net.trunk.state_dict().items()}
    head0 = {k: v.clone() for k, v in net.head.state_dict().items()}
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    for _ in range(5):
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        opt.step()
    for k, v in net.trunk.state_dict().items():
        assert torch.equal(v, trunk0[k]), k  # includes BatchNorm running stats
    assert any(not torch.equal(v, head0[k]) for k, v in net.head.state_dict().items())


def test_drill77_param_groups_split_and_dedupe():
    net = _Net()
    net.tied = net.head[0]  # same module registered twice
    X.freeze_module(net.trunk)
    groups = X.head_param_groups(net, weight_decay=0.1)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    ids0 = [id(p) for p in groups[0]["params"]]
    ids1 = [id(p) for p in groups[1]["params"]]
    assert len(ids0 + ids1) == len(set(ids0 + ids1))
    trunk_ids = {id(p) for p in net.trunk.parameters()}
    assert not trunk_ids & set(ids0 + ids1)
    assert set(ids0) == {id(net.head[0].weight), id(net.head[2].weight)}
    assert set(ids1) == {id(net.head[0].bias), id(net.head[1].weight), id(net.head[1].bias), id(net.head[2].bias)}


# ---------------------------------------------------------------------------
# Drill #78
# ---------------------------------------------------------------------------


def test_drill78_backoff_bounds_and_cap():
    rng = np.random.default_rng(0)
    for attempt, hi in [(0, 0.5), (3, 4.0), (10, 30.0), (2000, 30.0)]:
        d = np.array([X.backoff_delay(attempt, 0.5, 30.0, rng) for _ in range(2000)])
        assert d.min() >= 0.0 and d.max() <= hi
        assert abs(d.mean() - hi / 2) < 0.1 * hi  # full jitter, not a fixed delay


def test_drill78_poll_until_terminal():
    statuses = iter(["queued", "running", "running", "succeeded", "boom"])
    slept = []
    out = X.poll_until_terminal(lambda: next(statuses), slept.append, np.random.default_rng(0), base=1.0, cap=3.0)
    assert out == "succeeded"
    assert len(slept) == 3
    assert slept[0] <= 1.0 and slept[1] <= 2.0 and slept[2] <= 3.0
    calls, slept = [], []

    def status():
        calls.append(1)
        return "running"

    with pytest.raises(TimeoutError):
        X.poll_until_terminal(status, slept.append, np.random.default_rng(0), max_attempts=4)
    assert len(calls) == 4 and len(slept) == 3


def _fake_api(n):
    items = [{"id": f"id{k}"} for k in range(n)]
    cursors = []

    def fetch(after_id, limit):
        cursors.append(after_id)
        start = 0 if after_id is None else [it["id"] for it in items].index(after_id) + 1
        return [dict(it) for it in items[start:start + limit]]

    return items, cursors, fetch


@pytest.mark.parametrize("n", [0, 7, 6])
def test_drill78_fetch_all_pages(n):
    items, cursors, fetch = _fake_api(n)
    out = X.fetch_all_pages(fetch, 3)
    assert out == items
    assert cursors[0] is None
    if n == 7:
        assert cursors == [None, "id2", "id5"]
    assert len(cursors) <= n // 3 + 1


def test_drill78_resumable_download_refreshes_and_continues():
    content = bytes(range(256)) * 10
    state = {"gen": 0, "uses": 0}
    offsets = []

    def refresh():
        state["gen"] += 1
        state["uses"] = 0
        return f"url-{state['gen']}"

    def fetch(url, offset):
        if url != f"url-{state['gen']}" or state["uses"] >= 2:
            raise PermissionError("expired")
        state["uses"] += 1
        offsets.append(offset)
        return content[offset:offset + 300]

    out = X.resumable_download(fetch, refresh, len(content), max_refreshes=20)
    assert out == content
    assert offsets == sorted(set(offsets))  # never re-downloads from an earlier offset
    with pytest.raises(RuntimeError):
        def always_expired(url, offset):
            raise PermissionError

        X.resumable_download(always_expired, refresh, 10, max_refreshes=3)


def test_drill78_phase_state_machine():
    sm = X.PhaseStateMachine(["queued", "running", "uploading", "done"])
    assert sm.current == "queued" and not sm.is_done()
    sm.advance("running")
    sm.advance("running")  # idempotent
    assert sm.current == "running"
    with pytest.raises(ValueError):
        sm.advance("queued")
    with pytest.raises(KeyError):
        sm.advance("exploded")
    sm.advance("done")  # skipping is allowed
    assert sm.is_done() and sm.current == "done"
