"""Tests for src/drills/infra.py (Drills #50-#58)."""

import math
import tracemalloc

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from drills import infra


# ---------------------------------------------------------------------------
# Drill #50
# ---------------------------------------------------------------------------


def test_drill50_pad_to_max_1d_features():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100
    padded, mask = infra.pad_to_max([a, b], value=-1.0)
    assert padded.shape == (2, 4, 3)
    assert padded.dtype == torch.float32
    torch.testing.assert_close(padded[0, :2], a)
    torch.testing.assert_close(padded[0, 2:], torch.full((2, 3), -1.0))
    torch.testing.assert_close(padded[1], b)
    assert mask.dtype == torch.float32
    torch.testing.assert_close(mask, torch.tensor([[1.0, 1, 0, 0], [1, 1, 1, 1]]))


def test_drill50_pad_to_max_pads_every_dim():
    # Pair features (L, L, C): the pad tuple order matters.
    a = torch.ones(2, 2, 3)
    b = torch.full((3, 3, 3), 2.0)
    padded, mask = infra.pad_to_max([a, b])
    assert padded.shape == (2, 3, 3, 3)
    torch.testing.assert_close(padded[0, :2, :2], a)
    assert padded[0, 2].abs().sum() == 0 and padded[0, :, 2].abs().sum() == 0
    torch.testing.assert_close(mask, torch.tensor([[1.0, 1, 0], [1, 1, 1]]))


def test_drill50_pair_mask():
    mask = torch.tensor([[1.0, 1, 0], [1, 0, 0]])
    pm = infra.pair_mask(mask)
    assert pm.shape == (2, 3, 3)
    expected = torch.tensor(
        [
            [[1.0, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )
    torch.testing.assert_close(pm, expected)


def test_drill50_masked_attention_matches_sdpa():
    B, L, D = 2, 5, 4
    q, k, v = torch.randn(3, B, L, D).unbind(0)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    out = infra.masked_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool()[:, None, :])
    assert out.shape == (B, L, D)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_drill50_masked_attention_padding_invariance():
    D = 4
    xs = [torch.randn(3, D), torch.randn(5, D)]
    padded, mask = infra.pad_to_max(xs)
    # Huge junk in the padding must not leak into real rows.
    padded = padded.clone()
    padded[0, 3:] = 1e4
    out = infra.masked_attention(padded, padded, padded, mask)
    for i, x in enumerate(xs):
        L = x.shape[0]
        ref = F.scaled_dot_product_attention(x[None], x[None], x[None])[0]
        torch.testing.assert_close(out[i, :L], ref, atol=1e-5, rtol=1e-5)


def test_drill50_masked_mean_loss_value_and_grad():
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]], requires_grad=True)
    target = torch.zeros(1, 3, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss = infra.masked_mean_loss(pred, target, mask)
    # (1 + 4 + 9 + 16) / (2 positions * 2 channels) = 7.5
    torch.testing.assert_close(loss, torch.tensor(7.5))
    loss.backward()
    assert pred.grad[0, 2].abs().sum() == 0
    torch.testing.assert_close(pred.grad[0, 0], torch.tensor([0.5, 1.0]))


def test_drill50_masked_mean_loss_empty_mask():
    pred = torch.randn(2, 3, 4, requires_grad=True)
    loss = infra.masked_mean_loss(pred, torch.zeros(2, 3, 4), torch.zeros(2, 3))
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_drill50_padding_fraction_hand_case():
    lengths = [2, 4, 3, 3]
    # batch [0,1]: 2*4 = 8 slots, 6 real; batch [2,3]: 6 slots, 6 real
    frac = infra.padding_fraction(lengths, [[0, 1], [2, 3]])
    assert frac == pytest.approx(1 - 12 / 14)
    assert infra.padding_fraction(lengths, [[2, 3], [0], [1]]) == pytest.approx(0.0)


def test_drill50_bucket_by_length_budget_and_partition():
    rng = np.random.default_rng(0)
    lengths = [int(x) for x in rng.integers(5, 200, size=120)]
    max_tokens = 800
    batches = infra.bucket_by_length(lengths, max_tokens)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))
    for b in batches:
        assert len(b) >= 1
        assert len(b) * max(lengths[i] for i in b) <= max_tokens
    # Naive sequential batches with the same number of batches.
    nb = len(batches)
    size = math.ceil(len(lengths) / nb)
    naive = [list(range(s, min(s + size, len(lengths)))) for s in range(0, len(lengths), size)]
    after = infra.padding_fraction(lengths, batches)
    before = infra.padding_fraction(lengths, naive)
    assert after < before
    assert after < 0.15


def test_drill50_bucket_by_length_oversized_and_exact():
    batches = infra.bucket_by_length([5, 1, 5, 1, 50], max_tokens=10)
    flat = sorted(i for b in batches for i in b)
    assert flat == [0, 1, 2, 3, 4]
    assert [4] in batches  # longer than the budget: alone
    assert infra.padding_fraction([5, 1, 5, 1, 50], batches) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Drill #51
# ---------------------------------------------------------------------------


class _Spy(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.sizes = []

    def forward(self, x):
        self.sizes.append(x.shape[0])
        return self.inner(x)


def test_drill51_chunk_layer_matches_unchunked():
    lin = nn.Linear(8, 5)
    spy = _Spy(lin)
    x = torch.randn(3, 7, 8)  # 21 rows
    with torch.no_grad():
        out = infra.chunk_layer(spy, x, chunk_size=4, no_batch_dims=2)
        ref = lin(x)
    assert out.shape == (3, 7, 5)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    assert spy.sizes == [4, 4, 4, 4, 4, 1]


def test_drill51_chunk_layer_trailing_dims_and_none():
    # Layer acting on (n, 4, 6) -> (n, 4, 2); one batch dim.
    lin = nn.Linear(6, 2)
    spy = _Spy(lin)
    x = torch.randn(5, 4, 6)
    with torch.no_grad():
        out = infra.chunk_layer(spy, x, chunk_size=100, no_batch_dims=1)
        torch.testing.assert_close(out, lin(x), atol=1e-6, rtol=1e-6)
        assert spy.sizes == [5]
        spy.sizes.clear()
        out = infra.chunk_layer(spy, x, chunk_size=None, no_batch_dims=1)
        torch.testing.assert_close(out, lin(x), atol=1e-6, rtol=1e-6)
        assert spy.sizes == [5]


def test_drill51_chunk_size_rule():
    assert infra.chunk_size_for(1000, training=True) is None
    assert infra.chunk_size_for(100, training=True) is None
    assert infra.chunk_size_for(385, training=False) == 128
    assert infra.chunk_size_for(384, training=False) == 512
    assert infra.chunk_size_for(10, training=False) == 512


# ---------------------------------------------------------------------------
# Drill #52
# ---------------------------------------------------------------------------


class _CountBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tanh(self.lin(x)) + x


def _make_blocks(n=3, d=6):
    torch.manual_seed(1)
    return [_CountBlock(d) for _ in range(n)]


def test_drill52_checkpoint_same_grads_and_recompute():
    import copy

    blocks = _make_blocks()
    plain = nn.Sequential(*copy.deepcopy(blocks))
    stack = infra.CheckpointedStack(blocks, use_checkpoint=True)
    stack.train()
    x = torch.randn(4, 6, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    out = stack(x)
    out.pow(2).sum().backward()
    ref = plain(x2)
    ref.pow(2).sum().backward()

    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(x.grad, x2.grad)
    for b1, b2 in zip(stack.blocks, plain):
        torch.testing.assert_close(b1.lin.weight.grad, b2.lin.weight.grad)
        torch.testing.assert_close(b1.lin.bias.grad, b2.lin.bias.grad)
    # Each block ran twice: forward plus recompute in backward.
    assert [b.calls for b in stack.blocks] == [2, 2, 2]


def test_drill52_no_checkpoint_in_eval_or_when_disabled():
    blocks = _make_blocks()
    stack = infra.CheckpointedStack(blocks, use_checkpoint=True)
    stack.eval()
    x = torch.randn(4, 6, requires_grad=True)
    stack(x).sum().backward()
    assert [b.calls for b in stack.blocks] == [1, 1, 1]

    blocks = _make_blocks()
    stack = infra.CheckpointedStack(blocks, use_checkpoint=False)
    stack.train()
    stack(x).sum().backward()
    assert [b.calls for b in stack.blocks] == [1, 1, 1]


# ---------------------------------------------------------------------------
# Drill #53
# ---------------------------------------------------------------------------


class _GradSpy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)
        self.grad_flags = []

    def forward(self, x):
        self.grad_flags.append(torch.is_grad_enabled())
        return self.lin(x)


def test_drill53_zero_init_means_no_recycling_effect():
    d = 5
    trunk = _GradSpy(d)
    model = infra.Recycler(d, trunk)
    model.eval()
    s_init = torch.randn(3, d)
    with torch.no_grad():
        ref = trunk.lin(s_init)
        out0 = model(s_init, 0)
        out3 = model(s_init, 3)
    torch.testing.assert_close(out0, ref)
    torch.testing.assert_close(out3, ref)


def test_drill53_grad_only_on_last_pass():
    d = 5
    trunk = _GradSpy(d)
    model = infra.Recycler(d, trunk)
    model.train()
    s_init = torch.randn(3, d)
    model(s_init, 3)
    assert trunk.grad_flags == [False, False, False, True]

    trunk.grad_flags.clear()
    model.eval()
    model(s_init, 2)
    assert trunk.grad_flags == [False, False, False]


def test_drill53_gradient_does_not_flow_through_earlier_passes():
    d = 5
    trunk = _GradSpy(d)
    model = infra.Recycler(d, trunk)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5 * torch.randn_like(p))
    model.train()
    s_init = torch.randn(3, d, requires_grad=True)
    out = model(s_init, 2)
    out.sum().backward()
    # Only the last pass is on the graph, and s_prev is a constant there, so
    # d(sum(out))/d(s_init) = column sums of the trunk weight.
    expected = trunk.lin.weight.detach().sum(0).expand(3, d)
    torch.testing.assert_close(s_init.grad, expected, atol=1e-5, rtol=1e-5)
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)


# ---------------------------------------------------------------------------
# Drill #54
# ---------------------------------------------------------------------------


def test_drill54_chunked_sampling_matches_full():
    net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    sizes = []

    def sample_fn(z):
        sizes.append(z.shape[0])
        return net(z)

    noise = torch.randn(7, 4)
    with torch.no_grad():
        full = net(noise)
        out = infra.sample_in_chunks(sample_fn, noise, max_parallel_samples=3)
    assert out.shape == (7, 3)
    torch.testing.assert_close(out, full, atol=1e-6, rtol=1e-6)
    assert sizes == [3, 3, 1]


def test_drill54_cap_larger_than_m_and_none():
    sizes = []

    def sample_fn(z):
        sizes.append(z.shape[0])
        return z * 2

    noise = torch.randn(5, 2, 2)
    out = infra.sample_in_chunks(sample_fn, noise, max_parallel_samples=10)
    torch.testing.assert_close(out, noise * 2)
    assert sizes == [5]
    sizes.clear()
    out = infra.sample_in_chunks(sample_fn, noise, max_parallel_samples=None)
    torch.testing.assert_close(out, noise * 2)
    assert sizes == [5]
    sizes.clear()
    out = infra.sample_in_chunks(sample_fn, noise, max_parallel_samples=1)
    torch.testing.assert_close(out, noise * 2)
    assert sizes == [1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Drill #55
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("P", [2, 3, 4])
def test_drill55_ring_all_reduce_sum_and_bytes(P):
    rng = np.random.default_rng(P)
    arrays = [rng.standard_normal((4, 6)).astype(np.float32) for _ in range(P)]  # 24 elems
    copies = [a.copy() for a in arrays]
    outs, sent = infra.ring_all_reduce(arrays)
    total = np.sum(arrays, axis=0)
    assert len(outs) == P and len(sent) == P
    for o in outs:
        assert o.shape == (4, 6) and o.dtype == np.float32
        np.testing.assert_allclose(o, total, rtol=1e-5, atol=1e-5)
        assert np.array_equal(o, outs[0])
    for a, c in zip(arrays, copies):
        assert np.array_equal(a, c)  # inputs untouched
    nbytes = arrays[0].nbytes
    for s in sent:
        assert s == pytest.approx(2 * (P - 1) / P * nbytes)


def test_drill55_ring_all_reduce_uneven_and_single_worker():
    rng = np.random.default_rng(0)
    P = 3
    arrays = [rng.standard_normal(10) for _ in range(P)]  # 10 not divisible by 3
    outs, sent = infra.ring_all_reduce(arrays)
    for o in outs:
        np.testing.assert_allclose(o, np.sum(arrays, axis=0), rtol=1e-12, atol=1e-12)
    # Each step moves every segment once in total across workers.
    assert sum(sent) == 2 * (P - 1) * arrays[0].nbytes

    a = rng.standard_normal(5)
    outs, sent = infra.ring_all_reduce([a])
    np.testing.assert_array_equal(outs[0], a)
    assert sent == [0]


def test_drill55_sharded_matmul_backward_matches_autograd():
    rng = np.random.default_rng(1)
    B, D, H, P = 4, 5, 6, 3
    X = rng.standard_normal((B, D))
    W = rng.standard_normal((D, H))
    dY = rng.standard_normal((B, H))
    W_shards = np.split(W, P, axis=1)
    dY_shards = np.split(dY, P, axis=1)
    dX_list, dW_shards = infra.sharded_matmul_backward(X, W_shards, dY_shards)

    Xt = torch.tensor(X, requires_grad=True)
    Wt = torch.tensor(W, requires_grad=True)
    (Xt @ Wt).backward(torch.tensor(dY))
    assert len(dX_list) == P and len(dW_shards) == P
    for dX in dX_list:
        np.testing.assert_allclose(dX, Xt.grad.numpy(), rtol=1e-10, atol=1e-10)
    for p in range(P):
        assert dW_shards[p].shape == W_shards[p].shape
    np.testing.assert_allclose(
        np.concatenate(dW_shards, axis=1), Wt.grad.numpy(), rtol=1e-10, atol=1e-10
    )


def test_drill55_sharded_matmul_backward_uneven_shards():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 4))
    W_shards = [rng.standard_normal((4, 1)), rng.standard_normal((4, 3))]
    dY_shards = [rng.standard_normal((3, 1)), rng.standard_normal((3, 3))]
    dX_list, dW_shards = infra.sharded_matmul_backward(X, W_shards, dY_shards)
    W = np.concatenate(W_shards, axis=1)
    dY = np.concatenate(dY_shards, axis=1)
    for dX in dX_list:
        np.testing.assert_allclose(dX, dY @ W.T, rtol=1e-10, atol=1e-10)
    assert [w.shape for w in dW_shards] == [(4, 1), (4, 3)]


# ---------------------------------------------------------------------------
# Drill #56
# ---------------------------------------------------------------------------


def test_drill56_symmetric_hand_case():
    x = np.array([-1.27, 0.5, 1.27, 0.0, 0.333], dtype=np.float32)
    q, scale = infra.quantize_symmetric(x)
    assert q.dtype == np.int8
    assert scale == pytest.approx(0.01, rel=1e-6)
    np.testing.assert_array_equal(q, np.array([-127, 50, 127, 0, 33], dtype=np.int8))
    xh = infra.dequantize_symmetric(q, scale)
    assert xh.dtype == np.float32
    np.testing.assert_allclose(xh, [-1.27, 0.5, 1.27, 0.0, 0.33], atol=1e-6)


def test_drill56_symmetric_roundtrip_and_zeros():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((16, 8)) * 3).astype(np.float32)
    q, scale = infra.quantize_symmetric(x)
    err = np.abs(infra.dequantize_symmetric(q, scale) - x)
    assert err.max() <= scale / 2 + 1e-6
    assert np.abs(q.astype(np.int32)).max() == 127

    z = np.zeros((3, 3), dtype=np.float32)
    qz, sz = infra.quantize_symmetric(z)
    assert np.isfinite(sz) and sz > 0
    np.testing.assert_array_equal(infra.dequantize_symmetric(qz, sz), z)


def test_drill56_asymmetric_hand_case():
    x = np.array([-1.0, 0.0, 1.55, 0.5], dtype=np.float32)
    q, scale, zp = infra.quantize_asymmetric(x)
    assert q.dtype == np.uint8
    assert isinstance(zp, int)
    assert scale == pytest.approx(0.01, rel=1e-6)
    assert zp == 100
    np.testing.assert_array_equal(q, np.array([0, 100, 255, 150], dtype=np.uint8))
    xh = infra.dequantize_asymmetric(q, scale, zp)
    assert xh.dtype == np.float32
    np.testing.assert_allclose(xh, x, atol=1e-6)


def test_drill56_asymmetric_zero_exact_and_beats_symmetric_on_positive():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 4.0, size=200).astype(np.float32)
    x[0] = 0.0
    q, scale, zp = infra.quantize_asymmetric(x)
    xh = infra.dequantize_asymmetric(q, scale, zp)
    assert xh[0] == 0.0
    assert zp == 0
    assert np.abs(xh - x).max() <= scale / 2 + 1e-6
    qs, ss = infra.quantize_symmetric(x)
    err_sym = np.abs(infra.dequantize_symmetric(qs, ss) - x).mean()
    err_asym = np.abs(xh - x).mean()
    assert err_asym < err_sym

    neg = np.array([-3.0, -1.0, -2.0], dtype=np.float32)
    q, scale, zp = infra.quantize_asymmetric(neg)
    assert zp == 255
    np.testing.assert_allclose(infra.dequantize_asymmetric(q, scale, zp), neg, atol=scale / 2)

    z = np.zeros(4, dtype=np.float32)
    q, scale, zp = infra.quantize_asymmetric(z)
    np.testing.assert_array_equal(infra.dequantize_asymmetric(q, scale, zp), z)


# ---------------------------------------------------------------------------
# Drill #57
# ---------------------------------------------------------------------------


def test_drill57_dataloader_hand_cases():
    # load 10 ms, compute 5 ms, 4 batches
    assert infra.simulate_dataloader(4, 10.0, 5.0, 0) == pytest.approx(60.0)
    assert infra.simulate_dataloader(4, 10.0, 5.0, 1) == pytest.approx(45.0)
    assert infra.simulate_dataloader(4, 10.0, 5.0, 2) == pytest.approx(30.0)
    assert infra.simulate_dataloader(4, 10.0, 5.0, 4) == pytest.approx(30.0)


def test_drill57_dataloader_compute_bound_floor():
    # Compute-bound: more workers cannot beat first load + n * compute.
    n, load, comp = 20, 3.0, 7.0
    t8 = infra.simulate_dataloader(n, load, comp, 8)
    assert t8 == pytest.approx(load + n * comp)
    times = [infra.simulate_dataloader(50, 12.0, 4.0, w) for w in range(0, 6)]
    assert all(a >= b - 1e-9 for a, b in zip(times, times[1:]))
    # Data-bound with 1 worker: one batch per load time.
    assert infra.simulate_dataloader(50, 12.0, 4.0, 1) == pytest.approx(50 * 12.0 + 4.0)


def test_drill57_cuda_stream_async_vs_sync():
    k = ("kernel", 5.0)
    s = ("sync", 0.0)
    wall, idle = infra.simulate_cuda_stream([k, k, k], launch_ms=1.0)
    assert wall == pytest.approx(16.0)
    assert idle == pytest.approx(1.0)
    wall, idle = infra.simulate_cuda_stream([k, s, k, s, k, s], launch_ms=1.0)
    assert wall == pytest.approx(18.0)
    assert idle == pytest.approx(3.0)


def test_drill57_cuda_stream_host_work_overlap():
    k = ("kernel", 10.0)
    c = ("cpu", 8.0)
    # Host work overlaps with a running kernel when there is no sync.
    wall, idle = infra.simulate_cuda_stream([k, c, k], launch_ms=1.0)
    # k0: 1 -> 11; cpu busy to 9; launch at 10; k1: 11 -> 21
    assert wall == pytest.approx(21.0)
    assert idle == pytest.approx(1.0)
    # A sync before the host work serialises everything.
    wall, idle = infra.simulate_cuda_stream([k, ("sync", 0.0), c, k], launch_ms=1.0)
    # k0: 1 -> 11; wait to 11; cpu to 19; launch to 20; k1: 20 -> 30
    assert wall == pytest.approx(30.0)
    assert idle == pytest.approx(10.0)
    assert infra.simulate_cuda_stream([], launch_ms=1.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Drill #58
# ---------------------------------------------------------------------------


def test_drill58_pairwise_matches_cdist_and_chunk_invariance():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((37, 5))
    b = rng.standard_normal((11, 5))
    ref = cdist(a, b, "sqeuclidean")
    for cs in (None, 1, 8, 37, 1000):
        d = infra.pairwise_sq_dists(a, b, chunk_size=cs)
        assert d.shape == (37, 11) and d.dtype == np.float64
        np.testing.assert_allclose(d, ref, rtol=1e-9, atol=1e-9)


def test_drill58_pairwise_nonnegative_under_cancellation():
    rng = np.random.default_rng(1)
    a = 1e4 + 1e-3 * rng.standard_normal((20, 3))
    d = infra.pairwise_sq_dists(a, a, chunk_size=7)
    assert (d >= 0).all()
    assert np.abs(np.diag(d)).max() < 1e-2


def test_drill58_pairwise_does_not_build_nkd():
    rng = np.random.default_rng(2)
    N, K, D = 200, 200, 256  # N*K*D float64 = 80 MB; N*K = 0.3 MB
    a = rng.standard_normal((N, D))
    b = rng.standard_normal((K, D))
    tracemalloc.start()
    try:
        infra.pairwise_sq_dists(a, b, chunk_size=64)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 10 * 1024 * 1024


def test_drill58_nearest_center():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 4))
    centers = rng.standard_normal((6, 4))
    idx = infra.nearest_center(x, centers, chunk_size=16)
    assert idx.shape == (50,) and idx.dtype == np.int64
    np.testing.assert_array_equal(idx, cdist(x, centers, "sqeuclidean").argmin(1))
    # Ties go to the first index.
    tie = infra.nearest_center(np.zeros((1, 2)), np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert tie.tolist() == [0]
