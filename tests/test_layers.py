"""Tests for drills/layers.py (Drills #11-#19)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from drills.layers import (
    BatchNorm,
    CausalSelfAttention,
    Decoder,
    LayerNorm,
    MLP,
    RMSNorm,
    Transition,
    TransformerBlock,
    Value,
    conv2d_im2col,
    conv2d_loops,
    conv2d_output_size,
    count_params_per_block,
    dropout,
    layer_norm,
    mlp_forward_backward,
    rms_norm,
    rope,
    row_dropout_mask,
    sinusoidal_positional_encoding,
)


def _fill(module: nn.Module, scale: float = 0.3) -> None:
    """Set every parameter to fixed values, in sorted-name order (names are part of the spec)."""
    with torch.no_grad():
        for k, (_, p) in enumerate(sorted(module.named_parameters())):
            vals = scale * torch.sin(torch.arange(p.numel(), dtype=p.dtype) * 0.7 + k)
            p.copy_(vals.reshape(p.shape))


# ---------------------------------------------------------------------------
# Drill #11
# ---------------------------------------------------------------------------


def _mlp_params(rng, D=5, H=7, C=4, scale2=1.0):
    W1 = rng.normal(size=(D, H))
    b1 = rng.normal(size=H) * 0.1
    W2 = rng.normal(size=(H, C)) * scale2
    b2 = rng.normal(size=C) * 0.1
    return W1, b1, W2, b2


def _torch_mlp(X, y, W1, b1, W2, b2):
    ts = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
    logits = torch.relu(torch.tensor(X) @ ts[0] + ts[1]) @ ts[2] + ts[3]
    loss = F.cross_entropy(logits, torch.tensor(y))
    loss.backward()
    return loss.item(), [t.grad.numpy() for t in ts]


def test_drill11_grads_match_autograd():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 5))
    y = rng.integers(0, 4, size=6)
    params = _mlp_params(rng)
    loss, grads = mlp_forward_backward(X, y, *params)
    ref_loss, ref_grads = _torch_mlp(X, y, *params)
    assert loss == pytest.approx(ref_loss, rel=1e-10)
    for key, g in zip(["W1", "b1", "W2", "b2"], ref_grads):
        assert grads[key].shape == g.shape
        np.testing.assert_allclose(grads[key], g, rtol=1e-8, atol=1e-10)


def test_drill11_large_logits_are_stable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 5))
    y = rng.integers(0, 4, size=5)
    params = _mlp_params(rng, scale2=1e3)
    loss, grads = mlp_forward_backward(X, y, *params)
    ref_loss, ref_grads = _torch_mlp(X, y, *params)
    assert np.isfinite(loss)
    assert loss == pytest.approx(ref_loss, rel=1e-8)
    for key, g in zip(["W1", "b1", "W2", "b2"], ref_grads):
        assert np.all(np.isfinite(grads[key]))
        np.testing.assert_allclose(grads[key], g, rtol=1e-6, atol=1e-6)


def test_drill11_zero_output_layer_gives_ln_c():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 5))
    y = rng.integers(0, 4, size=8)
    W1, b1, _, _ = _mlp_params(rng)
    loss, grads = mlp_forward_backward(X, y, W1, b1, np.zeros((7, 4)), np.zeros(4))
    assert loss == pytest.approx(math.log(4), rel=1e-12)
    np.testing.assert_allclose(grads["W1"], 0.0, atol=1e-15)
    np.testing.assert_allclose(grads["b2"].sum(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Drill #12
# ---------------------------------------------------------------------------


def test_drill12_layer_norm_matches_torch():
    x = torch.randn(2, 3, 6, dtype=torch.float64) * 3 + 1
    w = torch.randn(6, dtype=torch.float64)
    b = torch.randn(6, dtype=torch.float64)
    out = layer_norm(x, w, b, eps=1e-5)
    torch.testing.assert_close(out, F.layer_norm(x, (6,), w, b, eps=1e-5), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(layer_norm(x), F.layer_norm(x, (6,)), rtol=1e-10, atol=1e-10)


def test_drill12_layer_norm_eps_inside_sqrt_on_constant_rows():
    x = torch.full((3, 4), 2.5, dtype=torch.float64)
    out = layer_norm(x, eps=1e-2)
    assert torch.all(torch.isfinite(out))
    torch.testing.assert_close(out, torch.zeros_like(x))
    # Tiny variance: eps must dominate, as in torch.
    x = torch.tensor([[0.0, 1e-3, 0.0, 1e-3]], dtype=torch.float64)
    torch.testing.assert_close(layer_norm(x, eps=1e-2), F.layer_norm(x, (4,), eps=1e-2), rtol=1e-10, atol=1e-12)


def test_drill12_rms_norm_matches_torch():
    x = torch.randn(4, 5, dtype=torch.float64) + 2
    w = torch.randn(5, dtype=torch.float64)
    torch.testing.assert_close(rms_norm(x, w, eps=1e-6), F.rms_norm(x, (5,), w, eps=1e-6), rtol=1e-10, atol=1e-10)
    # RMSNorm does not subtract the mean: a constant row maps to sign(c) * weight.
    c = torch.full((1, 5), -3.0, dtype=torch.float64)
    torch.testing.assert_close(rms_norm(c, w, eps=0.0), -w.unsqueeze(0), rtol=1e-12, atol=1e-12)


def test_drill12_modules_match_torch_modules():
    x = torch.randn(2, 3, 8)
    ln = LayerNorm(8)
    assert ln.weight.shape == (8,) and ln.bias.shape == (8,)
    assert sum(p.numel() for p in ln.parameters()) == 16
    with torch.no_grad():
        ln.weight.copy_(torch.randn(8))
        ln.bias.copy_(torch.randn(8))
    ref = nn.LayerNorm(8)
    ref.load_state_dict(ln.state_dict())
    torch.testing.assert_close(ln(x), ref(x), rtol=1e-5, atol=1e-5)

    rn = RMSNorm(8, eps=1e-6)
    assert sum(p.numel() for p in rn.parameters()) == 8
    with torch.no_grad():
        rn.weight.copy_(torch.randn(8))
    ref = nn.RMSNorm(8, eps=1e-6)
    ref.load_state_dict(rn.state_dict())
    torch.testing.assert_close(rn(x), ref(x), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Drill #13
# ---------------------------------------------------------------------------


def _copy_affine(mine: nn.Module, ref: nn.Module) -> None:
    with torch.no_grad():
        w = torch.randn_like(ref.weight)
        b = torch.randn_like(ref.bias)
        ref.weight.copy_(w)
        ref.bias.copy_(b)
        mine.weight.copy_(w)
        mine.bias.copy_(b)


def test_drill13_batchnorm_4d_train_then_eval_matches_torch():
    bn = BatchNorm(3, eps=1e-5, momentum=0.2).double()
    ref = nn.BatchNorm2d(3, eps=1e-5, momentum=0.2).double()
    _copy_affine(bn, ref)
    assert "running_mean" in dict(bn.named_buffers())
    assert "running_var" in dict(bn.named_buffers())
    assert sum(p.numel() for p in bn.parameters()) == 6
    for _ in range(3):
        x = torch.randn(4, 3, 5, 2, dtype=torch.float64) * 2 + 1
        torch.testing.assert_close(bn(x), ref(x), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(bn.running_mean, ref.running_mean, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(bn.running_var, ref.running_var, rtol=1e-8, atol=1e-10)
    bn.eval()
    ref.eval()
    x = torch.randn(2, 3, 5, 2, dtype=torch.float64)
    before = bn.running_mean.clone()
    torch.testing.assert_close(bn(x), ref(x), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(bn.running_mean, before)


def test_drill13_batchnorm_2d_input_and_gradient():
    bn = BatchNorm(4).double()
    ref = nn.BatchNorm1d(4).double()
    _copy_affine(bn, ref)
    x1 = (torch.randn(6, 4, dtype=torch.float64) * 3).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    g = torch.randn(6, 4, dtype=torch.float64)
    (bn(x1) * g).sum().backward()
    (ref(x2) * g).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(bn.running_var, ref.running_var, rtol=1e-8, atol=1e-10)


def test_drill13_batchnorm_rejects_3d_input():
    bn = BatchNorm(2)
    with pytest.raises(ValueError):
        bn(torch.randn(3, 2, 4))


# ---------------------------------------------------------------------------
# Drill #14
# ---------------------------------------------------------------------------


def test_drill14_dropout_eval_and_edge_probabilities():
    x = torch.randn(10, 10)
    assert torch.equal(dropout(x, 0.5, training=False), x)
    assert torch.equal(dropout(x, 0.0, training=True), x)
    out = dropout(x, 1.0, training=True)
    assert torch.all(torch.isfinite(out)) and torch.all(out == 0)
    with pytest.raises(ValueError):
        dropout(x, 1.5)


def test_drill14_dropout_scaling_and_expectation():
    p = 0.3
    x = torch.full((200_000,), 2.0)
    out = dropout(x, p)
    kept = out != 0
    torch.testing.assert_close(out[kept], torch.full_like(out[kept], 2.0 / (1 - p)))
    assert abs(kept.float().mean().item() - (1 - p)) < 0.01
    assert abs(out.mean().item() - 2.0) < 0.03


def test_drill14_row_mask_shape_and_values():
    z = torch.randn(2, 50, 50, 3)
    p = 0.25
    m = row_dropout_mask(z, p)
    assert m.shape == (2, 50, 1, 1) and m.dtype == z.dtype
    nz = m[m != 0]
    assert nz.numel() > 0 and nz.numel() < m.numel()
    torch.testing.assert_close(nz, torch.full_like(nz, 1 / (1 - p)))
    mc = row_dropout_mask(z, p, columnwise=True)
    assert mc.shape == (2, 1, 50, 1)
    # Shared along each row: broadcasting onto z gives identical values across j and c.
    full = (m * torch.ones_like(z))
    assert torch.all(full == full[:, :, :1, :1])


def test_drill14_row_mask_eval_and_p_one():
    z = torch.randn(1, 4, 4, 2)
    assert torch.equal(row_dropout_mask(z, 0.5, training=False), torch.ones(1, 4, 1, 1))
    m = row_dropout_mask(z, 1.0)
    assert torch.all(torch.isfinite(m)) and torch.all(m == 0)


# ---------------------------------------------------------------------------
# Drill #15
# ---------------------------------------------------------------------------

_TRANSITION_EXPECTED = [
    [[0.010964012862660827, 0.013026899753786566, 0.009242425040960043, 0.0013093192391280307],
     [0.004867141003835999, 0.004399784330395415, 0.001957504195081502, -0.0013634374089036515],
     [-0.015870285824405788, -0.005873578320409974, 0.006759591964270754, 0.016358596080690936]],
    [[0.0119182552907008, 0.015851840071785736, 0.012670037251856166, 0.003801057072201085],
     [0.006732844656048313, 0.006282973812483936, 0.0030128755530236797, -0.0016096068622382133],
     [-0.015517693705855456, -0.006352281487894721, 0.005664468160448528, 0.01513861793819794]],
]


def test_drill15_structure_and_zero_init():
    t = Transition(6)
    names = sorted(n for n, _ in t.named_parameters())
    assert names == ["fc1.weight", "fc2.weight", "fc3.weight", "norm.bias", "norm.weight"]
    assert t.fc1.weight.shape == (24, 6) and t.fc3.weight.shape == (6, 24)
    assert torch.all(t.fc3.weight == 0)
    x = torch.randn(2, 3, 6)
    out = t(x)
    assert out.shape == x.shape
    assert torch.all(out == 0)


def test_drill15_values():
    t = Transition(4, 8).double()
    _fill(t)
    x = torch.cos(torch.arange(24, dtype=torch.float64) * 0.5).reshape(2, 3, 4)
    expected = torch.tensor(_TRANSITION_EXPECTED, dtype=torch.float64)
    torch.testing.assert_close(t(x), expected, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("chunk_size", [1, 3, 16, 100])
def test_drill15_chunked_matches_unchunked(chunk_size):
    t = Transition(5, 16).double()
    with torch.no_grad():
        for p in t.parameters():
            p.copy_(torch.randn_like(p))
    x = torch.randn(2, 4, 5, dtype=torch.float64)
    torch.testing.assert_close(t(x, chunk_size=chunk_size), t(x), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Drill #16
# ---------------------------------------------------------------------------


def test_drill16_sinusoidal_values_odd_width():
    pe = sinusoidal_positional_encoding(3, 5)
    expected = np.array([
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.8414709848078965, 0.5403023058681398, 0.025116222909773774, 0.9996845379152098, 0.0006309573026154199],
        [0.9092974268256817, -0.4161468365471424, 0.050216599387465206, 0.9987383506934931, 0.0012619143540422218],
    ])
    assert pe.shape == (3, 5)
    np.testing.assert_allclose(pe, expected, rtol=1e-12, atol=1e-14)


def test_drill16_sinusoidal_even_width_properties():
    pe = sinusoidal_positional_encoding(50, 16)
    assert pe.shape == (50, 16)
    np.testing.assert_allclose(pe[0], np.tile([0.0, 1.0], 8), atol=1e-15)
    np.testing.assert_allclose(pe[:, 0::2] ** 2 + pe[:, 1::2] ** 2, 1.0, rtol=1e-12)


def test_drill16_rope_values_both_conventions():
    x = torch.cos(torch.arange(12, dtype=torch.float64) * 0.5).reshape(3, 4)
    half = torch.tensor([
        [1.0, 0.8775825618903728, 0.5403023058681398, 0.0707372016677029],
        [0.608204865700652, -0.7036515638210858, -0.8850707170793919, -1.0117592088216474],
        [0.014078429722150954, -0.3473848675845647, -0.7124015836408741, 0.6526649001018374],
    ], dtype=torch.float64)
    inter = torch.tensor([
        [1.0, 0.8775825618903728, 0.5403023058681398, 0.0707372016677029],
        [0.4492940117806842, -0.7830352311855614, -0.8915569870774245, -1.03061263784715],
        [0.4636878030596824, -0.5066344574217589, 0.13721687755678005, 0.750898537067004],
    ], dtype=torch.float64)
    torch.testing.assert_close(rope(x, base=100.0), half, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(rope(x, base=100.0, interleaved=True), inter, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("interleaved", [False, True])
def test_drill16_rope_relative_position_and_norm(interleaved):
    q = torch.randn(1, 1, 8, dtype=torch.float64).expand(1, 6, 8)
    k = torch.randn(1, 1, 8, dtype=torch.float64).expand(1, 6, 8)
    rq, rk = rope(q, interleaved=interleaved), rope(k, interleaved=interleaved)
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1), rtol=1e-12, atol=1e-12)
    # <rq[m], rk[n]> depends only on m - n.
    s = rq[0] @ rk[0].T
    torch.testing.assert_close(s[3, 1], s[5, 3], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(s[2, 2], (q[0, 0] @ k[0, 0]), rtol=1e-10, atol=1e-12)


def test_drill16_rope_offset_and_odd_dim():
    x = torch.randn(2, 3, 7, 8, dtype=torch.float64)
    full = rope(x)
    torch.testing.assert_close(rope(x[..., 5:6, :], offset=5), full[..., 5:6, :], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        rope(torch.randn(3, 5))


# ---------------------------------------------------------------------------
# Drill #17
# ---------------------------------------------------------------------------

_BLOCK_EXPECTED = [[
    [0.04325298725186966, 2.389527514053749, 0.32397809562049107, -0.20745995267855805,
     0.006022364270927505, -2.581078944063714, 0.20906975119755034, -2.5891692085161018],
    [-1.2780737512733682, 1.114237763167746, 0.10148943515662356, 0.3955388760881355,
     1.1419435877594548, -0.5874316928762958, 1.6638979786282984, -0.93797546146973],
    [-0.8276368179909102, 0.6474870857521015, -0.9844590974999923, -1.52104098176926,
     -0.4261393407869988, -2.238994922144829, 1.1177695747339698, -0.8828172517572423],
]]

_BLOCK_PARAM_NAMES = sorted([
    "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
    "attn.q_proj.weight", "attn.q_proj.bias", "attn.k_proj.weight", "attn.k_proj.bias",
    "attn.v_proj.weight", "attn.v_proj.bias", "attn.out_proj.weight", "attn.out_proj.bias",
    "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias",
])


def test_drill17_count_params_per_block():
    assert count_params_per_block(8, 2) == 872
    assert count_params_per_block(8, 4) == 872
    assert count_params_per_block(16, 2) == 3280


def test_drill17_block_param_names_and_numel():
    block = TransformerBlock(8, 2)
    assert sorted(n for n, _ in block.named_parameters()) == _BLOCK_PARAM_NAMES
    assert sum(p.numel() for p in block.parameters()) == 872
    assert sum(p.numel() for p in TransformerBlock(16, 4).parameters()) == 3280


def test_drill17_block_values():
    block = TransformerBlock(8, 2).double()
    _fill(block)
    x = torch.cos(torch.arange(24, dtype=torch.float64) * 0.5).reshape(1, 3, 8)
    torch.testing.assert_close(
        block(x), torch.tensor(_BLOCK_EXPECTED, dtype=torch.float64), rtol=1e-8, atol=1e-10
    )


def test_drill17_block_residual_goes_around_norm():
    block = TransformerBlock(8, 2)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.startswith(("attn.", "mlp.")):
                p.zero_()
    x = torch.randn(2, 5, 8) * 3 + 1
    torch.testing.assert_close(block(x), x)


def test_drill17_attention_is_causal_and_checks_heads():
    attn = CausalSelfAttention(8, 2)
    x = torch.randn(2, 6, 8)
    y = x.clone()
    y[:, 4:] = torch.randn(2, 2, 8)
    out_x, out_y = attn(x), attn(y)
    assert out_x.shape == (2, 6, 8)
    torch.testing.assert_close(out_x[:, :4], out_y[:, :4], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(out_x[:, 4:], out_y[:, 4:])
    with pytest.raises(ValueError):
        CausalSelfAttention(8, 3)


def test_drill17_mlp_shapes_and_gelu():
    mlp = MLP(8)
    assert mlp.fc1.weight.shape == (32, 8) and mlp.fc2.weight.shape == (8, 32)
    assert sum(p.numel() for p in mlp.parameters()) == 552
    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.eye(32, 8))
        mlp.fc1.bias.zero_()
        mlp.fc2.weight.copy_(torch.eye(8, 32))
        mlp.fc2.bias.zero_()
    x = torch.tensor([[-1.0, 0.0, 0.5, 1.0, 2.0, -2.0, 3.0, -0.5]])
    # With identity weights the MLP is exact GELU: gelu(1) = 0.5 * (1 + erf(1/sqrt 2)).
    assert mlp(x)[0, 3].item() == pytest.approx(0.8413447460685429, rel=1e-6)
    assert mlp(x)[0, 0].item() == pytest.approx(-0.15865525393145707, rel=1e-6)


def test_drill17_decoder_shape_params_and_causality():
    V, d, L, max_len = 11, 8, 2, 16
    model = Decoder(V, d, n_heads=2, n_layers=L, max_len=max_len)
    n = sum(p.numel() for p in model.parameters())
    assert n == V * d + max_len * d + L * 872 + 2 * d + d * V
    ids = torch.randint(0, V, (3, 7))
    logits = model(ids)
    assert logits.shape == (3, 7, V)
    ids2 = ids.clone()
    ids2[:, 5] = (ids2[:, 5] + 1) % V
    logits2 = model(ids2)
    torch.testing.assert_close(logits[:, :5], logits2[:, :5], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(logits[:, 5:], logits2[:, 5:])
    with pytest.raises(ValueError):
        model(torch.zeros(1, max_len + 1, dtype=torch.long))


# ---------------------------------------------------------------------------
# Drill #18
# ---------------------------------------------------------------------------


def test_drill18_reused_node_accumulates():
    a = Value(3.0)
    out = a * a + a
    out.backward()
    assert float(out.data) == 12.0
    assert float(a.grad) == 7.0
    # a is the left operand of two different additions: d/da (a+1)(a+2) = 2a + 3.
    a = Value(3.0)
    ((a + 1) * (a + 2)).backward()
    assert float(a.grad) == 9.0


def test_drill18_scalar_graph_matches_torch():
    a, b = Value(-1.5), Value(2.0)
    c = a * b + 3 - b
    d = (c * c + 2 * a).relu() + (1 - a) * c
    d.backward()
    ta = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tc = ta * tb + 3 - tb
    td = torch.relu(tc * tc + 2 * ta) + (1 - ta) * tc
    td.backward()
    assert float(d.data) == pytest.approx(td.item(), rel=1e-12)
    assert float(a.grad) == pytest.approx(ta.grad.item(), rel=1e-12)
    assert float(b.grad) == pytest.approx(tb.grad.item(), rel=1e-12)


def test_drill18_matmul_broadcast_bias_matches_torch():
    rng = np.random.default_rng(0)
    Xn, Wn, bn = rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=5)
    X, W, b = Value(Xn), Value(Wn), Value(bn)
    loss = ((X @ W + b).relu() * (X @ W)).sum()
    loss.backward()
    tX, tW, tb = (torch.tensor(a, requires_grad=True) for a in (Xn, Wn, bn))
    tl = (torch.relu(tX @ tW + tb) * (tX @ tW)).sum()
    tl.backward()
    assert float(loss.data) == pytest.approx(tl.item(), rel=1e-12)
    assert b.grad.shape == (5,)
    np.testing.assert_allclose(X.grad, tX.grad.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(W.grad, tW.grad.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b.grad, tb.grad.numpy(), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Drill #19
# ---------------------------------------------------------------------------


def test_drill19_output_size():
    assert conv2d_output_size(5, 3, 2, 1) == 3
    assert conv2d_output_size(7, 3, 1, 0) == 5
    assert conv2d_output_size(6, 3, 2, 0) == 2
    assert conv2d_output_size(4, 4, 1, 0) == 1


def test_drill19_cross_correlation_by_hand():
    x = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 0.0], [0.0, 0.0]]]])  # picks the top-left of each window
    expected = np.array([[[[0.0, 1.0], [3.0, 4.0]]]])
    np.testing.assert_allclose(conv2d_loops(x, w), expected)
    np.testing.assert_allclose(conv2d_im2col(x, w), expected)


@pytest.mark.parametrize("fn", [conv2d_loops, conv2d_im2col])
@pytest.mark.parametrize("stride,padding,bias", [(1, 0, True), (2, 1, False), (2, 2, True)])
def test_drill19_matches_torch_conv2d(fn, stride, padding, bias):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 7, 6))
    w = rng.normal(size=(4, 3, 3, 2))
    b = rng.normal(size=4) if bias else None
    out = fn(x, w, b, stride=stride, padding=padding)
    ref = F.conv2d(
        torch.tensor(x), torch.tensor(w), None if b is None else torch.tensor(b),
        stride=stride, padding=padding,
    ).numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)
