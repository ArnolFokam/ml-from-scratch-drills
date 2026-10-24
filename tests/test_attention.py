import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from drills.attention import (
    CachedCausalSelfAttention,
    MultiHeadAttention,
    aft_full,
    aft_simple,
    beam_search,
    flash_attention,
    kv_cache_bytes,
    repeat_kv,
    sample_next_token,
    scaled_dot_product_attention,
    scaled_dot_product_attention_np,
    sliding_window_attention,
    top_k_filter,
    top_p_filter,
)

D64 = torch.float64


def _qkv(B=2, H=3, Lq=5, Lk=7, d=8, dtype=D64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, Lq, d, generator=g, dtype=dtype)
    k = torch.randn(B, H, Lk, d, generator=g, dtype=dtype)
    v = torch.randn(B, H, Lk, d, generator=g, dtype=dtype)
    return q, k, v


def _tril(Lq, Lk):
    return torch.ones(Lq, Lk, dtype=torch.bool).tril(Lk - Lq)


# ---------------------------------------------------------------- Drill #20


def test_drill20_matches_torch_no_mask():
    q, k, v = _qkv()
    out = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 5, 8)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill20_padding_mask_matches_torch():
    q, k, v = _qkv()
    keep = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    keep[0, ..., 5:] = False
    keep[1, ..., 2:] = False
    out = scaled_dot_product_attention(q, k, v, mask=keep)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill20_padding_invariance():
    q, k, v = _qkv()
    keep = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    keep[..., 4:] = False
    out1 = scaled_dot_product_attention(q, k, v, mask=keep)
    k2, v2 = k.clone(), v.clone()
    k2[..., 4:, :] = 1e3
    v2[..., 4:, :] = -1e3
    out2 = scaled_dot_product_attention(q, k2, v2, mask=keep)
    torch.testing.assert_close(out1, out2, rtol=1e-10, atol=1e-10)


def test_drill20_causal_square():
    q, k, v = _qkv(Lq=6, Lk=6)
    out = scaled_dot_product_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill20_causal_rectangular_bottom_right():
    q, k, v = _qkv(Lq=3, Lk=7)
    out = scaled_dot_product_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=_tril(3, 7))
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill20_causal_and_padding_combined():
    q, k, v = _qkv(Lq=6, Lk=6)
    keep = torch.ones(2, 1, 1, 6, dtype=torch.bool)
    keep[1, ..., 0] = False
    out = scaled_dot_product_attention(q, k, v, mask=keep, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=keep & _tril(6, 6))
    # row 0 of batch 1 is fully blocked: only check rows that keep a key
    torch.testing.assert_close(out[0], ref[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(out[1, :, 1:], ref[1, :, 1:], rtol=1e-10, atol=1e-10)
    assert torch.isfinite(out).all()


def test_drill20_fully_masked_row_is_finite():
    q, k, v = _qkv(dtype=torch.float32)
    keep = torch.ones(2, 1, 5, 7, dtype=torch.bool)
    keep[0, :, 2, :] = False
    out = scaled_dot_product_attention(q, k, v, mask=keep)
    assert torch.isfinite(out).all()


def test_drill20_large_logits_stable():
    q, k, v = _qkv(dtype=torch.float32)
    q = q * 1e3
    out = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


def test_drill20_numpy_matches_torch():
    q, k, v = _qkv()
    keep = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    keep[1, ..., 3:] = False
    out = scaled_dot_product_attention_np(q.numpy(), k.numpy(), v.numpy(), mask=keep.numpy())
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=keep).numpy()
    assert isinstance(out, np.ndarray) and out.shape == (2, 3, 5, 8)
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)


def test_drill20_numpy_causal_and_large_logits():
    q, k, v = _qkv(Lq=4, Lk=6)
    q = q * 1e4
    out = scaled_dot_product_attention_np(q.numpy(), k.numpy(), v.numpy(), causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=_tril(4, 6)).numpy()
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------- Drill #21


def _torch_mha_from(m: MultiHeadAttention, d_model, n_heads):
    ref = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dtype=D64)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight]))
        ref.in_proj_bias.copy_(torch.cat([m.q_proj.bias, m.k_proj.bias, m.v_proj.bias]))
        ref.out_proj.weight.copy_(m.o_proj.weight)
        ref.out_proj.bias.copy_(m.o_proj.bias)
    return ref.eval()


def test_drill21_self_attention_matches_torch():
    m = MultiHeadAttention(16, 4).double()
    ref = _torch_mha_from(m, 16, 4)
    x = torch.randn(2, 5, 16, dtype=D64)
    out = m(x)
    exp, _ = ref(x, x, x, need_weights=False)
    assert out.shape == (2, 5, 16)
    torch.testing.assert_close(out, exp, rtol=1e-10, atol=1e-10)


def test_drill21_cross_attention_lq_ne_lk():
    m = MultiHeadAttention(16, 4).double()
    ref = _torch_mha_from(m, 16, 4)
    x = torch.randn(2, 3, 16, dtype=D64)
    ctx = torch.randn(2, 9, 16, dtype=D64)
    out = m(x, context=ctx)
    exp, _ = ref(x, ctx, ctx, need_weights=False)
    assert out.shape == (2, 3, 16)
    torch.testing.assert_close(out, exp, rtol=1e-10, atol=1e-10)


def test_drill21_key_padding_mask_keep_convention():
    m = MultiHeadAttention(12, 3).double()
    ref = _torch_mha_from(m, 12, 3)
    x = torch.randn(2, 4, 12, dtype=D64)
    ctx = torch.randn(2, 6, 12, dtype=D64)
    keep = torch.ones(2, 6, dtype=torch.bool)
    keep[0, 4:] = False
    keep[1, 1:] = False
    out = m(x, context=ctx, key_padding_mask=keep)
    # torch's key_padding_mask uses True = ignore, the opposite convention
    exp, _ = ref(x, ctx, ctx, key_padding_mask=~keep, need_weights=False)
    torch.testing.assert_close(out, exp, rtol=1e-10, atol=1e-10)


def test_drill21_causal_matches_torch():
    m = MultiHeadAttention(8, 2).double()
    ref = _torch_mha_from(m, 8, 2)
    x = torch.randn(1, 6, 8, dtype=D64)
    out = m(x, causal=True)
    blocked = ~_tril(6, 6)
    exp, _ = ref(x, x, x, attn_mask=blocked, need_weights=False)
    torch.testing.assert_close(out, exp, rtol=1e-10, atol=1e-10)


def test_drill21_param_count_and_names():
    m = MultiHeadAttention(16, 4)
    assert m.n_heads == 4 and m.d_model == 16
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(m, name), nn.Linear)
    assert sum(p.numel() for p in m.parameters()) == 4 * (16 * 16 + 16)
    m_nobias = MultiHeadAttention(16, 4, bias=False)
    assert sum(p.numel() for p in m_nobias.parameters()) == 4 * 16 * 16


def test_drill21_padding_invariance_and_noncontiguous_input():
    m = MultiHeadAttention(8, 2).double()
    x = torch.randn(2, 6, 8, dtype=D64)
    keep = torch.ones(2, 6, dtype=torch.bool)
    keep[:, 4:] = False
    out1 = m(x[:, :4], context=x, key_padding_mask=keep)
    x2 = x.clone()
    x2[:, 4:] = 50.0
    out2 = m(x2[:, :4], context=x2, key_padding_mask=keep)
    torch.testing.assert_close(out1, out2, rtol=1e-10, atol=1e-10)
    xt = x.transpose(0, 1).contiguous().transpose(0, 1)  # non-contiguous view
    torch.testing.assert_close(m(xt), m(x), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- Drill #22


def test_drill22_repeat_kv_grouping():
    x = torch.randn(2, 3, 5, 4)
    y = repeat_kv(x, 2)
    assert y.shape == (2, 6, 5, 4)
    for h in range(6):
        torch.testing.assert_close(y[:, h], x[:, h // 2], rtol=0, atol=0)
    torch.testing.assert_close(repeat_kv(x, 1), x, rtol=0, atol=0)


def test_drill22_kv_cache_bytes():
    assert kv_cache_bytes(32, 4096, 8, 128, 1, 2) == 536_870_912
    assert kv_cache_bytes(2, 10, 1, 4, 3, 4) == 2 * 2 * 10 * 1 * 4 * 3 * 4
    assert isinstance(kv_cache_bytes(1, 1, 1, 1, 1), int)


def _full_ref(m: CachedCausalSelfAttention, x):
    B, T, _ = x.shape
    H, G = m.n_heads, m.n_kv_heads
    dh = m.d_model // H
    q = m.q_proj(x).view(B, T, H, dh).transpose(1, 2)
    k = m.k_proj(x).view(B, T, G, dh).transpose(1, 2)
    v = m.v_proj(x).view(B, T, G, dh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return m.o_proj(o.transpose(1, 2).reshape(B, T, m.d_model))


@pytest.mark.parametrize("n_kv", [4, 2, 1])
def test_drill22_full_forward_matches_torch_gqa(n_kv):
    m = CachedCausalSelfAttention(16, 4, n_kv_heads=n_kv).double()
    assert m.k_proj.out_features == n_kv * 4
    x = torch.randn(2, 7, 16, dtype=D64)
    out, cache = m(x)
    torch.testing.assert_close(out, _full_ref(m, x), rtol=1e-10, atol=1e-10)
    assert cache["k"].shape == (2, n_kv, 7, 4)
    assert cache["v"].shape == (2, n_kv, 7, 4)


@pytest.mark.parametrize("n_kv", [4, 2, 1])
def test_drill22_prefill_then_decode_equals_uncached(n_kv):
    m = CachedCausalSelfAttention(16, 4, n_kv_heads=n_kv).double()
    x = torch.randn(2, 9, 16, dtype=D64)
    full, _ = m(x)
    out, cache = m(x[:, :4])
    outs = [out]
    for t in range(4, 9):
        o, cache = m(x[:, t : t + 1], cache)
        outs.append(o)
        assert cache["k"].shape[2] == t + 1
    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-10, atol=1e-10)


def test_drill22_chunked_decode_keeps_absolute_positions():
    m = CachedCausalSelfAttention(12, 3).double()
    x = torch.randn(1, 8, 12, dtype=D64)
    full, _ = m(x)
    o1, c = m(x[:, :3])
    o2, c = m(x[:, 3:6], c)  # 3 new queries against 6 keys
    o3, c = m(x[:, 6:], c)
    torch.testing.assert_close(torch.cat([o1, o2, o3], 1), full, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------- Drill #23


@pytest.mark.parametrize("block_size", [1, 3, 4, 16, 64])
def test_drill23_matches_naive(block_size):
    q, k, v = _qkv(Lq=10, Lk=13)
    out = flash_attention(q, k, v, block_size=block_size)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("block_size", [1, 4, 5])
def test_drill23_causal_square(block_size):
    q, k, v = _qkv(Lq=12, Lk=12)
    out = flash_attention(q, k, v, block_size=block_size, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill23_causal_rectangular_fully_masked_blocks():
    q, k, v = _qkv(Lq=4, Lk=11)
    out = flash_attention(q, k, v, block_size=3, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=_tril(4, 11))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill23_float32_tolerance_1e6():
    q, k, v = _qkv(Lq=16, Lk=32, dtype=torch.float32)
    out = flash_attention(q, k, v, block_size=8)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double()).float()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_drill23_large_logits_no_overflow():
    q, k, v = _qkv(Lq=6, Lk=10, dtype=torch.float32)
    q = q * 300.0
    out = flash_attention(q, k, v, block_size=3)
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- Drill #24


def test_drill24_top_k_filter():
    logits = np.array([[1.0, 3.0, -2.0, 0.5, 2.0], [0.0, -1.0, 4.0, 2.0, 1.0]])
    before = logits.copy()
    out = top_k_filter(logits, 2)
    expected = np.array(
        [[-np.inf, 3.0, -np.inf, -np.inf, 2.0], [-np.inf, -np.inf, 4.0, 2.0, -np.inf]]
    )
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(logits, before)


def test_drill24_top_k_larger_than_vocab_keeps_all():
    logits = np.array([0.3, -0.1, 2.0])
    np.testing.assert_array_equal(top_k_filter(logits, 10), logits)
    np.testing.assert_array_equal(top_k_filter(logits, 3), logits)


def test_drill24_top_p_filter_hand_case():
    probs = np.array([0.15, 0.5, 0.05, 0.3])
    logits = np.log(probs)
    out = top_p_filter(logits, 0.7)  # 0.5 then 0.3 reaches 0.8 >= 0.7
    keep = np.isfinite(out)
    np.testing.assert_array_equal(keep, [False, True, False, True])
    np.testing.assert_allclose(out[keep], logits[keep])
    out = top_p_filter(logits, 0.85)
    np.testing.assert_array_equal(np.isfinite(out), [True, True, False, True])
    out = top_p_filter(logits, 1.0)
    assert np.isfinite(out).all()


def test_drill24_top_p_always_keeps_one():
    logits = np.array([[0.0, 5.0, 1.0], [2.0, 2.1, 1.9]])
    out = top_p_filter(logits, 1e-6)
    np.testing.assert_array_equal(np.isfinite(out), [[False, True, False], [False, True, False]])


def test_drill24_greedy_and_shape():
    logits = np.array([[0.1, 2.0, -1.0], [3.0, 0.0, 2.9]])
    rng = np.random.default_rng(0)
    out = sample_next_token(logits, rng, temperature=0.0)
    np.testing.assert_array_equal(out, [1, 0])
    assert out.shape == (2,)


def test_drill24_sampling_frequencies_with_top_k():
    probs = np.array([0.1, 0.4, 0.2, 0.3])
    logits = np.tile(np.log(probs), (20000, 1))
    rng = np.random.default_rng(123)
    draws = sample_next_token(logits, rng, top_k=2)
    assert draws.shape == (20000,)
    freq = np.bincount(draws, minlength=4) / draws.size
    np.testing.assert_allclose(freq, [0.0, 4 / 7, 0.0, 3 / 7], atol=0.015)


def test_drill24_temperature_applied_before_top_p():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    logits = np.tile(np.log(probs), (2000, 1))
    rng = np.random.default_rng(1)
    # at T=0.5 token 0 alone has mass ~0.685 >= 0.6, so only it survives
    draws = sample_next_token(logits, rng, temperature=0.5, top_p=0.6)
    assert (draws == 0).all()
    # at T=1 tokens 0 and 1 survive with renormalised mass 0.625 / 0.375
    draws = sample_next_token(np.tile(np.log(probs), (20000, 1)), rng, top_p=0.6)
    freq = np.bincount(draws, minlength=4) / draws.size
    np.testing.assert_allclose(freq, [0.625, 0.375, 0.0, 0.0], atol=0.015)


def _toy_model(table, eos, max_prefix):
    """Next-token log-probs keyed by the generated prefix; eos forced after max_prefix tokens."""

    def step_fn(prefix):
        if len(prefix) >= max_prefix:
            p = np.full(3, 1e-9)
            p[eos] = 1.0
        else:
            p = np.asarray(table[tuple(prefix)], dtype=float) + 1e-9
        return np.log(p)

    return step_fn


# vocab: 0 = eos, 1 = "a", 2 = "b"
_GREEDY_TRAP = {
    (): [0.0, 0.55, 0.45],
    (1,): [0.4, 0.3, 0.3],
    (2,): [0.9, 0.05, 0.05],
}


def test_drill24_beam_search_beats_greedy():
    step = _toy_model(_GREEDY_TRAP, eos=0, max_prefix=2)
    seq, score = beam_search(step, eos=0, beam_width=2, max_len=4)
    assert seq == [2, 0]
    assert score == pytest.approx(math.log(0.45 * 0.9), abs=1e-6)


def test_drill24_beam_width_one_is_greedy():
    step = _toy_model(_GREEDY_TRAP, eos=0, max_prefix=2)
    seq, score = beam_search(step, eos=0, beam_width=1, max_len=4)
    assert seq == [1, 0]
    assert score == pytest.approx(math.log(0.55 * 0.4), abs=1e-6)


def test_drill24_beam_length_penalty_flips_choice():
    table = {(): [0.5, 0.5, 0.0], (1,): [0.9, 0.05, 0.05]}
    step = _toy_model(table, eos=0, max_prefix=2)
    seq, score = beam_search(step, eos=0, beam_width=2, max_len=4, length_penalty=0.0)
    assert seq == [0]
    assert score == pytest.approx(math.log(0.5), abs=1e-6)
    seq, score = beam_search(step, eos=0, beam_width=2, max_len=4, length_penalty=1.0)
    assert seq == [1, 0]
    assert score == pytest.approx(math.log(0.45) / 2, abs=1e-6)


# ---------------------------------------------------------------- Drill #25


def _window_loop_ref(q, k, v, window, causal):
    L = q.shape[-2]
    rows = []
    for i in range(L):
        lo = max(0, i - window + 1)
        hi = i + 1 if causal else min(L, i + window)
        rows.append(F.scaled_dot_product_attention(q[..., i : i + 1, :], k[..., lo:hi, :], v[..., lo:hi, :]))
    return torch.cat(rows, dim=-2)


@pytest.mark.parametrize("window,causal", [(1, True), (3, True), (3, False), (5, False)])
def test_drill25_sliding_window_matches_loop(window, causal):
    q, k, v = _qkv(Lq=9, Lk=9)
    out = sliding_window_attention(q, k, v, window=window, causal=causal)
    ref = _window_loop_ref(q, k, v, window, causal)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_drill25_sliding_window_edge_cases():
    q, k, v = _qkv(Lq=6, Lk=6)
    torch.testing.assert_close(
        sliding_window_attention(q, k, v, window=1, causal=True), v, rtol=1e-12, atol=1e-12
    )
    big = sliding_window_attention(q, k, v, window=100, causal=True)
    torch.testing.assert_close(
        big, F.scaled_dot_product_attention(q, k, v, is_causal=True), rtol=1e-10, atol=1e-10
    )
    big = sliding_window_attention(q, k, v, window=100, causal=False)
    torch.testing.assert_close(big, F.scaled_dot_product_attention(q, k, v), rtol=1e-10, atol=1e-10)


def _aft_inputs():
    q = torch.tensor([[[0.5, -1.0], [1.0, 0.0], [-0.5, 2.0]]], dtype=D64)
    k = torch.tensor([[[0.1, 0.3], [-0.2, 0.5], [0.4, -0.1]]], dtype=D64)
    v = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.0, 0.5]]], dtype=D64)
    w = torch.tensor([[0.0, 0.5, -0.5], [0.2, 0.0, 0.1], [-0.3, 0.4, 0.0]], dtype=D64)
    return q, k, v, w


def test_drill25_aft_full_hand_values():
    q, k, v, w = _aft_inputs()
    expected = torch.tensor(
        [[[0.9549843624916587, 0.014902992443675319],
          [0.7289022052491017, 0.25],
          [0.4791936083475671, -0.0014498995618558313]]],
        dtype=D64,
    )
    torch.testing.assert_close(aft_full(q, k, v, w), expected, rtol=1e-10, atol=1e-10)


def test_drill25_aft_full_causal_hand_values():
    q, k, v, w = _aft_inputs()
    expected = torch.tensor(
        [[[0.6224593312018546, 0.5378828427399902],
          [1.2830672680431923, 0.25],
          [0.4791936083475671, -0.0014498995618558313]]],
        dtype=D64,
    )
    out = aft_full(q, k, v, w, causal=True)
    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)
    # first position only sees itself
    torch.testing.assert_close(out[:, 0], torch.sigmoid(q[:, 0]) * v[:, 0])


def test_drill25_aft_simple_equals_full_with_zero_bias_and_large_k():
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, 6, 4, generator=g, dtype=D64) for _ in range(3))
    w = torch.zeros(6, 6, dtype=D64)
    s = aft_simple(q, k, v)
    assert s.shape == (2, 6, 4)
    torch.testing.assert_close(s, aft_full(q, k, v, w), rtol=1e-10, atol=1e-10)
    shift = torch.full((1, 1, 4), 1000.0, dtype=D64)  # softmax over time is shift invariant
    big = aft_simple(q, k + shift, v)
    assert torch.isfinite(big).all()
    torch.testing.assert_close(big, s, rtol=1e-8, atol=1e-8)


def test_drill25_aft_full_large_k_stable_and_gradcheck():
    q, k, v, w = _aft_inputs()
    big = aft_full(q, k + 1000.0, v, w)
    assert torch.isfinite(big).all()
    torch.testing.assert_close(big, aft_full(q, k, v, w), rtol=1e-8, atol=1e-8)
    inputs = tuple(t.clone().requires_grad_(True) for t in (q, k, v, w))
    assert torch.autograd.gradcheck(lambda a, b, c, d: aft_full(a, b, c, d, causal=True), inputs)
