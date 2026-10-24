"""Tests for Section 5 structure-prediction layers (Drills #42-#49, except #45).

Weights are loaded by parameter name (see each class docstring), so the
submodule names and shapes must match the spec. Expected numbers are
"fingerprints" of the output for fixed seeds: sum, sum of squares, and the
first six flattened values.
"""

import math

import pytest
import torch

from drills.structure_layers import (
    AdaLN,
    AttentionPairBias,
    ConditionedTransitionBlock,
    FourierEmbedding,
    OuterProductMean,
    PairWeightedAveraging,
    RelativePositionEncoder,
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    relative_position_features,
)

ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fill(module: torch.nn.Module, seed: int, scale: float = 0.5) -> torch.nn.Module:
    """Overwrite every parameter, in sorted-name order, with seeded N(0, scale^2)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return module


def _randn(seed: int, *shapes):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g) for s in shapes]


def _fp(t: torch.Tensor) -> list[float]:
    t = t.detach().double()
    return [t.sum().item(), (t**2).sum().item()] + t.flatten()[:6].tolist()


def _check_fp(t: torch.Tensor, expected: list[float]) -> None:
    got = torch.tensor(_fp(t), dtype=torch.float64)
    torch.testing.assert_close(got, torch.tensor(expected, dtype=torch.float64), atol=ATOL, rtol=RTOL)


def _param_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {n: tuple(p.shape) for n, p in module.named_parameters()}


# ---------------------------------------------------------------------------
# fixed cases (shared with the fingerprints below)
# ---------------------------------------------------------------------------


def _case_apb():
    mod = _fill(AttentionPairBias(c_s=8, c_z=4, num_heads=2), seed=1)
    s, z = _randn(2, (1, 4, 8), (1, 4, 4, 4))
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    return mod, (s, z, mask)


def _pair_mask_3():
    return torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]])


def _case_trimul(cls):
    mod = _fill(cls(dim=4), seed=3)
    (x,) = _randn(4, (1, 3, 3, 4))
    return mod, (x, _pair_mask_3())


def _case_opm():
    mod = _fill(OuterProductMean(c_in=4, c_hidden=3, c_out=5), seed=5)
    (m,) = _randn(6, (1, 3, 4, 4))
    mask = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]])
    return mod, (m, mask)


def _case_adaln():
    mod = _fill(AdaLN(dim=4, dim_single_cond=3), seed=7)
    a, s = _randn(8, (2, 3, 4), (2, 3, 3))
    return mod, (a, s)


def _case_transition():
    mod = _fill(ConditionedTransitionBlock(dim_single=4, dim_single_cond=3, expansion_factor=2), seed=9)
    a, s = _randn(10, (2, 3, 4), (2, 3, 3))
    return mod, (a, s)


def _case_pwa():
    mod = _fill(PairWeightedAveraging(c_m=4, c_z=3, c_h=2, num_heads=2), seed=11)
    m, z = _randn(12, (1, 2, 3, 4), (1, 3, 3, 3))
    return mod, (m, z, _pair_mask_3())


def _case_triatt(starting: bool):
    mod = _fill(TriangleAttention(c_in=4, c_hidden=3, num_heads=2, starting=starting), seed=13)
    (x,) = _randn(14, (1, 3, 3, 4))
    return mod, (x, _pair_mask_3())


FP_APB = [-10.6922236, 15.6294504, 0.5835705, -0.3476594, -0.6617428, -1.4136111, -0.2056361, 0.3115247]
FP_TRIMUL_OUT = [-0.6696059, 2.5283221, -0.0289504, -0.1783876, 0.0162804, 0.2259436, 0.0724634, -0.5371601]
FP_TRIMUL_IN = [-2.1501949, 2.9576726, -0.0102353, -0.2165835, 0.0239527, 0.26317, 0.3307644, -0.7672874]
FP_OPM = [-2.0877049, 13.9588793, 0.3889552, 0.9681106, 0.0390632, 0.0497819, -0.2387976, 0.2499498]
FP_ADALN = [-3.4395304, 13.6010412, -1.1119213, 0.5228429, 0.5267561, -0.6220392, -0.4884817, -0.0505232]
FP_TRANSITION = [-1.7495185, 1.2764228, 0.1595236, 0.0558458, 0.0323818, 0.2430741, -0.2697344, -0.1333643]
FP_PWA = [2.2363344, 0.7400029, 0.2649336, 0.0464363, -0.0909804, 0.1108019, 0.3357709, 0.0944873]
FP_TRIATT_START = [-6.6170515, 7.1524105, -0.1336051, -0.4241611, -0.2789353, -0.1451821, -0.1298099, -0.3095317]
FP_TRIATT_END = [-6.0736869, 7.4000469, 0.3679803, -0.8559684, 0.0180133, -0.201922, 0.0942058, -0.5004803]


# ---------------------------------------------------------------------------
# Drill #42: AttentionPairBias
# ---------------------------------------------------------------------------


def test_drill42_parameter_names_and_shapes():
    mod = AttentionPairBias(c_s=8, c_z=4, num_heads=2)
    assert _param_shapes(mod) == {
        "proj_q.weight": (8, 8),
        "proj_q.bias": (8,),
        "proj_k.weight": (8, 8),
        "proj_v.weight": (8, 8),
        "proj_g.weight": (8, 8),
        "norm_z.weight": (4,),
        "norm_z.bias": (4,),
        "proj_z.weight": (2, 4),
        "proj_o.weight": (8, 8),
    }
    assert torch.count_nonzero(mod.proj_o.weight) == 0


def test_drill42_matches_fingerprint():
    mod, (s, z, mask) = _case_apb()
    out = mod(s, z, mask)
    assert out.shape == (1, 4, 8)
    _check_fp(out, FP_APB)


def test_drill42_fresh_layer_outputs_zero():
    mod = AttentionPairBias(c_s=8, c_z=4, num_heads=2)
    s, z = _randn(20, (2, 5, 8), (2, 5, 5, 4))
    out = mod(s, z, torch.ones(2, 5))
    assert out.shape == (2, 5, 8)
    assert torch.all(out == 0)


def test_drill42_masked_key_is_ignored():
    mod, (s, z, mask) = _case_apb()
    out = mod(s, z, mask)
    s2 = s.clone()
    s2[:, 3] += 10.0  # token 3 is masked: it is never a key
    out2 = mod(s2, z, mask)
    torch.testing.assert_close(out2[:, :3], out[:, :3], atol=1e-5, rtol=1e-5)


def test_drill42_all_masked_and_large_bias_are_finite():
    mod, (s, z, _) = _case_apb()
    out = mod(s, z, torch.zeros(1, 4))
    assert torch.isfinite(out).all()
    with torch.no_grad():
        mod.proj_z.weight.mul_(1e4)  # pair bias of order 1e4
    out = mod(s, z, torch.ones(1, 4))
    assert torch.isfinite(out).all()


def test_drill42_multiplicity_repeats_bias():
    mod, (s, z, mask) = _case_apb()
    (s_extra,) = _randn(21, (1, 4, 8))
    s_rep = torch.cat([s, s_extra], dim=0)  # M = 2 copies of example 0
    mask_rep = torch.cat([mask, torch.ones(1, 4)], dim=0)
    out = mod(s_rep, z, mask_rep, multiplicity=2)
    torch.testing.assert_close(out[0:1], mod(s, z, mask), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1:2], mod(s_extra, z, torch.ones(1, 4)), atol=1e-5, rtol=1e-5)


def test_drill42_permutation_equivariant():
    mod, (s, z, mask) = _case_apb()
    perm = torch.tensor([2, 0, 3, 1])
    out = mod(s, z, mask)
    out_p = mod(s[:, perm], z[:, perm][:, :, perm], mask[:, perm])
    torch.testing.assert_close(out_p, out[:, perm], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Drill #43: triangle multiplicative update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
def test_drill43_parameter_names_and_shapes(cls):
    assert _param_shapes(cls(dim=4)) == {
        "norm_in.weight": (4,),
        "norm_in.bias": (4,),
        "p_in.weight": (8, 4),
        "g_in.weight": (8, 4),
        "norm_out.weight": (4,),
        "norm_out.bias": (4,),
        "p_out.weight": (4, 4),
        "g_out.weight": (4, 4),
    }


def test_drill43_outgoing_matches_fingerprint():
    mod, (x, mask) = _case_trimul(TriangleMultiplicationOutgoing)
    out = mod(x, mask)
    assert out.shape == (1, 3, 3, 4)
    _check_fp(out, FP_TRIMUL_OUT)


def test_drill43_incoming_matches_fingerprint():
    mod, (x, mask) = _case_trimul(TriangleMultiplicationIncoming)
    out = mod(x, mask)
    assert out.shape == (1, 3, 3, 4)
    _check_fp(out, FP_TRIMUL_IN)


def test_drill43_incoming_is_outgoing_on_transpose_with_halves_swapped():
    inc, (x, mask) = _case_trimul(TriangleMultiplicationIncoming)
    out_mod = TriangleMultiplicationOutgoing(dim=4)
    state = {k: v.clone() for k, v in inc.state_dict().items()}
    for name in ("p_in.weight", "g_in.weight"):
        w = state[name]
        state[name] = torch.cat([w[4:], w[:4]], dim=0)
    out_mod.load_state_dict(state)
    expected = inc(x, mask)
    got = out_mod(x.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
def test_drill43_masked_pairs_do_not_leak(cls):
    mod, (x, mask) = _case_trimul(cls)
    out = mod(x, mask)
    x2 = x.clone()
    x2[0, 0, 2] += 5.0  # (0,2) and (2,0) are masked
    x2[0, 2, 0] -= 5.0
    out2 = mod(x2, mask)
    keep = torch.ones(3, 3, dtype=torch.bool)
    keep[0, 2] = keep[2, 0] = False  # their own gate may change
    torch.testing.assert_close(out2[0][keep], out[0][keep], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Drill #44: outer product mean
# ---------------------------------------------------------------------------


def test_drill44_parameter_names_and_zero_init():
    mod = OuterProductMean(c_in=4, c_hidden=3, c_out=5)
    assert _param_shapes(mod) == {
        "norm.weight": (4,),
        "norm.bias": (4,),
        "proj_a.weight": (3, 4),
        "proj_b.weight": (3, 4),
        "proj_o.weight": (5, 9),
        "proj_o.bias": (5,),
    }
    (m,) = _randn(30, (2, 3, 4, 4))
    out = mod(m, torch.ones(2, 3, 4))
    assert out.shape == (2, 4, 4, 5)
    assert torch.all(out == 0)


def test_drill44_matches_fingerprint():
    mod, (m, mask) = _case_opm()
    out = mod(m, mask)
    assert out.shape == (1, 4, 4, 5)
    _check_fp(out, FP_OPM)


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 10])
def test_drill44_chunked_matches_unchunked(chunk_size):
    mod, (m, mask) = _case_opm()
    torch.testing.assert_close(mod(m, mask, chunk_size=chunk_size), mod(m, mask), atol=1e-5, rtol=1e-5)


def test_drill44_divides_by_pair_count_not_depth():
    mod, (m, mask) = _case_opm()
    (pad,) = _randn(31, (1, 2, 4, 4))
    m2 = torch.cat([m, pad], dim=1)  # two fully padded MSA rows
    mask2 = torch.cat([mask, torch.zeros(1, 2, 4)], dim=1)
    torch.testing.assert_close(mod(m2, mask2), mod(m, mask), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Drill #46: relative position encoding
# ---------------------------------------------------------------------------

# Two chains: tokens 0-2 are chain A (asym 0, entity 0, sym 0); token 2 is a second
# token of residue 1. Tokens 3-4 are chain B (asym 1, entity 0, sym 1).
_RP = dict(
    asym_id=torch.tensor([[0, 0, 0, 1, 1]]),
    residue_index=torch.tensor([[0, 1, 1, 0, 40]]),
    entity_id=torch.tensor([[0, 0, 0, 0, 1]]),
    token_index=torch.tensor([[0, 1, 2, 0, 1]]),
    sym_id=torch.tensor([[0, 0, 0, 1, 3]]),
)


def _blocks(f, r_max, s_max):
    w = 2 * r_max + 2
    return f[..., :w], f[..., w : 2 * w], f[..., 2 * w], f[..., 2 * w + 1 :]


def test_drill46_width_and_one_hot_blocks():
    f = relative_position_features(**_RP, r_max=4, s_max=2)
    assert f.shape == (1, 5, 5, 4 * 5 + 2 * 3 + 1)
    assert f.dtype == torch.float32
    pos, tok, ent, ch = _blocks(f, 4, 2)
    for blk in (pos, tok, ch):
        assert torch.all(blk.sum(-1) == 1)
    assert set(ent.unique().tolist()) <= {0.0, 1.0}


def test_drill46_hand_computed_buckets():
    r_max, s_max = 4, 2
    pos, tok, ent, ch = _blocks(relative_position_features(**_RP, r_max=r_max, s_max=s_max), r_max, s_max)
    pos, tok, ch, ent = pos.argmax(-1)[0], tok.argmax(-1)[0], ch.argmax(-1)[0], ent[0]
    # same chain: residue 0 - residue 1 + 4 = 3
    assert pos[0, 1] == 3 and pos[1, 0] == 5
    # different chains -> extra bucket 2*r_max+1 = 9
    assert pos[0, 3] == 9
    # same chain, clipped: residue 40 - 0 + 4 -> 8
    assert pos[4, 3] == 8 and pos[3, 4] == 0
    # token offset only inside the same residue: tokens 1 and 2 share residue 1
    assert tok[2, 1] == 5 and tok[1, 2] == 3
    assert tok[0, 1] == 9 and tok[0, 0] == 4
    # same entity bit
    assert ent[0, 3] == 1 and ent[0, 4] == 0
    # chain offset: same chain -> bucket 2*s_max+1 = 5; different chain -> clip(sym_i - sym_j + 2)
    assert ch[0, 1] == 5
    assert ch[0, 3] == 1 and ch[3, 0] == 3
    assert ch[0, 4] == 0 and ch[4, 0] == 4


def test_drill46_encoder_is_bias_free_linear_of_features():
    enc = _fill(RelativePositionEncoder(token_z=6, r_max=4, s_max=2), seed=40)
    assert _param_shapes(enc) == {"linear_layer.weight": (6, 27)}
    f = relative_position_features(**_RP, r_max=4, s_max=2)
    out = enc(**_RP)
    assert out.shape == (1, 5, 5, 6)
    torch.testing.assert_close(out, f @ enc.linear_layer.weight.T, atol=1e-6, rtol=1e-6)


def test_drill46_default_width():
    enc = RelativePositionEncoder(token_z=3)
    assert _param_shapes(enc) == {"linear_layer.weight": (3, 4 * 33 + 2 * 3 + 1)}


# ---------------------------------------------------------------------------
# Drill #47: AdaLN, conditioned transition, Fourier embedding
# ---------------------------------------------------------------------------


def test_drill47_adaln_parameters_and_fingerprint():
    mod = AdaLN(dim=4, dim_single_cond=3)
    assert _param_shapes(mod) == {
        "s_norm.weight": (3,),
        "s_scale.weight": (4, 3),
        "s_scale.bias": (4,),
        "s_bias.weight": (4, 3),
    }
    mod, (a, s) = _case_adaln()
    out = mod(a, s)
    assert out.shape == (2, 3, 4)
    _check_fp(out, FP_ADALN)


def test_drill47_adaln_invariant_to_affine_change_of_a():
    mod, (a, s) = _case_adaln()
    torch.testing.assert_close(mod(3.0 * a + 7.0, s), mod(a, s), atol=1e-4, rtol=1e-4)


def test_drill47_transition_init_and_parameters():
    mod = ConditionedTransitionBlock(dim_single=4, dim_single_cond=3, expansion_factor=2)
    shapes = _param_shapes(mod)
    assert shapes["swish_gate.weight"] == (16, 4)
    assert shapes["a_to_b.weight"] == (8, 4)
    assert shapes["b_to_a.weight"] == (4, 8)
    assert shapes["output_projection.weight"] == (4, 3)
    assert shapes["output_projection.bias"] == (4,)
    assert "adaln.s_scale.weight" in shapes
    assert len(shapes) == 9
    assert torch.all(mod.output_projection.weight == 0)
    assert torch.all(mod.output_projection.bias == -2.0)


def test_drill47_transition_fingerprint():
    mod, (a, s) = _case_transition()
    out = mod(a, s)
    assert out.shape == (2, 3, 4)
    _check_fp(out, FP_TRANSITION)


def test_drill47_fourier_frozen_and_formula():
    torch.manual_seed(0)
    mod = FourierEmbedding(dim=16)
    assert _param_shapes(mod) == {"proj.weight": (16, 1), "proj.bias": (16,)}
    assert not any(p.requires_grad for p in mod.parameters())
    assert "proj.weight" in mod.state_dict()
    t = torch.tensor([0.0, 0.25, 1.5])
    out = mod(t)
    assert out.shape == (3, 16)
    w, b = mod.proj.weight[:, 0], mod.proj.bias
    torch.testing.assert_close(out[1], torch.cos(2 * math.pi * (0.25 * w + b)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0], torch.cos(2 * math.pi * b), atol=1e-5, rtol=1e-5)


def test_drill47_fourier_weights_are_standard_normal():
    torch.manual_seed(0)
    mod = FourierEmbedding(dim=4096)
    w = mod.proj.weight.flatten()
    assert abs(w.mean().item()) < 0.1 and abs(w.std().item() - 1.0) < 0.1
    assert abs(mod.proj.bias.std().item() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Drill #48: pair-weighted averaging
# ---------------------------------------------------------------------------


def test_drill48_parameters_and_zero_init():
    mod = PairWeightedAveraging(c_m=4, c_z=3, c_h=2, num_heads=2)
    assert _param_shapes(mod) == {
        "norm_m.weight": (4,),
        "norm_m.bias": (4,),
        "norm_z.weight": (3,),
        "norm_z.bias": (3,),
        "proj_m.weight": (4, 4),
        "proj_g.weight": (4, 4),
        "proj_z.weight": (2, 3),
        "proj_o.weight": (4, 4),
    }
    m, z = _randn(50, (1, 2, 3, 4), (1, 3, 3, 3))
    out = mod(m, z, torch.ones(1, 3, 3))
    assert out.shape == (1, 2, 3, 4)
    assert torch.all(out == 0)


def test_drill48_matches_fingerprint():
    mod, (m, z, mask) = _case_pwa()
    out = mod(m, z, mask)
    assert out.shape == (1, 2, 3, 4)
    _check_fp(out, FP_PWA)


def test_drill48_msa_row_permutation_equivariant():
    mod, (m, z, mask) = _case_pwa()
    out = mod(m, z, mask)
    torch.testing.assert_close(mod(m.flip(1), z, mask), out.flip(1), atol=1e-5, rtol=1e-5)


def test_drill48_masked_pair_weight_is_zero():
    mod, (m, z, mask) = _case_pwa()
    out = mod(m, z, mask)
    z2 = z.clone()
    z2[0, 0, 2] += 100.0  # pair (0,2) is masked
    torch.testing.assert_close(mod(m, z2, mask), out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Drill #49: triangle attention
# ---------------------------------------------------------------------------


def test_drill49_parameter_names_and_shapes():
    mod = TriangleAttention(c_in=4, c_hidden=3, num_heads=2)
    assert _param_shapes(mod) == {
        "layer_norm.weight": (4,),
        "layer_norm.bias": (4,),
        "linear.weight": (2, 4),
        "linear_q.weight": (6, 4),
        "linear_k.weight": (6, 4),
        "linear_v.weight": (6, 4),
        "linear_g.weight": (6, 4),
        "linear_g.bias": (6,),
        "linear_o.weight": (4, 6),
        "linear_o.bias": (4,),
    }


def test_drill49_starting_matches_fingerprint():
    mod, (x, mask) = _case_triatt(True)
    out = mod(x, mask)
    assert out.shape == (1, 3, 3, 4)
    _check_fp(out, FP_TRIATT_START)


def test_drill49_ending_matches_fingerprint():
    mod, (x, mask) = _case_triatt(False)
    out = mod(x, mask)
    assert out.shape == (1, 3, 3, 4)
    _check_fp(out, FP_TRIATT_END)


def test_drill49_ending_is_starting_on_transpose():
    start, (x, mask) = _case_triatt(True)
    end = TriangleAttention(c_in=4, c_hidden=3, num_heads=2, starting=False)
    end.load_state_dict(start.state_dict())
    expected = start(x.transpose(1, 2), mask.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(end(x, mask), expected, atol=1e-5, rtol=1e-5)


def test_drill49_mask_none_and_all_masked_row():
    mod, (x, _) = _case_triatt(True)
    torch.testing.assert_close(mod(x, None), mod(x, torch.ones(1, 3, 3)), atol=1e-6, rtol=1e-6)
    mask = torch.ones(1, 3, 3)
    mask[0, 1] = 0.0  # every key in row 1 is masked
    out = mod(x, mask)
    assert torch.isfinite(out).all()


def test_drill49_masked_key_is_ignored():
    mod, (x, mask) = _case_triatt(True)
    out = mod(x, mask)
    x2 = x.clone()
    x2[0, 0, 2] += 5.0  # (0,2) is masked: never a key in row 0
    out2 = mod(x2, mask)
    torch.testing.assert_close(out2[0, 0, :2], out[0, 0, :2], atol=1e-5, rtol=1e-5)
