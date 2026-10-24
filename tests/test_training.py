"""Tests for Section 4: training mechanics (Drills #26-#35)."""

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drills import training as T


# ---------------------------------------------------------------------------
# Drill #26
# ---------------------------------------------------------------------------


def test_drill26_dataset_len_getitem_and_loader():
    x = torch.randn(10, 3)
    y = torch.arange(10) % 2
    ds = T.TensorPairDataset(x, y)
    assert len(ds) == 10
    xi, yi = ds[3]
    assert torch.equal(xi, x[3]) and int(yi) == int(y[3])
    batches = list(DataLoader(ds, batch_size=4, shuffle=False))
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]
    with pytest.raises(ValueError):
        T.TensorPairDataset(torch.randn(5, 3), torch.zeros(4, dtype=torch.long))


def test_drill26_mlp_shapes_and_returns_logits():
    m = T.MLPClassifier(5, 16, 3, dropout=0.1)
    assert isinstance(m.fc1, nn.Linear) and isinstance(m.fc2, nn.Linear)
    assert m.fc1.in_features == 5 and m.fc2.out_features == 3
    out = m.eval()(10 * torch.randn(7, 5))
    assert out.shape == (7, 3) and out.dtype == torch.float32
    # logits, not probabilities
    assert not torch.allclose(out.sum(-1), torch.ones(7), atol=1e-3)


def test_drill26_train_one_epoch_weights_uneven_last_batch():
    torch.manual_seed(1)
    model = nn.Linear(3, 4)
    x, y = torch.randn(10, 3), torch.randint(0, 4, (10,))
    expected = F.cross_entropy(model(x), y).item()
    loader = DataLoader(list(zip(x, y)), batch_size=4, shuffle=False)
    got = T.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.0))
    assert isinstance(got, float)
    assert abs(got - expected) < 1e-5


def test_drill26_train_one_epoch_clears_stale_grads():
    torch.manual_seed(2)
    model = nn.Linear(3, 4)
    x, y = torch.randn(8, 3), torch.randint(0, 4, (8,))
    w0 = model.weight.detach().clone()
    g_true = torch.autograd.grad(F.cross_entropy(model(x), y), model.weight)[0]
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e3)  # stale gradients from somewhere else
    loader = DataLoader(list(zip(x, y)), batch_size=8, shuffle=False)
    T.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1))
    torch.testing.assert_close(model.weight.detach(), w0 - 0.1 * g_true, atol=1e-6, rtol=0)


def test_drill26_train_one_epoch_switches_to_train_mode():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 2)
            self.modes = []

        def forward(self, x):
            self.modes.append(self.training)
            return self.lin(x)

    m = Recorder().eval()
    x, y = torch.randn(4, 3), torch.tensor([0, 1, 0, 1])
    T.train_one_epoch(m, [(x, y)], torch.optim.SGD(m.parameters(), lr=0.1))
    assert m.modes and all(m.modes)


def test_drill26_evaluate_accuracy_eval_mode_and_no_grad():
    lin = nn.Linear(3, 3)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(3))
        lin.bias.zero_()
    model = nn.Sequential(lin, nn.Dropout(0.9))
    # logits equal x in eval mode; argmax per row is 0, 2, 1, 1
    x = torch.tensor([[-1.0, -3.0, -2.0], [0.0, 1.0, 5.0], [0.1, 0.3, 0.2], [-5.0, -4.0, -6.0]])
    y = torch.tensor([0, 2, 1, 0])
    loss, acc = T.evaluate(model, [(x[:3], y[:3]), (x[3:], y[3:])])
    assert abs(acc - 0.75) < 1e-12
    assert abs(loss - F.cross_entropy(x, y).item()) < 1e-5
    assert model.training is False
    assert all(p.grad is None for p in model.parameters())


def test_drill26_overfit_one_batch_collapses_loss():
    torch.manual_seed(3)
    model = T.MLPClassifier(4, 32, 3)
    x = torch.randn(16, 4)
    y = torch.arange(16) % 3
    losses = T.overfit_one_batch(model, x, y, steps=200, lr=1e-2)
    assert len(losses) == 200 and all(isinstance(v, float) for v in losses)
    assert abs(losses[0] - math.log(3)) < 0.3
    assert losses[-1] < 0.05


def test_drill26_pad_collate_shapes_mask_and_padding():
    a, b, c = torch.randn(2, 3), torch.randn(4, 3), torch.randn(1, 3)
    x, mask, y = T.pad_collate([(a, 1), (b, 0), (c, 2)])
    assert x.shape == (3, 4, 3) and mask.shape == (3, 4) and mask.dtype == torch.bool
    assert y.dtype == torch.int64 and y.tolist() == [1, 0, 2]
    assert mask.sum(1).tolist() == [2, 4, 1]
    assert torch.equal(x[0, :2], a) and torch.equal(x[1], b) and torch.equal(x[2, :1], c)
    assert torch.all(x[~mask] == 0)


# ---------------------------------------------------------------------------
# Drill #27
# ---------------------------------------------------------------------------


def test_drill27_numerical_gradient_quadratic_and_input_untouched():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    A = M + M.T
    x = rng.standard_normal(4)
    x0 = x.copy()
    g = T.numerical_gradient(lambda v: 0.5 * v @ A @ v, x)
    np.testing.assert_allclose(g, A @ x0, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(x, x0)


def test_drill27_numerical_gradient_2d_array():
    W = np.random.default_rng(1).standard_normal((3, 4))
    g = T.numerical_gradient(lambda v: float(np.tanh(v).sum()), W)
    assert g.shape == (3, 4)
    np.testing.assert_allclose(g, 1 - np.tanh(W) ** 2, rtol=0, atol=1e-8)


def test_drill27_max_relative_error_hand_cases():
    a = np.array([1.0, 0.0, 2.0])
    b = np.array([1.1, 0.0, 2.0])
    assert abs(T.max_relative_error(a, b) - 0.1 / 1.1) < 1e-12
    assert T.max_relative_error(np.zeros(3), np.zeros(3)) == 0.0


def test_drill27_check_gradient_flags_wrong_gradient():
    x = np.random.default_rng(2).standard_normal(5)
    f = lambda v: float((v**2).sum())
    ok, err = T.check_gradient(f, lambda v: 2 * v, x)
    assert ok and err < 1e-6
    ok, err = T.check_gradient(f, lambda v: v, x)
    assert not ok and abs(err - 0.5) < 1e-6


def test_drill27_check_gradient_relu_kink():
    f = lambda v: float(np.maximum(v, 0).sum())
    grad = lambda v: (v > 0).astype(np.float64)
    assert T.check_gradient(f, grad, np.array([0.5, -0.7, 1.3]))[0]
    # an input within eps of the kink breaks the check
    assert not T.check_gradient(f, grad, np.array([0.5, 3e-6, 1.3]))[0]


# ---------------------------------------------------------------------------
# Drill #28
# ---------------------------------------------------------------------------


def _make_params(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3, 4, generator=g).requires_grad_(), torch.randn(4, generator=g).requires_grad_()]


def _run(opt, params, steps=6):
    g = torch.Generator().manual_seed(123)
    xs = [torch.randn(5, 3, generator=g) for _ in range(steps)]
    for x in xs:
        opt.zero_grad()
        loss = ((x @ params[0] + params[1]) ** 2).mean() + params[0].abs().sum() * 0.01
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in params]


def test_drill28_sgd_momentum_matches_torch():
    p_ref, p_me = _make_params(0), _make_params(0)
    ref = _run(torch.optim.SGD(p_ref, lr=0.05, momentum=0.9, weight_decay=0.01), p_ref)
    me = _run(T.SGDMomentum(p_me, lr=0.05, momentum=0.9, weight_decay=0.01), p_me)
    for a, b in zip(me, ref):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_drill28_adam_matches_torch_with_l2():
    p_ref, p_me = _make_params(1), _make_params(1)
    ref = _run(torch.optim.Adam(p_ref, lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1), p_ref)
    me = _run(T.Adam(p_me, lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1), p_me)
    for a, b in zip(me, ref):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_drill28_adam_first_step_is_lr_sized():
    # with bias correction the first step moves each coordinate by about lr
    p = torch.tensor([3.0, -2.0, 0.5], requires_grad=True)
    opt = T.Adam([p], lr=0.1)
    p.grad = torch.tensor([1e-3, -5.0, 200.0])
    opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor([2.9, -1.9, 0.4]), atol=1e-4, rtol=0)


def test_drill28_adamw_matches_torch_with_param_groups():
    p_ref, p_me = _make_params(2), _make_params(2)
    groups = lambda ps: [{"params": [ps[0]], "weight_decay": 0.5}, {"params": [ps[1]], "weight_decay": 0.0}]
    ref = _run(torch.optim.AdamW(groups(p_ref), lr=1e-2), p_ref)
    me = _run(T.AdamW(groups(p_me), lr=1e-2), p_me)
    for a, b in zip(me, ref):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    # decoupled decay is not the same as Adam with L2
    p_l2 = _make_params(2)
    l2 = _run(torch.optim.Adam([{"params": [p_l2[0]], "weight_decay": 0.5}, {"params": [p_l2[1]]}], lr=1e-2), p_l2)
    assert (me[0] - l2[0]).abs().max() > 1e-4


def test_drill28_zero_grad_sets_none():
    for cls in (T.SGDMomentum, T.Adam, T.AdamW):
        ps = _make_params(3)
        opt = cls(ps, lr=0.1)
        for p in ps:
            p.grad = torch.ones_like(p)
        opt.zero_grad()
        assert all(p.grad is None for p in ps)


def test_drill28_decay_param_groups_excludes_norm_and_bias():
    model = nn.Sequential()
    model.add_module("proj", nn.Linear(4, 8))
    model.add_module("layer_norm", nn.LayerNorm(8))
    model.add_module("out", nn.Linear(8, 2, bias=False))
    model.out.weight.requires_grad_(False)
    model.add_module("head", nn.Linear(8, 2))
    groups = T.decay_param_groups(model, 0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    ids = lambda ps: [id(p) for p in ps]
    assert ids(groups[0]["params"]) == ids([model.proj.weight, model.head.weight])
    assert ids(groups[1]["params"]) == ids(
        [model.proj.bias, model.layer_norm.weight, model.layer_norm.bias, model.head.bias]
    )


# ---------------------------------------------------------------------------
# Drill #29
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step,min_lr,expected",
    [(0, 0.0, 0.0), (5, 0.0, 0.5), (10, 0.0, 1.0), (60, 0.0, 0.5), (110, 0.0, 0.0),
     (500, 0.0, 0.0), (60, 0.1, 0.55), (35, 0.0, 0.5 * (1 + math.cos(math.pi / 4))), (200, 0.1, 0.1)],
)
def test_drill29_warmup_cosine_values(step, min_lr, expected):
    assert abs(T.warmup_cosine_lr(step, 1.0, 10, 110, min_lr) - expected) < 1e-12


@pytest.mark.parametrize(
    "step,base_lr,expected",
    [(0, 0.0, 0.0), (500, 0.0, 0.9e-3), (1000, 0.0, 1.8e-3), (50_000, 0.0, 1.8e-3),
     (50_001, 0.0, 1.8e-3 * 0.95), (99_999, 0.0, 1.8e-3 * 0.95), (100_000, 0.0, 1.8e-3 * 0.95**2),
     (500, 1e-4, 1e-4 + 0.5 * 1.7e-3), (1000, 1e-4, 1.8e-3)],
)
def test_drill29_af3_lr_values(step, base_lr, expected):
    assert abs(T.af3_lr(step, base_lr=base_lr) - expected) < 1e-15


def test_drill29_af3_lr_warmup_never_overshoots():
    lrs = [T.af3_lr(s, base_lr=5e-4) for s in range(0, 1001, 50)]
    assert max(lrs) <= 1.8e-3 + 1e-15


def test_drill29_scheduler_tracks_formula_per_optimiser_step():
    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.SGD([{"params": [w], "lr": 0.5}, {"params": [b], "lr": 0.1}], lr=0.5)
    sched = T.make_warmup_cosine_scheduler(opt, warmup_steps=4, total_steps=20, min_lr=0.0)
    for k in range(0, 25):
        for g, m in zip(opt.param_groups, (0.5, 0.1)):
            assert abs(g["lr"] - T.warmup_cosine_lr(k, m, 4, 20, 0.0)) < 1e-12
        opt.step()
        sched.step()


# ---------------------------------------------------------------------------
# Drill #30
# ---------------------------------------------------------------------------


def _model_with_grads(seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    m(torch.randn(6, 4)).pow(2).sum().backward()
    return m


def _copy_with_grads(m):
    c = copy.deepcopy(m)
    for p, q in zip(c.parameters(), m.parameters()):
        p.grad = None if q.grad is None else q.grad.clone()
    return c


def test_drill30_global_grad_norm_matches_torch_and_skips_none():
    m = _model_with_grads()
    ref = _copy_with_grads(m)
    expected = torch.nn.utils.clip_grad_norm_(ref.parameters(), float("inf"))
    extra = torch.zeros(3, requires_grad=True)  # grad is None
    got = T.global_grad_norm(list(m.parameters()) + [extra])
    assert got.dim() == 0
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    assert float(T.global_grad_norm([extra])) == 0.0


@pytest.mark.parametrize("max_norm", [0.5, 1e6])
def test_drill30_clip_grad_norm_matches_torch(max_norm):
    m = _model_with_grads(1)
    ref = _copy_with_grads(m)
    exp_norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), max_norm)
    got_norm = T.clip_grad_norm(m.parameters(), max_norm)
    torch.testing.assert_close(got_norm, exp_norm, atol=1e-6, rtol=1e-6)
    for p, q in zip(m.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-6)


def test_drill30_accumulation_equals_full_batch_with_uneven_chunks():
    torch.manual_seed(4)
    model = nn.Linear(3, 2).double()
    x, y = torch.randn(10, 3, dtype=torch.float64), torch.randn(10, 2, dtype=torch.float64)
    full = F.mse_loss(model(x), y)
    g_full = torch.autograd.grad(full, list(model.parameters()))
    for p in model.parameters():
        p.grad = torch.full_like(p, 7.0)  # stale
    loss = T.accumulate_gradients(model, F.mse_loss, x, y, micro_batch_size=4)
    assert abs(loss - full.item()) < 1e-10
    for p, g in zip(model.parameters(), g_full):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=0)


def test_drill30_per_module_grad_norms():
    m = nn.Sequential()
    m.add_module("a", nn.Linear(2, 2))
    m.add_module("act", nn.ReLU())
    m.add_module("b", nn.Linear(2, 2))
    m.a.weight.grad = torch.ones(2, 2)
    m.a.bias.grad = torch.ones(2)
    norms = T.per_module_grad_norms(m)
    assert set(norms) == {"a"}
    assert isinstance(norms["a"], float) and abs(norms["a"] - math.sqrt(6)) < 1e-6


@pytest.mark.parametrize(
    "norm,step,expected",
    [(5.0, 0, False), (50.0, 0, False), (50.0, 100, True), (5.0, 100, False), (10.0, 100, False),
     (float("nan"), 0, True), (float("inf"), 0, True), (torch.tensor(float("nan")), 200, True),
     (torch.tensor(11.0), 100, True)],
)
def test_drill30_should_skip_step(norm, step, expected):
    assert T.should_skip_step(norm, step, threshold=10.0, warmup_steps=100) is expected


# ---------------------------------------------------------------------------
# Drill #31
# ---------------------------------------------------------------------------


def _amp_setup(seed=5):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    mbs = [(torch.randn(6, 4), torch.randint(0, 3, (6,))) for _ in range(3)]
    return model, mbs


def _fp32_reference_step(model, mbs, max_norm, lr):
    model.zero_grad()
    x = torch.cat([m[0] for m in mbs])
    y = torch.cat([m[1] for m in mbs])
    F.cross_entropy(model(x), y).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    torch.optim.SGD(model.parameters(), lr=lr).step()
    return float(norm)


def test_drill31_amp_step_matches_fp32_with_unscale_before_clip():
    model, mbs = _amp_setup()
    ref = copy.deepcopy(model)
    exp_norm = _fp32_reference_step(ref, mbs, max_norm=0.1, lr=0.5)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    norm = T.amp_train_step(model, opt, scaler, mbs, F.cross_entropy, max_norm=0.1, amp_enabled=False)
    assert abs(norm - exp_norm) < 1e-5
    for p, q in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_drill31_amp_step_skips_on_nonfinite_grads():
    model, mbs = _amp_setup(6)
    before = [p.detach().clone() for p in model.parameters()]
    bad = mbs[1][0].clone()
    bad[0, 0] = float("inf")
    mbs[1] = (bad, mbs[1][1])
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    norm = T.amp_train_step(model, opt, scaler, mbs, F.cross_entropy, max_norm=1.0, amp_enabled=False)
    assert not math.isfinite(norm)
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p.detach(), q)
    assert scaler.get_scale() < 2.0**10


def test_drill31_amp_bf16_step_close_to_fp32():
    model, mbs = _amp_setup(7)
    ref = copy.deepcopy(model)
    _fp32_reference_step(ref, mbs, max_norm=1.0, lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    T.amp_train_step(model, opt, scaler, mbs, F.cross_entropy, max_norm=1.0, amp_dtype=torch.bfloat16)
    for p, q in zip(model.parameters(), ref.parameters()):
        assert p.dtype == torch.float32
        torch.testing.assert_close(p, q, atol=2e-3, rtol=2e-2)


def test_drill31_fp32_island_inside_bf16_autocast():
    a, b = torch.randn(8, 16), torch.randn(16, 4)
    exact = a @ b
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.mm(a, b).dtype == torch.bfloat16  # sanity: autocast is active
        out = T.fp32_island(torch.mm, a, b)
        out_from_bf16 = T.fp32_island(torch.mm, a.bfloat16(), b.bfloat16())
    assert out.dtype == torch.float32 and out_from_bf16.dtype == torch.float32
    torch.testing.assert_close(out, exact, atol=1e-5, rtol=1e-5)


def test_drill31_fp32_island_passes_integer_args_through():
    x = torch.randn(5, 3).bfloat16()
    idx = torch.tensor([4, 0, 2])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = T.fp32_island(lambda t, i, s: t[i] * s, x, idx, 2.0)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.float()[idx] * 2.0)


# ---------------------------------------------------------------------------
# Drill #32
# ---------------------------------------------------------------------------


def _scalar_model(v=0.0):
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(v)
    return m


def test_drill32_ema_warmup_hand_values():
    m = _scalar_model(0.0)
    ema = T.EMA(m, decay=0.999)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)  # d = 2/11
    ema.update(m)  # d = 3/12
    ema.copy_to(m)
    assert abs(m.weight.item() - 10.5 / 11) < 1e-6


def test_drill32_ema_without_warmup_and_decay_cap():
    m = _scalar_model(0.0)
    ema = T.EMA(m, decay=0.9, use_warmup=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    ema.update(m)
    ema.copy_to(m)
    assert abs(m.weight.item() - 0.19) < 1e-6

    m = _scalar_model(0.0)
    ema = T.EMA(m, decay=0.5)
    for _ in range(100):
        ema.update(m)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)  # warm-up value is above 0.5 now, so d = 0.5
    ema.copy_to(m)
    assert abs(m.weight.item() - 0.5) < 1e-6


def test_drill32_store_copy_to_restore_round_trip():
    torch.manual_seed(8)
    m = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    ema = T.EMA(m, decay=0.9)
    m.train()(torch.randn(16, 3))  # moves BatchNorm running stats
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    original = {k: v.clone() for k, v in m.state_dict().items()}
    ema.store(m)
    ema.copy_to(m)
    assert not torch.equal(m[0].weight, original["0.weight"])
    assert not torch.equal(m[1].running_mean, original["1.running_mean"])
    ema.restore(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, original[k]), k


def test_drill32_integer_buffers_untouched_and_no_graph():
    m = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = T.EMA(m)
    m.train()(torch.randn(8, 2))
    m.train()(torch.randn(8, 2))
    assert int(m[1].num_batches_tracked) == 2
    ema.update(m)
    ema.copy_to(m)
    assert int(m[1].num_batches_tracked) == 2
    assert all(not p.grad_fn for p in m.parameters())
    assert all(p.grad is None for p in m.parameters())


# ---------------------------------------------------------------------------
# Drill #33
# ---------------------------------------------------------------------------


def test_drill33_structure_wise_mean_vs_atom_mean():
    per_atom = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0], requires_grad=True)
    idx = torch.tensor([0, 0, 0, 0, 1])
    out = T.structure_wise_mean(per_atom, idx, 2)
    assert out.dim() == 0
    assert abs(out.item() - 3.0) < 1e-6
    assert abs(per_atom.mean().item() - 1.8) < 1e-6  # the big structure dominates
    out.backward()
    torch.testing.assert_close(per_atom.grad, torch.tensor([0.125, 0.125, 0.125, 0.125, 0.5]))


def test_drill33_structure_wise_mean_unsorted_and_empty_structures():
    per_atom = torch.tensor([2.0, 4.0, 6.0, 1.0])
    idx = torch.tensor([3, 0, 3, 0])  # structures 1 and 2 are empty
    assert abs(T.structure_wise_mean(per_atom, idx, 4).item() - (2.5 + 4.0) / 2) < 1e-6
    empty = T.structure_wise_mean(torch.zeros(0), torch.zeros(0, dtype=torch.long), 3)
    assert torch.isfinite(empty) and empty.item() == 0.0


def test_drill33_ddp_scaled_loss_averages_to_global_mean():
    rank_losses = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0])]
    global_count = torch.tensor(4.0)
    vals = [T.ddp_scaled_loss(l.sum(), global_count, 2) for l in rank_losses]
    ddp_avg = sum(vals) / 2
    assert abs(ddp_avg.item() - 16.0 / 4) < 1e-6
    # the naive mean of local means is not the global mean
    assert abs(sum(l.mean() for l in rank_losses).item() / 2 - 4.0) > 1.0


def test_drill33_temperature_sampling_probs():
    np.testing.assert_allclose(T.temperature_sampling_probs([100, 1], 1.0), [100 / 101, 1 / 101])
    np.testing.assert_allclose(T.temperature_sampling_probs([100, 1], 2.0), [10 / 11, 1 / 11])
    np.testing.assert_allclose(T.temperature_sampling_probs([100, 1], 1e9), [0.5, 0.5], atol=1e-6)
    p = T.temperature_sampling_probs([9, 0, 4], 2.0)
    assert p.dtype == np.float64
    np.testing.assert_allclose(p, [0.6, 0.0, 0.4])


def test_drill33_sample_dataset_indices_distribution_and_range():
    sizes = [1000, 10, 1]
    out = T.sample_dataset_indices(sizes, 2.0, 20000, seed=0)
    assert out.shape == (20000, 2) and out.dtype == np.int64
    ds, idx = out[:, 0], out[:, 1]
    assert np.all(idx >= 0) and np.all(idx < np.asarray(sizes)[ds])
    freq = np.bincount(ds, minlength=3) / len(ds)
    np.testing.assert_allclose(freq, T.temperature_sampling_probs(sizes, 2.0), atol=0.015)
    assert np.array_equal(out, T.sample_dataset_indices(sizes, 2.0, 20000, seed=0))


# ---------------------------------------------------------------------------
# Drill #34
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale,fan,fan_value", [(2.0, "fan_in", 512), (1.0, "fan_out", 256), (1.0, "fan_avg", 384)])
def test_drill34_trunc_normal_variance_and_bounds(scale, fan, fan_value):
    w = torch.empty(256, 512)
    out = T.trunc_normal_init_(w, scale=scale, fan=fan)
    assert out is w
    target_std = math.sqrt(scale / fan_value)
    assert abs(w.std().item() / target_std - 1) < 0.03
    bound = 2 * target_std / 0.87962566103423978
    assert w.abs().max().item() <= bound + 1e-6
    assert w.abs().max().item() > 0.95 * bound


def test_drill34_zero_and_gating_init():
    lin = T.zero_init_(nn.Linear(4, 3))
    assert torch.all(lin.weight == 0) and torch.all(lin.bias == 0)
    gate = T.gating_init_(nn.Linear(4, 3))
    assert torch.all(gate.weight == 0) and torch.all(gate.bias == 1.0)
    gate = T.gating_init_(nn.Linear(4, 3), bias=-2.0)
    x = torch.randn(5, 4)
    torch.testing.assert_close(torch.sigmoid(gate(x)), torch.full((5, 3), 1 / (1 + math.e**2)))


def test_drill34_zero_init_residual_is_identity_at_init():
    torch.manual_seed(9)
    blk = T.ZeroInitResidualMLP(64, 128)
    x = torch.randn(2, 5, 64)
    assert torch.equal(blk(x), x)
    assert abs(blk.in_proj.weight.std().item() / math.sqrt(2 / 64) - 1) < 0.1
    blk(x).pow(2).sum().backward()
    assert torch.all(blk.in_proj.weight.grad == 0)
    assert blk.out_proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Drill #35
# ---------------------------------------------------------------------------


def test_drill35_fixed_train_learns():
    out = T.fixed_train(seed=0, n_steps=200)
    losses = out["losses"]
    assert len(losses) == 200 and all(isinstance(v, float) for v in losses)
    assert abs(losses[0] - math.log(4)) < 0.2  # initial loss is about ln C
    assert losses[-1] < 0.2
    assert out["val_acc"] > 0.9


def test_drill35_fixed_train_returns_logits_and_is_deterministic():
    a = T.fixed_train(seed=1, n_steps=50)
    b = T.fixed_train(seed=1, n_steps=50)
    assert a["losses"] == b["losses"] and a["val_acc"] == b["val_acc"]
    model = a["model"].eval()
    with torch.no_grad():
        logits = model(torch.randn(16, 8))
    assert logits.shape == (16, 4)
    assert not torch.allclose(logits.sum(-1), torch.ones(16), atol=1e-3)


def test_drill35_fixed_beats_buggy():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = T.buggy_train(seed=2)
    good = T.fixed_train(seed=2)
    assert good["val_acc"] > bad["val_acc"] + 0.5
    assert good["losses"][-1] < bad["losses"][-1] - 0.5
