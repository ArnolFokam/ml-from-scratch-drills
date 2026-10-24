"""Section 4: training mechanics (Drills #26-#35).

PyTorch unless a docstring says otherwise. Every stub raises
NotImplementedError until you fill it in. Drill #35 also ships a
deliberately buggy training function (`buggy_train`) that you must not edit;
you write `fixed_train` instead.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Drill #26: end-to-end training script, closed book
# ---------------------------------------------------------------------------


class TensorPairDataset(Dataset):
    """A map-style Dataset over paired tensors.

    Drill #26. Tier A. Target: part of the 25 min for this drill.

    Args:
        x: (N, ...) float tensor of inputs.
        y: (N,) int64 tensor of class labels.

    `len(ds)` is N and `ds[i]` returns the tuple `(x[i], y[i])`.
    Raise ValueError if x and y disagree on N.

    Hints:
        - Which two methods does DataLoader need from a map-style Dataset?
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y
        raise NotImplementedError("Drill #26")

    def __len__(self) -> int:
        raise NotImplementedError("Drill #26")

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Drill #26")


class MLPClassifier(nn.Module):
    """Two-layer MLP: Linear -> ReLU -> Dropout -> Linear, returning logits.

    Drill #26. Tier A.

    Args:
        d_in: input features.
        d_hidden: hidden width.
        n_classes: number of classes C.
        dropout: dropout probability after the ReLU.

    forward: x (B, d_in) float32 -> logits (B, C) float32 (no softmax).
    Name the layers `fc1` and `fc2`.

    Hints:
        - Should the model or the loss apply the softmax?
    """

    def __init__(self, d_in: int, d_hidden: int, n_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_classes = n_classes
        self.dropout = dropout
        raise NotImplementedError("Drill #26")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #26")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """One pass over `loader` with cross-entropy loss and one optimiser step per batch.

    Drill #26. Tier A.

    Puts the model in train mode first. Returns the mean loss per EXAMPLE over
    the epoch (batches may have different sizes, e.g. a short last batch),
    measured on each batch before its update, as a Python float.

    Hints:
        - What happens to gradients left over from a previous call?
        - If the last batch has 2 examples and the others 32, how should its loss be weighted?
    """
    raise NotImplementedError("Drill #26")


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    """Mean cross-entropy per example and argmax accuracy over `loader`.

    Drill #26. Tier A.

    Must put the model in eval mode, build no autograd graph, and leave
    parameter gradients untouched. Returns `(mean_loss, accuracy)` as floats.

    Hints:
        - `torch.max(x, dim)` returns two things; which one is the class?
        - What do dropout and batch norm do differently in eval mode?
    """
    raise NotImplementedError("Drill #26")


def overfit_one_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-2,
) -> list[float]:
    """Sanity check: train on a single batch with Adam until the loss collapses.

    Drill #26. Tier A.

    Args:
        model: maps x -> logits (B, C).
        x, y: one batch; y is (B,) int64.
        steps: number of optimiser steps.
        lr: Adam learning rate.

    Returns the list of `steps` cross-entropy values (Python floats), each
    measured before its update.

    Hints:
        - What should the very first loss be for C balanced classes at random init?
    """
    raise NotImplementedError("Drill #26")


def pad_collate(
    batch: Sequence[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """collate_fn for variable-length sequences (Drill #26 follow-up).

    Drill #26. Tier A.

    Args:
        batch: list of (seq, label) with seq of shape (L_i, D) float and label an int.

    Returns:
        x: (B, L_max, D) float, zero-padded at the end.
        mask: (B, L_max) bool, True where a real token is (False on padding).
        y: (B,) int64 labels.

    Hints:
        - What is L_max when every sequence has the same length?
    """
    raise NotImplementedError("Drill #26")


# ---------------------------------------------------------------------------
# Drill #27: finite-difference gradient check
# ---------------------------------------------------------------------------


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference gradient of a scalar function. NumPy, float64.

    Drill #27. Tier A. Target: about 10 min (with the next two functions).

    Args:
        f: maps an array shaped like x to a scalar.
        x: float64 array of any shape. It must be unchanged when you return.
        eps: step size.

    Returns: array shaped like x with df/dx_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps).

    Hints:
        - If you perturb x in place, what must you do before moving to the next element?
        - Why is float32 a bad idea with eps = 1e-5?
    """
    raise NotImplementedError("Drill #27")


def max_relative_error(a: np.ndarray, b: np.ndarray, tiny: float = 1e-12) -> float:
    """max_i |a_i - b_i| / max(|a_i|, |b_i|, tiny).

    Drill #27. Tier A.

    Returns 0.0 when a and b are both all zeros. Python float.

    Hints:
        - What should happen where both entries are exactly 0?
    """
    raise NotImplementedError("Drill #27")


def check_gradient(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-6,
) -> tuple[bool, float]:
    """Compare an analytic gradient against the central difference.

    Drill #27. Tier A.

    Returns `(passed, err)` with err the max relative error and passed = err < tol.

    Hints:
        - A ReLU kink at 0: what happens if some x_i sits within eps of it?
    """
    raise NotImplementedError("Drill #27")


# ---------------------------------------------------------------------------
# Drill #28: optimisers (SGD with momentum, Adam, AdamW)
# ---------------------------------------------------------------------------


class SGDMomentum:
    """SGD with heavy-ball momentum, matching torch.optim.SGD (dampening=0, nesterov=False).

    Drill #28. Tier B. Target: about 15 min for the three optimisers.

    Args:
        params: iterable of tensors with requires_grad=True.
        lr, momentum, weight_decay: as in torch.optim.SGD (weight decay is L2: g += wd * p).

    Methods: `step()` updates params in place under no_grad (skip params whose
    .grad is None); `zero_grad()` sets every .grad to None.

    Hints:
        - Is the buffer mu * buf + g or mu * buf + (1 - mu) * g?
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        raise NotImplementedError("Drill #28")

    def step(self) -> None:
        raise NotImplementedError("Drill #28")

    def zero_grad(self) -> None:
        raise NotImplementedError("Drill #28")


class Adam:
    """Adam with bias correction, matching torch.optim.Adam (amsgrad=False).

    Drill #28. Tier B.

    weight_decay here is the classic L2 form (added to the gradient), as in
    torch.optim.Adam. Methods as in SGDMomentum.

    Hints:
        - Does the step counter t start at 0 or at 1 when you divide by 1 - beta**t?
        - Is eps added to sqrt(v_hat) or inside the sqrt?
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        raise NotImplementedError("Drill #28")

    def step(self) -> None:
        raise NotImplementedError("Drill #28")

    def zero_grad(self) -> None:
        raise NotImplementedError("Drill #28")


class AdamW:
    """AdamW (decoupled weight decay), matching torch.optim.AdamW (amsgrad=False).

    Drill #28. Tier B.

    Accepts either a list of tensors or a list of param-group dicts
    `{"params": [...], "weight_decay": wd}` (a group without the key uses the
    default weight_decay). Methods as in SGDMomentum.

    Hints:
        - Is the decay applied to the gradient or directly to the parameter, and scaled by what?
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        raise NotImplementedError("Drill #28")

    def step(self) -> None:
        raise NotImplementedError("Drill #28")

    def zero_grad(self) -> None:
        raise NotImplementedError("Drill #28")


def decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_keywords: Sequence[str] = ("bias", "norm"),
) -> list[dict]:
    """Split parameters into a decayed and a non-decayed group.

    Drill #28. Tier B.

    A parameter goes to the no-decay group if its full name (from
    `named_parameters`, lower-cased) contains any keyword, or if it is 1-D.
    Returns `[{"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0}]`, each list in
    `named_parameters` order. Skip parameters with requires_grad=False.
    Compare the Boltz code, src/boltz/model/models/boltz2.py::configure_optimizers.

    Hints:
        - Why should LayerNorm gains not be pulled towards 0?
    """
    raise NotImplementedError("Drill #28")


# ---------------------------------------------------------------------------
# Drill #29: LR schedules
# ---------------------------------------------------------------------------


def warmup_cosine_lr(
    step: int,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup then cosine decay. Pure Python.

    Drill #29. Tier B. Target: about 10 min for this and af3_lr.

    step is 0-indexed (number of optimiser steps already taken).
      - step < warmup_steps: max_lr * step / warmup_steps.
      - warmup_steps <= step <= total_steps: with p = (step - warmup_steps) / (total_steps - warmup_steps),
        min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * p)).
      - step > total_steps: min_lr.

    Hints:
        - Where do the brackets go in 0.5 * (1 + cos(...))?
        - What do you get at step == warmup_steps and at step == total_steps?
    """
    raise NotImplementedError("Drill #29")


def af3_lr(
    step: int,
    base_lr: float = 0.0,
    max_lr: float = 1.8e-3,
    warmup_steps: int = 1000,
    start_decay_after: int = 50_000,
    decay_every: int = 50_000,
    decay_factor: float = 0.95,
) -> float:
    """AlphaFold3-style schedule: linear warmup, plateau, then step decay.

    Drill #29. Tier B.

      - step <= warmup_steps: base_lr + (max_lr - base_lr) * step / warmup_steps.
      - warmup_steps < step <= start_decay_after: max_lr.
      - step > start_decay_after: max_lr * decay_factor ** ((step - start_decay_after) // decay_every + 1).

    Compare the Boltz code, src/boltz/model/optim/scheduler.py::AlphaFoldLRScheduler.

    Hints:
        - If base_lr > 0, does base_lr + max_lr * step / warmup_steps overshoot max_lr?
        - At step = start_decay_after + 1, how many decays have been applied?
    """
    raise NotImplementedError("Drill #29")


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Wrap warmup_cosine_lr as a torch scheduler.

    Drill #29. Tier B.

    The optimiser's current lr (each group's "lr" at construction) is max_lr.
    After k calls to `scheduler.step()` (one per optimiser step, called after
    `optimizer.step()`), each group's lr must equal
    `warmup_cosine_lr(k, max_lr, warmup_steps, total_steps, min_lr)`.

    Hints:
        - LambdaLR multiplies the initial lr by your lambda; what should the lambda return?
    """
    raise NotImplementedError("Drill #29")


# ---------------------------------------------------------------------------
# Drill #30: global-norm clipping, gradient accumulation, grad-norm logging, skip guard
# ---------------------------------------------------------------------------


def global_grad_norm(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """L2 norm of all gradients concatenated, as a 0-dim float tensor.

    Drill #30. Tier B. Target: about 20 min for this block.

    Parameters whose .grad is None are skipped; returns tensor(0.) if none have a grad.

    Hints:
        - Did you `.sum()` after `.pow(2)` before adding across tensors?
    """
    raise NotImplementedError("Drill #30")


def clip_grad_norm(
    params: Iterable[torch.Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip gradients in place by global L2 norm, like torch.nn.utils.clip_grad_norm_.

    Drill #30. Tier B.

    coef = max_norm / (total_norm + eps); if coef < 1 multiply every grad by coef.
    Returns the total norm BEFORE clipping (0-dim tensor).

    Hints:
        - Do you scale each tensor by its own norm or by the global one?
    """
    raise NotImplementedError("Drill #30")


def accumulate_gradients(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    micro_batch_size: int,
) -> float:
    """Gradient accumulation over micro-batches, exact for an uneven last chunk.

    Drill #30. Tier B.

    Zeroes the model grads, splits (x, y) along dim 0 into chunks of
    `micro_batch_size` (the last chunk may be smaller), and calls backward on
    each chunk so that afterwards every .grad equals the gradient of
    `loss_fn(model(x), y)` on the full batch (to 1e-6). `loss_fn` returns the
    MEAN over its chunk. Returns the full-batch mean loss as a float.
    Does not step any optimiser.

    Hints:
        - With chunk sizes 4, 4, 2, is dividing each chunk loss by 3 right?
    """
    raise NotImplementedError("Drill #30")


def per_module_grad_norms(model: nn.Module) -> dict[str, float]:
    """L2 gradient norm per top-level child module, for logging.

    Drill #30. Tier B.

    Keys are the names from `model.named_children()`; a child whose parameters
    all have .grad None is left out. Values are Python floats.

    Hints:
        - Is a module's norm the sum of its parameters' norms, or sqrt of the sum of squares?
    """
    raise NotImplementedError("Drill #30")


def should_skip_step(
    grad_norm: float | torch.Tensor,
    step: int,
    threshold: float,
    warmup_steps: int,
) -> bool:
    """Skip-step guard.

    Drill #30. Tier B.

    Returns True if grad_norm is NaN or inf (at any step), or if
    step >= warmup_steps and grad_norm > threshold. Otherwise False.

    Hints:
        - Why does the threshold not apply during warm-up, but the NaN check does?
    """
    raise NotImplementedError("Drill #30")


# ---------------------------------------------------------------------------
# Drill #31: mixed precision
# ---------------------------------------------------------------------------


def amp_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_norm: float,
    device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.bfloat16,
    amp_enabled: bool = True,
) -> float:
    """One optimiser step with autocast, GradScaler, accumulation and clipping.

    Drill #31. Tier B. Target: about 15 min with fp32_island.

    All micro-batches have the same size; the step must equal one step on the
    mean loss over all of them. Order: zero grads; for each micro-batch run
    the forward under `torch.autocast(device_type, dtype=amp_dtype,
    enabled=amp_enabled)` and call `scaler.scale(loss).backward()` on the
    loss divided by the number of micro-batches; then `scaler.unscale_` once,
    clip to max_norm by global norm, `scaler.step`, `scaler.update`.

    Returns the unscaled pre-clip global grad norm as a float (inf or nan if
    the step was skipped).

    Hints:
        - Clipping before unscale_: what threshold are you really applying?
        - What happens if you call unscale_ inside the micro-batch loop?
    """
    raise NotImplementedError("Drill #31")


def fp32_island(fn: Callable[..., torch.Tensor], *args: object) -> torch.Tensor:
    """Run fn with autocast disabled and all floating tensor args cast to float32.

    Drill #31. Tier B.

    Use it for SVD, det, cdist and losses inside a bf16 region. Non-tensor and
    non-floating tensor args (e.g. int64 indices) pass through unchanged.
    The output is whatever fn returns (float32 for float inputs).

    Hints:
        - Is `autocast(enabled=False)` alone enough if the inputs are already bf16?
    """
    raise NotImplementedError("Drill #31")


# ---------------------------------------------------------------------------
# Drill #32: parameter EMA
# ---------------------------------------------------------------------------


class EMA:
    """Exponential moving average of a model's floating-point state.

    Drill #32. Tier B. Target: about 10 min.

    Tracks every floating-point entry of `model.state_dict()` (parameters and
    float buffers); non-floating entries (e.g. BatchNorm num_batches_tracked)
    are not tracked and never touched. Shadows start as detached copies.

    update(model): n += 1; d = min(decay, (1 + n) / (10 + n)) if use_warmup else decay;
        shadow -= (1 - d) * (shadow - current), under no_grad. The first update uses n = 1.
    store(model): save a copy of the tracked current values.
    copy_to(model): load the shadows into the model.
    restore(model): put back what store saved.

    As in the Boltz code, src/boltz/model/modules/utils.py::ExponentialMovingAverage.

    Hints:
        - What does d look like on the first few updates, and why?
        - What breaks if copy_to is called on an integer buffer?
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, use_warmup: bool = True) -> None:
        self.decay = decay
        self.use_warmup = use_warmup
        raise NotImplementedError("Drill #32")

    def update(self, model: nn.Module) -> None:
        raise NotImplementedError("Drill #32")

    def store(self, model: nn.Module) -> None:
        raise NotImplementedError("Drill #32")

    def copy_to(self, model: nn.Module) -> None:
        raise NotImplementedError("Drill #32")

    def restore(self, model: nn.Module) -> None:
        raise NotImplementedError("Drill #32")


# ---------------------------------------------------------------------------
# Drill #33: structure-wise reduction and temperature sampling
# ---------------------------------------------------------------------------


def structure_wise_mean(
    per_atom: torch.Tensor,
    batch_index: torch.Tensor,
    num_structures: int,
) -> torch.Tensor:
    """Mean per structure first, then mean over structures.

    Drill #33. Tier B. Target: about 15 min for this block.

    Args:
        per_atom: (N_atoms,) float per-atom loss values (atoms of all structures concatenated).
        batch_index: (N_atoms,) int64, structure id in [0, num_structures).
        num_structures: number of structures S in the batch.

    Returns a 0-dim tensor, differentiable w.r.t. per_atom. Structures with no
    atoms are left out of the outer mean; if every structure is empty, return 0.
    No Python loop over structures.

    Hints:
        - index_add_ or scatter_add_ for the sums; what is the denominator for an empty structure?
        - Compare with per_atom.mean(): which structures dominate it?
    """
    raise NotImplementedError("Drill #33")


def ddp_scaled_loss(local_sum: torch.Tensor, global_count: torch.Tensor, world_size: int) -> torch.Tensor:
    """Per-rank loss whose DDP-averaged gradient equals the gradient of the global mean.

    Drill #33. Tier B.

    DDP averages gradients over ranks. Given this rank's summed loss and the
    total count across all ranks (already all-reduced), return
    local_sum * world_size / global_count.

    Hints:
        - Why is the plain local mean wrong when ranks hold different numbers of atoms?
    """
    raise NotImplementedError("Drill #33")


def temperature_sampling_probs(sizes: Sequence[int], temperature: float) -> np.ndarray:
    """Dataset sampling probabilities p_i proportional to n_i ** (1 / T). NumPy.

    Drill #33. Tier B.

    T = 1 is proportional to size; large T tends to uniform. Returns a (K,)
    float64 array summing to 1. Datasets of size 0 get probability 0.

    Hints:
        - Is the exponent T or 1/T?
    """
    raise NotImplementedError("Drill #33")


def sample_dataset_indices(
    sizes: Sequence[int],
    temperature: float,
    num_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Draw (dataset_id, index_within_dataset) pairs with temperature sampling. NumPy.

    Drill #33. Tier B.

    Pick the dataset with temperature_sampling_probs, then an index uniformly
    in [0, sizes[dataset_id]). Use np.random.default_rng(seed).
    Returns a (num_samples, 2) int64 array.

    Hints:
        - Can you draw all dataset ids in one call and all indices in one call?
    """
    raise NotImplementedError("Drill #33")


# ---------------------------------------------------------------------------
# Drill #34: initialisation
# ---------------------------------------------------------------------------


def trunc_normal_init_(weight: torch.Tensor, scale: float = 1.0, fan: str = "fan_in") -> torch.Tensor:
    """Truncated-normal init with variance scale / fan, in place.

    Drill #34. Tier C. Target: about 10 min for this block.

    weight has the nn.Linear layout (out_features, in_features). fan is
    "fan_in", "fan_out" or "fan_avg". Samples come from a normal truncated at
    +-2 std, with std = sqrt(scale / max(1, fan)) / 0.87962566103423978, so the
    variance after truncation is scale / fan. scale=1 is LeCun, scale=2 is
    He/Kaiming. Returns `weight`. As in the Boltz code,
    src/boltz/model/layers/initialize.py.

    Hints:
        - Truncation shrinks the variance; where does 0.8796... come from?
        - Should this run under no_grad?
    """
    raise NotImplementedError("Drill #34")


def zero_init_(linear: nn.Linear) -> nn.Linear:
    """Zero the weight and bias (if any) of a Linear, in place. Returns it.

    Drill #34. Tier C.

    Hints:
        - Why zero the final projection of a residual branch?
    """
    raise NotImplementedError("Drill #34")


def gating_init_(linear: nn.Linear, bias: float = 1.0) -> nn.Linear:
    """Gate init: weight 0, bias constant `bias` (1.0 as in AF2 gating; -2.0 as in AF3 Alg. 26). Returns it.

    Drill #34. Tier C.

    Hints:
        - What is sigmoid(1) and sigmoid(-2): is the gate mostly open or closed at init?
    """
    raise NotImplementedError("Drill #34")


class ZeroInitResidualMLP(nn.Module):
    """x + out_proj(relu(in_proj(x))) with in_proj He-initialised and out_proj zero-initialised.

    Drill #34. Tier C.

    Args:
        d: feature size (input and output).
        d_hidden: hidden size.

    Layers must be named `in_proj` and `out_proj`, both with bias. At init the
    block is exactly the identity. forward: (..., d) -> (..., d).

    Hints:
        - At init, which of the two weights receives a non-zero gradient?
    """

    def __init__(self, d: int, d_hidden: int) -> None:
        super().__init__()
        self.d = d
        self.d_hidden = d_hidden
        raise NotImplementedError("Drill #34")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #34")


# ---------------------------------------------------------------------------
# Drill #35: debug a training script that runs but does not learn
# ---------------------------------------------------------------------------


def make_debug_data(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toy 4-class Gaussian-blob data for Drill #35. Provided, not a drill.

    Returns (x_train (512, 8), y_train (512,), x_val (256, 8), y_val (256,)),
    float32 inputs and int64 labels. The raw features have a large offset and
    scale, so they need standardising.
    """
    g = torch.Generator().manual_seed(seed)
    centers = 3.0 * torch.randn(4, 8, generator=g)
    n = 768
    y = torch.randint(0, 4, (n,), generator=g)
    x = centers[y] + torch.randn(n, 8, generator=g)
    x = 50.0 + 20.0 * x
    return x[:512], y[:512], x[512:], y[512:]


def buggy_train(seed: int = 0, n_steps: int = 200) -> dict:
    """DELIBERATELY BUGGY. Do not fix this function; read it and write fixed_train.

    It runs without errors but does not learn. It contains several bugs from
    the checklist in the Drill #35 entry. Same return format as fixed_train.
    """
    torch.manual_seed(seed)
    x_train, y_train, x_val, y_val = make_debug_data(seed)

    # standardise
    x_all = torch.cat([x_train, x_val])
    mean, std = x_all.mean(0), x_all.std(0)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    # shuffle the training set once
    perm = torch.randperm(x_train.shape[0])
    x_train = x_train[perm]

    model = nn.Sequential(
        nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 4), nn.Softmax(dim=0)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    val_loss = 0.0
    for step in range(n_steps):
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        scheduler.step()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            model.eval()
            val_logits = model(x_val)
            val_loss = val_loss + loss_fn(val_logits, y_val)

    model.eval()
    with torch.no_grad():
        pred = model(x_val).max(dim=0)[1]
    val_acc = (pred == y_val[: pred.shape[0]]).float().mean().item()
    return {"losses": losses, "val_acc": val_acc, "model": model}


def fixed_train(seed: int = 0, n_steps: int = 200) -> dict:
    """The fixed version of buggy_train. Same data, same model sizes, same step count.

    Drill #35. Checklist drill (no tier). Target: find every bug, then about 10 min.

    Full-batch training on make_debug_data(seed) for n_steps optimiser steps,
    seeded with torch.manual_seed(seed). Standardise with training-set
    statistics only. Keep the architecture 8 -> 64 -> 4 with ReLU and dropout,
    any reasonable optimiser and schedule.

    Returns a dict:
        "losses": list of n_steps Python floats, the training loss at each step before its update.
        "val_acc": Python float, argmax accuracy on the validation set in eval mode.
        "model": the trained nn.Module (mapping standardised x to logits).

    Hints:
        - What should losses[0] be for 4 balanced classes, and what does buggy_train give?
        - Walk the Drill #35 checklist line by line against buggy_train.
        - Which tensors must be permuted together?
    """
    raise NotImplementedError("Drill #35")
