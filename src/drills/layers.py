"""Section 2: core layers and architecture (Drills #11-#19).

Every function, method and class here raises NotImplementedError until you
fill it in. Classes that subclass torch.nn.Module raise in both __init__ and
forward: create the parameters yourself, with the attribute names given in
each docstring (the tests look them up by name).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Drill #11: two-layer MLP with a manual backward pass
# ---------------------------------------------------------------------------


def mlp_forward_backward(
    X: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    """Forward and manual backward pass of Affine -> ReLU -> Affine -> softmax-CE.

    Drill #11. Tier A. Target: about 20 min. NumPy only (no autograd).

    Args:
        X: (N, D) float array.
        y: (N,) int array of class indices in [0, C).
        W1: (D, H), b1: (H,), W2: (H, C), b2: (C,) float arrays.

    Returns:
        loss: Python float, the MEAN cross-entropy over the N rows.
        grads: dict with keys "W1", "b1", "W2", "b2"; each array has the same
            shape as the matching parameter and is d(loss)/d(param).

    Requirements:
        - Numerically stable softmax (logits of size 1e3 must not overflow).
        - Fully vectorised, no Python loop over rows.

    Hints:
        - Which tensor does the ReLU mask come from: the input or the output of the ReLU?
        - Where does the 1/N of the mean enter the backward pass?
        - Which axis do you sum over to get db?
    """
    raise NotImplementedError("Drill #11")


# ---------------------------------------------------------------------------
# Drill #12: LayerNorm and RMSNorm
# ---------------------------------------------------------------------------


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm over the last dimension. Must match torch.nn.functional.layer_norm.

    Drill #12. Tier A. Target: about 10 min (do it inside #17). PyTorch,
    without F.layer_norm or nn.LayerNorm.

    Args:
        x: (..., D) float tensor.
        weight: (D,) scale (gamma) or None for no scale.
        bias: (D,) shift (beta) or None for no shift.
        eps: added to the variance, inside the square root.

    Returns:
        Tensor with the shape and dtype of x. Differentiable.

    Hints:
        - Biased or unbiased variance?
        - Over which dimensions do you take the mean?
    """
    raise NotImplementedError("Drill #12")


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm over the last dimension. Must match torch.nn.functional.rms_norm.

    Drill #12. Tier A. PyTorch, without F.rms_norm or nn.RMSNorm.

    out = x / sqrt(mean(x^2 over last dim) + eps) * weight. No mean subtraction, no bias.

    Args:
        x: (..., D) float tensor.
        weight: (D,) scale or None.
        eps: added inside the square root.

    Returns:
        Tensor with the shape and dtype of x.

    Hints:
        - What does RMSNorm skip compared with LayerNorm?
    """
    raise NotImplementedError("Drill #12")


class LayerNorm(nn.Module):
    """LayerNorm module with an affine scale and shift.

    Drill #12. Tier A.

    Attributes to create:
        weight: nn.Parameter of shape (dim,), initialised to ones.
        bias: nn.Parameter of shape (dim,), initialised to zeros.
        eps: float.

    forward(x: (..., dim)) -> same shape; must match nn.LayerNorm(dim, eps).

    Hints:
        - Can forward just call your layer_norm function?
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        raise NotImplementedError("Drill #12")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #12")


class RMSNorm(nn.Module):
    """RMSNorm module with a learnable scale.

    Drill #12. Tier A.

    Attributes to create:
        weight: nn.Parameter of shape (dim,), initialised to ones.
        eps: float.

    forward(x: (..., dim)) -> same shape; must match nn.RMSNorm(dim, eps=eps).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        raise NotImplementedError("Drill #12")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #12")


# ---------------------------------------------------------------------------
# Drill #13: BatchNorm in train and eval, with running statistics
# ---------------------------------------------------------------------------


class BatchNorm(nn.Module):
    """BatchNorm for (N, C) or (N, C, H, W) input. Must match nn.BatchNorm1d / nn.BatchNorm2d.

    Drill #13. Tier B. Target: about 15 min. PyTorch tensors, without
    F.batch_norm or nn.BatchNorm*.

    Attributes to create:
        weight: nn.Parameter (C,), ones.     bias: nn.Parameter (C,), zeros.
        running_mean: buffer (C,), zeros.    running_var: buffer (C,), ones.
        (use self.register_buffer so they are not parameters)

    forward(x):
        - train mode (self.training): normalise with the batch mean and the
          BIASED batch variance, then update
          running = (1 - momentum) * running + momentum * batch_stat,
          using the UNBIASED batch variance for running_var.
        - eval mode: normalise with running_mean and running_var; no update.
        - Statistics are per channel C. Gradients must flow through the
          batch statistics in train mode.
        - Any other input rank: raise ValueError.

    Hints:
        - For a 4-D input, which axes do you reduce over?
        - Should the running-stat update be tracked by autograd?
        - What is n in the n / (n - 1) correction for a 4-D input?
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        raise NotImplementedError("Drill #13")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #13")


# ---------------------------------------------------------------------------
# Drill #14: inverted dropout and row-shared pair dropout
# ---------------------------------------------------------------------------


def dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Inverted dropout.

    Drill #14. Tier B. Target: about 5 min. PyTorch, without F.dropout.

    Args:
        x: any-shape float tensor.
        p: drop probability in [0, 1]; otherwise raise ValueError.
        training: if False, return x unchanged.

    Returns:
        Tensor like x: each element is zeroed with probability p and the
        survivors are scaled by 1 / (1 - p), so E[out] = x.
        p = 0 returns x unchanged; p = 1 returns all zeros (no division by zero).
        Use torch's global RNG (torch.rand_like).

    Hints:
        - Why scale at train time rather than at eval time?
        - What happens to 1 / (1 - p) at p = 1?
    """
    raise NotImplementedError("Drill #14")


def row_dropout_mask(
    z: torch.Tensor,
    p: float,
    training: bool = True,
    columnwise: bool = False,
) -> torch.Tensor:
    """Dropout mask shared along one axis of a pair tensor, as in the Boltz-1 code
    (src/boltz/model/layers/dropout.py::get_dropout_mask).

    Drill #14. Tier B. PyTorch.

    Args:
        z: (B, N, N, C) pair tensor (only its shape, dtype and device are used).
        p: drop probability in [0, 1]; otherwise raise ValueError.
        training: if False, the mask is all ones.
        columnwise: False -> one draw per row i, mask shape (B, N, 1, 1)
            (the shape of z[:, :, 0:1, 0:1]); True -> one draw per column j,
            mask shape (B, 1, N, 1).

    Returns:
        Mask tensor with dtype of z and values in {0, 1 / (1 - p)}, meant to be
        multiplied onto a residual update: z = z + mask * update(z).
        p = 1 gives all zeros.

    Hints:
        - Which slice of z has exactly the mask shape you need?
        - Why might sharing one draw along a row help for pair updates?
    """
    raise NotImplementedError("Drill #14")


# ---------------------------------------------------------------------------
# Drill #15: SwiGLU transition, plus a version chunked over the hidden dim
# ---------------------------------------------------------------------------


class Transition(nn.Module):
    """SwiGLU transition: out = fc3(silu(fc1(LN(x))) * fc2(LN(x))), as in the
    Boltz-1 code (src/boltz/model/layers/transition.py::Transition).

    Drill #15. Tier B. Target: about 12 min. PyTorch.

    Attributes to create:
        norm: nn.LayerNorm(dim) (default eps 1e-5, affine).
        fc1: nn.Linear(dim, hidden, bias=False)   (gate branch, goes through silu)
        fc2: nn.Linear(dim, hidden, bias=False)   (linear branch)
        fc3: nn.Linear(hidden, dim, bias=False), weight initialised to ZERO.
        hidden defaults to 4 * dim.

    forward(x: (..., dim), chunk_size: Optional[int]) -> (..., dim):
        chunk_size None: the plain computation above.
        chunk_size k: split the hidden dimension into slices of at most k units,
        compute each slice's contribution to fc3 and sum them. The result must
        be allclose to the unchunked one for any k >= 1, including k that does
        not divide hidden and k > hidden.

    Hints:
        - Which rows of fc1/fc2 and which columns of fc3 belong to one hidden slice?
        - With fc3 at zero, what does a freshly built block add to the residual stream?
    """

    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = 4 * dim if hidden is None else hidden
        raise NotImplementedError("Drill #15")

    def forward(self, x: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError("Drill #15")


# ---------------------------------------------------------------------------
# Drill #16: sinusoidal positional encoding and RoPE
# ---------------------------------------------------------------------------


def sinusoidal_positional_encoding(
    n_pos: int, d_model: int, base: float = 10000.0
) -> np.ndarray:
    """Sinusoidal positional encoding (Vaswani et al.), NumPy.

    Drill #16. Tier B. Target: about 10 min (with RoPE).

    PE[p, 2i]   = sin(p / base^(2i / d_model))
    PE[p, 2i+1] = cos(p / base^(2i / d_model))

    Args:
        n_pos: number of positions (0 .. n_pos - 1).
        d_model: encoding width; may be ODD (then the last column is a sin).

    Returns:
        (n_pos, d_model) float64 array.

    Hints:
        - With odd d_model, how many sin columns and how many cos columns are there?
        - Does column 2i+1 use frequency index i or i+1?
    """
    raise NotImplementedError("Drill #16")


def rope(
    x: torch.Tensor,
    offset: int = 0,
    base: float = 10000.0,
    interleaved: bool = False,
) -> torch.Tensor:
    """Rotary position embedding, PyTorch.

    Drill #16. Tier B.

    Args:
        x: (..., T, D) float tensor, D even (odd D: raise ValueError).
            Position of row t is offset + t (offset > 0 is the KV-cache case).
        base: frequency base; pair i uses theta_i = base^(-2i / D), i = 0 .. D/2 - 1,
            and is rotated by angle pos * theta_i.
        interleaved: False -> half-split (GPT-NeoX) pairing: (x[i], x[i + D/2]).
                     True  -> interleaved (original RoFormer) pairing: (x[2i], x[2i+1]).
            Rotation of a pair (a, b) by angle t: (a cos t - b sin t, a sin t + b cos t).

    Returns:
        Tensor with the shape and dtype of x.

    Hints:
        - Should <rope(q) at m, rope(k) at n> depend on m and n, or only on m - n?
        - When decoding token 7 alone with a cache, which position does it get?
    """
    raise NotImplementedError("Drill #16")


# ---------------------------------------------------------------------------
# Drill #17: full pre-norm transformer block and decoder, with a parameter count
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. Do not use F.scaled_dot_product_attention
    or nn.MultiheadAttention.

    Drill #17. Tier A. Target: about 35 min for all of #17, including #12.

    Attributes to create (all nn.Linear(d_model, d_model) WITH bias):
        q_proj, k_proj, v_proj, out_proj.
    d_model must be divisible by n_heads (else raise ValueError);
    head_dim = d_model // n_heads; scores are scaled by 1 / sqrt(head_dim).

    forward(x: (B, T, d_model)) -> (B, T, d_model). Position t attends only to
    positions <= t.

    Hints:
        - After transpose(1, 2), can you still call .view, or do you need .reshape?
        - Which entries of the (T, T) score matrix are set to -inf?
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        raise NotImplementedError("Drill #17")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #17")


class MLP(nn.Module):
    """Position-wise MLP: fc2(gelu(fc1(x))), exact (erf) GELU.

    Drill #17. Tier A.

    Attributes to create:
        fc1: nn.Linear(d_model, mult * d_model) with bias.
        fc2: nn.Linear(mult * d_model, d_model) with bias.
    """

    def __init__(self, d_model: int, mult: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.mult = mult
        raise NotImplementedError("Drill #17")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #17")


class TransformerBlock(nn.Module):
    """Pre-norm transformer block:
        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))

    Drill #17. Tier A.

    Attributes to create:
        ln1, ln2: affine LayerNorm(d_model) with eps 1e-5 (your #12 LayerNorm
            or nn.LayerNorm; parameters named weight and bias).
        attn: CausalSelfAttention(d_model, n_heads).
        mlp: MLP(d_model, 4).
    So the parameter names are ln1.weight, ln1.bias, attn.q_proj.weight, ...,
    mlp.fc2.bias.

    forward(x: (B, T, d_model)) -> (B, T, d_model).

    Hints:
        - Does the residual go around the norm or through it?
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        raise NotImplementedError("Drill #17")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #17")


class Decoder(nn.Module):
    """Decoder-only language model: ids (B, T) -> logits (B, T, vocab_size).

    Drill #17. Tier A.

    Attributes to create:
        tok_emb: nn.Embedding(vocab_size, d_model)
        pos_emb: nn.Embedding(max_len, d_model)   (learned absolute positions 0..T-1)
        blocks: nn.ModuleList of n_layers TransformerBlock(d_model, n_heads)
        ln_f: final affine LayerNorm(d_model)
        lm_head: nn.Linear(d_model, vocab_size, bias=False)   (not tied)

    forward(ids: (B, T) int64) -> (B, T, vocab_size) float logits.
    T > max_len: raise ValueError.

    Hints:
        - What shape does pos_emb(torch.arange(T)) have, and how does it broadcast?
        - How would you test that logits at position t ignore tokens after t?
    """

    def __init__(
        self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        raise NotImplementedError("Drill #17")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Drill #17")


def count_params_per_block(d_model: int, n_heads: int) -> int:
    """Closed-form parameter count of one TransformerBlock (as specified above).

    Drill #17. Tier A. Pure arithmetic: do not build the module.

    Counts: 2 affine LayerNorms, 4 biased d x d attention projections, and the
    biased 4d MLP. Excludes embeddings and the LM head.
    Example: count_params_per_block(8, 2) == 872.

    Hints:
        - Why does n_heads not appear in the answer?
    """
    raise NotImplementedError("Drill #17")


# ---------------------------------------------------------------------------
# Drill #18: minimal reverse-mode autograd (micrograd-style)
# ---------------------------------------------------------------------------


class Value:
    """A node in a reverse-mode autograd graph, holding a NumPy array.

    Drill #18. Tier C. Target: about 30 min. Plain Python + NumPy, no torch.

    Attributes:
        data: np.ndarray, float64 (a 0-d array for scalars).
        grad: np.ndarray of the same shape as data, zeros until backward().

    Operations (each returns a new Value; Python numbers and arrays on either
    side are wrapped automatically):
        a + b, a * b, a - b, -a   elementwise, with NumPy broadcasting
        a @ b                     2-D matrix product
        a.relu()                  elementwise max(a, 0); gradient 0 at a == 0
        a.sum()                   sum of all elements -> 0-d Value
    backward(): call on a 0-d Value; seeds its grad with 1 and fills .grad of
    every node it depends on, in reverse topological order.

    Rules:
        - Gradients ACCUMULATE: a node used twice receives both contributions
          (a * a + a at a = 3 gives a.grad == 7).
        - When an operand was broadcast, its gradient is summed back to its
          own shape.

    Hints:
        - Why must the grad update be += and not =?
        - Why can't you call each child's backward recursively in any order?
        - A bias of shape (H,) added to (N, H): which axis do you sum over?
    """

    def __init__(self, data, _children: tuple = (), _op: str = "") -> None:
        raise NotImplementedError("Drill #18")

    def __add__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def __radd__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def __mul__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def __rmul__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def __neg__(self) -> "Value":
        raise NotImplementedError("Drill #18")

    def __sub__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def __rsub__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def __matmul__(self, other) -> "Value":
        raise NotImplementedError("Drill #18")

    def relu(self) -> "Value":
        raise NotImplementedError("Drill #18")

    def sum(self) -> "Value":
        raise NotImplementedError("Drill #18")

    def backward(self) -> None:
        raise NotImplementedError("Drill #18")


# ---------------------------------------------------------------------------
# Drill #19: Conv2d forward, loops first, then im2col
# ---------------------------------------------------------------------------


def conv2d_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Output length of a convolution along one spatial axis.

    Drill #19. Tier C.

    Returns floor((size + 2 * padding - kernel) / stride) + 1.

    Hints:
        - Check it on size 5, kernel 3, stride 2, padding 1.
    """
    raise NotImplementedError("Drill #19")


def conv2d_loops(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """2-D convolution forward with explicit loops. Must match torch.nn.functional.conv2d.

    Drill #19. Tier C. Target: about 20 min (with im2col). NumPy.

    Args:
        x: (N, C_in, H, W) float array (NCHW).
        w: (C_out, C_in, KH, KW) float array; KH may differ from KW.
        b: (C_out,) or None.
        stride, padding: ints applied to both spatial axes; zero padding.

    Returns:
        (N, C_out, H_out, W_out) array.

    Hints:
        - Do you flip the kernel? (What does a deep-learning "convolution" compute?)
        - Which axis of w matches which axis of x?
    """
    raise NotImplementedError("Drill #19")


def conv2d_im2col(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """2-D convolution forward via im2col and one matrix multiply.

    Drill #19. Tier C. NumPy. Same contract as conv2d_loops.

    No Python loop over N, C_out, H_out or W_out (a loop over the KH x KW kernel
    offsets is fine).

    Hints:
        - What shape is the column matrix, and in what order must its rows be
          to line up with w.reshape(C_out, -1)?
    """
    raise NotImplementedError("Drill #19")
