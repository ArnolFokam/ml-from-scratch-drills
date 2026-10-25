"""Section 3: attention family and decoding (drills #20-#25).

Shape names: B batch, H heads, Lq query length, Lk key length, d head dim,
T sequence length, V vocabulary size.

Mask convention for this whole module: a boolean mask where True means
"keep / may attend" and False means "blocked".
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Drill #20  Scaled dot-product attention with padding and causal masks
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal: bool = False,
) -> Tensor:
    """Scaled dot-product attention in PyTorch.

    Drill #20 | Tier A | ~15 min | PyTorch

    Inputs:
        q: (B, H, Lq, d) float.
        k: (B, H, Lk, d) float.
        v: (B, H, Lk, dv) float.
        mask: optional bool tensor broadcastable to (B, H, Lq, Lk).
            True = keep, False = blocked. A padding mask over keys is
            usually given as (B, 1, 1, Lk).
        causal: if True, query i may attend key j only when
            j <= i + (Lk - Lq), i.e. tril(ones(Lq, Lk), diagonal=Lk - Lq).
            Combined with `mask` by logical AND.

    Output: (B, H, Lq, dv), same dtype as q.

    Requirements:
        - Scores are q @ k^T / sqrt(d); softmax over the key axis.
        - Masking happens before the softmax.
        - A query row with every key blocked must give finite values
          (no NaN). Its exact value is not tested.
        - Stable for large logits.

    Hints:
        - Which axis does the softmax run over, and why?
        - What does softmax do to a row that is all -inf?
        - Does your mask mean "keep" or "block" at the point you apply it?
    """
    # Step 1: Get the attention scores (B, H, Lq, Lk)
    scores = q @ k.transpose(-1, -2) / (k.shape[-1] ** 0.5)

    # Step 2: Get the valid mask (B, H, Lq, Lk) and mask scores
    Lq, Lk = scores.shape[-2:]
    valid_mask = torch.ones((Lq, Lk), device=scores.device, dtype=bool)

    if causal:
        # We are not assuming that when causal is set, Lk >= Lq always holds
        causal_mask = torch.tril(valid_mask, diagonal=Lk - Lq)
        valid_mask = valid_mask & causal_mask

    if mask is not None:
        valid_mask = valid_mask & mask

    scores = scores.masked_fill(~valid_mask, -torch.inf)

    # Step 3: Compute the attention weights (B, H, Lq, Lk)
    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = row_max.masked_fill(row_max == -torch.inf, 0)
    weights = (scores - row_max).exp()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Step 4: Compute the output (B, H, Lq, dv)
    return weights @ v


    



def scaled_dot_product_attention_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    causal: bool = False,
) -> np.ndarray:
    """Scaled dot-product attention in NumPy.

    Drill #20 | Tier A | ~15 min | NumPy

    Same contract as `scaled_dot_product_attention`, on float64 arrays:
    q (B, H, Lq, d), k (B, H, Lk, d), v (B, H, Lk, dv), optional bool mask
    broadcastable to (B, H, Lq, Lk) with True = keep, causal diagonal
    Lk - Lq. Returns (B, H, Lq, dv). Fully blocked rows must be finite.

    Hints:
        - NumPy has no softmax: how do you make exp() safe for logits of 1e4?
        - How do you transpose only the last two axes of a 4-D array?
    """
    # Step 1: Get the attention scores (B, H, Lq, Lk)
    scores = q @ k.swapaxes(-1, -2) / (k.shape[-1] ** 0.5)

    # Step 2: Compute a valid mask and mask the scores (B, H, Lq, dv)
    Lq, Lk = scores.shape[-2:]
    valid_mask = np.ones((Lq, Lk), dtype=bool)

    if causal:
        causal_mask = np.tril(valid_mask, k=Lk - Lq)
        valid_mask = valid_mask & causal_mask

    if mask is not None:
        valid_mask = valid_mask & mask

    scores = np.where(valid_mask, scores, -np.inf)

    # Step 3: Compute a numerically stable softmax (B, H, Lq, dv)
    row_max = scores.max(axis=-1, keepdims=True)
    row_max = np.where(row_max == -np.inf, 0, row_max)
    weights = np.exp(scores - row_max)
    weights = weights / weights.sum(axis=-1, keepdims=True).clip(min=1e-12)

    # Step 4: Compute the final output (B, H, Lq, dv)
    return weights @ v
    




# ---------------------------------------------------------------------------
# Drill #21  Multi-head attention module, self- and cross-attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V/O projections.

    Drill #21 | Tier A | ~20 min | PyTorch

    __init__(d_model, n_heads, bias=True):
        Store d_model, n_heads and d_head = d_model // n_heads (assert it
        divides). Create four nn.Linear(d_model, d_model, bias=bias) layers
        named exactly q_proj, k_proj, v_proj, o_proj (the tests copy their
        weights into a reference).

    forward(x, context=None, key_padding_mask=None, causal=False):
        x: (B, Lq, d_model) queries.
        context: (B, Lk, d_model) keys/values source; None means
            self-attention (context = x). Lq may differ from Lk.
        key_padding_mask: optional bool (B, Lk), True = real token (keep),
            False = padding.
        causal: same rule as drill #20 (diagonal Lk - Lq).
        Returns (B, Lq, d_model). No dropout.

    Hints:
        - Split heads as (B, L, H, d_head) then move H forward: why view
          before transpose and not the other way round?
        - After merging heads back, is the tensor still contiguous?
        - How does a (B, Lk) mask need to be reshaped to broadcast against
          (B, H, Lq, Lk)?
    """

    def __init__(self, d_model: int, n_heads: int, bias: bool = True) -> None:
        super().__init__()

        assert d_model % n_heads == 0, f"d_model: {d_model} must be divisible by n_heads: {n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.bias = bias

        self.d_head = d_model // n_heads

        # Initialize the layers
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        causal: bool = False,
    ) -> Tensor:

        # Step 1: Project the inputs (B, Lq, d_model)
        q = self.q_proj(x)

        if context is not None:
            k = self.k_proj(context)
            v = self.v_proj(context)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Step 2: reshape the QKV values (B, H, L{q/k/v}, d_head)
        q = q.reshape(
            *q.shape[:-1],
            self.n_heads,
            self.d_head
        ).transpose(-2, -3)
        k = k.reshape(
            *k.shape[:-1],
            self.n_heads,
            self.d_head
        ).transpose(-2, -3)
        v = v.reshape(
            *v.shape[:-1],
            self.n_heads,
            self.d_head
        ).transpose(-2, -3)

        # Step 3: Carry out attention (B, H, Lv, d_head)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
        out = scaled_dot_product_attention(q, k, v, mask=key_padding_mask, causal=causal)

        # Step 4: Reshape back to the original shape
        out = out.transpose(-2, -3).reshape(
            *x.shape[:2],
            self.d_model
        )

        return self.o_proj(out)



# ---------------------------------------------------------------------------
# Drill #22  KV cache, then GQA/MQA on the cache
# ---------------------------------------------------------------------------


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand grouped key/value heads to match the number of query heads.

    Drill #22 | Tier C | part of ~20 min | PyTorch

    Input: x (B, G, L, d) with G key/value heads.
    Output: (B, G * n_rep, L, d) where output head h is input head
    h // n_rep (heads of the same group are adjacent).

    Hints:
        - Should query heads 0 and 1 share a KV head, or heads 0 and G?
        - Which dim do you repeat along?
    """
    raise NotImplementedError("Drill #22")


def kv_cache_bytes(
    n_layers: int,
    seq_len: int,
    n_kv_heads: int,
    d_head: int,
    batch: int,
    bytes_per_elem: int = 2,
) -> int:
    """Memory taken by a KV cache, in bytes.

    Drill #22 | Tier C | ~2 min | Python

    Counts both K and V for every layer. Returns a Python int.

    Hints:
        - What is the factor in front that people forget?
    """
    raise NotImplementedError("Drill #22")


class CachedCausalSelfAttention(nn.Module):
    """Causal self-attention with a KV cache and grouped KV heads (GQA/MQA).

    Drill #22 | Tier C | ~20 min | PyTorch

    __init__(d_model, n_heads, n_kv_heads=None, bias=False):
        n_kv_heads = None means n_heads (plain MHA); 1 means MQA.
        Assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0.
        d_head = d_model // n_heads. Create layers named exactly:
            q_proj: nn.Linear(d_model, d_model)
            k_proj: nn.Linear(d_model, n_kv_heads * d_head)
            v_proj: nn.Linear(d_model, n_kv_heads * d_head)
            o_proj: nn.Linear(d_model, d_model)
        all with bias=bias. Query head h uses KV head h // (n_heads // n_kv_heads).

    forward(x, cache=None) -> (out, new_cache):
        x: (B, T, d_model), the next T tokens (T = prompt length for prefill,
            T = 1 for a decode step, any T >= 1 in general).
        cache: None, or dict {"k": (B, n_kv_heads, L_past, d_head),
            "v": (B, n_kv_heads, L_past, d_head)} from the previous call.
        out: (B, T, d_model). Token t of x sits at absolute position
            L_past + t and attends to all positions <= its own.
        new_cache: dict with keys "k" and "v" of shape
            (B, n_kv_heads, L_past + T, d_head). Store the un-repeated heads.

    Running forward on the full sequence once must equal prefill followed
    by decode steps. No dropout, no positional encoding.

    Hints:
        - With L_past cached keys and T new queries, which diagonal does the
          causal mask need?
        - Why cache n_kv_heads heads rather than n_heads?
        - If you added RoPE, would you cache keys before or after rotating?
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.bias = bias
        raise NotImplementedError("Drill #22")

    def forward(
        self, x: Tensor, cache: Optional[dict[str, Tensor]] = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError("Drill #22")


# ---------------------------------------------------------------------------
# Drill #23  FlashAttention-style tiled attention (online softmax)
# ---------------------------------------------------------------------------


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_size: int = 16,
    causal: bool = False,
) -> Tensor:
    """Exact attention computed block by block over K/V with an online softmax.

    Drill #23 | Tier B | ~25 min | PyTorch

    Inputs: q (B, H, Lq, d), k (B, H, Lk, d), v (B, H, Lk, dv).
    block_size: number of keys per block; Lk need not be a multiple of it.
    causal: same rule as drill #20 (diagonal Lk - Lq). With causal=True you
        may assume Lq <= Lk, so every query row keeps at least one key.
    Output: (B, H, Lq, dv), equal to softmax(q k^T / sqrt(d)) v to ~1e-6.

    Rules: never build the full (Lq, Lk) score matrix. Loop over K/V blocks,
    keeping per query row a running max m, a running sum l and an
    unnormalised accumulator o. Divide by l once at the end.

    Hints:
        - When the max grows from m_old to m_new, what factor fixes the old
          l and o?
        - What are m, l and o before the first block?
        - A block can be fully masked for some rows: does exp(-inf - -inf)
          appear anywhere in your code?
    """
    raise NotImplementedError("Drill #23")


# ---------------------------------------------------------------------------
# Drill #24  Decoding: temperature, greedy, top-k, top-p, beam search
# ---------------------------------------------------------------------------


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits on the last axis, set the rest to -inf.

    Drill #24 | Tier C | part of ~15 min | NumPy

    Input: logits (..., V) float, k >= 1. Output: same shape, new array
    (do not modify the input). If k >= V nothing is removed. Ties at the
    k-th value are not tested.

    Hints:
        - What should happen when k > V?
        - Can you find the k-th largest value without a full sort?
    """
    raise NotImplementedError("Drill #24")


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter on the last axis: keep the smallest top set with mass >= p.

    Drill #24 | Tier C | part of ~15 min | NumPy

    Input: logits (..., V) float, 0 < p <= 1. Sort tokens by probability
    (softmax of logits), descending. Keep a token if the cumulative
    probability of the tokens strictly before it is < p. The most likely
    token is always kept. Others become -inf. Output has the input's shape
    and token order (do not modify the input).

    Hints:
        - Why can "keep while cumsum <= p" keep nothing at all?
        - After sorting, how do you put the keep-flags back in vocab order?
    """
    raise NotImplementedError("Drill #24")


def sample_next_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> np.ndarray:
    """Sample one token id per row of logits.

    Drill #24 | Tier C | ~15 min | NumPy

    Input: logits (..., V) float; rng a seeded numpy Generator.
    Order: divide by temperature first, then top-k, then top-p, then
    softmax and sample. temperature == 0 means greedy (argmax, rng unused).
    Output: int array of shape logits.shape[:-1].

    Hints:
        - Does temperature change which tokens survive top-p? Then where
          must it be applied?
        - How do you draw from many categorical rows at once with one
          uniform sample per row?
    """
    raise NotImplementedError("Drill #24")


def beam_search(
    step_fn: Callable[[list[int]], np.ndarray],
    eos: int,
    beam_width: int,
    max_len: int,
    length_penalty: float = 0.0,
) -> tuple[list[int], float]:
    """Beam search over a toy next-token model.

    Drill #24 | Tier C | ~15 min | Python + NumPy

    step_fn(prefix) takes the list of tokens generated so far (starts as [])
    and returns log-probabilities over the vocabulary, shape (V,).

    Algorithm to implement:
        - Start with one empty hypothesis of log-prob 0.
        - Each step: extend every live hypothesis by every token, rank all
          candidates by summed log-prob, keep the best beam_width. Those
          ending in eos move to a finished pool; the rest stay live.
        - Stop when no live hypothesis remains or after max_len steps.
        - Score every finished and remaining live hypothesis as
          sum_logprob / len(tokens) ** length_penalty, where len counts
          generated tokens including eos.
    Returns (tokens, score) of the best one. tokens includes eos if it ended.

    Hints:
        - What does length_penalty = 0 favour, and why?
        - Should a finished hypothesis keep being extended?
    """
    raise NotImplementedError("Drill #24")


# ---------------------------------------------------------------------------
# Drill #25  Sliding-window attention and the attention-free transformer
# ---------------------------------------------------------------------------


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window: int,
    causal: bool = True,
) -> Tensor:
    """Local self-attention: each query sees only nearby keys.

    Drill #25 | Tier C | ~8 min | PyTorch

    Inputs: q, k, v of shape (B, H, L, d) (self-attention, same L).
    window >= 1.
        causal=True: query i attends key j iff i - window < j <= i
            (itself plus the window - 1 previous tokens).
        causal=False: query i attends key j iff |i - j| < window.
    Output: (B, H, L, d), scaled by 1/sqrt(d) like drill #20.

    Hints:
        - What is the output when window == 1?
        - Can you reuse your drill #20 function with the right mask?
    """
    raise NotImplementedError("Drill #25")


def aft_full(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    causal: bool = False,
) -> Tensor:
    """Attention-free transformer, AFT-full (Zhai et al., 2021).

    Drill #25 | Tier C | ~10 min | PyTorch

    Inputs: q, k, v (B, T, d) (already projected); w (T, T) learned
    pairwise position bias.
    Output (B, T, d):
        y[t] = sigmoid(q[t]) * sum_s exp(k[s] + w[t, s]) * v[s]
                             / sum_s exp(k[s] + w[t, s])
    with products element-wise over the d channels. causal=True restricts
    the sums to s <= t. Must stay finite when k has entries around 1e3.

    Hints:
        - For a fixed t and channel, the weights over s are a softmax: over
          which axis?
        - What happens to exp(k + w) when k = 1000?
    """
    raise NotImplementedError("Drill #25")


def aft_simple(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """AFT-simple: AFT-full with w = 0, computed in O(T d).

    Drill #25 | Tier C | ~5 min | PyTorch

    Inputs: q, k, v (B, T, d). Output (B, T, d):
        y[t] = sigmoid(q[t]) * sum_s softmax_s(k)[s] * v[s]
    The context vector is the same for every t. Must stay finite for large k.

    Hints:
        - Why does this cost O(T d) rather than O(T^2 d)?
    """
    raise NotImplementedError("Drill #25")
