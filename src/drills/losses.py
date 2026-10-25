"""Section 1: losses and metrics (Drills #1-#10).

Every function raises NotImplementedError until you fill it in.
Run the tests with:  python -m pytest tests/test_losses.py
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

__all__ = [
    "softmax",
    "log_softmax",
    "softmax_cross_entropy",
    "masked_cross_entropy",
    "bce_with_logits",
    "binary_focal_loss",
    "clip_loss",
    "censored_huber_loss",
    "pairwise_difference_loss",
    "affinity_loss",
    "kl_discrete",
    "gaussian_kl",
    "vae_loss",
    "precision_recall_f1",
    "rankdata_average",
    "roc_auc",
    "spearman",
    "enrichment_factor",
    "top_k_recall",
    "grouped_bootstrap_ci",
]


# ---------------------------------------------------------------------------
# Drill #1: stable softmax and log-softmax
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``.

    Drill #1 | Tier A | about 8 min | NumPy.

    Args:
        x: float array of any shape, e.g. (..., C). Must not be modified.
        axis: axis to normalise over (negative values allowed).

    Returns:
        Array of the same shape and dtype as ``x``; entries are >= 0 and sum
        to 1 along ``axis``. Must stay finite for inputs around +-1000 and for
        float16 inputs.

    Hints:
        - Which constant can you subtract from every logit without changing the
          result, and does it need ``keepdims``?
        - Does your function write into the caller's array?
    """
    logits = np.exp(x - x.max(axis=axis, keepdims=True))
    return logits / logits.sum(axis=axis, keepdims=True).clip(1e-12)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax along ``axis``.

    Drill #1 | Tier A | about 8 min | NumPy.

    Args:
        x: float array of any shape. Must not be modified.
        axis: axis to normalise over.

    Returns:
        Array of the same shape and dtype as ``x``, equal to x - logsumexp(x).
        Must be finite (no -inf) even when a probability underflows to 0.

    Hints:
        - Why is log(softmax(x)) a bad way to compute this?
        - Can you reuse the max-shift trick inside logsumexp?
    """
    logits = x - x.max(axis=axis, keepdims=True)
    return logits - np.log(np.sum(np.exp(logits), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Drill #2: softmax cross-entropy forward and backward
# ---------------------------------------------------------------------------


def softmax_cross_entropy(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean softmax cross-entropy and its gradient w.r.t. the logits.

    Drill #2 | Tier A | about 15 min | NumPy (no autograd).

    Args:
        logits: float array (N, C).
        targets: int array (N,), class indices in [0, C).

    Returns:
        (loss, dlogits) where ``loss`` is a Python float, the MEAN over the N
        rows, and ``dlogits`` is a float array (N, C), the gradient of that
        mean loss. Must be finite for logits around +-1000.

    Hints:
        - Derive d(loss)/d(logit_j) for one row first. Where does 1/N enter?
        - With random small logits, what value should the loss be close to?
    """
    log_probs = log_softmax(logits)

    n = targets.shape[0]
    loss = float(-log_probs[np.arange(n), targets].mean())

    dLogits = np.exp(log_probs)
    dLogits[np.arange(n), targets] -= 1
    dLogits = dLogits / n

    return (loss, dLogits)


# ---------------------------------------------------------------------------
# Drill #3: masked cross-entropy with ignore_index, smoothing, temperature
# ---------------------------------------------------------------------------


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy averaged over valid positions only.

    Drill #3 | Tier A | about 15 min | PyTorch (do not call F.cross_entropy).

    Args:
        logits: float tensor (B, T, C).
        targets: long tensor (B, T). Positions equal to ``ignore_index`` are
            invalid.
        mask: optional bool tensor (B, T); True means "valid". A position counts
            only if mask is True AND target != ignore_index.
        ignore_index: target id that marks padding.
        label_smoothing: eps in [0, 1). Smoothed target is
            (1 - eps) * onehot + eps / C. Applied to valid positions only.
        temperature: logits are divided by this before the softmax.

    Returns:
        Scalar tensor: sum of per-position losses over valid positions divided
        by the number of valid positions. If there are no valid positions,
        return 0.0 (not NaN). Must be differentiable w.r.t. ``logits``.

    Hints:
        - What do you divide by: T, B*T, or the count of valid positions?
        - What happens to gather() when a target equals ignore_index (-100)?
        - What does an all-False mask give you, and how do you avoid 0/0?
    """
    # Step 1: Get a valid mask (B, T)
    valid = targets != ignore_index
    if mask is not None:
        valid = valid & mask

    # Step 2: Get the log_probs (B, T, C)
    logits = (logits - logits.amax(dim=-1, keepdim=True)) / temperature
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # Step 3: Create the smoothed targets
    C = log_probs.shape[-1]
    targets_smoothed = torch.zeros_like(log_probs)
    targets_masked = targets.masked_fill(~valid, 0)
    targets_smoothed.scatter_(dim=-1, index=targets_masked.unsqueeze(-1), value=1.0)
    targets_smoothed = targets_smoothed * (1 - label_smoothing) + label_smoothing / C

    # Step 4: Compute the final loss (B, T) -> scalar
    loss = -(log_probs * targets_smoothed).sum(dim=-1)
    n_valid = valid.sum().clamp_min(1)
    return (loss * valid).sum() / n_valid



# ---------------------------------------------------------------------------
# Drill #4: stable BCE-with-logits and its gradient
# ---------------------------------------------------------------------------


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean binary cross-entropy from logits, and its gradient.

    Drill #4 | Tier A | about 10 min | NumPy.

    Args:
        z: float array (N,), logits (NOT probabilities).
        y: float array (N,), labels in {0, 1}.

    Returns:
        (loss, dz): ``loss`` is a Python float, the MEAN over N; ``dz`` is a
        float array (N,), the gradient of that mean. Must be finite and exact
        for z around +-1000.

    Hints:
        - Can you rewrite -[y log s(z) + (1-y) log(1-s(z))] so no exp ever
          overflows and no log ever sees 0?
        - What is d/dz of the per-sample loss? Does it need the stable form?
    """
    raise NotImplementedError("Drill #4")


# ---------------------------------------------------------------------------
# Drill #5: binary focal loss from logits
# ---------------------------------------------------------------------------


def binary_focal_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss built on top of a stable BCE-with-logits.

    Drill #5 | Tier A | about 10 min | PyTorch.

    Per element: loss = alpha_t * (1 - p_t) ** gamma * bce(z, y), where
    p_t is the model probability of the true class and
    alpha_t = alpha if y == 1 else (1 - alpha). If ``alpha`` is None, no
    class weighting (alpha_t = 1).

    Args:
        z: float tensor (N,), logits.
        y: float tensor (N,), labels in {0, 1}.
        alpha: weight on the positive class, or None.
        gamma: focusing parameter >= 0. gamma=0 and alpha=None is plain BCE.
        reduction: "none" -> (N,), "mean" -> scalar, "sum" -> scalar.

    Returns:
        Tensor as described by ``reduction``. Finite for z around +-1000.

    Hints:
        - Given the stable BCE value, how do you get p_t without calling
          sigmoid and risking log(0)?
        - Does your gamma=0, alpha=None case match torch's BCE exactly?
    """
    raise NotImplementedError("Drill #5")


# ---------------------------------------------------------------------------
# Drill #6: symmetric InfoNCE / CLIP loss
# ---------------------------------------------------------------------------


def clip_loss(
    img: torch.Tensor, txt: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """Symmetric contrastive (CLIP / InfoNCE) loss.

    Drill #6 | Tier A | about 25 min | PyTorch.

    Args:
        img: float tensor (B, D), rows already L2-normalised (do not
            renormalise).
        txt: float tensor (B, D), rows already L2-normalised. Row i of img
            matches row i of txt.
        log_scale: scalar tensor (may require grad). The logit scale is
            exp(log_scale).

    Returns:
        Scalar float32 tensor: the average of the image->text and text->image
        cross-entropies over the (B, B) logit matrix, each a mean over B rows.
        fp16 inputs must still give a float32 result.

    Hints:
        - Which shape is img @ txt? Will a test with B != D catch a missing .T?
        - Do you multiply or divide by the scale, and is it scale or exp(scale)?
        - Are you taking a softmax twice?
    """
    raise NotImplementedError("Drill #6")


# ---------------------------------------------------------------------------
# Drill #7: censored Huber + pairwise difference loss (affinity)
# ---------------------------------------------------------------------------
# Huber convention used throughout (same as torch.nn.functional.huber_loss):
#   h(r) = 0.5 * r**2                  if |r| < delta
#   h(r) = delta * (|r| - 0.5 * delta) otherwise
# The default constants below follow one reading of the Boltz-2 preprint,
# App. C.2.5; treat them as parameters, not facts.


def censored_huber_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    is_greater: torch.Tensor,
    delta: float = 0.5,
) -> torch.Tensor:
    """Absolute-value Huber loss with right-censored ('>') labels.

    Drill #7 | Tier A | about 25 min (whole drill) | PyTorch.

    Args:
        pred: float tensor (B,), predicted values.
        y: float tensor (B,), measured values.
        is_greater: bool tensor (B,). True means the label is "> y": the row
            only contributes when pred < y (a prediction above the bound is
            not penalised). False means an exact ('=') label.
        delta: Huber threshold.

    Returns:
        Scalar tensor: mean over all B rows (censored rows that are satisfied
        contribute 0 but still count in the denominator).

    Hints:
        - Can you express "only penalise when pred < y" by editing the residual
          instead of using Python if-statements?
        - Is the gradient zero for a satisfied '>' row?
    """
    raise NotImplementedError("Drill #7")


def pairwise_difference_loss(
    pred: torch.Tensor, y: torch.Tensor, delta: float = 0.5
) -> torch.Tensor:
    """Huber loss on all pairwise differences within one assay.

    Drill #7 | Tier A | PyTorch.

    For every ordered pair (i, j) with i != j, the residual is
    (pred_i - pred_j) - (y_i - y_j). Return the mean Huber loss over the
    B * (B - 1) off-diagonal pairs.

    Args:
        pred: float tensor (B,), B >= 2.
        y: float tensor (B,).
        delta: Huber threshold.

    Returns:
        Scalar tensor. Unchanged when a constant is added to all of ``pred``
        (or all of ``y``).

    Hints:
        - Can you build the (B, B) difference matrix with broadcasting?
        - How do you exclude the diagonal from both the sum and the count?
    """
    raise NotImplementedError("Drill #7")


def affinity_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    is_greater: torch.Tensor,
    binary_loss: Optional[torch.Tensor] = None,
    delta: float = 0.5,
    w_dif: float = 0.9,
    w_abs: float = 0.1,
) -> torch.Tensor:
    """Total affinity loss: w_dif * L_dif + w_abs * L_abs + L_binary.

    Drill #7 | Tier A | PyTorch.

    Args:
        pred, y, is_greater: as in ``censored_huber_loss`` (B compounds from
            one assay).
        binary_loss: optional scalar tensor (e.g. a focal loss from Drill #5);
            treated as 0 if None.
        delta, w_dif, w_abs: Huber threshold and mixing weights.

    Returns:
        Scalar tensor.

    Hints:
        - Reuse the two functions above; what should L_dif use as y for
          censored rows? (Here: the raw y; keep it simple.)
    """
    raise NotImplementedError("Drill #7")


# ---------------------------------------------------------------------------
# Drill #8: KL divergence and the VAE loss
# ---------------------------------------------------------------------------


def kl_discrete(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """KL(p || q) for discrete distributions along ``dim``.

    Drill #8 | Tier C | about 15 min | PyTorch.

    Args:
        p, q: float tensors of the same shape, non-negative, each summing to 1
            along ``dim``. q > 0 wherever p > 0.
        dim: event dimension.

    Returns:
        Tensor with ``dim`` reduced. Uses the convention 0 * log 0 = 0, so
        entries where p == 0 contribute exactly 0 (no NaN).

    Hints:
        - What does 0 * log(0) give in floating point, and how do you mask it?
    """
    raise NotImplementedError("Drill #8")


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Analytic KL( N(mu, diag(exp(logvar))) || N(0, I) ).

    Drill #8 | Tier C | PyTorch.

    Args:
        mu: float tensor (B, Z).
        logvar: float tensor (B, Z), log of the variance (not sigma).

    Returns:
        Float tensor (B,): KL summed over the Z latent dims, one per sample.

    Hints:
        - Write KL for one 1-D Gaussian first; which terms need exp(logvar)?
    """
    raise NotImplementedError("Drill #8")


def vae_loss(
    recon_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Negative ELBO for a Bernoulli decoder.

    Drill #8 | Tier C | PyTorch.

    loss = mean over batch of [ sum over features of BCE-with-logits(recon, x)
                                + gaussian_kl(mu, logvar) ].

    Args:
        recon_logits: float tensor (B, F), decoder logits.
        x: float tensor (B, F), targets in [0, 1].
        mu, logvar: float tensors (B, Z).

    Returns:
        Scalar tensor.

    Hints:
        - Sum over features and latents first, then average over the batch:
          does your reconstruction term use reduction="sum"/B or "mean"?
    """
    raise NotImplementedError("Drill #8")


# ---------------------------------------------------------------------------
# Drill #9: classification metrics
# ---------------------------------------------------------------------------


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    zero_division: float = 0.0,
) -> tuple[float, float, float]:
    """Precision, recall and F1.

    Drill #9 | Tier B | about 15 min (whole drill) | NumPy.

    Args:
        y_true: int array (N,), class labels.
        y_pred: int array (N,), predicted labels.
        average:
            "binary": positive class is 1.
            "macro": compute P, R, F1 per class (classes = union of labels in
                y_true and y_pred), then take the unweighted mean of each.
            "micro": pool TP/FP/FN over classes first.
        zero_division: value used whenever a ratio has denominator 0
            (also for F1 when P + R == 0).

    Returns:
        (precision, recall, f1) as Python floats.

    Hints:
        - In "macro", is F1 the mean of per-class F1s or F1 of mean P and R?
          (Here: mean of per-class F1s.)
        - What is micro-F1 equal to for single-label multi-class data?
    """
    raise NotImplementedError("Drill #9")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (no scipy).

    Drill #9 | Tier B | NumPy.

    Args:
        x: 1-D float array (N,).

    Returns:
        float64 array (N,). Example: [10, 20, 20, 30] -> [1, 2.5, 2.5, 4].

    Hints:
        - After argsort, how do you find the start and end of each tie block?
    """
    raise NotImplementedError("Drill #9")


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, ties counted as one half.

    Drill #9 | Tier B | NumPy.

    Args:
        y_true: array (N,) in {0, 1}, both classes present.
        scores: float array (N,), higher means more likely positive.

    Returns:
        P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg), a float.
        Should run in O(N log N), not O(N^2).

    Hints:
        - How does the sum of positive ranks relate to the Mann-Whitney U?
        - Do your ranks handle ties?
    """
    raise NotImplementedError("Drill #9")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties.

    Drill #9 | Tier B | NumPy.

    Args:
        x, y: 1-D float arrays (N,), N >= 2, neither constant.

    Returns:
        Pearson correlation of the average ranks, a float in [-1, 1].

    Hints:
        - Is the 1 - 6 sum(d^2) / (n(n^2-1)) shortcut still exact with ties?
    """
    raise NotImplementedError("Drill #9")


# ---------------------------------------------------------------------------
# Drill #10: screening metrics
# ---------------------------------------------------------------------------
# Tie convention for #10: when the cutoff falls inside a block of tied scores,
# count the EXPECTED number of actives you would select if ties were broken
# uniformly at random (fractional counts allowed).


def enrichment_factor(
    scores: np.ndarray, labels: np.ndarray, fraction: float
) -> float:
    """Enrichment factor at the top ``fraction`` of a ranked library.

    Drill #10 | Tier C | about 15 min (whole drill) | NumPy.

    n_sel = ceil(fraction * N) items are selected by descending score.
    EF = (actives_selected / n_sel) / (total_actives / N).

    Args:
        scores: float array (N,), higher = predicted more active.
        labels: array (N,) in {0, 1}, at least one active.
        fraction: in (0, 1].

    Returns:
        EF as a float (1.0 means no better than random).

    Hints:
        - If 3 items share the score at the cutoff and only 1 slot is left,
          how many actives do you expect to pick?
    """
    raise NotImplementedError("Drill #10")


def top_k_recall(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of all actives found in the top ``k`` by score.

    Drill #10 | Tier C | NumPy.

    Args:
        scores: float array (N,).
        labels: array (N,) in {0, 1}.
        k: number selected, 1 <= k <= N.

    Returns:
        expected actives in top k / total actives, a float; NaN if there are
        no actives. Same tie convention as ``enrichment_factor``.

    Hints:
        - Can you share the tie-aware "expected actives in top n" helper with
          enrichment_factor?
    """
    raise NotImplementedError("Drill #10")


def grouped_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI that resamples whole groups (targets/assays).

    Drill #10 | Tier C | NumPy.

    Each replicate draws G group ids with replacement (G = number of unique
    groups), concatenates all rows of the drawn groups (a group drawn twice
    appears twice), and evaluates ``metric(scores_rep, labels_rep)``.

    Args:
        scores: float array (N,).
        labels: array (N,).
        groups: array (N,) of group ids.
        metric: function (scores, labels) -> float.
        n_boot: number of replicates.
        ci: coverage, e.g. 0.95.
        seed: seed for np.random.default_rng.

    Returns:
        (lo, hi): the (1-ci)/2 and 1-(1-ci)/2 quantiles of the replicate
        values (np.quantile, default linear interpolation).

    Hints:
        - Why is resampling single rows wrong when rows within a target are
          correlated?
        - Are you drawing group ids or row indices?
    """
    raise NotImplementedError("Drill #10")
