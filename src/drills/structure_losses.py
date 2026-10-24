"""Section 5: structure-prediction losses and metrics (Drills #36-#41, #45).

All drills here are PyTorch. Coordinates are in Angstrom, shape (..., N, 3),
one point per row (row-vector convention). Masks are 1/True for a real
(resolved, non-padding) atom or token and 0/False for padding; accept bool or
float masks. Every function raises NotImplementedError until you fill it in.

Repo paths refer to the public Boltz code (github.com/jwohlwend/boltz) and
algorithm numbers to the AlphaFold3 supplementary information.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Drill #36: distogram target and masked distogram cross-entropy
# ---------------------------------------------------------------------------


def distogram_target(
    coords: Tensor,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
) -> Tensor:
    """One-hot distogram target from coordinates.

    Drill #36. Tier A. Target: about 20 min together with distogram_loss.

    Args:
        coords: (B, L, 3) float, one representative atom per token.
        min_dist, max_dist: first and last bin boundary in Angstrom.
        num_bins: number of bins K.

    Returns:
        (B, L, L, K) float32 one-hot. With boundaries = linspace(min_dist,
        max_dist, K - 1), the bin of distance d is the number of boundaries
        strictly smaller than d. So d <= 2 is bin 0 and d > 22 is bin K - 1.

    As in the Boltz featurizer (src/boltz/data/feature/featurizer.py).

    Hints:
        - How many boundaries do K bins need?
        - Is a distance exactly equal to a boundary in the lower or the upper bin?
    """
    raise NotImplementedError("Drill #36")


def distogram_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Masked distogram cross-entropy with soft or one-hot targets.

    Drill #36. Tier A.

    Args:
        logits: (B, L, L, K) float, unnormalised.
        target: (B, L, L, K) float, each pair's row sums to 1 (soft or one-hot).
        mask: (B, L) token mask.

    Returns:
        Scalar. Per pair: err_ij = -sum_k target * log_softmax(logits).
        Pair mask = m_i * m_j * (1 - eye). Per example: sum of err * pair_mask
        over i and j, divided by (1e-5 + pair_mask.sum()). Then the mean over
        the batch. Compute in float32 and stay finite for logits around 1e4.

    As in src/boltz/model/loss/distogramv2.py::distogram_loss.

    Hints:
        - Is the denominator the count of one row, or of the whole example?
        - What happens to exp(logits) when logits are 1e4?
        - Should a token's distance to itself count?
    """
    raise NotImplementedError("Drill #36")


# ---------------------------------------------------------------------------
# Drill #37: weighted Kabsch alignment, aligned weighted MSE and RMSD
# ---------------------------------------------------------------------------


def weighted_rigid_align(
    true_coords: Tensor,
    pred_coords: Tensor,
    weights: Tensor,
    mask: Tensor,
) -> Tensor:
    """Rigidly move true_coords onto pred_coords with weighted Kabsch.

    Drill #37. Tier A. Target: about 35 min for the whole drill.

    Args:
        true_coords: (B, N, 3) ground truth.
        pred_coords: (B, N, 3) prediction (the frame to align into).
        weights: (B, N) non-negative per-atom weights.
        mask: (B, N) atom mask. The effective weight is weights * mask.

    Returns:
        (B, N, 3), true_coords after the best proper rotation and translation
        (det R = +1, never a reflection), in the input dtype, DETACHED from the
        graph. All N atoms are moved, masked ones included.

    Recipe: weighted centroids, covariance of the centred sets, SVD in
    float32, fix the sign of the last singular direction with det(U V^T).
    As in src/boltz/model/loss/diffusionv2.py::weighted_rigid_align.

    Hints:
        - Which matrix's determinant tells you the SVD solution is a reflection?
        - With row vectors, do you apply R or R^T on the right?
        - Why is the output detached when it is used as a loss target?
    """
    raise NotImplementedError("Drill #37")


def aligned_weighted_mse(
    pred_coords: Tensor,
    true_coords: Tensor,
    weights: Tensor,
    mask: Tensor,
) -> Tensor:
    """Weighted MSE between pred and the ground truth aligned onto pred.

    Drill #37. Tier A.

    Args:
        pred_coords, true_coords: (B, N, 3).
        weights, mask: (B, N).

    Returns:
        (B,) per structure:
        sum_n w_n m_n ||pred_n - aligned_n||^2 / (3 * sum_n w_n m_n + 1e-5),
        where aligned = weighted_rigid_align(true, pred, weights, mask).
        Gradients flow into pred_coords only.

    Hints:
        - Where does the factor 3 come from?
        - Which argument receives the gradient, given that the target is detached?
    """
    raise NotImplementedError("Drill #37")


def weighted_rmsd(
    pred_coords: Tensor,
    true_coords: Tensor,
    weights: Tensor,
    mask: Tensor,
) -> Tensor:
    """Weighted RMSD after optimal rigid alignment.

    Drill #37. Tier A.

    Args:
        pred_coords, true_coords: (B, N, 3).
        weights, mask: (B, N).

    Returns:
        (B,) sqrt(sum_n w_n m_n ||pred_n - aligned_n||^2 / (sum_n w_n m_n + 1e-5)).
        No factor 3 here.

    Hints:
        - Is RMSD of a rotated, shifted copy of the same structure exactly 0?
        - What should a mirror image give?
    """
    raise NotImplementedError("Drill #37")


def weighted_minimum_rmsd(
    pred_samples: Tensor,
    true_coords: Tensor,
    weights: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Best weighted RMSD over several predicted samples.

    Drill #37. Tier A.

    Args:
        pred_samples: (B, S, N, 3), S samples per structure.
        true_coords: (B, N, 3).
        weights, mask: (B, N).

    Returns:
        (rmsd, best): rmsd is (B,) the minimum over S of weighted_rmsd, and
        best is (B,) int64 index of that sample.

    As in src/boltz/model/loss/validation.py::weighted_minimum_rmsd.

    Hints:
        - Can you flatten (B, S) into one batch instead of looping?
        - repeat_interleave or repeat: which keeps sample s of example b at b*S + s?
    """
    raise NotImplementedError("Drill #37")


# ---------------------------------------------------------------------------
# Drill #38: smooth lDDT loss and hard lDDT
# ---------------------------------------------------------------------------


def smooth_lddt_loss(
    pred_coords: Tensor,
    true_coords: Tensor,
    is_nucleotide: Tensor,
    mask: Tensor,
) -> Tensor:
    """Differentiable lDDT loss (AlphaFold3 Alg. 27), vectorised.

    Drill #38. Tier A. Target: about 20 min together with lddt.

    Args:
        pred_coords, true_coords: (B, N, 3).
        is_nucleotide: (B, N) bool/float, 1 for nucleotide atoms.
        mask: (B, N) atom mask.

    Returns:
        Scalar 1 - mean_b(lddt_b), where
        dd_ij = | ||t_i - t_j|| - ||p_i - p_j|| |,
        eps_ij = mean over thr in {0.5, 1, 2, 4} of sigmoid(thr - dd_ij),
        pair_mask_ij = [d_true_ij < cutoff_i] * m_i * m_j * (i != j),
        cutoff_i = 30 if atom i (the ROW) is a nucleotide else 15,
        lddt_b = sum(eps * pair_mask) / (sum(pair_mask) + 1e-5).

    As in src/boltz/model/loss/diffusionv2.py::smooth_lddt_loss.

    Hints:
        - Is the inclusion radius measured on the true or the predicted distances?
        - Does the nucleotide flag go on i, on j, or on both?
        - What is the loss when pred equals true? (It is not 0.)
        - Is the gradient of your distance finite at d = 0 (the diagonal)?
    """
    raise NotImplementedError("Drill #38")


def lddt(
    pred_coords: Tensor,
    true_coords: Tensor,
    is_nucleotide: Tensor,
    mask: Tensor,
) -> Tensor:
    """Hard (non-differentiable) global lDDT.

    Drill #38. Tier A.

    Args: as smooth_lddt_loss.

    Returns:
        (B,) float in [0, 1]. Same pair mask and reduction as smooth_lddt_loss,
        but eps_ij = mean over thr of the indicator [dd_ij < thr].

    Hints:
        - Can you share the pair-mask code with the smooth version?
        - Which value is returned for a structure with no valid pair?
    """
    raise NotImplementedError("Drill #38")


# ---------------------------------------------------------------------------
# Drill #39: EDM diffusion training loss on coordinates
# ---------------------------------------------------------------------------


def sample_training_sigma(
    batch_size: int,
    sigma_data: float = 16.0,
    p_mean: float = -1.2,
    p_std: float = 1.5,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Training noise levels sigma = sigma_data * exp(p_mean + p_std * N(0, 1)).

    Drill #39. Tier B. Target: about 35-40 min for the whole drill.

    Returns:
        (batch_size,) float32, drawn with torch.randn(..., generator=generator).

    As in src/boltz/model/modules/diffusionv2.py::AtomDiffusion.noise_distribution.

    Hints:
        - What distribution does log(sigma / sigma_data) follow?
    """
    raise NotImplementedError("Drill #39")


def edm_preconditioning(
    sigma: Tensor, sigma_data: float = 16.0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """EDM (Karras et al. 2022) preconditioning coefficients.

    Drill #39. Tier B.

    Args:
        sigma: (B,) positive noise levels.

    Returns:
        (c_skip, c_out, c_in, c_noise), each (B,):
        c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
        c_noise = 0.25 * log(clamp(sigma / sigma_data, min=1e-20))

    Hints:
        - What do c_skip and c_out tend to as sigma -> 0? What does that mean for D?
    """
    raise NotImplementedError("Drill #39")


def edm_loss_weight(sigma: Tensor, sigma_data: float = 16.0) -> Tensor:
    """EDM loss weight (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2, shape (B,).

    Drill #39. Tier B.

    Hints:
        - Is this 1 / c_out^2?
    """
    raise NotImplementedError("Drill #39")


def edm_denoise(
    net: Callable[[Tensor, Tensor], Tensor],
    x_noisy: Tensor,
    sigma: Tensor,
    sigma_data: float = 16.0,
) -> Tensor:
    """Preconditioned denoiser D(x; sigma) = c_skip x + c_out F(c_in x, c_noise).

    Drill #39. Tier B.

    Args:
        net: F, called as net(scaled_x (B, N, 3), c_noise (B,)) -> (B, N, 3).
        x_noisy: (B, N, 3).
        sigma: (B,).

    Returns:
        (B, N, 3) denoised coordinates.

    Hints:
        - Which shape must the (B,) coefficients take to broadcast against (B, N, 3)?
    """
    raise NotImplementedError("Drill #39")


def edm_diffusion_loss(
    net: Callable[[Tensor, Tensor], Tensor],
    coords: Tensor,
    mask: Tensor,
    is_nucleotide: Tensor,
    is_ligand: Tensor,
    sigma: Tensor,
    noise: Tensor,
    sigma_data: float = 16.0,
    add_smooth_lddt: bool = False,
) -> Tensor:
    """EDM training loss on (already centred and augmented) coordinates.

    Drill #39. Tier B. Reuse Drills #37 and #38.

    Args:
        net: F as in edm_denoise.
        coords: (B, N, 3) ground truth, already centred and randomly rotated.
        mask: (B, N) resolved-atom mask.
        is_nucleotide, is_ligand: (B, N) bool/float flags.
        sigma: (B,) noise levels (e.g. from sample_training_sigma).
        noise: (B, N, 3) standard normal noise.
        add_smooth_lddt: add smooth_lddt_loss(D, coords, is_nucleotide, mask).

    Returns:
        Scalar. Steps:
        x_noisy = coords + sigma * noise; D = edm_denoise(net, x_noisy, sigma);
        w = 1 + 5 * is_nucleotide + 10 * is_ligand;
        mse_b = aligned_weighted_mse(D, coords, w, mask) (ground truth aligned
        onto D, target detached); loss = mean_b(mse_b * edm_loss_weight(sigma_b)),
        plus the smooth lDDT term if requested.

    As in src/boltz/model/modules/diffusionv2.py::AtomDiffusion.compute_loss.

    Hints:
        - Which tensor is aligned onto which, and which one carries the gradient?
        - Are the alignment weights the same as the MSE weights?
    """
    raise NotImplementedError("Drill #39")


# ---------------------------------------------------------------------------
# Drill #40: Karras noise schedule and stochastic sampler (AF3 Alg. 18)
# ---------------------------------------------------------------------------


def karras_schedule(
    num_steps: int,
    sigma_data: float = 16.0,
    sigma_max: float = 160.0,
    sigma_min: float = 1e-4,
    rho: float = 7.0,
) -> Tensor:
    """Karras noise schedule scaled by sigma_data, with a final 0 appended.

    Drill #40. Tier C. Target: about 20 min together with karras_sample.

    Args:
        num_steps: N >= 2.

    Returns:
        (N + 1,) float32, decreasing:
        sigma_i = sigma_data * (sigma_max^(1/rho)
                  + i / (N - 1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        for i = 0..N-1, then 0.
        Defaults are the Boltz-2 ones (Boltz-1 used sigma_min = 4e-4, rho = 8).

    As in src/boltz/model/modules/diffusionv2.py::AtomDiffusion.sample_schedule.

    Hints:
        - What are the first and the last non-zero values, in Angstrom?
        - Is the interpolation done in sigma, or in sigma^(1/rho)?
    """
    raise NotImplementedError("Drill #40")


def karras_sample(
    denoise_fn: Callable[[Tensor, float], Tensor],
    shape: tuple[int, ...],
    sigmas: Tensor,
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale: float = 1.003,
    step_scale: float = 1.5,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Stochastic EDM sampler (AlphaFold3 Alg. 18, without the augmentation step).

    Drill #40. Tier C.

    Args:
        denoise_fn: D, called as denoise_fn(x_noisy, t_hat) with t_hat a float.
        shape: output shape, e.g. (S, N, 3).
        sigmas: (T + 1,) schedule ending in 0 (from karras_schedule).
        generator: use it for every torch.randn call.

    Returns:
        Tensor of `shape`. Algorithm:
        x = sigmas[0] * randn(shape)
        for each consecutive pair (s_prev, s_t):
            gamma = gamma_0 if s_t > gamma_min else 0
            t_hat = s_prev * (gamma + 1)
            x_noisy = x + noise_scale * sqrt(t_hat^2 - s_prev^2) * randn(shape)
            delta = (x_noisy - D(x_noisy, t_hat)) / t_hat
            x = x_noisy + step_scale * (s_t - t_hat) * delta

    Hints:
        - Is gamma decided by the current or by the next sigma?
        - Does the last step (s_t = 0) inject new noise?
    """
    raise NotImplementedError("Drill #40")


# ---------------------------------------------------------------------------
# Drill #41: random SO(3) rotation and centring augmentation (AF3 Alg. 19)
# ---------------------------------------------------------------------------


def random_quaternions(
    n: int, generator: Optional[torch.Generator] = None
) -> Tensor:
    """Uniformly random unit quaternions, real part first, real part >= 0.

    Drill #41. Tier B. Target: about 10 min for the whole drill.

    Returns:
        (n, 4) float32 (w, x, y, z), unit norm, w >= 0. Draw a 4-D standard
        normal with torch.randn(n, 4, generator=generator), normalise, flip sign.

    As in src/boltz/model/modules/utils.py::random_quaternions.

    Hints:
        - Why does a normalised Gaussian give a uniform rotation?
        - Why is flipping q to -q allowed?
    """
    raise NotImplementedError("Drill #41")


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrix of a quaternion (real part first).

    Drill #41. Tier B.

    Args:
        q: (..., 4) quaternions (w, x, y, z); need not be exactly unit norm.

    Returns:
        (..., 3, 3) R such that the rotated column vector is R @ v (the same
        matrix as scipy's Rotation.as_matrix()). For row vectors use x @ R^T.

    Hints:
        - Divide by |q|^2 so a slightly non-unit q still gives a rotation.
        - Check R^T R = I and det R = +1.
    """
    raise NotImplementedError("Drill #41")


def center_random_augmentation(
    coords: Tensor,
    mask: Tensor,
    s_trans: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Centre, randomly rotate and randomly translate each structure.

    Drill #41. Tier B.

    Args:
        coords: (B, N, 3).
        mask: (B, N) atom mask.
        s_trans: std of the random translation in Angstrom.

    Returns:
        (B, N, 3): subtract the masked centroid, apply one random rotation per
        structure (from random_quaternions), then add s_trans * N(0, I_3) per
        structure. Rigid: pairwise distances and handedness are preserved.

    As in src/boltz/model/modules/utils.py::center_random_augmentation.

    Hints:
        - Do padding atoms belong in the centroid?
        - Rotate first, then translate, or the other way round?
    """
    raise NotImplementedError("Drill #41")


# ---------------------------------------------------------------------------
# Drill #45: binned confidence heads (pLDDT, PDE, PAE) and pTM / ipTM
# ---------------------------------------------------------------------------


def bin_values(x: Tensor, max_value: float, num_bins: int) -> Tensor:
    """Bin index floor(x * num_bins / max_value), clamped to [0, num_bins - 1].

    Drill #45. Tier A. Target: about 20 min for the core of the drill.

    Returns:
        int64 tensor with the shape of x.

    Hints:
        - Which bin does x == max_value fall in?
    """
    raise NotImplementedError("Drill #45")


def expected_value_from_bins(logits: Tensor, max_value: float) -> Tensor:
    """Decode a binned head: sum_k softmax(logits)_k * centre_k.

    Drill #45. Tier A.

    Args:
        logits: (..., K).
        max_value: the bins split [0, max_value] into K equal bins.

    Returns:
        (...) expected value, with centre_k = (k + 0.5) * max_value / K.

    Hints:
        - What does uniform logits decode to?
    """
    raise NotImplementedError("Drill #45")


def plddt_loss(logits: Tensor, target_lddt: Tensor, mask: Tensor) -> Tensor:
    """pLDDT head cross-entropy.

    Drill #45. Tier A.

    Args:
        logits: (B, N, K) with K = 50 in Boltz.
        target_lddt: (B, N) per-token lDDT in [0, 1].
        mask: (B, N) token mask.

    Returns:
        Scalar. Target bin = bin_values(target_lddt, 1.0, K); per-token CE;
        per structure sum(ce * mask) / (sum(mask) + 1e-5); mean over batch.
        Stable for large logits.

    As in src/boltz/model/loss/confidencev2.py::plddt_loss.

    Hints:
        - Which bin does lDDT = 1.0 land in?
    """
    raise NotImplementedError("Drill #45")


def pde_loss(
    logits: Tensor,
    pred_coords: Tensor,
    true_coords: Tensor,
    mask: Tensor,
    max_dist: float = 32.0,
) -> Tensor:
    """PDE head cross-entropy: binned |d_true - d_pred| per token pair.

    Drill #45. Tier A.

    Args:
        logits: (B, N, N, K) with K = 64 in Boltz.
        pred_coords, true_coords: (B, N, 3).
        mask: (B, N) token mask.

    Returns:
        Scalar. Target bin = bin_values(|cdist(true) - cdist(pred)|, max_dist, K);
        pair mask = m_i * m_j (diagonal kept); per structure masked mean of
        the CE; mean over batch.

    As in src/boltz/model/loss/confidencev2.py::pde_loss.

    Hints:
        - Is the target symmetric in i and j? Are the logits?
        - Should the target carry gradients into pred_coords?
    """
    raise NotImplementedError("Drill #45")


def compute_ptm(
    pae_logits: Tensor,
    mask: Tensor,
    asym_id: Optional[Tensor] = None,
    max_value: float = 32.0,
) -> Tensor:
    """pTM (or ipTM when asym_id is given) from PAE logits.

    Drill #45. Tier B.

    Args:
        pae_logits: (B, N, N, K), bins over [0, max_value].
        mask: (B, N) token mask.
        asym_id: optional (B, N) chain ids. If given, only pairs from
            different chains count (ipTM).

    Returns:
        (B,). With n = sum(mask) per structure,
        d0 = 1.24 * (max(n, 19) - 15)^(1/3) - 1.8,
        tm(e) = 1 / (1 + (e / d0)^2) evaluated at the bin centres,
        E_ij = sum_k softmax(pae_logits)_ijk * tm(centre_k),
        pair_mask_ij = m_i * m_j (* [asym_i != asym_j] for ipTM),
        row_i = sum_j E_ij pair_mask_ij / (sum_j pair_mask_ij + 1e-5),
        result = max over i of row_i.

    As in src/boltz/model/layers/confidence_utils.py::compute_ptms.

    Hints:
        - Why clamp n at 19?
        - What does ipTM return for a single-chain input?
    """
    raise NotImplementedError("Drill #45")
