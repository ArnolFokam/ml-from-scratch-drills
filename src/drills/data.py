"""Section 8: data and evaluation hygiene (Drills #64-#67).

Leakage-aware splits, Tanimoto similarity and diversity picking,
multi-dataset sampling weights, and label-noise detection.
Everything here is plain NumPy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Drill #64: leakage-aware splits (group, temporal, similarity)
# ---------------------------------------------------------------------------


def group_split(
    groups: np.ndarray, test_frac: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split items so that every group lands wholly in train or wholly in test.

    Drill #64. Tier B. About 20 min (whole drill). NumPy.

    Args:
        groups: (N,) group id per item (int or str), e.g. assay or document id.
        test_frac: target fraction of ITEMS (not groups) in test, in [0, 1].
        seed: seed for np.random.default_rng.

    Returns:
        (train_idx, test_idx): sorted int arrays of item indices. Together they
        cover range(N) exactly once.

    Convention: shuffle the unique group ids with the seeded generator, then
    move whole groups into test until the test item count is >= test_frac * N.
    test_frac = 0 gives an empty test set.

    Hints:
        - Can any group id appear in both outputs?
        - Are you counting items or groups when you compare with test_frac?
    """
    raise NotImplementedError("Drill #64")


def temporal_split(
    dates: np.ndarray, cutoff: np.datetime64 | float
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal split: train is strictly before the cutoff, test is on/after it.

    Drill #64. Tier B. NumPy.

    Args:
        dates: (N,) datetime64 or numeric release dates.
        cutoff: same type as dates.

    Returns:
        (train_idx, test_idx): sorted int arrays; train has dates < cutoff,
        test has dates >= cutoff.

    Hints:
        - Which side does an item dated exactly on the cutoff go to?
    """
    raise NotImplementedError("Drill #64")


def flag_similar_to_train(
    sim_test_train: np.ndarray, threshold: float = 0.45
) -> np.ndarray:
    """Flag test items too similar to any train item.

    Drill #64. Tier B. NumPy.

    Args:
        sim_test_train: (N_test, N_train) float similarity (e.g. Tanimoto).
        threshold: flag when the max similarity to train is strictly above it.

    Returns:
        (N_test,) bool mask. With N_train == 0 nothing is flagged.

    Hints:
        - What does .max(axis=1) do on an array with zero columns?
    """
    raise NotImplementedError("Drill #64")


def similarity_components(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph "sim[i, j] > threshold", via union-find.

    Drill #64. Tier B. NumPy plus a small union-find.

    Args:
        sim: (N, N) symmetric float similarity matrix. The diagonal is ignored.
        threshold: an edge exists when sim[i, j] > threshold (strict).

    Returns:
        (N,) int64 component labels 0..K-1, numbered in order of first
        appearance (item 0 is always in component 0).

    Hints:
        - Is similarity transitive? Why must A~B and B~C put A and C together?
        - Do you use path compression in find()?
    """
    raise NotImplementedError("Drill #64")


def similarity_split(
    sim: np.ndarray, threshold: float, test_frac: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Group split where the groups are similarity components.

    Drill #64. Tier B. NumPy.

    Same contract as group_split, with groups = similarity_components(sim,
    threshold). No pair with sim > threshold may cross the split.

    Hints:
        - Can you reuse the two functions above?
    """
    raise NotImplementedError("Drill #64")


# ---------------------------------------------------------------------------
# Drill #65: vectorised Tanimoto and diversity picking
# ---------------------------------------------------------------------------


def tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between binary fingerprints, as a matrix.

    Drill #65. Tier C. About 15 min (whole drill). NumPy, no Python loops.

    T(A, B) = |A and B| / (|A| + |B| - |A and B|)

    Args:
        a: (N, F) bool or {0, 1} fingerprints.
        b: (M, F) bool or {0, 1} fingerprints.

    Returns:
        (N, M) float64 in [0, 1]. Convention: two all-zero fingerprints have
        similarity 0.0 (no division by zero, no NaN).

    Hints:
        - Can the intersection counts for all pairs come from one matmul?
        - What happens to bool @ bool? Cast first.
    """
    raise NotImplementedError("Drill #65")


def maxmin_pick(fps: np.ndarray, n_pick: int, first: int = 0) -> np.ndarray:
    """Greedy MaxMin diversity picking with Tanimoto distance (1 - T).

    Drill #65. Tier C. NumPy.

    Start from index `first`. At each step pick the unpicked item whose
    minimum distance to the already-picked set is largest. Ties go to the
    lowest index.

    Args:
        fps: (N, F) binary fingerprints.
        n_pick: number of items to pick, 0 <= n_pick <= N, else ValueError.
        first: index of the first pick.

    Returns:
        (n_pick,) int64 indices in pick order, no repeats.

    Hints:
        - Can you keep a running (N,) "min distance to picked" vector instead
          of recomputing it each step?
        - How do you stop an already-picked item from being picked again?
    """
    raise NotImplementedError("Drill #65")


def nearest_neighbour_similarity(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Max Tanimoto of each query fingerprint to the train set (OOD check).

    Drill #65. Tier C. NumPy.

    Args:
        query: (Q, F) binary fingerprints.
        train: (M, F) binary fingerprints.

    Returns:
        (Q,) float64. With M == 0 return zeros.

    Hints:
        - Low nearest-neighbour similarity means what for the prediction?
    """
    raise NotImplementedError("Drill #65")


# ---------------------------------------------------------------------------
# Drill #66: weighted and temperature multi-dataset sampling
# (merged into Drill #33; kept here as standalone functions)
# ---------------------------------------------------------------------------


def temperature_sampling_probs(sizes: np.ndarray, temperature: float) -> np.ndarray:
    """Dataset sampling probabilities p_i proportional to (n_i / sum n)^(1/T).

    Drill #66. Tier B. About 5 min. NumPy. As in the JMP paper (T = 2).

    Args:
        sizes: (D,) positive dataset sizes.
        temperature: T > 0. T = 1 is size-proportional; T -> inf is uniform.

    Returns:
        (D,) float64 probabilities summing to 1.

    Hints:
        - Did you renormalise after the power?
    """
    raise NotImplementedError("Drill #66")


def cluster_sample_weights(
    cluster_ids: np.ndarray,
    type_counts: np.ndarray,
    alphas: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """AF3-style cluster weights: w_i = beta / |cluster(i)| * (alphas . counts_i).

    Drill #66. Tier B. NumPy. As in AlphaFold3 Sec. 2.5 and the Boltz-1 code,
    src/boltz/data/sample/cluster.py.

    Args:
        cluster_ids: (N,) cluster id per example.
        type_counts: (N, C) counts per molecule type (e.g. protein, nucleic,
            ligand) in each example.
        alphas: (C,) per-type weights.
        beta: global scale (cancels after normalisation, keep it anyway).

    Returns:
        (N,) float64 sampling probabilities summing to 1.

    Hints:
        - How do you get each example's cluster size without a Python loop?
          (np.unique has return_inverse and return_counts.)
    """
    raise NotImplementedError("Drill #66")


# ---------------------------------------------------------------------------
# Drill #67: label-noise detection with out-of-fold probabilities
# ---------------------------------------------------------------------------


def out_of_fold_probs(
    x: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Out-of-fold predicted probabilities via K-fold cross-fitting.

    Drill #67. Tier C. About 15 min (whole drill). NumPy.

    Fold of item i is i % n_folds. For each fold f, call
    fit_predict(x_train, y_train, x_test) with the items NOT in f as train and
    the items in f as test; it returns (n_test, K) probabilities.

    Args:
        x: (N, D) features.
        y: (N,) int labels.
        n_folds: number of folds, 2 <= n_folds <= N.
        fit_predict: the model, treated as a black box.

    Returns:
        (N, K) float64; row i comes from a model that never saw item i.

    Hints:
        - Why is in-sample probability useless for finding mislabelled items?
    """
    raise NotImplementedError("Drill #67")


def flag_label_issues(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.2
) -> np.ndarray:
    """Flag items whose out-of-fold probability of the GIVEN label is low.

    Drill #67. Tier C. NumPy.

    Args:
        probs: (N, K) out-of-fold probabilities.
        labels: (N,) int given labels.
        threshold: flag when probs[i, labels[i]] < threshold (strict).

    Returns:
        (N,) bool mask.

    Hints:
        - Fancy indexing: probs[np.arange(N), labels].
    """
    raise NotImplementedError("Drill #67")


def confident_learning_flags(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Confident-learning style flags with per-class thresholds.

    Drill #67 follow-up. Tier C. NumPy.

    1. t_j = mean of probs[i, j] over items with labels[i] == j
       (t_j = +inf if class j has no items).
    2. Item i's candidate classes are those k with probs[i, k] >= t_k.
    3. If there is no candidate, the item is not flagged. Otherwise its
       predicted class is the candidate with the largest probs[i, k]
       (ties to the lowest index), and it is flagged when that class differs
       from labels[i].

    Args:
        probs: (N, K) out-of-fold probabilities.
        labels: (N,) int given labels.

    Returns:
        (N,) bool mask.

    Hints:
        - Why does a per-class threshold beat one global threshold when the
          model is under-confident on a rare class?
    """
    raise NotImplementedError("Drill #67")
