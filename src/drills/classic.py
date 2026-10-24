"""Section 7: classic ML from scratch (Drills #59-#63).

All drills here are NumPy only. Every function raises NotImplementedError
until you fill it in.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Drill #59: k-means (Lloyd's algorithm), vectorised
# ---------------------------------------------------------------------------


def kmeans_plusplus_init(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means++ seeding.

    Drill #59 (follow-up). Tier B. Target: part of the 20 min below.

    Args:
        X: (N, D) float array.
        k: number of centroids, 1 <= k <= number of distinct rows of X.
        seed: seed for np.random.default_rng.

    Returns:
        (k, D) float array of initial centroids. Every centroid is a row of X
        and no two centroids are equal.

    Rule: the first centroid is uniform over the rows; each next one is drawn
    with probability proportional to the squared distance to the nearest
    centroid chosen so far.

    Hints:
        - What is the probability of picking a point that is already a centroid?
        - Can you update the "distance to nearest centroid" vector in O(N D) per step?
    """
    raise NotImplementedError("Drill #59")


def kmeans(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means by Lloyd's algorithm, fully vectorised (no Python loop over points).

    Drill #59. Tier B. Target: about 20 min.

    Args:
        X: (N, D) float array.
        k: number of clusters, 1 <= k <= number of distinct rows of X.
        max_iters: maximum number of assign/update rounds.
        tol: stop when no centroid moves by more than tol (or labels stop changing).
        seed: seed for np.random.default_rng (initialisation and any re-seeding).

    Returns:
        centroids: (k, D) float array.
        labels: (N,) int64 array in [0, k), labels[i] = index of the nearest
            centroid of X[i] (squared Euclidean distance).

    Requirements:
        - Initialise without repeated points (k-means++ or sampling distinct rows).
        - No empty clusters in the result: if a cluster empties, re-seed it
          (for example with the point farthest from its centroid).
        - The returned labels must be consistent with the returned centroids.
        - Pairwise distances via ||a||^2 - 2 a.b + ||b||^2, never an (N, k, D) tensor.

    Hints:
        - What happens to the mean of a cluster with zero members?
        - If X has many duplicate rows, can your initialisation pick the same point twice?
        - After the last update step, are your labels still the argmin for the final centroids?
    """
    raise NotImplementedError("Drill #59")


# ---------------------------------------------------------------------------
# Drill #60: kNN, 1-NN, and 1-NN as a linear layer
# ---------------------------------------------------------------------------


def knn_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int
) -> np.ndarray:
    """k-nearest-neighbour classification with a majority vote, vectorised.

    Drill #60. Tier B. Target: about 12 min (with the function below).

    Args:
        X_train: (N, D) float array.
        y_train: (N,) int array of non-negative class labels.
        X_test: (M, D) float array.
        k: number of neighbours, 1 <= k <= N.

    Returns:
        (M,) int64 array of predicted labels.

    Conventions:
        - Distance is Euclidean (squared is fine, same ordering).
        - Vote ties are broken by the smallest class label.
        - Use np.argpartition to get the k nearest, not a full sort.

    Hints:
        - Does argpartition return the k nearest in sorted order? Does it matter for a vote?
        - How can np.bincount (or a one-hot sum) count votes for all test points at once?
    """
    raise NotImplementedError("Drill #60")


def nn1_as_linear(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express 1-NN as a linear layer: argmax over (X_test @ W.T + b).

    Drill #60. Tier B.

    Args:
        X_train: (N, D) float array.

    Returns:
        W: (N, D) float array.
        b: (N,) float array.
        such that argmax_j (x @ W[j] + b[j]) == argmin_j ||x - X_train[j]||.

    Hints:
        - Expand ||x - x_j||^2. Which term does not depend on j?
        - Minimising a quantity is maximising its negative times what constant?
    """
    raise NotImplementedError("Drill #60")


# ---------------------------------------------------------------------------
# Drill #61: logistic regression by gradient descent
# ---------------------------------------------------------------------------


def logistic_loss_and_grad(
    w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, l2: float = 0.0
) -> tuple[float, np.ndarray, float]:
    """Mean binary cross-entropy of a logistic model, with its gradient.

    Drill #61. Tier B. Target: about 10 min (with the function below).

    Model: z = X @ w + b (logits), p = sigmoid(z).
    Loss: mean_i BCE(z_i, y_i) + 0.5 * l2 * ||w||^2 (the bias is not penalised).

    Args:
        w: (D,) float array.
        b: scalar bias.
        X: (N, D) float array.
        y: (N,) array of 0/1 labels.
        l2: L2 strength.

    Returns:
        loss: float.
        grad_w: (D,) float array.
        grad_b: float.

    Requirements:
        - Work on logits, not probabilities: finite for |z| up to 1e4.

    Hints:
        - Can you write log(1 + exp(z)) with np.logaddexp or a max-shift?
        - What is d(loss)/dz for one example, in terms of p and y?
    """
    raise NotImplementedError("Drill #61")


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    n_iters: int = 2000,
    l2: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fit logistic regression by full-batch gradient descent.

    Drill #61. Tier B.

    Minimises the loss of logistic_loss_and_grad starting from w = 0, b = 0
    (deterministic, no random init).

    Args:
        X: (N, D) float array.
        y: (N,) array of 0/1 labels.
        lr: step size.
        n_iters: number of full-batch steps.
        l2: L2 strength on w.

    Returns:
        w: (D,) float array.
        b: float.

    Hints:
        - Did you check your gradient against finite differences before looping?
        - Why is starting from zeros fine here but not for a neural net?
    """
    raise NotImplementedError("Drill #61")


# ---------------------------------------------------------------------------
# Drill #62: linear regression, normal equation, GD, ridge, LOO residuals
# ---------------------------------------------------------------------------


def ridge_normal_equation(
    X: np.ndarray, y: np.ndarray, l2: float = 0.0
) -> tuple[np.ndarray, float]:
    """Linear / ridge regression in closed form, with an unpenalised intercept.

    Drill #62. Tier C. Target: about 15 min (whole drill).

    Minimises sum_i (x_i @ w + b - y_i)^2 + l2 * ||w||^2  (sum, not mean;
    b is not penalised). l2 = 0 gives ordinary least squares.

    Args:
        X: (N, D) float array (N > D when l2 = 0).
        y: (N,) float array.
        l2: ridge strength >= 0.

    Returns:
        w: (D,) float array.
        b: float.

    Requirements:
        - Use np.linalg.solve, never an explicit matrix inverse.

    Hints:
        - Append a column of ones; which diagonal entry of the penalty matrix stays 0?
        - Alternatively: centre X and y first. What is b then?
    """
    raise NotImplementedError("Drill #62")


def linear_regression_gd(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 5000
) -> tuple[np.ndarray, float]:
    """Ordinary least squares by full-batch gradient descent.

    Drill #62. Tier C.

    Minimises the mean squared error (1/N) sum_i (x_i @ w + b - y_i)^2,
    starting from w = 0, b = 0.

    Args:
        X: (N, D) float array.
        y: (N,) float array.
        lr: step size.
        n_iters: number of steps.

    Returns:
        w: (D,) float array.
        b: float.

    Hints:
        - Mean or sum in the loss: how does that change the safe step size?
        - What is the factor 2 in the gradient doing?
    """
    raise NotImplementedError("Drill #62")


def ridge_loo_residuals(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Leave-one-out residuals of ridge regression without refitting N times.

    Drill #62. Tier C.

    Same model and objective as ridge_normal_equation (unpenalised intercept).
    Returns r_i = y_i - yhat_{-i}(x_i), where yhat_{-i} is the model fitted
    without row i. Use the hat-matrix shortcut r_i = e_i / (1 - h_ii), where
    e = y - H y is the in-sample residual and H the hat matrix.

    Args:
        X: (N, D) float array.
        y: (N,) float array.
        l2: ridge strength > 0.

    Returns:
        (N,) float array of LOO residuals.

    Hints:
        - With the ones column appended, what is H in terms of the design and penalty?
        - Do you need all of H, or only its diagonal?
        - Check against an explicit loop that refits N times.
    """
    raise NotImplementedError("Drill #62")


# ---------------------------------------------------------------------------
# Drill #63: PCA via SVD
# ---------------------------------------------------------------------------


def pca(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal component analysis via the SVD of the centred data.

    Drill #63. Tier B. Target: about 10 min.

    Args:
        X: (N, D) float array.
        n_components: k, 1 <= k <= min(N, D).

    Returns:
        components: (k, D) float array, orthonormal rows, sorted by decreasing
            variance. Sign convention: the entry with the largest absolute
            value in each row is positive.
        explained_variance: (k,) float array, variance along each component,
            with the unbiased (N - 1) denominator.
        scores: (N, k) float array, the centred data projected on the components.

    Hints:
        - Did you centre X before the SVD?
        - How does a singular value s relate to a variance?
        - If you flip the sign of a component, what else has to flip?
    """
    raise NotImplementedError("Drill #63")
