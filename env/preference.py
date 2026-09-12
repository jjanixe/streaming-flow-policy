from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreferenceFit:
    weights: np.ndarray
    objective: float
    gradient_norm: float
    converged: bool
    iterations: int
    feature_rank: int


def _float32_array(values: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validated_features(delta_features: np.ndarray) -> np.ndarray:
    features = _float32_array(delta_features, "delta_features")
    if features.ndim != 2:
        raise ValueError("delta_features must be two-dimensional")
    if features.shape[1] == 0:
        raise ValueError("delta_features must contain at least one feature")
    return features


def _validated_weights(weights: np.ndarray, feature_count: int) -> np.ndarray:
    values = _float32_array(weights, "weights")
    if values.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if values.shape != (feature_count,):
        raise ValueError(f"weights must have shape ({feature_count},)")
    return values


def _validated_labels(labels: np.ndarray, row_count: int) -> np.ndarray:
    signs = _float32_array(labels, "labels")
    if signs.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if signs.shape[0] != row_count:
        raise ValueError("labels and delta_features must have the same number of rows")
    if not np.all((signs == -1.0) | (signs == 1.0)):
        raise ValueError("labels must contain only -1 or +1")
    return signs


def _logits(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features.astype(np.float64) @ weights.astype(np.float64)


def preference_probabilities(
    delta_features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    features = _validated_features(delta_features)
    values = _validated_weights(weights, features.shape[1])
    logits = _logits(features, values)
    return np.exp(-np.logaddexp(0.0, -logits)).astype(np.float32)


def preference_metrics(
    delta_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    features = _validated_features(delta_features)
    signs = _validated_labels(labels, features.shape[0])
    values = _validated_weights(weights, features.shape[1])
    if features.shape[0] == 0:
        raise ValueError("preference metrics require at least one comparison")
    logits = _logits(features, values)
    losses = np.logaddexp(0.0, -(signs.astype(np.float64) * logits))
    predictions = np.where(logits >= 0.0, 1.0, -1.0)
    return {
        "log_loss": float(losses.mean(dtype=np.float64)),
        "accuracy": float(np.mean(predictions == signs)),
    }


def fit_bradley_terry(
    delta_features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 0.1,
    prior_mean: np.ndarray | None = None,
    max_steps: int = 100,
    tolerance: float = 1e-5,
) -> PreferenceFit:
    if (
        isinstance(l2, (bool, np.bool_))
        or not isinstance(l2, (int, float, np.integer, np.floating))
        or not np.isfinite(l2)
        or float(l2) <= 0.0
    ):
        raise ValueError("l2 must be finite and positive")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a non-negative integer")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(tolerance)
        or float(tolerance) <= 0.0
    ):
        raise ValueError("tolerance must be finite and positive")
    features = _validated_features(delta_features)
    signs = _validated_labels(labels, features.shape[0])
    feature_count = features.shape[1]
    prior = (
        np.zeros(feature_count, dtype=np.float32)
        if prior_mean is None
        else _float32_array(prior_mean, "prior_mean")
    )
    if prior.shape != (feature_count,):
        raise ValueError(f"prior_mean must have shape ({feature_count},)")
    features64 = features.astype(np.float64)
    signs64 = signs.astype(np.float64)
    prior64 = prior.astype(np.float64)
    weights64 = prior64.copy()
    identity = np.eye(feature_count, dtype=np.float64)

    def state(values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        margins = signs64 * (features64 @ values)
        losses = np.logaddexp(0.0, -margins)
        negative_probabilities = np.exp(-np.logaddexp(0.0, margins))
        residual = values - prior64
        objective = float(
            losses.sum(dtype=np.float64)
            + 0.5 * float(l2) * np.dot(residual, residual)
        )
        gradient = (
            features64.T @ (-signs64 * negative_probabilities)
            + float(l2) * residual
        )
        curvature = negative_probabilities * (1.0 - negative_probabilities)
        hessian = (
            features64.T @ (curvature[:, None] * features64)
            + float(l2) * identity
        )
        return objective, gradient, hessian

    def singular_newton_direction(
        values: np.ndarray,
    ) -> np.ndarray:
        margins = signs64 * (features64 @ values)
        probabilities = np.exp(-np.logaddexp(0.0, margins))
        curvature = probabilities * (1.0 - probabilities)
        positive = curvature > 0.0
        weighted_features = (
            np.sqrt(curvature[positive])[:, None] * features64[positive]
        )
        left, singular_values, right = np.linalg.svd(
            weighted_features,
            full_matrices=False,
        )

        likelihood_residual = -signs64 * probabilities
        scaled_residual = (
            likelihood_residual[positive] / np.sqrt(curvature[positive])
        )
        likelihood_components = singular_values * (left.T @ scaled_residual)

        zero_curvature_gradient = (
            features64[~positive].T @ likelihood_residual[~positive]
        )
        regularization_and_zero = (
            float(l2) * (values - prior64) + zero_curvature_gradient
        )
        extra_components = right @ regularization_and_zero
        resolved = right.T @ (
            (likelihood_components + extra_components)
            / (singular_values * singular_values + float(l2))
        )
        unresolved = regularization_and_zero - right.T @ extra_components
        direction = -(resolved + unresolved / float(l2))

        if not np.isfinite(direction).all():
            raise np.linalg.LinAlgError(
                "regularized SVD Newton solve produced a nonfinite direction"
            )
        return direction

    iterations = 0
    objective, gradient, hessian = state(weights64)
    gradient_norm = float(np.linalg.norm(gradient))
    converged = gradient_norm <= tolerance
    while not converged and iterations < max_steps:
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            # A small ridge can round away when X.T W X is very large. Solving
            # through the weighted design's singular values retains it.
            direction = singular_newton_direction(weights64)
        directional_derivative = float(np.dot(gradient, direction))
        step_size = 1.0
        accepted = False
        for _ in range(1075):
            candidate = weights64 + step_size * direction
            if np.array_equal(candidate, weights64):
                break
            candidate_objective, _, _ = state(candidate)
            sufficient_decrease = (
                objective + 1e-4 * step_size * directional_derivative
            )
            if candidate_objective <= sufficient_decrease:
                weights64 = candidate
                accepted = True
                break
            step_size *= 0.5
        iterations += 1
        if not accepted:
            break
        objective, gradient, hessian = state(weights64)
        gradient_norm = float(np.linalg.norm(gradient))
        converged = gradient_norm <= tolerance

    weights = weights64.astype(np.float32)
    objective, gradient, _ = state(weights.astype(np.float64))
    gradient_norm = float(np.linalg.norm(gradient))
    converged = gradient_norm <= tolerance

    return PreferenceFit(
        weights=weights,
        objective=objective,
        gradient_norm=gradient_norm,
        converged=converged,
        iterations=iterations,
        feature_rank=(
            0 if features.shape[0] == 0 else int(np.linalg.matrix_rank(features))
        ),
    )
