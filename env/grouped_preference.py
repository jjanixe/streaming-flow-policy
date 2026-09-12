"""Bounded two-group preference features and Bradley--Terry fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from env.demonstrations import evaluate_path


_SCOPES = ("direction", "width", "overall")
_SCOPE_PATTERN = (
    "direction", "width", "overall", "direction", "width",
    "overall", "direction", "width", "overall", "overall",
)
_HORIZON = 64


@dataclass(frozen=True)
class GroupedPreference:
    """Within-group and between-group two-element probability simplexes."""

    weights: np.ndarray
    alpha: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights)
        alpha = np.asarray(self.alpha)
        if weights.shape != (2, 2) or alpha.shape != (2,):
            raise ValueError("weights and alpha must have shapes (2, 2) and (2,)")
        if weights.dtype != np.float64 or alpha.dtype != np.float64:
            raise ValueError("weights and alpha must use float64")
        if not np.isfinite(weights).all() or not np.isfinite(alpha).all():
            raise ValueError("weights and alpha must be finite")
        if np.any(weights < 0.0) or np.any(alpha < 0.0):
            raise ValueError("weights and alpha must be nonnegative")
        if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("each row of weights must sum to one")
        if not np.isclose(alpha.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("alpha must sum to one")
        weights = weights.copy()
        alpha = alpha.copy()
        weights.flags.writeable = False
        alpha.flags.writeable = False
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "alpha", alpha)


@dataclass(frozen=True)
class GroupedFit:
    preference: GroupedPreference
    objective: float
    converged: bool
    iterations: int
    projected_gradient_norm: float
    jacobian_rank: int
    scope_counts: dict[str, int]
    staged_initialization: np.ndarray
    start_objectives: tuple[float, ...]
    start_converged: tuple[bool, ...]


def _finite_positive(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite_nonnegative(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _validated_delta(delta_features: np.ndarray, *, nonempty: bool = False) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(delta_features, dtype=np.float64)
    if delta.ndim != 3 or delta.shape[1:] != (2, 2):
        raise ValueError("delta_features must have shape [N, 2, 2]")
    if nonempty and delta.shape[0] == 0:
        raise ValueError("at least one comparison is required")
    if not np.isfinite(delta).all():
        raise ValueError("delta_features must be finite")
    return delta


def _validated_scopes(scopes, row_count: int) -> np.ndarray:
    values = np.asarray(scopes)
    if values.ndim != 1 or values.shape[0] != row_count:
        raise ValueError("scopes must have one entry per comparison")
    if not all(isinstance(value, (str, np.str_)) and value in _SCOPES for value in values):
        raise ValueError("scopes must contain direction, width, or overall")
    return values.astype("U9")


def _validated_labels(labels, row_count: int) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != row_count:
        raise ValueError("labels must have one entry per comparison")
    if not np.all((values == -1.0) | (values == 1.0)):
        raise ValueError("labels must contain only -1 or +1")
    return values


def _validated_features(features: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(features, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (2, 2):
        raise ValueError("features must end with shape [2, 2]")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    return values


def _require_representable(values, context: str):
    if not np.isfinite(values).all():
        raise ValueError(f"{context} exceeds representable float64 range")
    return values


def _representable_mean(values: np.ndarray, context: str) -> float:
    """Compute a mean without overflowing an otherwise representable result."""
    _require_representable(values, context)
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(values / len(values), dtype=np.float64))
    return float(_require_representable(np.asarray(result), context))


def _representable_add(left, right, context: str):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left) + np.asarray(right)
    return _require_representable(result, context)


def classify_modes(midpoint_y: np.ndarray) -> np.ndarray:
    """Classify midpoint heights as UN, UW, LN, LW, or other (IDs 0--4)."""
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(midpoint_y, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("midpoint_y must be finite")
    return np.select(
        (
            (values >= .12) & (values < .4),
            values >= .4,
            (values > -.4) & (values <= -.12),
            values <= -.4,
        ),
        (0, 1, 2, 3),
        default=4,
    ).astype(np.int64, copy=False)


def grouped_features(
    positions: np.ndarray, *, y_scale: float = 0.35,
    feature_kind: str = "continuous",
) -> np.ndarray:
    """Map complete 64-action paths to continuous or mode-aligned features."""
    scale = _finite_positive(y_scale, "y_scale")
    if feature_kind not in ("continuous", "mode"):
        raise ValueError("feature_kind must be continuous or mode")
    with np.errstate(over="ignore", invalid="ignore"):
        paths = np.asarray(positions, dtype=np.float64)
    if paths.ndim < 2 or paths.shape[-2:] != (65, 2):
        raise ValueError("positions must end with shape [65, 2]")
    if not np.isfinite(paths).all():
        raise ValueError("positions must be finite")
    if feature_kind == "mode":
        modes = classify_modes(paths[..., 32, 1])
        features = np.zeros((*modes.shape, 2, 2), dtype=np.float64)
        features[..., 0, 0] = (modes == 0) | (modes == 1)
        features[..., 0, 1] = (modes == 2) | (modes == 3)
        features[..., 1, 0] = (modes == 1) | (modes == 3)
        features[..., 1, 1] = (modes == 0) | (modes == 2)
        return features
    rho = np.tanh(paths[..., 1:, 1] / scale)
    direction = rho.mean(axis=-1, dtype=np.float64)
    energy = np.square(rho).mean(axis=-1, dtype=np.float64)
    return np.stack(
        (
            np.stack(((1.0 + direction) / 2.0, (1.0 - direction) / 2.0), axis=-1),
            np.stack((energy, 1.0 - energy), axis=-1),
        ),
        axis=-2,
    ).astype(np.float64, copy=False)


def grouped_utility(features: np.ndarray, preference: GroupedPreference) -> np.ndarray:
    """Return alpha-weighted utility, including zero-valued failure features."""
    values = _validated_features(features)
    if not isinstance(preference, GroupedPreference):
        raise ValueError("preference must be a GroupedPreference")
    within = np.sum(values * preference.weights, axis=-1, dtype=np.float64)
    return np.sum(within * preference.alpha, axis=-1, dtype=np.float64)


def _parameters(preference: GroupedPreference) -> np.ndarray:
    if not isinstance(preference, GroupedPreference):
        raise ValueError("preference must be a GroupedPreference")
    return np.array(
        [preference.weights[0, 0], preference.weights[1, 0], preference.alpha[0]],
        dtype=np.float64,
    )


def _preference(parameters: np.ndarray) -> GroupedPreference:
    direction, width, alpha = (float(value) for value in parameters)
    return GroupedPreference(
        weights=np.array(
            [[direction, 1.0 - direction], [width, 1.0 - width]], dtype=np.float64
        ),
        alpha=np.array([alpha, 1.0 - alpha], dtype=np.float64),
    )


def _logits_and_jacobian(
    delta: np.ndarray,
    scopes: np.ndarray,
    parameters: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require_representable(parameters, "preference parameters")
    direction, width, alpha = parameters
    with np.errstate(over="ignore", invalid="ignore"):
        direction_slope = delta[:, 0, 0] - delta[:, 0, 1]
        width_slope = delta[:, 1, 0] - delta[:, 1, 1]
        direction_value = delta[:, 0, 1] + direction * direction_slope
        width_value = delta[:, 1, 1] + width * width_slope
        logits = np.empty(len(delta), dtype=np.float64)
        jacobian = np.zeros((len(delta), 3), dtype=np.float64)

        direction_rows = scopes == "direction"
        width_rows = scopes == "width"
        overall_rows = scopes == "overall"
        logits[direction_rows] = rho * direction_value[direction_rows]
        jacobian[direction_rows, 0] = rho * direction_slope[direction_rows]
        logits[width_rows] = rho * width_value[width_rows]
        jacobian[width_rows, 1] = rho * width_slope[width_rows]
        logits[overall_rows] = rho * (
            alpha * direction_value[overall_rows]
            + (1.0 - alpha) * width_value[overall_rows]
        )
        jacobian[overall_rows, 0] = rho * alpha * direction_slope[overall_rows]
        jacobian[overall_rows, 1] = rho * (1.0 - alpha) * width_slope[overall_rows]
        jacobian[overall_rows, 2] = rho * (
            direction_value[overall_rows] - width_value[overall_rows]
        )
    _require_representable(logits, "scaled logits")
    _require_representable(jacobian, "scaled logit Jacobian")
    return logits, jacobian


def grouped_probabilities(
    delta_features: np.ndarray,
    scopes,
    preference: GroupedPreference,
    *,
    rho: float = 16.0,
) -> np.ndarray:
    delta = _validated_delta(delta_features)
    scope_values = _validated_scopes(scopes, len(delta))
    scale = _finite_positive(rho, "rho")
    logits, _ = _logits_and_jacobian(delta, scope_values, _parameters(preference), scale)
    return expit(logits).astype(np.float64, copy=False)


def _objective_state(
    parameters: np.ndarray,
    delta: np.ndarray,
    labels: np.ndarray,
    scopes: np.ndarray,
    rho: float,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits, jacobian = _logits_and_jacobian(delta, scopes, parameters, rho)
    objective = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    for scope in _SCOPES:
        mask = scopes == scope
        if not mask.any():
            continue
        margins = labels[mask] * logits[mask]
        losses = np.logaddexp(0.0, -margins)
        scope_objective = _representable_mean(losses, "logistic objective")
        objective = float(
            _representable_add(objective, scope_objective, "logistic objective")
        )
        loss_logit_gradient = -labels[mask] * expit(-margins)
        with np.errstate(over="ignore", invalid="ignore"):
            contributions = loss_logit_gradient[:, None] * jacobian[mask]
            scope_gradient = np.sum(
                contributions / int(mask.sum()), axis=0, dtype=np.float64
            )
        _require_representable(scope_gradient, "logistic gradient")
        gradient = _representable_add(gradient, scope_gradient, "logistic gradient")
    residual = parameters - 0.5
    with np.errstate(over="ignore", invalid="ignore"):
        regularizer = l2 * float(residual @ residual)
        regularizer_gradient = l2 * (2.0 * residual)
    _require_representable(np.asarray(regularizer), "regularized objective")
    _require_representable(regularizer_gradient, "regularized gradient")
    objective = float(
        _representable_add(objective, regularizer, "regularized objective")
    )
    gradient = _representable_add(gradient, regularizer_gradient, "regularized gradient")
    return objective, gradient, jacobian


def _staged_initialization(
    delta: np.ndarray,
    labels: np.ndarray,
    scopes: np.ndarray,
    rho: float,
    l2: float,
    max_steps: int,
) -> np.ndarray:
    parameters = np.full(3, 0.5, dtype=np.float64)
    for coordinate, scope in ((0, "direction"), (1, "width")):
        mask = scopes == scope
        if not mask.any():
            continue

        def scoped_objective(value: float) -> float:
            candidate = parameters.copy()
            candidate[coordinate] = value
            logits, _ = _logits_and_jacobian(delta[mask], scopes[mask], candidate, rho)
            margins = labels[mask] * logits
            loss = _representable_mean(
                np.logaddexp(0.0, -margins), "staged logistic objective"
            )
            with np.errstate(over="ignore", invalid="ignore"):
                regularizer = l2 * (value - 0.5) ** 2
            return float(
                _representable_add(loss, regularizer, "staged regularized objective")
            )

        result = minimize_scalar(
            scoped_objective,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"maxiter": max_steps, "xatol": 1e-10},
        )
        _require_representable(
            np.asarray([result.x, result.fun]), "staged optimizer result"
        )
        parameters[coordinate] = float(result.x)

    overall = scopes == "overall"
    if overall.any():
        def alpha_objective(value: float) -> float:
            candidate = parameters.copy()
            candidate[2] = value
            logits, _ = _logits_and_jacobian(
                delta[overall], scopes[overall], candidate, rho
            )
            margins = labels[overall] * logits
            loss = _representable_mean(
                np.logaddexp(0.0, -margins), "staged logistic objective"
            )
            with np.errstate(over="ignore", invalid="ignore"):
                regularizer = l2 * (value - 0.5) ** 2
            return float(
                _representable_add(loss, regularizer, "staged regularized objective")
            )

        result = minimize_scalar(
            alpha_objective,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"maxiter": max_steps, "xatol": 1e-10},
        )
        _require_representable(
            np.asarray([result.x, result.fun]), "staged optimizer result"
        )
        parameters[2] = float(result.x)
    return parameters


def _projected_gradient(parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    projected = gradient.copy()
    lower_stationary = (parameters <= 1e-10) & (gradient > 0.0)
    upper_stationary = (parameters >= 1.0 - 1e-10) & (gradient < 0.0)
    projected[lower_stationary | upper_stationary] = 0.0
    return projected


def fit_grouped_preference(
    delta_features: np.ndarray,
    labels,
    scopes,
    *,
    rho: float = 16.0,
    l2: float = 0.1,
    max_steps: int = 300,
) -> GroupedFit:
    """Fit three bounded simplex coordinates with staged and joint optimization."""
    delta = _validated_delta(delta_features, nonempty=True)
    signs = _validated_labels(labels, len(delta))
    scope_values = _validated_scopes(scopes, len(delta))
    scale = _finite_positive(rho, "rho")
    penalty = _finite_nonnegative(l2, "l2")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or int(max_steps) <= 0
    ):
        raise ValueError("max_steps must be a positive integer")
    step_limit = int(max_steps)

    staged = _staged_initialization(
        delta, signs, scope_values, scale, penalty, step_limit
    )
    starts = [staged, np.full(3, 0.5, dtype=np.float64)]
    starts.extend(
        np.array([first, second, third], dtype=np.float64)
        for first in (0.15, 0.85)
        for second in (0.15, 0.85)
        for third in (0.15, 0.85)
    )

    results = []
    for start in starts:
        result = minimize(
            lambda values: _objective_state(
                values, delta, signs, scope_values, scale, penalty
            )[:2],
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=((0.0, 1.0),) * 3,
            options={"maxiter": step_limit, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        _require_representable(
            np.concatenate(
                (
                    np.atleast_1d(result.fun),
                    np.asarray(result.x),
                    np.asarray(result.jac),
                )
            ),
            "joint optimizer result",
        )
        results.append(result)
    chosen = min(results, key=lambda result: (float(result.fun), not result.success))
    parameters = np.clip(np.asarray(chosen.x, dtype=np.float64), 0.0, 1.0)
    # With no relevant likelihood term, positive L2 has the exact optimum 0.5.
    if penalty > 0.0:
        if not np.any((scope_values == "direction") | (scope_values == "overall")):
            parameters[0] = 0.5
        if not np.any((scope_values == "width") | (scope_values == "overall")):
            parameters[1] = 0.5
        if not np.any(scope_values == "overall"):
            parameters[2] = 0.5
    objective, gradient, jacobian = _objective_state(
        parameters, delta, signs, scope_values, scale, penalty
    )
    projected_norm = float(np.linalg.norm(_projected_gradient(parameters, gradient), ord=np.inf))
    _require_representable(np.asarray(projected_norm), "projected gradient norm")
    jacobian_scale = float(np.max(np.abs(jacobian)))
    jacobian_rank = (
        0
        if jacobian_scale == 0.0
        else int(np.linalg.matrix_rank(jacobian / jacobian_scale))
    )
    return GroupedFit(
        preference=_preference(parameters),
        objective=float(objective),
        converged=bool(chosen.success and projected_norm <= 1e-5),
        iterations=int(chosen.nit),
        projected_gradient_norm=projected_norm,
        jacobian_rank=jacobian_rank,
        scope_counts={scope: int(np.sum(scope_values == scope)) for scope in _SCOPES},
        staged_initialization=staged.copy(),
        start_objectives=tuple(float(result.fun) for result in results),
        start_converged=tuple(bool(result.success) for result in results),
    )


def grouped_metrics(
    delta_features: np.ndarray,
    labels,
    scopes,
    preference: GroupedPreference,
    *,
    rho: float = 16.0,
) -> dict[str, dict[str, float | int | None]]:
    delta = _validated_delta(delta_features, nonempty=True)
    signs = _validated_labels(labels, len(delta))
    scope_values = _validated_scopes(scopes, len(delta))
    scale = _finite_positive(rho, "rho")
    logits, _ = _logits_and_jacobian(delta, scope_values, _parameters(preference), scale)

    def metrics(mask: np.ndarray) -> dict[str, float | int | None]:
        count = int(mask.sum())
        if count == 0:
            return {"count": 0, "cross_entropy": None, "accuracy": None}
        losses = np.logaddexp(0.0, -(signs[mask] * logits[mask]))
        predictions = np.where(logits[mask] >= 0.0, 1.0, -1.0)
        return {
            "count": count,
            "cross_entropy": float(losses.mean(dtype=np.float64)),
            "accuracy": float(np.mean(predictions == signs[mask])),
        }

    output = {"aggregate": metrics(np.ones(len(delta), dtype=bool))}
    output.update({scope: metrics(scope_values == scope) for scope in _SCOPES})
    return output


def build_grouped_queries(seed: int, count: int = 40) -> dict[str, np.ndarray]:
    """Build a deterministic, user-independent nested bank of analytic path pairs."""
    if type(seed) is not int or seed < 0 or type(count) is not int or count <= 0:
        raise ValueError("seed must be nonnegative and count positive integers")
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 1.0, _HORIZON + 1, dtype=np.float32)
    trajectories = np.empty((count, 2, _HORIZON + 1, 2), dtype=np.float32)
    scopes = np.array([_SCOPE_PATTERN[index % 10] for index in range(count)])
    for index, scope in enumerate(scopes):
        sign = float(rng.choice((-1, 1)))
        narrow = float(rng.uniform(0.22, 0.32))
        wide = float(rng.uniform(0.48, 0.58))
        if scope == "direction":
            amplitude = float(rng.choice((narrow, wide)))
            pair = ((sign, amplitude), (-sign, amplitude))
        elif scope == "width":
            pair = ((sign, narrow), (sign, wide))
        else:
            pair = ((sign, narrow), (-sign, wide))
        if rng.integers(2):
            pair = pair[::-1]
        for option, (direction, amplitude) in enumerate(pair):
            trajectories[index, option] = evaluate_path(direction, amplitude, times)[0]
    ids = np.array(
        [[f"grouped-query-{seed}-{index}-{option}" for option in range(2)]
         for index in range(count)]
    )
    return {"trajectories": trajectories, "ids": ids, "scopes": scopes}


def synthetic_grouped_users() -> dict[str, GroupedPreference]:
    """Return the four geometric modes crossed with two group priorities."""
    users = {}
    modes = (
        ("upper_narrow", 0.9, 0.1),
        ("upper_wide", 0.9, 0.9),
        ("lower_narrow", 0.1, 0.1),
        ("lower_wide", 0.1, 0.9),
    )
    for mode, direction, width in modes:
        weights = np.array(
            [[direction, 1.0 - direction], [width, 1.0 - width]], dtype=np.float64
        )
        for priority, alpha in (
            ("direction", np.array([0.8, 0.2], dtype=np.float64)),
            ("width", np.array([0.2, 0.8], dtype=np.float64)),
        ):
            users[f"{mode}_{priority}"] = GroupedPreference(weights.copy(), alpha.copy())
    return users
