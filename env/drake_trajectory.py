"""Drake-backed trajectory targets for normalized PushT action windows."""

from typing import Any

import numpy as np


def _validate_action(action: np.ndarray) -> None:
    if not isinstance(action, np.ndarray):
        raise ValueError("action must be a NumPy array")
    if action.shape != (16, 2):
        raise ValueError("action must have shape (16, 2)")
    if action.dtype != np.float32:
        raise ValueError("action must use float32")
    if not np.isfinite(action).all():
        raise ValueError("action must contain finite values")


def _validate_tau(tau: np.float32) -> None:
    if not isinstance(tau, np.float32):
        raise ValueError("tau must be a float32 scalar")
    if not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and in [0, 1]")


def evaluate_drake_foh(
    action: np.ndarray,
    tau: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 16-knot Drake first-order hold at normalized time ``tau``."""
    _validate_action(action)
    _validate_tau(tau)
    try:
        from pydrake.trajectories import PiecewisePolynomial
    except ImportError as error:
        raise RuntimeError("Drake FirstOrderHold dependency is unavailable") from error

    try:
        breaks = np.linspace(0.0, 1.0, 16, dtype=np.float64)
        trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks,
            action.astype(np.float64, copy=False).T,
        )
        position = trajectory.value(float(tau)).T.astype(np.float32)
        derivative = trajectory.EvalDerivative(float(tau)).T.astype(np.float32)
    except Exception as error:
        raise RuntimeError("Drake FirstOrderHold evaluation failed") from error
    return position, derivative


class SFPDDrakeTransform:
    """Sample noisy SFPD targets from a Drake first-order-hold trajectory."""

    def __init__(self, sigma: float, rng: np.random.Generator) -> None:
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and nonnegative")
        self.sigma = np.float32(sigma)
        self.rng = rng

    def __call__(self, datum: dict[str, Any]) -> dict[str, Any]:
        tau = np.float32(self.rng.random())
        xi, xi_dot = evaluate_drake_foh(datum["action"], tau)
        noise = self.rng.standard_normal(xi.shape, dtype=np.float32)
        transformed = {key: value for key, value in datum.items() if key != "action"}
        transformed.update(
            x=(xi + self.sigma * noise).astype(np.float32),
            v=xi_dot.astype(np.float32, copy=False),
            t=np.asarray(tau, dtype=np.float32),
        )
        return transformed
