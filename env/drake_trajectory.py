"""Drake-backed trajectory targets for normalized PushT action windows."""

import math
from typing import Any

import numpy as np
import torch


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


def evaluate_uniform_foh_batch(
    actions: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Drake-equivalent uniform 16-knot FOHs as one Torch batch."""
    if not isinstance(actions, torch.Tensor) or actions.shape[1:] != (16, 2):
        raise ValueError("actions must be a torch.Tensor with shape [B, 16, 2]")
    if actions.dtype != torch.float32:
        raise ValueError("actions must use float32")
    if not isinstance(times, torch.Tensor) or times.shape != (actions.shape[0],):
        raise ValueError("times must be a torch.Tensor with shape [B]")
    if times.dtype != torch.float32:
        raise ValueError("times must use float32")
    if times.device != actions.device:
        raise ValueError("actions and times must share a device")
    if not torch.isfinite(actions).all() or not torch.isfinite(times).all():
        raise ValueError("actions and times must contain finite values")
    if torch.any((times < 0.0) | (times > 1.0)):
        raise ValueError("times must be in [0, 1]")

    scaled_times = times * 15.0
    lower_indices = torch.floor(scaled_times).to(torch.int64).clamp(max=14)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    left = actions[batch_indices, lower_indices]
    right = actions[batch_indices, lower_indices + 1]
    alpha = (scaled_times - lower_indices.to(torch.float32)).unsqueeze(1)
    positions = ((1.0 - alpha) * left + alpha * right).unsqueeze(1)
    derivatives = (15.0 * (right - left)).unsqueeze(1)
    return positions, derivatives


def sample_sfps_training_targets(
    actions: torch.Tensor,
    *,
    sigma0: float,
    sigma1: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample one vectorized SFPS target per normalized action window."""
    if not math.isfinite(sigma0) or not math.isfinite(sigma1):
        raise ValueError("sigma0 and sigma1 must be finite")
    if not 0.0 <= sigma0 <= sigma1:
        raise ValueError("sigma0 and sigma1 must satisfy 0 <= sigma0 <= sigma1")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    if not isinstance(actions, torch.Tensor):
        raise ValueError("actions must be a torch.Tensor")
    if torch.device(generator.device) != actions.device:
        raise ValueError("generator and actions must share a device")

    batch_size = actions.shape[0]
    times = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=actions.device,
        generator=generator,
    )
    xi, xi_dot = evaluate_uniform_foh_batch(actions, times)
    z0 = torch.randn(
        (batch_size, 1, 2),
        dtype=torch.float32,
        device=actions.device,
        generator=generator,
    )
    epsilon_a0 = float(sigma0) * torch.randn(
        (batch_size, 1, 2),
        dtype=torch.float32,
        device=actions.device,
        generator=generator,
    )
    sigma_r = math.sqrt(sigma1**2 - sigma0**2)
    expanded_times = times[:, None, None]
    action = xi + epsilon_a0 + sigma_r * expanded_times * z0
    latent = (1.0 - (1.0 - sigma1) * expanded_times) * z0 + expanded_times * xi
    action_velocity = xi_dot + sigma_r * z0
    latent_velocity = xi + expanded_times * xi_dot - (1.0 - sigma1) * z0
    return {
        "a": action,
        "z": latent,
        "va": action_velocity,
        "vz": latent_velocity,
        "t": times,
    }


def sample_sfpd_training_targets(
    actions: torch.Tensor,
    *,
    sigma: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample one vectorized SFPD target per normalized action window."""
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    if not isinstance(actions, torch.Tensor):
        raise ValueError("actions must be a torch.Tensor")
    if torch.device(generator.device) != actions.device:
        raise ValueError("generator and actions must share a device")

    batch_size = actions.shape[0]
    times = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=actions.device,
        generator=generator,
    )
    xi, xi_dot = evaluate_uniform_foh_batch(actions, times)
    noise = torch.randn(
        (batch_size, 1, 2),
        dtype=torch.float32,
        device=actions.device,
        generator=generator,
    )
    return {
        "x": xi + float(sigma) * noise,
        "v": xi_dot,
        "t": times,
    }


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


class SFPSDrakeTransform:
    """Sample seeded joint action-latent SFPS targets from a Drake FOH."""

    def __init__(
        self,
        sigma0: float,
        sigma1: float,
        rng: np.random.Generator,
    ) -> None:
        if not np.isfinite(sigma0) or sigma0 < 0.0:
            raise ValueError("sigma0 must be finite and nonnegative")
        if not np.isfinite(sigma1) or sigma1 < sigma0:
            raise ValueError("sigma1 must be finite and at least sigma0")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a NumPy Generator")
        self.sigma0 = np.float32(sigma0)
        self.sigma1 = np.float32(sigma1)
        self.sigma_r = np.float32(
            np.sqrt(np.float32(self.sigma1**2 - self.sigma0**2))
        )
        self.rng = rng

    def __call__(self, datum: dict[str, Any]) -> dict[str, Any]:
        tau = np.float32(self.rng.random())
        xi, xi_dot = evaluate_drake_foh(datum["action"], tau)
        z0 = self.rng.standard_normal(xi.shape, dtype=np.float32)
        epsilon_a0 = self.sigma0 * self.rng.standard_normal(
            xi.shape,
            dtype=np.float32,
        )
        one = np.float32(1.0)
        action = xi + epsilon_a0 + self.sigma_r * tau * z0
        latent = (one - (one - self.sigma1) * tau) * z0 + tau * xi
        action_velocity = xi_dot + self.sigma_r * z0
        latent_velocity = xi + tau * xi_dot - (one - self.sigma1) * z0
        transformed = {key: value for key, value in datum.items() if key != "action"}
        transformed.update(
            a=action.astype(np.float32, copy=False),
            z=latent.astype(np.float32, copy=False),
            va=action_velocity.astype(np.float32, copy=False),
            vz=latent_velocity.astype(np.float32, copy=False),
            t=np.asarray(tau, dtype=np.float32),
        )
        return transformed
