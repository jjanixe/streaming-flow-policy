"""Deterministic streaming flow policy for normalized PushT action chunks."""

from __future__ import annotations

from collections.abc import Mapping
import sys
import types

import torch
from torch import nn

# Torchdyn imports torchvision through Lightning.  The supplied Python runtime
# omits its optional ``_lzma`` extension, while torchvision only stores
# ``lzma.open`` for archive extraction during that import.  Keep Torchdyn
# usable for ODE integration and fail clearly if that unrelated feature is used.
try:
    import lzma as _lzma
except ModuleNotFoundError:
    _lzma = types.ModuleType("lzma")

    def _unavailable_lzma_open(*args: object, **kwargs: object) -> object:
        raise RuntimeError("lzma support is unavailable in this Python runtime")

    _lzma.open = _unavailable_lzma_open
    sys.modules["lzma"] = _lzma

from torchdyn.core import NeuralODE


def _require_float32_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32")
    return value


class _ConditionedVectorField(nn.Module):
    """Adapt a batched conditional velocity model to Torchdyn's ODE interface."""

    def __init__(self, velocity_net: nn.Module, condition: torch.Tensor) -> None:
        super().__init__()
        self.velocity_net = velocity_net
        self.register_buffer("condition", condition)

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.dtype != torch.float32 or t.dtype != torch.float32:
            raise ValueError("ODE state and time must have dtype float32")
        if x.numel() != 2:
            raise ValueError("ODE state must contain exactly two action values")
        if t.numel() != 1:
            raise ValueError("ODE time must be scalar")

        sample = x.reshape(1, 1, 2)
        timestep = t.reshape(1)
        velocity = self.velocity_net(
            sample=sample,
            timestep=timestep,
            global_cond=self.condition,
        )
        if velocity.dtype != torch.float32:
            raise ValueError("velocity network output must have dtype float32")
        return velocity.reshape(2)


class StreamingFlowPolicyDeterministic(nn.Module):
    """Learn and integrate a deterministic conditional PushT velocity field."""

    def __init__(
        self,
        velocity_net: nn.Module,
        pred_horizon: int = 16,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if pred_horizon != 16:
            raise ValueError("pred_horizon must be 16")
        self.velocity_net = velocity_net
        self.device = torch.device(device)
        self.register_buffer("pred_horizon", torch.tensor(16, dtype=torch.int32))

    def _validate_model_device(self, device: torch.device) -> None:
        model_state = [*self.velocity_net.parameters(), *self.velocity_net.buffers()]
        devices = {value.device for value in model_state}
        if len(devices) > 1:
            raise ValueError("velocity network state must share a device")
        if devices and devices != {device}:
            raise ValueError("velocity network and inputs must share a device")
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in model_state
        ):
            raise ValueError("velocity network state must have dtype float32")

    @staticmethod
    def _validate_loss_batch(batch: Mapping[str, torch.Tensor]) -> None:
        tensors = {name: _require_float32_tensor(batch[name], name) for name in ("obs", "x", "v", "t")}
        if tensors["obs"].ndim != 3 or tensors["obs"].shape[1:] != (2, 3):
            raise ValueError("obs shape must be [B, 2, 3]")
        if tensors["x"].ndim != 3 or tensors["x"].shape[1:] != (1, 2):
            raise ValueError("x shape must be [B, 1, 2]")
        if tensors["v"].shape != tensors["x"].shape:
            raise ValueError("v shape must match x shape")
        if tensors["t"].ndim != 1 or tensors["t"].shape[0] != tensors["x"].shape[0]:
            raise ValueError("t shape must be [B]")
        if tensors["obs"].shape[0] != tensors["x"].shape[0]:
            raise ValueError("obs, x, v, and t must share a batch size")
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("obs, x, v, and t must contain finite values")

    def loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the float32 velocity mean-squared error for one training batch."""
        self._validate_loss_batch(batch)
        self._validate_model_device(self.device)
        obs = batch["obs"].to(self.device)
        x = batch["x"].to(self.device)
        target = batch["v"].to(self.device)
        timestep = batch["t"].to(self.device)
        condition = obs.flatten(start_dim=1)
        prediction = self.velocity_net(x, timestep, condition)
        if prediction.dtype != torch.float32:
            raise ValueError("velocity network output must have dtype float32")
        return torch.nn.functional.mse_loss(prediction, target)

    @torch.inference_mode()
    def predict(
        self,
        nobs: torch.Tensor,
        num_actions: int,
        integration_steps_per_action: int,
    ) -> torch.Tensor:
        """Integrate the field from the current normalized observation anchor."""
        _require_float32_tensor(nobs, "nobs")
        if nobs.shape != (2, 3):
            raise ValueError("nobs shape must be [2, 3]")
        if not torch.isfinite(nobs).all():
            raise ValueError("nobs must contain finite values")
        if not isinstance(num_actions, int) or isinstance(num_actions, bool):
            raise ValueError("num_actions must be an integer")
        if not 1 <= num_actions <= self.pred_horizon.item():
            raise ValueError("num_actions must be in [1, 16]")
        if (
            not isinstance(integration_steps_per_action, int)
            or isinstance(integration_steps_per_action, bool)
            or integration_steps_per_action <= 0
        ):
            raise ValueError("integration_steps_per_action must be positive")
        if nobs.device != self.device:
            raise ValueError("nobs and policy must share a device")
        self._validate_model_device(nobs.device)

        condition = nobs.unsqueeze(0).flatten(start_dim=1)
        num_future_actions = num_actions - 1
        t_max = num_future_actions / (self.pred_horizon.item() - 1)
        total_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(
            0.0,
            t_max,
            total_steps,
            dtype=torch.float32,
            device=nobs.device,
        )
        solver = NeuralODE(
            _ConditionedVectorField(self.velocity_net, condition),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )
        trajectory = solver.trajectory(x=nobs[-1, :2], t_span=t_span)
        if trajectory.dtype != torch.float32:
            raise ValueError("ODE trajectory must have dtype float32")
        indices = torch.arange(
            0,
            total_steps,
            integration_steps_per_action,
            device=nobs.device,
        )
        return trajectory[indices].unsqueeze(0)
