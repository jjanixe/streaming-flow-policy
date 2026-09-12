"""Deterministic streaming flow policy for normalized PushT action chunks."""

from __future__ import annotations

from collections.abc import Mapping
import re
import sys
import types

import torch
from torch import nn

# Torchdyn imports torchvision through Lightning. The supplied Python runtime
# omits its optional ``_lzma`` extension, while torchvision only stores
# ``lzma.open`` for archive extraction during that import. Keep Torchdyn usable
# for ODE integration without changing the runtime's global lzma semantics.
_MISSING_MODULE = object()
_previous_lzma = sys.modules.get("lzma", _MISSING_MODULE)
try:
    import lzma
except ModuleNotFoundError as error:
    if error.name != "_lzma":
        raise
    _lzma_stub = types.ModuleType("lzma")

    def _unavailable_lzma_open(*args: object, **kwargs: object) -> object:
        raise RuntimeError("lzma support is unavailable in this Python runtime")

    _lzma_stub.open = _unavailable_lzma_open
    sys.modules["lzma"] = _lzma_stub
    try:
        from torchdyn.core import NeuralODE
    finally:
        if _previous_lzma is _MISSING_MODULE:
            sys.modules.pop("lzma", None)
        else:
            sys.modules["lzma"] = _previous_lzma
else:
    from torchdyn.core import NeuralODE


class PolicyNumericalError(RuntimeError):
    """A genuine non-finite or numerical-integration policy failure."""


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
        if not torch.isfinite(x).all():
            raise PolicyNumericalError("ODE state contains non-finite values")
        if not torch.isfinite(t).all():
            raise PolicyNumericalError("ODE time contains non-finite values")

        sample = x.reshape(1, 1, 2)
        timestep = t.reshape(1)
        velocity = self.velocity_net(
            sample=sample,
            timestep=timestep,
            global_cond=self.condition,
        )
        if not isinstance(velocity, torch.Tensor):
            raise ValueError("velocity network output must be a torch.Tensor")
        if velocity.shape != sample.shape:
            raise ValueError("velocity network output shape must match ODE sample")
        if velocity.dtype != torch.float32:
            raise ValueError("velocity network output must have dtype float32")
        if velocity.device != x.device:
            raise ValueError("velocity network output must share the ODE state device")
        if not torch.isfinite(velocity).all():
            raise PolicyNumericalError(
                "velocity network output contains non-finite values"
            )
        return velocity.reshape(2)


class _JointConditionedVectorField(nn.Module):
    """Adapt a joint action-latent velocity model to Torchdyn's ODE interface."""

    def __init__(self, velocity_net: nn.Module, condition: torch.Tensor) -> None:
        super().__init__()
        self.velocity_net = velocity_net
        self.register_buffer("condition", condition)

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.dtype != torch.float32 or t.dtype != torch.float32:
            raise ValueError("ODE state and time must have dtype float32")
        if x.ndim != 3 or x.shape[1:] != (2, 2):
            raise ValueError("joint ODE state must have shape [B, 2, 2]")
        if t.numel() != 1:
            raise ValueError("ODE time must be scalar")
        if not torch.isfinite(x).all():
            raise PolicyNumericalError("ODE state contains non-finite values")
        if not torch.isfinite(t).all():
            raise PolicyNumericalError("ODE time contains non-finite values")
        velocity = self.velocity_net(
            sample=x,
            timestep=t.reshape(1).expand(x.shape[0]),
            global_cond=self.condition,
        )
        if not isinstance(velocity, torch.Tensor):
            raise ValueError("velocity network output must be a torch.Tensor")
        if velocity.shape != x.shape:
            raise ValueError("velocity network output shape must match ODE sample")
        if velocity.dtype != torch.float32:
            raise ValueError("velocity network output must have dtype float32")
        if velocity.device != x.device:
            raise ValueError("velocity network output must share the ODE state device")
        if not torch.isfinite(velocity).all():
            raise PolicyNumericalError(
                "velocity network output contains non-finite values"
            )
        return velocity


def _is_numerical_solver_runtime_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return re.search(
        r"\b(?:non[- ]?finite|nan|inf|infinite|infinity|overflow|underflow)\b"
        r"|\b(?:failed|fails|unable) to converge\b"
        r"|\bdid not converge\b"
        r"|\bconvergence (?:failure|failed|error)\b"
        r"|\bstep ?size (?:underflow|became too small|is too small)\b",
        message,
    ) is not None


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
        if num_actions == 1:
            return nobs[-1, :2].reshape(1, 1, 2)

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
        try:
            trajectory = solver.trajectory(x=nobs[-1, :2], t_span=t_span)
        except PolicyNumericalError:
            raise
        except (FloatingPointError, OverflowError) as error:
            raise PolicyNumericalError(str(error)) from error
        except RuntimeError as error:
            if _is_numerical_solver_runtime_error(error):
                raise PolicyNumericalError(str(error)) from error
            raise
        if not isinstance(trajectory, torch.Tensor):
            raise ValueError("ODE trajectory must be a torch.Tensor")
        if trajectory.shape != (total_steps, 2):
            raise ValueError(
                f"ODE trajectory shape must be {(total_steps, 2)}, "
                f"got {tuple(trajectory.shape)}"
            )
        if trajectory.dtype != torch.float32:
            raise ValueError("ODE trajectory must have dtype float32")
        if trajectory.device != nobs.device:
            raise ValueError("ODE trajectory must share the policy input device")
        if not torch.isfinite(trajectory).all():
            raise PolicyNumericalError("ODE trajectory contains non-finite values")
        indices = torch.arange(
            0,
            total_steps,
            integration_steps_per_action,
            device=nobs.device,
        )
        return trajectory[indices].unsqueeze(0)


class StreamingFlowPolicyStochastic(nn.Module):
    """Learn and integrate the repository-style joint action-latent SFPS field."""

    def __init__(
        self,
        velocity_net: nn.Module,
        pred_horizon: int = 16,
        sigma0: float = 0.1,
        sigma1: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if pred_horizon != 16:
            raise ValueError("pred_horizon must be 16")
        if not 0.0 <= sigma0 <= sigma1:
            raise ValueError("sigma0 and sigma1 must satisfy 0 <= sigma0 <= sigma1")
        self.velocity_net = velocity_net
        self.device = torch.device(device)
        sigma_r = (sigma1**2 - sigma0**2) ** 0.5
        self.register_buffer("pred_horizon", torch.tensor(16, dtype=torch.int32))
        self.register_buffer("sigma0", torch.tensor(sigma0, dtype=torch.float32))
        self.register_buffer("sigma1", torch.tensor(sigma1, dtype=torch.float32))
        self.register_buffer("sigma_r", torch.tensor(sigma_r, dtype=torch.float32))

    def _validate_model_device(self, device: torch.device) -> None:
        state = [*self.velocity_net.parameters(), *self.velocity_net.buffers()]
        devices = {value.device for value in state}
        if len(devices) > 1:
            raise ValueError("velocity network state must share a device")
        if devices and devices != {device}:
            raise ValueError("velocity network and inputs must share a device")
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in state
        ):
            raise ValueError("velocity network state must have dtype float32")

    @staticmethod
    def _validate_loss_batch(batch: Mapping[str, torch.Tensor]) -> None:
        names = ("obs", "a", "z", "va", "vz", "t")
        tensors = {name: _require_float32_tensor(batch[name], name) for name in names}
        if tensors["obs"].ndim != 3 or tensors["obs"].shape[1:] != (2, 3):
            raise ValueError("obs shape must be [B, 2, 3]")
        for name in ("a", "z", "va", "vz"):
            if tensors[name].ndim != 3 or tensors[name].shape[1:] != (1, 2):
                raise ValueError(f"{name} shape must be [B, 1, 2]")
        batch_size = tensors["obs"].shape[0]
        if tensors["t"].shape != (batch_size,):
            raise ValueError("t shape must be [B]")
        if any(value.shape[0] != batch_size for value in tensors.values()):
            raise ValueError("obs, joint state, velocities, and t must share a batch size")
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("SFPS loss batch must contain finite values")

    def loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self._validate_loss_batch(batch)
        self._validate_model_device(self.device)
        obs = batch["obs"].to(self.device)
        joint_state = torch.cat((batch["a"], batch["z"]), dim=1).to(self.device)
        target = torch.cat((batch["va"], batch["vz"]), dim=1).to(self.device)
        timestep = batch["t"].to(self.device)
        prediction = self.velocity_net(
            joint_state,
            timestep,
            obs.flatten(start_dim=1),
        )
        if prediction.dtype != torch.float32:
            raise ValueError("velocity network output must have dtype float32")
        return torch.nn.functional.mse_loss(prediction, target)

    def sample_latent(self, generator: torch.Generator) -> torch.Tensor:
        if not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator")
        if torch.device(generator.device) != self.device:
            raise ValueError("generator and policy must share a device")
        return torch.randn(
            (2,),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

    @torch.inference_mode()
    def predict(
        self,
        nobs: torch.Tensor,
        num_actions: int,
        integration_steps_per_action: int,
        *,
        generator: torch.Generator | None = None,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if num_actions == 1:
            return nobs[-1, :2].reshape(1, 1, 2)
        if (generator is None) == (latent is None):
            raise ValueError("provide exactly one of generator or latent")
        if latent is None:
            if generator is None:
                raise AssertionError("validated generator is unavailable")
            latent = self.sample_latent(generator)
        else:
            _require_float32_tensor(latent, "latent")
            if latent.shape != (2,):
                raise ValueError("latent shape must be [2]")
            if latent.device != nobs.device:
                raise ValueError("latent and nobs must share a device")
            if not torch.isfinite(latent).all():
                raise ValueError("latent must contain finite values")

        return self.predict_batch(
            nobs.unsqueeze(0),
            num_actions=num_actions,
            integration_steps_per_action=integration_steps_per_action,
            latents=latent.unsqueeze(0),
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        nobs: torch.Tensor,
        num_actions: int,
        integration_steps_per_action: int,
        *,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate multiple explicit action-latent initial states together."""
        _require_float32_tensor(nobs, "nobs")
        _require_float32_tensor(latents, "latents")
        if nobs.ndim != 3 or nobs.shape[1:] != (2, 3):
            raise ValueError("nobs shape must be [B, 2, 3]")
        if latents.shape != (nobs.shape[0], 2):
            raise ValueError("latents shape must be [B, 2]")
        if not torch.isfinite(nobs).all() or not torch.isfinite(latents).all():
            raise ValueError("nobs and latents must contain finite values")
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
        if nobs.device != self.device or latents.device != self.device:
            raise ValueError("nobs, latents, and policy must share a device")
        self._validate_model_device(nobs.device)
        if num_actions == 1:
            return nobs[:, -1:, :2]

        condition = nobs.flatten(start_dim=1)
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
            _JointConditionedVectorField(self.velocity_net, condition),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )
        initial_state = torch.stack((nobs[:, -1, :2], latents), dim=1)
        try:
            trajectory = solver.trajectory(x=initial_state, t_span=t_span)
        except PolicyNumericalError:
            raise
        except (FloatingPointError, OverflowError) as error:
            raise PolicyNumericalError(str(error)) from error
        except RuntimeError as error:
            if _is_numerical_solver_runtime_error(error):
                raise PolicyNumericalError(str(error)) from error
            raise
        expected_shape = (total_steps, nobs.shape[0], 2, 2)
        if not isinstance(trajectory, torch.Tensor):
            raise ValueError("ODE trajectory must be a torch.Tensor")
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"ODE trajectory shape must be {expected_shape}, got {tuple(trajectory.shape)}"
            )
        if trajectory.dtype != torch.float32:
            raise ValueError("ODE trajectory must have dtype float32")
        if trajectory.device != nobs.device:
            raise ValueError("ODE trajectory must share the policy input device")
        if not torch.isfinite(trajectory).all():
            raise PolicyNumericalError("ODE trajectory contains non-finite values")
        indices = torch.arange(
            0,
            total_steps,
            integration_steps_per_action,
            device=nobs.device,
        )
        return trajectory[indices, :, 0, :].permute(1, 0, 2)
