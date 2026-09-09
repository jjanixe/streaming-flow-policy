"""Toy conditional velocity model used by the SFPD policy."""

from __future__ import annotations

import torch
from torch import nn


class LocalTimeFeatures(nn.Module):
    """Encode normalized time with fixed Fourier features."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * torch.pi * timestep[:, None] * self.frequencies[None, :]
        return torch.cat(
            (timestep[:, None], torch.sin(angles), torch.cos(angles)), dim=1
        )


class SFPDVelocityMLP(nn.Module):
    """Predict a 2-D velocity conditioned on state, time, and observations."""

    def __init__(self, hidden_dim: int = 128, hidden_layers: int = 3) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_layers <= 0:
            raise ValueError("hidden_dim and hidden_layers must be positive")
        dimensions = [2 + 9 + 6] + [hidden_dim] * hidden_layers + [2]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.time_features = LocalTimeFeatures()
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        tensors = {"sample": sample, "timestep": timestep, "global_cond": global_cond}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if value.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype float32")

        if sample.ndim != 3 or sample.shape[1:] != (1, 2):
            raise ValueError(f"sample shape must be [B, 1, 2], got {tuple(sample.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"timestep shape must be [B], got {tuple(timestep.shape)}")
        if global_cond.ndim != 2 or global_cond.shape[1] != 6:
            raise ValueError(
                f"global_cond shape must be [B, 6], got {tuple(global_cond.shape)}"
            )
        batch_size = sample.shape[0]
        if timestep.shape[0] != batch_size or global_cond.shape[0] != batch_size:
            raise ValueError("sample, timestep, and global_cond must have the same batch size")

        devices = {sample.device, timestep.device, global_cond.device}
        parameter = next(self.parameters())
        devices.add(parameter.device)
        if len(devices) != 1:
            raise ValueError("sample, timestep, global_cond, and model must share a device")
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("sample, timestep, and global_cond must contain finite values")
        if torch.any((timestep < 0.0) | (timestep > 1.0)):
            raise ValueError("timestep values must be in the range [0, 1]")

        features = torch.cat(
            (sample[:, 0, :], self.time_features(timestep), global_cond), dim=1
        )
        return self.network(features).unsqueeze(1)
