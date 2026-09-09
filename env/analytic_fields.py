import math

import numpy as np
import torch

from env.config import StageAConfig
from env.demonstrations import DemonstrationBank


class AnalyticField:
    def __init__(
        self,
        train_bank: DemonstrationBank,
        config: StageAConfig,
    ) -> None:
        if len(train_bank) == 0 or not np.all(train_bank.splits == "train"):
            raise ValueError("AnalyticField requires a non-empty train-only bank")
        self.signs = torch.from_numpy(train_bank.signs.copy())
        self.amplitudes = torch.from_numpy(train_bank.amplitudes.copy())
        self.num_components = len(train_bank)
        self.initial_sigma = config.environment.initial_sigma
        self.k = config.field.stabilization_k
        self.kappa = config.field.diffusion_kappa

    @staticmethod
    def _validate_times(times: torch.Tensor) -> torch.Tensor:
        if not isinstance(times, torch.Tensor) or times.ndim != 1:
            raise ValueError("times must be a Torch tensor with shape [B]")
        if times.dtype != torch.float32:
            raise ValueError("times must use float32")
        if not torch.isfinite(times).all():
            raise ValueError("times must be finite")
        if torch.any(times < 0.0) or torch.any(times > 1.0):
            raise ValueError("times must lie in [0, 1]")
        if times.device.type != "cpu":
            raise ValueError("Stage A analytic fields run on CPU")
        return times

    @classmethod
    def _validate_inputs(
        cls,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(actions, torch.Tensor) or (
            actions.ndim != 2 or actions.shape[-1] != 2
        ):
            raise ValueError("actions must be a Torch tensor with shape [B, 2]")
        if actions.dtype != torch.float32:
            raise ValueError("actions must use float32")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        if actions.device.type != "cpu":
            raise ValueError("Stage A analytic fields run on CPU")
        valid_times = cls._validate_times(times)
        if len(actions) != len(valid_times):
            raise ValueError("actions and times must have the same batch size")
        return actions, valid_times

    def sigma(self, times: torch.Tensor) -> torch.Tensor:
        valid_times = self._validate_times(times)
        return (
            torch.tensor(self.initial_sigma, dtype=torch.float32)
            * torch.exp(-self.k * valid_times)
        )

    def epsilon(self, times: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma(times)
        return self.kappa * sigma.square()

    def centers_and_derivatives(
        self,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_times = self._validate_times(times)
        t = valid_times[:, None]
        progress = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        lateral = 16.0 * t**2 * (1.0 - t) ** 2
        progress_derivative = 60.0 * t**2 * (1.0 - t) ** 2
        lateral_derivative = 32.0 * t * (1.0 - t) * (1.0 - 2.0 * t)

        signed_amplitudes = self.signs * self.amplitudes
        x = (-1.0 + 2.0 * progress).expand(-1, self.num_components)
        y = lateral * signed_amplitudes[None, :]
        dx = progress_derivative.expand(-1, self.num_components)
        dy = lateral_derivative * signed_amplitudes[None, :]
        centers = torch.stack((x, y), dim=-1)
        derivatives = torch.stack((dx, dy), dim=-1)
        return centers.to(torch.float32), derivatives.to(torch.float32)

    def _logits(
        self,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers, derivatives = self.centers_and_derivatives(times)
        sigma = self.sigma(times)
        delta = actions[:, None, :] - centers
        logits = -0.5 * (
            delta / sigma[:, None, None]
        ).square().sum(dim=-1)
        return logits, centers, derivatives

    def responsibilities(
        self,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        valid_actions, valid_times = self._validate_inputs(actions, times)
        logits, _, _ = self._logits(valid_actions, valid_times)
        return torch.softmax(logits, dim=-1)

    def log_density(
        self,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        valid_actions, valid_times = self._validate_inputs(actions, times)
        logits, _, _ = self._logits(valid_actions, valid_times)
        sigma = self.sigma(valid_times)
        two_pi = torch.tensor(2.0 * math.pi, dtype=torch.float32)
        log_normalizer = torch.log(two_pi * sigma.square())
        return (
            torch.logsumexp(logits, dim=-1)
            - math.log(self.num_components)
            - log_normalizer
        )

    def velocity_pf(
        self,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        valid_actions, valid_times = self._validate_inputs(actions, times)
        logits, centers, derivatives = self._logits(
            valid_actions,
            valid_times,
        )
        weights = torch.softmax(logits, dim=-1)
        conditional = derivatives - self.k * (
            valid_actions[:, None, :] - centers
        )
        return (weights[:, :, None] * conditional).sum(dim=1)

    def score(
        self,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        valid_actions, valid_times = self._validate_inputs(actions, times)
        logits, centers, _ = self._logits(valid_actions, valid_times)
        weights = torch.softmax(logits, dim=-1)
        sigma_squared = self.sigma(valid_times).square()[:, None, None]
        conditional = -(valid_actions[:, None, :] - centers) / sigma_squared
        return (weights[:, :, None] * conditional).sum(dim=1)

    def base_sde_drift(
        self,
        actions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.velocity_pf(actions, times)
            + self.epsilon(times)[:, None] * self.score(actions, times)
        )

    def sample_marginal(
        self,
        times: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        valid_times = self._validate_times(times)
        centers, _ = self.centers_and_derivatives(valid_times)
        indices = torch.randint(
            self.num_components,
            (len(valid_times),),
            generator=generator,
        )
        selected = centers[torch.arange(len(valid_times)), indices]
        noise = torch.randn(
            (len(valid_times), 2),
            dtype=torch.float32,
            generator=generator,
        )
        return selected + self.sigma(valid_times)[:, None] * noise
