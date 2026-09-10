from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class StageB1Config:
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    sigma: float = 0.1
    hidden_dim: int = 128
    hidden_layers: int = 3
    batch_size: int = 1024
    max_updates: int = 20_000
    validation_interval: int = 1_000
    warmup_updates: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    ema_decay: float = 0.999
    integration_steps_per_action: int = 6
    rollout_count: int = 1024

    def __post_init__(self) -> None:
        integer_fields = (
            "pred_horizon",
            "obs_horizon",
            "action_horizon",
            "hidden_dim",
            "hidden_layers",
            "batch_size",
            "max_updates",
            "validation_interval",
            "warmup_updates",
            "integration_steps_per_action",
            "rollout_count",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an int")
        float_fields = (
            "sigma",
            "learning_rate",
            "weight_decay",
            "ema_decay",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")

        if (self.pred_horizon, self.obs_horizon, self.action_horizon) != (16, 2, 8):
            changed = next(
                name
                for name, value, expected in zip(
                    ("pred_horizon", "obs_horizon", "action_horizon"),
                    (self.pred_horizon, self.obs_horizon, self.action_horizon),
                    (16, 2, 8),
                )
                if value != expected
            )
            raise ValueError(
                f"{changed} is incompatible; Stage B1 requires horizons 16/2/8"
            )
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        for name in (
            "hidden_dim",
            "hidden_layers",
            "batch_size",
            "max_updates",
            "validation_interval",
            "integration_steps_per_action",
            "rollout_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be nonnegative")
        if self.warmup_updates > self.max_updates:
            raise ValueError("warmup_updates cannot exceed max_updates")
        if self.rollout_count > 2**32:
            raise ValueError("rollout_count exceeds the uint32 seed space")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")


DEFAULT_STAGE_B1_CONFIG = StageB1Config()


def stage_b1_config_to_dict(config: StageB1Config) -> dict[str, Any]:
    return asdict(config)


@dataclass(frozen=True)
class StageB2Config:
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    latent_dim: int = 2
    sigma0: float = 0.1
    sigma1: float = 0.1
    hidden_dim: int = 128
    hidden_layers: int = 3
    batch_size: int = 1024
    max_updates: int = 20_000
    validation_interval: int = 1_000
    warmup_updates: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    ema_decay: float = 0.999
    integration_steps_per_action: int = 6
    rollout_count: int = 1024

    def __post_init__(self) -> None:
        integer_fields = (
            "pred_horizon",
            "obs_horizon",
            "action_horizon",
            "latent_dim",
            "hidden_dim",
            "hidden_layers",
            "batch_size",
            "max_updates",
            "validation_interval",
            "warmup_updates",
            "integration_steps_per_action",
            "rollout_count",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an int")
        float_fields = (
            "sigma0",
            "sigma1",
            "learning_rate",
            "weight_decay",
            "ema_decay",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")

        expected_horizons = (16, 2, 8)
        horizons = (self.pred_horizon, self.obs_horizon, self.action_horizon)
        if horizons != expected_horizons:
            changed = next(
                name
                for name, value, expected in zip(
                    ("pred_horizon", "obs_horizon", "action_horizon"),
                    horizons,
                    expected_horizons,
                )
                if value != expected
            )
            raise ValueError(
                f"{changed} is incompatible; Stage B2 requires horizons 16/2/8"
            )
        if self.latent_dim != 2:
            raise ValueError("latent_dim must equal the action dimension 2")
        if self.sigma0 < 0.0 or self.sigma0 > self.sigma1:
            raise ValueError("sigma0 and sigma1 must satisfy 0 <= sigma0 <= sigma1")
        for name in (
            "hidden_dim",
            "hidden_layers",
            "batch_size",
            "max_updates",
            "validation_interval",
            "integration_steps_per_action",
            "rollout_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be nonnegative")
        if self.warmup_updates > self.max_updates:
            raise ValueError("warmup_updates cannot exceed max_updates")
        if self.rollout_count > 2**32:
            raise ValueError("rollout_count exceeds the uint32 seed space")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")


DEFAULT_STAGE_B2_CONFIG = StageB2Config()


def stage_b2_config_to_dict(config: StageB2Config) -> dict[str, Any]:
    return asdict(config)
