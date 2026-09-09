from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    start: tuple[float, float] = (-1.0, 0.0)
    goal: tuple[float, float] = (1.0, 0.0)
    physical_duration_s: float = 2.0
    horizon_steps: int = 64
    initial_sigma: float = 0.04
    goal_tolerance: float = 0.10
    visualization_x: tuple[float, float] = (-1.3, 1.3)
    visualization_y: tuple[float, float] = (-0.9, 0.9)
    max_step_distance: float = 0.075

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.max_step_distance)
            or self.max_step_distance <= 0.0
        ):
            raise ValueError("max_step_distance must be finite and positive")

    def start_array(self) -> np.ndarray:
        return np.asarray(self.start, dtype=np.float32)

    def goal_array(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=np.float32)


@dataclass(frozen=True)
class DemonstrationConfig:
    train_per_mode: int = 128
    validation_per_mode: int = 32
    test_per_mode: int = 32
    amplitude_narrow: tuple[float, float] = (0.22, 0.32)
    amplitude_wide: tuple[float, float] = (0.48, 0.58)


@dataclass(frozen=True)
class FieldConfig:
    stabilization_k: float = 1.0
    diffusion_kappa: float = 4.0


@dataclass(frozen=True)
class FeatureConfig:
    y_scale: float = 0.35


@dataclass(frozen=True)
class StageAConfig:
    environment: EnvironmentConfig = dataclass_field(default_factory=EnvironmentConfig)
    demonstrations: DemonstrationConfig = dataclass_field(
        default_factory=DemonstrationConfig
    )
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    features: FeatureConfig = dataclass_field(default_factory=FeatureConfig)
    root_seed: int = 0
    diagnostic_rollouts: int = 4096


DEFAULT_CONFIG = StageAConfig()


def config_to_dict(config: StageAConfig) -> dict[str, Any]:
    return asdict(config)
