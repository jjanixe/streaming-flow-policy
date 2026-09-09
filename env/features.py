from dataclasses import dataclass

import numpy as np

from env.config import StageAConfig
from env.demonstrations import DemonstrationBank


@dataclass(frozen=True)
class FeatureNormalizer:
    mu: np.ndarray
    scale: np.ndarray
    y_scale: float

    def __post_init__(self) -> None:
        if self.mu.shape != (2,) or self.scale.shape != (2,):
            raise ValueError("feature normalizer arrays must have shape (2,)")
        if self.mu.dtype != np.float32 or self.scale.dtype != np.float32:
            raise ValueError("feature normalizer arrays must use float32")
        if not np.isfinite(self.mu).all() or not np.isfinite(self.scale).all():
            raise ValueError("feature normalizer arrays must be finite")
        if np.any(self.scale <= 0.0):
            raise ValueError("feature normalizer scale must be positive")
        if not isinstance(self.y_scale, (float, int, np.floating)) or (
            not np.isfinite(self.y_scale) or self.y_scale <= 0.0
        ):
            raise ValueError("feature normalizer y_scale must be finite and positive")


def _positions(values: np.ndarray) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float32)
    if positions.ndim < 2 or positions.shape[-1] != 2:
        raise ValueError("positions must end with shape [time, 2]")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    return positions


def raw_step_features(
    positions: np.ndarray,
    y_scale: float,
) -> np.ndarray:
    values = _positions(positions)
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        raise ValueError("y_scale must be finite and positive")
    rho = np.tanh(values[..., 1] / np.float32(y_scale))
    return np.stack((rho, rho * rho), axis=-1).astype(np.float32)


def raw_trajectory_features(
    trajectories: np.ndarray,
    y_scale: float,
    dt: float,
) -> np.ndarray:
    positions = _positions(trajectories)
    if positions.shape[-2] < 2:
        raise ValueError("a trajectory must contain an initial state and one step")
    features = raw_step_features(positions[..., 1:, :], y_scale)
    return (
        np.float32(dt) * features.sum(axis=-2, dtype=np.float32)
    ).astype(np.float32)


def fit_feature_normalizer(
    bank: DemonstrationBank,
    config: StageAConfig,
) -> FeatureNormalizer:
    train = bank.select("train")
    if len(train) == 0:
        raise ValueError("the demonstration bank has no train trajectories")
    raw = raw_trajectory_features(
        train.positions,
        config.features.y_scale,
        1.0 / config.environment.horizon_steps,
    )
    mu = raw.mean(axis=0, dtype=np.float32).astype(np.float32)
    scale = raw.std(axis=0, ddof=0, dtype=np.float32).astype(np.float32)
    return FeatureNormalizer(
        mu=mu,
        scale=scale,
        y_scale=config.features.y_scale,
    )


def step_contributions(
    trajectories: np.ndarray,
    normalizer: FeatureNormalizer,
    dt: float,
) -> np.ndarray:
    positions = _positions(trajectories)
    if positions.shape[-2] < 2:
        raise ValueError("a trajectory must contain an initial state and one step")
    features = raw_step_features(
        positions[..., 1:, :],
        normalizer.y_scale,
    )
    centered = (features - normalizer.mu) / normalizer.scale
    return (np.float32(dt) * centered).astype(np.float32)


def trajectory_features(
    trajectories: np.ndarray,
    normalizer: FeatureNormalizer,
    dt: float,
) -> np.ndarray:
    contributions = step_contributions(trajectories, normalizer, dt)
    return contributions.sum(axis=-2, dtype=np.float32).astype(np.float32)


def trajectory_feature_parts(
    trajectories: np.ndarray,
    normalizer: FeatureNormalizer,
    prefix_steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    contributions = step_contributions(trajectories, normalizer, dt)
    if not 0 <= prefix_steps <= contributions.shape[-2]:
        raise ValueError("prefix_steps is outside the trajectory horizon")
    prefix = contributions[..., :prefix_steps, :].sum(
        axis=-2,
        dtype=np.float32,
    )
    future = contributions[..., prefix_steps:, :].sum(
        axis=-2,
        dtype=np.float32,
    )
    return prefix.astype(np.float32), future.astype(np.float32)
