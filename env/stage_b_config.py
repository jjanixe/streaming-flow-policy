from dataclasses import asdict, dataclass
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
        if (self.pred_horizon, self.obs_horizon, self.action_horizon) != (16, 2, 8):
            raise ValueError("Stage B1 requires PushT horizons 16/2/8")
        if not 0.0 <= self.sigma:
            raise ValueError("sigma must be nonnegative")
        if self.max_updates <= 0 or self.batch_size <= 0:
            raise ValueError("training counts must be positive")


DEFAULT_STAGE_B1_CONFIG = StageB1Config()


def stage_b1_config_to_dict(config: StageB1Config) -> dict[str, Any]:
    return asdict(config)
