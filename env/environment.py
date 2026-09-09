from typing import Any

import gym
import numpy as np
from gym import spaces

from env.config import DEFAULT_CONFIG, EnvironmentConfig
from env.rendering import PointReachRenderer


class PointReach2DPreferenceEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 32,
    }
    reward_range = (0.0, 0.0)

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        if render_mode not in (*self.metadata["render_modes"], None):
            raise ValueError(
                f"unsupported render_mode {render_mode!r}; expected "
                "'human', 'rgb_array', or None"
            )
        self.config = config or DEFAULT_CONFIG.environment
        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._position = self.config.start_array()
        self._trajectory = [self._position.copy()]
        self._renderer: PointReachRenderer | None = None
        self._step_index = 0
        self._done = False
        self._numerical_failure = False
        self._action_limit_failure = False
        self._action_attempt_count = 0
        self._action_limit_activation_count = 0
        self._requested_step_distance: float | None = None
        self._max_requested_step_distance = 0.0
        self._max_abs_x = float(abs(self._position[0]))
        self._max_abs_y = float(abs(self._position[1]))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        center_init = bool((options or {}).get("center_init", False))
        if center_init:
            self._position = self.config.start_array()
        else:
            noise = self.np_random.normal(
                0.0,
                self.config.initial_sigma,
                size=2,
            ).astype(np.float32)
            self._position = self.config.start_array() + noise
        self._trajectory = [self._position.copy()]
        self._step_index = 0
        self._done = False
        self._numerical_failure = False
        self._action_limit_failure = False
        self._action_attempt_count = 0
        self._action_limit_activation_count = 0
        self._requested_step_distance = None
        self._max_requested_step_distance = 0.0
        self._max_abs_x = float(abs(self._position[0]))
        self._max_abs_y = float(abs(self._position[1]))
        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("episode has already terminated")

        value = np.asarray(action, dtype=np.float32)
        if value.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {value.shape}")

        self._action_attempt_count += 1
        self._requested_step_distance = None
        if not np.isfinite(value).all():
            self._done = True
            self._numerical_failure = True
            observation = self._observation()
            info = self._info()
            if self.render_mode == "human":
                self.render()
            return observation, 0.0, True, False, info

        requested_step_distance_value = np.linalg.norm(value - self._position)
        requested_step_distance = float(requested_step_distance_value)
        self._requested_step_distance = requested_step_distance
        self._max_requested_step_distance = max(
            self._max_requested_step_distance,
            requested_step_distance,
        )
        if requested_step_distance_value > np.float32(
            self.config.max_step_distance
        ):
            self._done = True
            self._action_limit_failure = True
            self._action_limit_activation_count += 1
            observation = self._observation()
            info = self._info()
            if self.render_mode == "human":
                self.render()
            return observation, 0.0, True, False, info

        self._position = value.copy()
        self._trajectory.append(self._position.copy())
        self._step_index += 1
        self._max_abs_x = max(self._max_abs_x, float(abs(value[0])))
        self._max_abs_y = max(self._max_abs_y, float(abs(value[1])))
        self._done = self._step_index == self.config.horizon_steps
        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, 0.0, self._done, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = PointReachRenderer(self.config)
        trajectory = np.stack(self._trajectory).astype(np.float32, copy=False)
        return self._renderer.render(trajectory, self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observation(self) -> np.ndarray:
        normalized_time = np.float32(
            self._step_index / self.config.horizon_steps
        )
        return np.array(
            [self._position[0], self._position[1], normalized_time],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        goal_error = float(np.linalg.norm(self._position - self.config.goal_array()))
        x_low, x_high = self.config.visualization_x
        y_low, y_high = self.config.visualization_y
        outside_visualization = not (
            x_low <= float(self._position[0]) <= x_high
            and y_low <= float(self._position[1]) <= y_high
        )
        success = (
            self._done
            and not self._numerical_failure
            and not self._action_limit_failure
            and self._step_index == self.config.horizon_steps
            and goal_error <= self.config.goal_tolerance
        )
        action_limit_activation_rate = (
            self._action_limit_activation_count / self._action_attempt_count
            if self._action_attempt_count
            else 0.0
        )
        return {
            "step_index": self._step_index,
            "normalized_time": float(
                np.float32(self._step_index / self.config.horizon_steps)
            ),
            "goal_error": goal_error,
            "success": success,
            "numerical_failure": self._numerical_failure,
            "action_limit_failure": self._action_limit_failure,
            "action_attempt_count": self._action_attempt_count,
            "action_limit_activation_count": self._action_limit_activation_count,
            "action_limit_activation_rate": action_limit_activation_rate,
            "requested_step_distance": self._requested_step_distance,
            "max_requested_step_distance": self._max_requested_step_distance,
            "outside_visualization": outside_visualization,
            "max_abs_x": self._max_abs_x,
            "max_abs_y": self._max_abs_y,
        }
