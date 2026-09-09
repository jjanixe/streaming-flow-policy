import gym
import numpy as np
import pytest
import warnings
from gym.utils.env_checker import check_env

import env
from env.config import DEFAULT_CONFIG, EnvironmentConfig
from env.environment import PointReach2DPreferenceEnv


def _import_pygame_without_dependency_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Deprecated call to `pkg_resources.declare_namespace.*",
            category=DeprecationWarning,
        )
        import pygame
    return pygame


def test_gym_registration_and_api():
    wrapped = gym.make("PointReach2DPreference-v0")
    observation, info = wrapped.reset(seed=7, options={"center_init": True})

    np.testing.assert_array_equal(
        observation, np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert info["step_index"] == 0
    assert observation.dtype == np.float32
    wrapped.close()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Box.*")
        check_env(PointReach2DPreferenceEnv(), skip_render_check=True)


def test_module_prefixed_gym_registration():
    wrapped = gym.make("env:PointReach2DPreference-v0")
    observation, _ = wrapped.reset(seed=3, options={"center_init": True})

    assert wrapped.observation_space.contains(observation)
    wrapped.close()


def test_seeded_gaussian_reset_is_reproducible():
    instance = PointReach2DPreferenceEnv()

    first, _ = instance.reset(seed=11)
    second, _ = instance.reset(seed=11)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[:2], DEFAULT_CONFIG.environment.start_array())


def test_actions_within_limit_are_followed_exactly_until_horizon():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0, options={"center_init": True})
    positions = np.linspace(
        DEFAULT_CONFIG.environment.start_array(),
        DEFAULT_CONFIG.environment.goal_array(),
        DEFAULT_CONFIG.environment.horizon_steps + 1,
        dtype=np.float32,
    )[1:]

    for step, position in enumerate(positions):
        observation, reward, terminated, truncated, info = instance.step(position)
        assert reward == 0.0
        assert truncated is False
        assert terminated is (
            step == DEFAULT_CONFIG.environment.horizon_steps - 1
        )

    np.testing.assert_array_equal(observation[:2], positions[-1])
    assert info["success"] is True
    assert info["action_attempt_count"] == 64
    assert info["action_limit_activation_count"] == 0
    assert info["action_limit_activation_rate"] == 0.0


def test_action_above_limit_terminates_without_moving_or_clipping():
    instance = PointReach2DPreferenceEnv()
    initial, _ = instance.reset(seed=0, options={"center_init": True})
    requested = np.array([-0.9, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = instance.step(requested)

    np.testing.assert_array_equal(observation[:2], initial[:2])
    np.testing.assert_array_equal(instance._trajectory, initial[None, :2])
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["step_index"] == 0
    assert info["success"] is False
    assert info["action_limit_failure"] is True
    assert info["action_attempt_count"] == 1
    assert info["action_limit_activation_count"] == 1
    assert info["action_limit_activation_rate"] == 1.0
    assert info["requested_step_distance"] == pytest.approx(0.1)
    assert info["max_requested_step_distance"] == pytest.approx(0.1)


def test_action_limit_metrics_accumulate_and_reset():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0, options={"center_init": True})
    _, _, terminated, _, info = instance.step(
        np.array([-0.95, 0.0], dtype=np.float32)
    )

    assert terminated is False
    assert info["action_attempt_count"] == 1
    assert info["action_limit_activation_count"] == 0
    assert info["action_limit_activation_rate"] == 0.0
    assert info["requested_step_distance"] == pytest.approx(0.05)

    _, _, terminated, _, info = instance.step(
        np.array([-0.85, 0.0], dtype=np.float32)
    )

    assert terminated is True
    assert info["action_attempt_count"] == 2
    assert info["action_limit_activation_count"] == 1
    assert info["action_limit_activation_rate"] == 0.5

    _, reset_info = instance.reset(seed=0, options={"center_init": True})

    assert reset_info["action_attempt_count"] == 0
    assert reset_info["action_limit_activation_count"] == 0
    assert reset_info["action_limit_activation_rate"] == 0.0
    assert reset_info["requested_step_distance"] is None
    assert reset_info["max_requested_step_distance"] == 0.0
    assert reset_info["action_limit_failure"] is False


def test_action_at_float32_limit_is_accepted():
    instance = PointReach2DPreferenceEnv()
    initial, _ = instance.reset(seed=0, options={"center_init": True})
    requested = initial[:2] + np.array(
        [0.0, DEFAULT_CONFIG.environment.max_step_distance],
        dtype=np.float32,
    )

    observation, _, terminated, _, info = instance.step(requested)

    np.testing.assert_array_equal(observation[:2], requested)
    assert terminated is False
    assert info["action_limit_failure"] is False
    assert info["action_limit_activation_count"] == 0


@pytest.mark.parametrize("max_step_distance", [0.0, -0.1, float("nan")])
def test_environment_config_rejects_invalid_max_step_distance(max_step_distance):
    with pytest.raises(ValueError, match="max_step_distance"):
        EnvironmentConfig(max_step_distance=max_step_distance)


def test_numerical_failure_preserves_last_state():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0, options={"center_init": True})
    last_valid, _, _, _, _ = instance.step(
        np.array([-0.95, 0.0], dtype=np.float32)
    )

    observation, reward, terminated, truncated, info = instance.step(
        np.array([np.nan, 0.0], dtype=np.float32)
    )

    np.testing.assert_array_equal(observation[:2], last_valid[:2])
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["numerical_failure"] is True
    assert info["success"] is False
    assert info["action_attempt_count"] == 2
    assert info["action_limit_activation_count"] == 0
    assert info["action_limit_activation_rate"] == 0.0
    assert info["requested_step_distance"] is None
    assert info["max_requested_step_distance"] == pytest.approx(0.05)


def test_bad_action_shape_and_post_terminal_step_raise():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0)

    with pytest.raises(ValueError, match="shape"):
        instance.step(np.zeros(3, dtype=np.float32))

    instance.step(np.array([np.inf, 0.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="terminated"):
        instance.step(np.zeros(2, dtype=np.float32))


def test_invalid_render_mode_is_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        PointReach2DPreferenceEnv(render_mode="ansi")


def test_rgb_array_render_has_stable_shape_and_tracks_episode():
    instance = PointReach2DPreferenceEnv(render_mode="rgb_array")
    try:
        instance.reset(seed=0, options={"center_init": True})
        initial_frame = instance.render()

        assert initial_frame.shape == (520, 720, 3)
        assert initial_frame.dtype == np.uint8
        assert np.unique(initial_frame.reshape(-1, 3), axis=0).shape[0] > 4

        next_position = np.array([-0.95, -0.02], dtype=np.float32)
        observation, _, _, _, info = instance.step(next_position)
        stepped_frame = instance.render()
        assert not np.array_equal(stepped_frame, initial_frame)
        np.testing.assert_array_equal(observation[:2], next_position)
        np.testing.assert_array_equal(instance._position, next_position)
        assert info["outside_visualization"] is False

        instance.reset(seed=0, options={"center_init": True})
        reset_frame = instance.render()
        np.testing.assert_array_equal(reset_frame, initial_frame)
    finally:
        instance.close()


def test_human_render_is_automatic_and_closes_owned_pygame(monkeypatch):
    class CountingHumanEnv(PointReach2DPreferenceEnv):
        def __init__(self):
            self.render_calls = 0
            super().__init__(render_mode="human")

        def render(self):
            self.render_calls += 1
            return super().render()

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    instance = CountingHumanEnv()
    try:
        instance.reset(seed=0, options={"center_init": True})
        assert instance.render_calls == 1
        instance.step(np.array([-0.95, 0.02], dtype=np.float32))
        assert instance.render_calls == 2

        import pygame

        assert pygame.get_init() is True
    finally:
        instance.close()
        instance.close()

    assert pygame.get_init() is False


def test_human_close_preserves_externally_initialized_pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = _import_pygame_without_dependency_warnings()

    pygame.init()
    assert pygame.display.get_surface() is None
    instance = PointReach2DPreferenceEnv(render_mode="human")
    try:
        instance.reset(seed=0, options={"center_init": True})
        assert pygame.display.get_surface() is not None
        instance.close()

        assert pygame.get_init() is True
        assert pygame.display.get_init() is True
        assert pygame.display.get_surface() is None
    finally:
        instance.close()
        pygame.quit()


def test_human_render_refuses_to_replace_external_display(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = _import_pygame_without_dependency_warnings()
    pygame.init()
    external_surface = pygame.display.set_mode((32, 32))
    instance = PointReach2DPreferenceEnv(render_mode="human")
    try:
        with pytest.raises(RuntimeError, match="display surface"):
            instance.reset(seed=0, options={"center_init": True})
        assert pygame.display.get_surface() is external_surface
    finally:
        instance.close()
        pygame.quit()


def test_human_close_preserves_external_display_only_init(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = _import_pygame_without_dependency_warnings()
    pygame.quit()
    pygame.display.init()
    assert pygame.get_init() is False
    assert pygame.display.get_init() is True
    instance = PointReach2DPreferenceEnv(render_mode="human")
    try:
        instance.reset(seed=0, options={"center_init": True})
        instance.close()

        assert pygame.get_init() is False
        assert pygame.display.get_init() is True
        assert pygame.display.get_surface() is None
    finally:
        instance.close()
        pygame.display.quit()
