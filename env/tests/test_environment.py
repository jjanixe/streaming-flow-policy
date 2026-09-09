import gym
import numpy as np
import pytest
import warnings
from gym.utils.env_checker import check_env

import env
from env.config import DEFAULT_CONFIG
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


def test_action_is_followed_without_clipping_and_horizon_terminates():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0, options={"center_init": True})
    far_position = np.array([9.0, -4.0], dtype=np.float32)

    for step in range(DEFAULT_CONFIG.environment.horizon_steps):
        observation, reward, terminated, truncated, info = instance.step(far_position)
        assert reward == 0.0
        assert truncated is False
        assert terminated is (
            step == DEFAULT_CONFIG.environment.horizon_steps - 1
        )

    np.testing.assert_array_equal(observation[:2], far_position)
    assert info["outside_visualization"] is True


def test_numerical_failure_preserves_last_state():
    instance = PointReach2DPreferenceEnv()
    initial, _ = instance.reset(seed=0, options={"center_init": True})

    observation, reward, terminated, truncated, info = instance.step(
        np.array([np.nan, 0.0], dtype=np.float32)
    )

    np.testing.assert_array_equal(observation[:2], initial[:2])
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["numerical_failure"] is True
    assert info["success"] is False


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

        far_position = np.array([9.0, -4.0], dtype=np.float32)
        observation, _, _, _, info = instance.step(far_position)
        stepped_frame = instance.render()
        assert not np.array_equal(stepped_frame, initial_frame)
        np.testing.assert_array_equal(observation[:2], far_position)
        np.testing.assert_array_equal(instance._position, far_position)
        assert info["outside_visualization"] is True

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
        instance.step(np.array([-0.8, 0.1], dtype=np.float32))
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
