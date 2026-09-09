import gym
import numpy as np
import pytest
import warnings
from gym.utils.env_checker import check_env

import env
from env.config import DEFAULT_CONFIG
from env.environment import PointReach2DPreferenceEnv


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
