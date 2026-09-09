import numpy as np

# Gym 0.26 still checks np.bool8, which NumPy 2 removed in favor of np.bool_.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
from gym.envs.registration import register

from env.environment import PointReach2DPreferenceEnv


ENV_ID = "PointReach2DPreference-v0"

if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point="env.environment:PointReach2DPreferenceEnv",
    )


__all__ = ["ENV_ID", "PointReach2DPreferenceEnv"]
