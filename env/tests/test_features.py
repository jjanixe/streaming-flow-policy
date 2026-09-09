import copy

import numpy as np
import pytest

from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.features import (
    fit_feature_normalizer,
    raw_trajectory_features,
    trajectory_feature_parts,
    trajectory_features,
)


def test_normalizer_matches_train_raw_feature_statistics():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=3)

    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    train = bank.select("train")
    raw = raw_trajectory_features(
        train.positions,
        DEFAULT_CONFIG.features.y_scale,
        1.0 / DEFAULT_CONFIG.environment.horizon_steps,
    )

    np.testing.assert_allclose(normalizer.mu, raw.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        normalizer.scale,
        raw.std(axis=0, ddof=0),
        atol=1e-6,
    )
    assert normalizer.mu.dtype == np.float32
    assert normalizer.scale.dtype == np.float32


def test_normalizer_ignores_validation_and_test_trajectories():
    original = generate_demonstration_bank(DEFAULT_CONFIG, seed=31)
    changed = copy.deepcopy(original)
    changed.positions[changed.splits != "train", :, 1] = np.float32(100.0)

    original_normalizer = fit_feature_normalizer(original, DEFAULT_CONFIG)
    changed_normalizer = fit_feature_normalizer(changed, DEFAULT_CONFIG)

    np.testing.assert_array_equal(original_normalizer.mu, changed_normalizer.mu)
    np.testing.assert_array_equal(
        original_normalizer.scale,
        changed_normalizer.scale,
    )


def test_full_feature_equals_prefix_plus_future():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=4)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    trajectories = bank.select("test").positions[:8]
    dt = 1.0 / DEFAULT_CONFIG.environment.horizon_steps
    full = trajectory_features(trajectories, normalizer, dt=dt)

    for prefix_steps in (0, 1, 17, 63, 64):
        prefix, future = trajectory_feature_parts(
            trajectories,
            normalizer,
            prefix_steps=prefix_steps,
            dt=dt,
        )
        np.testing.assert_allclose(
            full,
            prefix + future,
            rtol=1e-5,
            atol=1e-6,
        )

    assert full.dtype == np.float32


def test_invalid_prefix_boundary_is_rejected():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=4)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    trajectories = bank.select("test").positions[:2]

    with pytest.raises(ValueError, match="prefix_steps"):
        trajectory_feature_parts(
            trajectories,
            normalizer,
            prefix_steps=65,
            dt=1.0 / 64.0,
        )
