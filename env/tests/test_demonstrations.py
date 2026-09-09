from dataclasses import replace

import numpy as np
import pytest

from env.config import DEFAULT_CONFIG
from env.demonstrations import evaluate_path, generate_demonstration_bank


def test_all_modes_have_exact_float32_endpoints():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=13)

    np.testing.assert_allclose(bank.positions[:, 0, 0], -1.0, atol=1e-6)
    np.testing.assert_allclose(bank.positions[:, 0, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(bank.positions[:, -1, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(bank.positions[:, -1, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(bank.derivatives[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(bank.derivatives[:, -1], 0.0, atol=1e-6)
    assert bank.times.shape == (65,)
    assert bank.positions.shape == (768, 65, 2)
    assert bank.positions.dtype == np.float32
    assert bank.derivatives.dtype == np.float32
    assert bank.times.dtype == np.float32
    assert bank.signs.dtype == np.float32
    assert bank.amplitudes.dtype == np.float32
    step_distances = np.linalg.norm(np.diff(bank.positions, axis=1), axis=-1)
    assert (
        float(step_distances.max())
        <= DEFAULT_CONFIG.environment.max_step_distance
    )


def test_generation_rejects_expert_steps_above_environment_limit():
    constrained = replace(
        DEFAULT_CONFIG,
        environment=replace(
            DEFAULT_CONFIG.environment,
            max_step_distance=0.01,
        ),
    )

    with pytest.raises(ValueError, match="expert.*max_step_distance"):
        generate_demonstration_bank(constrained, seed=13)


def test_analytic_derivative_matches_float32_central_difference():
    times = np.array([0.2, 0.37, 0.63, 0.8], dtype=np.float32)
    step = np.float32(1e-3)

    _, derivative = evaluate_path(1.0, 0.53, times)
    plus, _ = evaluate_path(1.0, 0.53, times + step)
    minus, _ = evaluate_path(1.0, 0.53, times - step)
    finite_difference = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=2e-3,
        atol=2e-4,
    )


def test_splits_are_balanced_isolated_and_reproducible():
    first = generate_demonstration_bank(DEFAULT_CONFIG, seed=17)
    second = generate_demonstration_bank(DEFAULT_CONFIG, seed=17)

    np.testing.assert_array_equal(first.amplitudes, second.amplitudes)
    np.testing.assert_array_equal(first.replay_seeds, second.replay_seeds)
    split_ids = {
        split: set(first.trajectory_ids[first.splits == split].tolist())
        for split in ("train", "validation", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])

    for split, per_mode in (("train", 128), ("validation", 32), ("test", 32)):
        for mode in ("upper-narrow", "upper-wide", "lower-narrow", "lower-wide"):
            assert np.sum((first.splits == split) & (first.modes == mode)) == per_mode


def test_replay_seed_reconstructs_each_sampled_amplitude():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=19)

    for index in range(12):
        mode = str(bank.modes[index])
        bounds = (
            DEFAULT_CONFIG.demonstrations.amplitude_narrow
            if mode.endswith("narrow")
            else DEFAULT_CONFIG.demonstrations.amplitude_wide
        )
        replayed = np.random.default_rng(int(bank.replay_seeds[index])).uniform(*bounds)
        np.testing.assert_allclose(
            bank.amplitudes[index],
            np.float32(replayed),
            rtol=0.0,
            atol=0.0,
        )


def test_bank_rejects_nonfinite_times_and_wrong_seed_dtype():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=19)
    invalid_times = bank.times.copy()
    invalid_times[1] = np.float32(np.nan)

    with pytest.raises(ValueError, match="finite"):
        replace(bank, times=invalid_times)
    with pytest.raises(ValueError, match="uint32"):
        replace(bank, replay_seeds=bank.replay_seeds.astype(np.int64))
