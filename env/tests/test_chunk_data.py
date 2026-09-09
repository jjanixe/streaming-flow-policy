import copy
from dataclasses import replace

import numpy as np
import pytest

from env.chunk_data import (
    PushTChunkDataset,
    build_episode_arrays,
    fit_pusht_stats,
    normalize_actions,
    unnormalize_actions,
)
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.stage_b_config import DEFAULT_STAGE_B1_CONFIG


def test_stage_b1_defaults_match_pusht_contract():
    config = DEFAULT_STAGE_B1_CONFIG
    assert config.pred_horizon == 16
    assert config.obs_horizon == 2
    assert config.action_horizon == 8
    assert config.sigma == 0.1
    assert config.integration_steps_per_action == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("pred_horizon", True),
        ("obs_horizon", 2.0),
        ("action_horizon", "8"),
        ("sigma", np.float32(0.1)),
        ("hidden_dim", True),
        ("hidden_layers", 3.0),
        ("batch_size", np.int64(4)),
        ("max_updates", False),
        ("validation_interval", 1.0),
        ("warmup_updates", np.int64(1)),
        ("learning_rate", np.float32(1e-4)),
        ("weight_decay", 0),
        ("ema_decay", np.float64(0.999)),
        ("integration_steps_per_action", True),
        ("rollout_count", 1.0),
    ],
)
def test_stage_b1_config_rejects_nonexact_field_types(field, value):
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_STAGE_B1_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("sigma", float("nan")),
        ("sigma", -0.1),
        ("hidden_dim", 0),
        ("hidden_layers", 0),
        ("batch_size", 0),
        ("max_updates", 0),
        ("validation_interval", 0),
        ("warmup_updates", -1),
        ("learning_rate", float("inf")),
        ("learning_rate", 0.0),
        ("weight_decay", -1.0),
        ("ema_decay", 1.0),
        ("integration_steps_per_action", 0),
        ("rollout_count", 0),
        ("rollout_count", 2**32 + 1),
    ],
)
def test_stage_b1_config_rejects_nonfinite_or_out_of_range_fields(field, value):
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_STAGE_B1_CONFIG, **{field: value})


def test_stage_b1_config_rejects_warmup_beyond_training_run():
    with pytest.raises(ValueError, match="warmup_updates"):
        replace(DEFAULT_STAGE_B1_CONFIG, max_updates=10, warmup_updates=11)


def test_episode_actions_are_shifted_next_positions_with_terminal_repeat():
    positions = np.arange(10, dtype=np.float32).reshape(5, 2)
    obs, action = build_episode_arrays(positions)
    np.testing.assert_array_equal(obs[:, :2], positions)
    np.testing.assert_array_equal(action[:-1], positions[1:])
    np.testing.assert_array_equal(action[-1], positions[-1])
    np.testing.assert_array_equal(
        obs[:, 2], np.linspace(0.0, 1.0, 5, dtype=np.float32)
    )
    assert obs.dtype == action.dtype == np.float32


def test_stats_are_fit_only_from_train_trajectories():
    original = generate_demonstration_bank(DEFAULT_CONFIG, seed=11)
    changed = copy.deepcopy(original)
    changed.positions[changed.splits != "train"] += np.float32(100.0)
    first = fit_pusht_stats(original)
    second = fit_pusht_stats(changed)
    np.testing.assert_array_equal(first.obs_min, second.obs_min)
    np.testing.assert_array_equal(first.obs_max, second.obs_max)
    np.testing.assert_array_equal(first.action_min, second.action_min)
    np.testing.assert_array_equal(first.action_max, second.action_max)


def test_action_normalization_round_trips_in_float32():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=12)
    stats = fit_pusht_stats(bank)
    actions = bank.select("validation").positions[:2, :4]
    restored = unnormalize_actions(normalize_actions(actions, stats), stats)
    np.testing.assert_allclose(restored, actions, atol=2e-7)
    assert restored.dtype == np.float32


def test_each_65_state_episode_produces_58_pusht_windows():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=13)
    stats = fit_pusht_stats(bank)
    dataset = PushTChunkDataset(bank, split="validation", stats=stats)
    assert len(dataset) == len(bank.select("validation")) * 58


def test_first_and_last_windows_match_pusht_padding_and_anchor():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=14)
    split = bank.select("validation")
    stats = fit_pusht_stats(bank)
    dataset = PushTChunkDataset(bank, split="validation", stats=stats)
    first = dataset[0]
    last = dataset[57]
    assert first["sequence_start"] == np.int64(-1)
    assert first["anchor_index"] == np.int64(0)
    np.testing.assert_array_equal(first["obs"][0], first["obs"][1])
    np.testing.assert_array_equal(first["action"][0], first["obs"][-1, :2])
    assert first["anchor_physical_l2"] > np.float32(0.0)
    assert first["anchor_physical_l2"] < np.float32(1e-3)
    assert last["sequence_start"] == np.int64(56)
    assert last["anchor_index"] == np.int64(57)
    np.testing.assert_array_equal(last["action"][-1], last["action"][-2])
    assert first["obs"].shape == (2, 3)
    assert first["action"].shape == (16, 2)
    assert first["trajectory_id"] == split.trajectory_ids[0]
    assert first["source_split"] == "validation"
