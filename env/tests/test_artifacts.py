import json
from dataclasses import replace

import numpy as np
import pytest

from env.artifacts import (
    load_demonstration_bank,
    load_feature_normalizer,
    save_demonstration_bank,
    save_feature_normalizer,
    save_json,
    save_rollouts,
    train_data_digest,
)
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.features import fit_feature_normalizer


def test_bank_and_normalizer_round_trip_without_pickle(tmp_path):
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=5)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    bank_path = tmp_path / "demonstrations.npz"
    normalizer_path = tmp_path / "normalizer.npz"

    save_demonstration_bank(bank_path, bank)
    save_feature_normalizer(
        normalizer_path,
        normalizer,
        train_data_digest(bank),
    )
    loaded_bank = load_demonstration_bank(bank_path)
    loaded_normalizer, digest = load_feature_normalizer(normalizer_path)

    np.testing.assert_array_equal(loaded_bank.positions, bank.positions)
    np.testing.assert_array_equal(
        loaded_bank.trajectory_ids,
        bank.trajectory_ids,
    )
    np.testing.assert_array_equal(loaded_normalizer.mu, normalizer.mu)
    np.testing.assert_array_equal(
        loaded_normalizer.scale,
        normalizer.scale,
    )
    assert digest == train_data_digest(bank)
    with np.load(bank_path, allow_pickle=False) as payload:
        assert payload["positions"].dtype == np.float32
        assert payload["trajectory_ids"].dtype.kind == "U"


def test_digest_changes_when_train_data_changes():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=6)
    changed = generate_demonstration_bank(DEFAULT_CONFIG, seed=6)
    changed.positions[0, 1, 1] += np.float32(0.01)

    assert train_data_digest(bank) != train_data_digest(changed)


def test_rollouts_and_json_are_written_in_portable_formats(tmp_path):
    initial = np.zeros((3, 2), dtype=np.float32)
    ode = np.zeros((3, 65, 2), dtype=np.float32)
    sde = np.ones((3, 65, 2), dtype=np.float32)
    rollout_path = tmp_path / "rollouts.npz"
    json_path = tmp_path / "diagnostics.json"

    save_rollouts(rollout_path, initial, ode, sde)
    save_json(json_path, {"accepted": True, "count": 3})

    with np.load(rollout_path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["initial_states"], initial)
        assert payload["ode_positions"].dtype == np.float32
        assert payload["sde_positions"].dtype == np.float32
    with json_path.open(encoding="utf-8") as stream:
        assert json.load(stream) == {"accepted": True, "count": 3}


def test_demonstration_bank_rejects_pickle_backed_metadata():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=5)

    with pytest.raises(ValueError, match="Unicode"):
        replace(bank, trajectory_ids=bank.trajectory_ids.astype(object))


def test_json_rejects_nonfinite_numbers(tmp_path):
    with pytest.raises(ValueError, match="JSON compliant"):
        save_json(tmp_path / "invalid.json", {"metric": float("nan")})
