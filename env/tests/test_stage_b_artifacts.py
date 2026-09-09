import numpy as np
import pytest
import torch

from env.artifacts import train_data_digest
from env.chunk_data import fit_pusht_stats
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.models import SFPDVelocityMLP
from env.stage_b_artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    load_pusht_stats,
    load_sfpd_checkpoint,
    save_pusht_stats,
    save_sfpd_checkpoint,
)


def _complete_metadata(**overrides):
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": "sfpd",
        "architecture": {
            "name": "SFPDVelocityMLP",
            "hidden_dim": 16,
            "hidden_layers": 1,
            "pred_horizon": 16,
        },
        "stage_b1_config": {"pred_horizon": 16},
        "selected_update": 2,
        "validation_loss": 0.25,
        "root_seed": 23,
        "seed_streams": {"model_initialization": 123},
        "train_data_digest": "correct",
        "drake_version": "1.26.0",
        "numpy_version": "1.26.4",
    }
    metadata.update(overrides)
    return metadata


def test_pusht_stats_round_trip_without_pickle(tmp_path):
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=21)
    stats = fit_pusht_stats(bank)
    path = tmp_path / "nested" / "pusht_stats.npz"
    save_pusht_stats(path, stats, train_data_digest(bank))

    loaded, digest = load_pusht_stats(path)

    np.testing.assert_array_equal(loaded.obs_min, stats.obs_min)
    np.testing.assert_array_equal(loaded.obs_max, stats.obs_max)
    np.testing.assert_array_equal(loaded.action_min, stats.action_min)
    np.testing.assert_array_equal(loaded.action_max, stats.action_max)
    assert digest == train_data_digest(bank)
    with np.load(path, allow_pickle=False) as payload:
        assert payload["train_data_digest"].dtype.kind == "U"
        assert all(payload[name].dtype == np.float32 for name in (
            "obs_min",
            "obs_max",
            "action_min",
            "action_max",
        ))


def test_checkpoint_rejects_wrong_demonstration_digest_before_other_metadata(tmp_path):
    model = SFPDVelocityMLP(hidden_dim=16, hidden_layers=1)
    checkpoint = tmp_path / "model.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=model.state_dict(),
        ema_state=model.state_dict(),
        metadata={"model_type": "sfpd", "train_data_digest": "correct"},
    )

    with pytest.raises(ValueError, match="digest"):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="wrong")


@pytest.mark.parametrize(
    "metadata,match",
    [
        (_complete_metadata(format_version=999), "format version"),
        (_complete_metadata(model_type="sfps"), "model type"),
        (_complete_metadata(architecture={"name": "OtherModel"}), "architecture"),
    ],
)
def test_checkpoint_rejects_incompatible_metadata(tmp_path, metadata, match):
    model = SFPDVelocityMLP(hidden_dim=16, hidden_layers=1)
    checkpoint = tmp_path / f"{match.replace(' ', '_')}.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=model.state_dict(),
        ema_state=model.state_dict(),
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=match):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="correct")


def test_checkpoint_requires_all_metadata_and_can_match_exact_architecture(tmp_path):
    model = SFPDVelocityMLP(hidden_dim=16, hidden_layers=1)
    checkpoint = tmp_path / "model.pt"
    metadata = _complete_metadata()
    del metadata["selected_update"]
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=model.state_dict(),
        ema_state=model.state_dict(),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="metadata"):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="correct")

    metadata = _complete_metadata()
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=model.state_dict(),
        ema_state=model.state_dict(),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="architecture"):
        load_sfpd_checkpoint(
            checkpoint,
            expected_train_data_digest="correct",
            expected_architecture={**metadata["architecture"], "hidden_dim": 32},
        )


def test_checkpoint_round_trip_keeps_finite_float32_weights_on_cpu(tmp_path):
    model = SFPDVelocityMLP(hidden_dim=16, hidden_layers=1)
    checkpoint = tmp_path / "nested" / "model.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=model.state_dict(),
        ema_state=model.state_dict(),
        metadata=_complete_metadata(),
    )

    loaded = load_sfpd_checkpoint(
        checkpoint,
        expected_train_data_digest="correct",
        expected_architecture=_complete_metadata()["architecture"],
    )

    assert loaded["metadata"]["model_type"] == "sfpd"
    for state_name in ("raw_state", "ema_state"):
        for value in loaded[state_name].values():
            if value.is_floating_point():
                assert value.dtype == torch.float32
                assert torch.isfinite(value).all()
            assert value.device.type == "cpu"
