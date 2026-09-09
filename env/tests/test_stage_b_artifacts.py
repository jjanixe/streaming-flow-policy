import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from env.artifacts import train_data_digest
from env.chunk_data import fit_pusht_stats
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.models import SFPDVelocityMLP
from env.sfp_policies import StreamingFlowPolicyDeterministic
from env.stage_b_artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    load_pusht_stats,
    load_sfpd_checkpoint,
    save_pusht_stats,
    save_sfpd_checkpoint,
)
from env.stage_b_config import DEFAULT_STAGE_B1_CONFIG, stage_b1_config_to_dict


_TEST_CONFIG = replace(
    DEFAULT_STAGE_B1_CONFIG,
    hidden_dim=16,
    hidden_layers=1,
    max_updates=4,
    validation_interval=2,
    warmup_updates=1,
)
_SEED_STREAM_NAMES = (
    "model_initialization",
    "dataloader_shuffle",
    "train_transform",
    "validation_transform",
    "rollout_initialization",
    "checkpoint_replay",
)


def _expected_seed_streams(root_seed):
    children = np.random.SeedSequence(root_seed).spawn(len(_SEED_STREAM_NAMES))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(_SEED_STREAM_NAMES, children)
    }


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
        "stage_b1_config": stage_b1_config_to_dict(_TEST_CONFIG),
        "selected_update": 2,
        "validation_loss": 0.25,
        "root_seed": 23,
        "seed_streams": _expected_seed_streams(23),
        "train_data_digest": "correct",
        "drake_version": "1.26.0",
        "numpy_version": "1.26.4",
        "torch_version": "2.7.1+cu126",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "warmup_updates": 1,
            "schedule": "linear_warmup_cosine",
        },
        "solver": {
            "name": "dopri5",
            "sensitivity": "adjoint",
            "atol": 1e-4,
            "rtol": 1e-4,
            "integration_steps_per_action": 6,
        },
    }
    metadata.update(overrides)
    return metadata


def _policy_state(hidden_dim=16, hidden_layers=1):
    return StreamingFlowPolicyDeterministic(
        SFPDVelocityMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    ).state_dict()


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
    state = _policy_state()
    checkpoint = tmp_path / "nested" / "model.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=state,
        ema_state=state,
        metadata=_complete_metadata(),
    )
    torch.manual_seed(734)
    rng_before_load = torch.random.get_rng_state().clone()

    loaded = load_sfpd_checkpoint(
        checkpoint,
        expected_train_data_digest="correct",
        expected_architecture=_complete_metadata()["architecture"],
    )

    assert torch.equal(rng_before_load, torch.random.get_rng_state())
    assert loaded["metadata"]["model_type"] == "sfpd"
    for state_name in ("raw_state", "ema_state"):
        for value in loaded[state_name].values():
            if value.is_floating_point():
                assert value.dtype == torch.float32
                assert torch.isfinite(value).all()
            assert value.device.type == "cpu"


def test_checkpoint_rejects_state_shape_that_disagrees_with_declared_hidden_dim(
    tmp_path,
):
    declared_16_but_actual_32 = _policy_state(hidden_dim=32)
    checkpoint = tmp_path / "mismatched_hidden_dim.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=declared_16_but_actual_32,
        ema_state=declared_16_but_actual_32,
        metadata=_complete_metadata(),
    )

    with pytest.raises(ValueError, match="state schema"):
        load_sfpd_checkpoint(
            checkpoint,
            expected_train_data_digest="correct",
        )


@pytest.mark.parametrize(
    "corruption",
    ["bare_velocity", "unexpected_key", "missing_key", "wrong_buffer_dtype"],
)
def test_checkpoint_rejects_noncanonical_wrapper_state(tmp_path, corruption):
    state = dict(_policy_state())
    if corruption == "bare_velocity":
        state = dict(SFPDVelocityMLP(hidden_dim=16, hidden_layers=1).state_dict())
    elif corruption == "unexpected_key":
        state["unexpected"] = torch.zeros(1, dtype=torch.float32)
    elif corruption == "missing_key":
        del state["velocity_net.network.0.bias"]
    else:
        state["pred_horizon"] = state["pred_horizon"].to(torch.int64)
    checkpoint = tmp_path / f"{corruption}.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=state,
        ema_state=state,
        metadata=_complete_metadata(),
    )

    with pytest.raises(ValueError, match="state schema"):
        load_sfpd_checkpoint(
            checkpoint,
            expected_train_data_digest="correct",
        )


@pytest.mark.parametrize("missing", ["torch_version", "optimizer", "solver"])
def test_checkpoint_requires_runtime_optimizer_and_solver_metadata(tmp_path, missing):
    metadata = _complete_metadata()
    del metadata[missing]
    checkpoint = tmp_path / f"missing_{missing}.pt"
    state = _policy_state()
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=state,
        ema_state=state,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="missing keys"):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="correct")


@pytest.mark.parametrize(
    "case,match",
    [
        ("empty_version", "version"),
        ("root_seed_bool", "root_seed"),
        ("seed_names", "seed_streams"),
        ("seed_out_of_range", "seed_streams"),
        ("seed_wrong_value", "seed_streams"),
        ("config_missing", "stage_b1_config"),
        ("config_wrong_type", "max_updates"),
        ("architecture_mismatch", "architecture"),
        ("optimizer_mismatch", "optimizer"),
        ("solver_mismatch", "solver"),
        ("selected_update_out_of_range", "selected_update"),
    ],
)
def test_checkpoint_rejects_semantically_corrupt_metadata(tmp_path, case, match):
    metadata = copy.deepcopy(_complete_metadata())
    if case == "empty_version":
        metadata["torch_version"] = ""
    elif case == "root_seed_bool":
        metadata["root_seed"] = True
    elif case == "seed_names":
        metadata["seed_streams"].pop("checkpoint_replay")
    elif case == "seed_out_of_range":
        metadata["seed_streams"]["checkpoint_replay"] = 2**32
    elif case == "seed_wrong_value":
        metadata["seed_streams"]["checkpoint_replay"] ^= 1
    elif case == "config_missing":
        metadata["stage_b1_config"].pop("sigma")
    elif case == "config_wrong_type":
        metadata["stage_b1_config"]["max_updates"] = True
    elif case == "architecture_mismatch":
        metadata["architecture"]["hidden_dim"] = 32
    elif case == "optimizer_mismatch":
        metadata["optimizer"]["learning_rate"] = 2e-4
    elif case == "solver_mismatch":
        metadata["solver"]["integration_steps_per_action"] = 5
    else:
        metadata["selected_update"] = 5
    checkpoint = tmp_path / f"semantic_{case}.pt"
    state = _policy_state()
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=state,
        ema_state=state,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=match):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="correct")


@pytest.mark.parametrize("state_name", ["raw_state", "ema_state"])
@pytest.mark.parametrize(
    "buffer_name",
    ["pred_horizon", "velocity_net.time_features.frequencies"],
)
def test_checkpoint_rejects_corrupt_immutable_canonical_buffers(
    tmp_path,
    state_name,
    buffer_name,
):
    raw_state = dict(_policy_state())
    ema_state = {name: value.clone() for name, value in raw_state.items()}
    target = raw_state if state_name == "raw_state" else ema_state
    target[buffer_name] = target[buffer_name].clone()
    target[buffer_name].reshape(-1)[0] += 1
    checkpoint = tmp_path / f"{state_name}_{buffer_name.replace('.', '_')}.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=raw_state,
        ema_state=ema_state,
        metadata=_complete_metadata(),
    )

    with pytest.raises(ValueError, match="immutable buffer"):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="correct")


def test_existing_canonical_checkpoint_remains_compatible():
    artifact_dir = Path(__file__).parents[1] / "artifacts" / "stage_b" / "b1-seed0"
    checkpoint = artifact_dir / "sfpd_best.pt"
    stats_path = artifact_dir / "pusht_stats.npz"
    if not checkpoint.is_file() or not stats_path.is_file():
        pytest.skip("canonical Stage B1 artifacts are not present")
    _, digest = load_pusht_stats(stats_path)

    loaded = load_sfpd_checkpoint(
        checkpoint,
        expected_train_data_digest=digest,
    )

    assert loaded["metadata"]["selected_update"] == 20_000
