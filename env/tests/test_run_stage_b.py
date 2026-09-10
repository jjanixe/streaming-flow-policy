import json
from dataclasses import replace
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import env.run_stage_b as run_stage_b_module
from env.artifacts import save_demonstration_bank, train_data_digest
from env.chunk_data import fit_pusht_stats, normalize_actions
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.evaluate_stage_b import (
    SFPDRollout,
    SFPDRolloutBatch,
    SFPSRolloutBatch,
    rollout_sfpd,
    rollout_sfps,
)
from env.run_stage_b import (
    derive_rollout_seeds,
    load_trained_sfpd,
    load_trained_sfps,
    run_stage_b1,
    run_stage_b2,
)
from env.stage_b_artifacts import (
    load_sfpd_checkpoint,
    load_sfps_checkpoint,
    save_sfpd_rollouts,
    save_sfps_rollouts,
)
from env.stage_b_config import DEFAULT_STAGE_B1_CONFIG, DEFAULT_STAGE_B2_CONFIG
from env.visualize_stage_b import (
    MODE_COLORS,
    _occupancy_plot_data,
    _rollout_display_mode,
    plot_mode_occupancy,
    plot_trajectory_comparison,
    write_representative_gifs,
)


def _small_bank(seed: int):
    demonstrations = replace(
        DEFAULT_CONFIG.demonstrations,
        train_per_mode=2,
        validation_per_mode=1,
        test_per_mode=1,
    )
    return generate_demonstration_bank(
        replace(DEFAULT_CONFIG, demonstrations=demonstrations),
        seed=seed,
    )


class _ScriptedPolicy:
    def __init__(self, normalized_positions: np.ndarray) -> None:
        self.normalized_positions = normalized_positions
        self.calls = 0

    def predict(self, nobs, num_actions, integration_steps_per_action):
        start = self.calls * 8
        self.calls += 1
        return torch.from_numpy(self.normalized_positions[start : start + 9][None])


class _SingleChunkPolicy:
    def __init__(self, normalized_chunk: np.ndarray) -> None:
        self.normalized_chunk = normalized_chunk

    def predict(self, nobs, num_actions, integration_steps_per_action):
        return torch.from_numpy(self.normalized_chunk[None])


class _ScriptedStochasticPolicy:
    device = torch.device("cpu")

    def __init__(self, normalized_positions: np.ndarray) -> None:
        self.normalized_positions = normalized_positions
        self.calls = 0

    def sample_latent(self, generator):
        return torch.randn((2,), dtype=torch.float32, generator=generator)

    def predict(
        self,
        nobs,
        num_actions,
        integration_steps_per_action,
        *,
        latent,
    ):
        start = self.calls * 8
        self.calls += 1
        return torch.from_numpy(self.normalized_positions[start : start + 9][None])


def test_reduced_stage_b1_run_writes_complete_safe_artifacts(tmp_path):
    bank = _small_bank(seed=41)
    config = replace(
        DEFAULT_STAGE_B1_CONFIG,
        hidden_dim=16,
        hidden_layers=1,
        batch_size=64,
        max_updates=2,
        validation_interval=1,
        warmup_updates=1,
        rollout_count=4,
        integration_steps_per_action=1,
    )

    result = run_stage_b1(tmp_path, bank=bank, config=config, seed=42)

    assert result["model_type"] == "sfpd"
    assert result["deterministic_replay"] is True
    assert np.isfinite(result["teacher_forced_test_field_loss"])
    assert result["gaussian_metrics"]["acceptance_applicable"] is True
    assert result["centered_metrics"]["acceptance_applicable"] is False
    assert result["centered_metrics"]["unique_executed_trajectory_count"] == 1
    for name in (
        "resolved_config.json",
        "seed_streams.json",
        "pusht_stats.npz",
        "sfpd_best.pt",
        "training_history.json",
        "diagnostics.json",
        "rollouts.npz",
        "trajectory_comparison.png",
        "mode_occupancy.png",
    ):
        assert (tmp_path / name).is_file(), name

    with np.load(tmp_path / "rollouts.npz", allow_pickle=False) as payload:
        assert not any(payload[name].dtype.hasobject for name in payload.files)
        assert payload["rollout_seeds"].tolist() == result["rollout_seeds"]
        assert payload["executed_positions"].dtype == np.float32
        assert payload["requested_actions"].dtype == np.float32
        assert payload["raw_predicted_chunks"].dtype == np.float32
        assert payload["executed_position_lengths"].dtype.kind in "iu"
        assert payload["failure_mask"].dtype == np.bool_
        assert str(payload["checkpoint_digest"].item()) == result["checkpoint_digest"]
        assert str(payload["train_data_digest"].item()) == result["train_data_digest"]

    for name in (
        "resolved_config.json",
        "seed_streams.json",
        "training_history.json",
        "diagnostics.json",
    ):
        with (tmp_path / name).open(encoding="utf-8") as stream:
            json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ))
    with (tmp_path / "diagnostics.json").open(encoding="utf-8") as stream:
        diagnostics = json.load(stream)
    assert diagnostics["checkpoint_environment"]["matches_runtime"] == {
        "drake": True,
        "numpy": True,
        "torch": True,
    }
    audit = diagnostics["audit"]
    assert set(audit["dataset_splits"]) == {"train", "validation", "test"}
    for split, summary in audit["dataset_splits"].items():
        assert summary["source_split"] == split
        assert summary["window_count"] > 0
        assert summary["boundary_anchor_correction_raw_l2"]["count"] == summary[
            "window_count"
        ]
        assert summary["normalization_endpoint_mismatch_l2"]["count"] == summary[
            "window_count"
        ]
    boundary = audit["rollout_boundary_jump_l2"]
    assert "raw_predicted_chunks[c, 0]" in boundary["definition"]
    assert boundary["gaussian"]["count"] >= 0
    assert boundary["centered"]["count"] >= 0

    checkpoint = load_sfpd_checkpoint(
        tmp_path / "sfpd_best.pt",
        expected_train_data_digest=train_data_digest(bank),
    )
    policy, _, metadata = load_trained_sfpd(
        tmp_path / "sfpd_best.pt",
        tmp_path / "pusht_stats.npz",
        expected_train_data_digest=train_data_digest(bank),
        device="cpu",
    )
    assert metadata == checkpoint["metadata"]
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, checkpoint["ema_state"][name])

    if result["gaussian_metrics"]["goal_success_count"]:
        assert (tmp_path / "representative_success.gif").is_file()
    if result["gaussian_metrics"]["goal_success_count"] < config.rollout_count:
        assert (tmp_path / "representative_failure.gif").is_file()


def test_rollout_npz_padding_is_explicit_and_pickle_free(tmp_path):
    bank = _small_bank(seed=43)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    successful = rollout_sfpd(
        _ScriptedPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=44,
        center_init=True,
        integration_steps_per_action=1,
    )
    anchor = DEFAULT_CONFIG.environment.start_array()
    failed_chunk = np.repeat(anchor[None], 9, axis=0)
    failed_chunk[1] = anchor + np.array([0.2, 0.0], dtype=np.float32)
    failed = rollout_sfpd(
        _SingleChunkPolicy(normalize_actions(failed_chunk, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=45,
        center_init=True,
        integration_steps_per_action=1,
    )
    path = tmp_path / "nested" / "rollouts.npz"

    save_sfpd_rollouts(
        path,
        SFPDRolloutBatch.from_rollouts((successful, failed)),
        checkpoint_digest="checkpoint-sha256",
        train_data_digest="train-sha256",
    )

    with np.load(path, allow_pickle=False) as payload:
        assert payload["format_version"].item() == 1
        assert str(payload["model_type"].item()) == "sfpd"
        assert payload["rollout_seeds"].tolist() == [44, 45]
        assert payload["executed_position_lengths"].tolist() == [65, 2]
        assert payload["requested_action_lengths"].tolist() == [64, 1]
        assert payload["raw_predicted_chunk_lengths"].tolist() == [8, 1]
        assert payload["executed_position_mask"].dtype == np.bool_
        assert payload["requested_action_mask"].dtype == np.bool_
        assert payload["raw_predicted_chunk_mask"].dtype == np.bool_
        assert payload["failure_mask"].tolist() == [False, True]
        assert np.isnan(payload["executed_positions"][1, 2:]).all()
        assert np.isnan(payload["requested_actions"][1, 1:]).all()
        assert np.isnan(payload["raw_predicted_chunks"][1, 1:]).all()
        np.testing.assert_array_equal(
            payload["raw_predicted_chunks"][0, :8],
            successful.raw_predicted_chunks,
        )
        assert not any(payload[name].dtype.hasobject for name in payload.files)


def test_sfps_rollout_npz_preserves_separate_seed_and_latent_provenance(tmp_path):
    bank = _small_bank(seed=401)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    rollout = rollout_sfps(
        _ScriptedStochasticPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        environment_seed=402,
        latent_seed=403,
        center_init=True,
        integration_steps_per_action=1,
    )
    path = tmp_path / "sfps_rollouts.npz"

    save_sfps_rollouts(
        path,
        SFPSRolloutBatch.from_rollouts((rollout,)),
        checkpoint_digest="checkpoint-sha256",
        train_data_digest="train-sha256",
    )

    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["model_type"].item()) == "sfps"
        assert payload["environment_seeds"].tolist() == [402]
        assert payload["latent_seeds"].tolist() == [403]
        assert payload["chunk_latents"].shape == (1, 8, 2)
        assert payload["chunk_latents"].dtype == np.float32
        assert payload["chunk_latent_mask"].dtype == np.bool_
        assert payload["chunk_latent_lengths"].tolist() == [8]
        assert not any(payload[name].dtype.hasobject for name in payload.files)


def test_reduced_stage_b2_run_writes_controlled_diversity_artifacts(tmp_path):
    bank = _small_bank(seed=404)
    config = replace(
        DEFAULT_STAGE_B2_CONFIG,
        hidden_dim=16,
        hidden_layers=1,
        batch_size=64,
        max_updates=2,
        validation_interval=1,
        warmup_updates=1,
        rollout_count=4,
        integration_steps_per_action=1,
    )
    b1_diagnostics = tmp_path / "compatible_b1_diagnostics.json"
    b1_diagnostics.write_text(
        json.dumps(
            {
                "model_type": "sfpd",
                "environment_id": "PointReach2DPreference-v0",
                "train_data_digest": train_data_digest(bank),
                "teacher_forced_test_field_loss": 0.5,
                "checkpoint": {"architecture": {"pred_horizon": 16}},
                "gaussian": {
                    "rollout_count": 4,
                    "midpoint_classified_count": 4,
                    "midpoint_occupancy": {
                        "upper-narrow": 0.25,
                        "upper-wide": 0.25,
                        "lower-narrow": 0.25,
                        "lower-wide": 0.25,
                        "other": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_stage_b2(
        tmp_path,
        bank=bank,
        config=config,
        seed=405,
        b1_diagnostics_path=b1_diagnostics,
    )

    assert result["model_type"] == "sfps"
    assert result["seeded_replay"] is True
    assert result["centered_metrics"]["same_state_diversity_applicable"] is True
    assert result["centered_metrics"]["same_state_unique_raw_trajectory_count"] > 1
    assert result["b1_comparison"]["available"] is True
    for name in (
        "resolved_config.json",
        "sfps_seed_streams.json",
        "pusht_stats.npz",
        "sfps_best.pt",
        "sfps_training_history.json",
        "sfps_diagnostics.json",
        "sfps_rollouts.npz",
        "sfps_centered_rollouts.npz",
        "sfps_trajectory_comparison.png",
        "sfps_mode_occupancy.png",
        "stage_b1_b2_comparison.json",
        "stage_b1_b2_mode_comparison.png",
    ):
        assert (tmp_path / name).is_file(), name

    checkpoint = load_sfps_checkpoint(
        tmp_path / "sfps_best.pt",
        expected_train_data_digest=train_data_digest(bank),
    )
    policy, _, metadata = load_trained_sfps(
        tmp_path / "sfps_best.pt",
        tmp_path / "pusht_stats.npz",
        expected_train_data_digest=train_data_digest(bank),
        device="cpu",
    )
    assert metadata == checkpoint["metadata"]
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, checkpoint["ema_state"][name])

    with (tmp_path / "sfps_diagnostics.json").open(encoding="utf-8") as stream:
        diagnostics = json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    assert diagnostics["dtypes"]["learned_model_and_ode"] == "float32"


def test_rollout_seed_derivation_is_stable_and_validated(monkeypatch):
    assert derive_rollout_seeds(123, 3) == [
        1514383052,
        2306788707,
        2261827930,
    ]
    with pytest.raises(ValueError, match="nonnegative integer"):
        derive_rollout_seeds(True, 3)
    with pytest.raises(ValueError, match="positive integer"):
        derive_rollout_seeds(123, 0)
    seeds = derive_rollout_seeds(456, 1024)
    assert len(set(seeds)) == len(seeds)

    class _MustNotConstructSeedSequence:
        def __init__(self, seed):
            raise AssertionError("capacity must be checked before seed spawning")

    monkeypatch.setattr(
        run_stage_b_module.np.random,
        "SeedSequence",
        _MustNotConstructSeedSequence,
    )
    with pytest.raises(ValueError, match="uint32 seed space"):
        derive_rollout_seeds(123, 2**32 + 1)


def test_rollout_seed_derivation_resolves_child_collisions(monkeypatch):
    class _CollidingChild:
        def generate_state(self, count, dtype):
            assert count == 1
            assert dtype == np.uint32
            return np.array([7], dtype=np.uint32)

    class _CollidingSeedSequence:
        def __init__(self, seed):
            assert seed == 123

        def spawn(self, count):
            return [_CollidingChild() for _ in range(count)]

    monkeypatch.setattr(
        run_stage_b_module.np.random,
        "SeedSequence",
        _CollidingSeedSequence,
    )

    assert derive_rollout_seeds(123, 3) == [7, 8, 9]


@pytest.mark.parametrize(
    "field",
    ("rollout_count", "integration_steps_per_action"),
)
def test_config_rejects_invalid_evaluation_counts_before_training(
    tmp_path, field
):
    with pytest.raises(ValueError, match=field):
        replace(
            DEFAULT_STAGE_B1_CONFIG,
            **{field: 0},
            max_updates=1,
            warmup_updates=0,
        )
    assert not (tmp_path / "sfpd_best.pt").exists()


def test_static_plots_close_figures_and_reject_nonfinite_experts(tmp_path):
    import matplotlib.pyplot as plt

    bank = _small_bank(seed=46)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    rollout = rollout_sfpd(
        _ScriptedPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=47,
        center_init=True,
        integration_steps_per_action=1,
    )
    batch = SFPDRolloutBatch.from_rollouts((rollout,))

    plot_trajectory_comparison(
        tmp_path / "nested" / "trajectories.png",
        bank.select("test").positions,
        batch,
        DEFAULT_CONFIG.environment,
    )
    plot_mode_occupancy(
        tmp_path / "occupancy.png",
        {
            "upper-narrow": 0.25,
            "upper-wide": 0.25,
            "lower-narrow": 0.25,
            "lower-wide": 0.25,
            "other": 0.0,
        },
        {
            "upper-narrow": 1.0,
            "upper-wide": 0.0,
            "lower-narrow": 0.0,
            "lower-wide": 0.0,
            "other": 0.0,
        },
        sfpd_classified_count=1,
        sfpd_rollout_count=1,
    )

    assert (tmp_path / "nested" / "trajectories.png").is_file()
    assert (tmp_path / "occupancy.png").is_file()
    assert plt.get_fignums() == []
    invalid = bank.select("test").positions.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plot_trajectory_comparison(
            tmp_path / "invalid.png",
            invalid,
            batch,
            DEFAULT_CONFIG.environment,
        )


def test_occupancy_plot_states_conditional_denominator_and_failed_share():
    occupancy = {
        "upper-narrow": 0.25,
        "upper-wide": 0.25,
        "lower-narrow": 0.25,
        "lower-wide": 0.25,
        "other": 0.0,
    }

    labels, expert_values, sfpd_values, title = _occupancy_plot_data(
        occupancy,
        occupancy,
        sfpd_classified_count=4,
        sfpd_rollout_count=5,
    )

    assert labels[-1] == "failed-before-midpoint"
    assert expert_values[-1] == 0.0
    assert sfpd_values[-1] == pytest.approx(0.2)
    assert "among rollouts reaching state 32" in title
    assert "4/5" in title


def test_failed_attempt_after_31_transitions_has_no_midpoint_mode():
    bank = _small_bank(seed=49)
    positions = bank.select("test").positions[0]
    invalid = positions[31] + np.array([0.2, 0.0], dtype=np.float32)
    requested = np.concatenate((positions[1:32], invalid[None]), axis=0)
    executed = np.concatenate((positions[:32], positions[[31]]), axis=0)
    chunks = np.repeat(positions[[0]][None], 4 * 9, axis=1).reshape(4, 9, 2)
    for chunk_index in range(4):
        start = chunk_index * 8
        chunks[chunk_index, 0] = positions[start]
        chunks[chunk_index, 1:] = requested[start : start + 8]
    rollout = SFPDRollout(
        seed=50,
        initial_observation=np.array(
            [positions[0, 0], positions[0, 1], 0.0], dtype=np.float32
        ),
        executed_positions=executed.astype(np.float32),
        requested_actions=requested.astype(np.float32),
        raw_predicted_chunks=chunks.astype(np.float32),
        executed_action_count=32,
        success=False,
        numerical_failure=False,
        action_limit_failure=True,
        info={
            "action_attempt_count": 32,
            "step_index": 31,
            "action_limit_activation_count": 1,
            "max_requested_step_distance": 0.2,
            "success": False,
            "numerical_failure": False,
            "action_limit_failure": True,
            "gym_numerical_failure": False,
            "policy_generation_numerical_failure": False,
            "policy_generation_nonfinite_chunk_indices": [],
            "normalized_anchor_discrepancies": np.zeros(4, dtype=np.float32),
            "physical_anchor_discrepancies": np.zeros(4, dtype=np.float32),
        },
    )

    assert len(rollout.executed_positions) == 33
    assert _rollout_display_mode(rollout) == "failed-before-midpoint"
    assert MODE_COLORS["failed-before-midpoint"] == "#8f949e"


def test_representative_gif_rerun_removes_stale_outcome_labels(tmp_path):
    bank = _small_bank(seed=51)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    successful = rollout_sfpd(
        _ScriptedPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=52,
        center_init=True,
        integration_steps_per_action=1,
    )
    anchor = DEFAULT_CONFIG.environment.start_array()
    failed_chunk = np.repeat(anchor[None], 9, axis=0)
    failed_chunk[1] = anchor + np.array([0.2, 0.0], dtype=np.float32)
    failed = rollout_sfpd(
        _SingleChunkPolicy(normalize_actions(failed_chunk, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=53,
        center_init=True,
        integration_steps_per_action=1,
    )
    success_path = tmp_path / "representative_success.gif"
    failure_path = tmp_path / "representative_failure.gif"
    success_path.write_bytes(b"stale-success")
    failure_path.write_bytes(b"stale-failure")

    first = write_representative_gifs(
        tmp_path,
        SFPDRolloutBatch.from_rollouts((failed,)),
        DEFAULT_CONFIG.environment,
        center_init=True,
    )

    assert set(first) == {"failure"}
    assert not success_path.exists()
    assert failure_path.read_bytes().startswith((b"GIF87a", b"GIF89a"))

    second = write_representative_gifs(
        tmp_path,
        SFPDRolloutBatch.from_rollouts((successful,)),
        DEFAULT_CONFIG.environment,
        center_init=True,
    )

    assert set(second) == {"success"}
    assert success_path.read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert not failure_path.exists()


def test_stage_b_cli_help_exposes_b1_b2_and_all():
    clean_environment = {
        key: value for key, value in os.environ.items() if key != "MPLCONFIGDIR"
    }
    help_result = subprocess.run(
        [sys.executable, "-m", "env.run_stage_b", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=clean_environment,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "Matplotlib created a temporary cache" not in help_result.stderr
    for flag in (
        "--stage",
        "--demonstrations",
        "--output-dir",
        "--seed",
        "--device",
        "--max-updates",
        "--rollout-count",
        "--integration-steps-per-action",
        "--enforce-acceptance",
        "--b1-record",
    ):
        assert flag in help_result.stdout
    assert "{b1,b2,all}" in help_result.stdout


def test_stage_b2_cli_refuses_missing_b1_record_before_training(tmp_path):
    demonstrations = tmp_path / "demonstrations.npz"
    save_demonstration_bank(demonstrations, _small_bank(seed=601))
    output = tmp_path / "b2"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "env.run_stage_b",
            "--stage",
            "b2",
            "--demonstrations",
            str(demonstrations),
            "--b1-record",
            str(tmp_path / "missing-b1.json"),
            "--output-dir",
            str(output),
            "--max-updates",
            "1",
            "--rollout-count",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "B1" in result.stderr
    assert not (output / "sfps_best.pt").exists()


def test_stage_b2_cli_refuses_incompatible_b1_record_before_training(tmp_path):
    demonstrations = tmp_path / "demonstrations.npz"
    save_demonstration_bank(demonstrations, _small_bank(seed=603))
    b1_record = tmp_path / "incompatible-b1.json"
    b1_record.write_text(
        json.dumps(
            {
                "model_type": "sfpd",
                "environment_id": "PointReach2DPreference-v0",
                "train_data_digest": "different-demonstrations",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "b2"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "env.run_stage_b",
            "--stage",
            "b2",
            "--demonstrations",
            str(demonstrations),
            "--b1-record",
            str(b1_record),
            "--output-dir",
            str(output),
            "--max-updates",
            "1",
            "--rollout-count",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "B1" in result.stderr
    assert not (output / "sfps_best.pt").exists()


def test_stage_b_all_cli_runs_sequential_models_and_emits_valid_json(tmp_path):
    demonstrations = tmp_path / "demonstrations.npz"
    save_demonstration_bank(demonstrations, _small_bank(seed=602))
    output = tmp_path / "all"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "env.run_stage_b",
            "--stage",
            "all",
            "--demonstrations",
            str(demonstrations),
            "--output-dir",
            str(output),
            "--max-updates",
            "1",
            "--rollout-count",
            "2",
            "--integration-steps-per-action",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(
        result.stdout,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert summary["stage"] == "all"
    assert summary["b1"]["model_type"] == "sfpd"
    assert summary["b2"]["model_type"] == "sfps"
    assert summary["b1_seed"] != summary["b2_seed"]
    assert (output / "b1" / "diagnostics.json").is_file()
    assert (output / "b2" / "sfps_diagnostics.json").is_file()
    assert (output / "stage_b_summary.json").is_file()


def test_cli_requires_saved_demonstrations_instead_of_regenerating(tmp_path):
    missing = tmp_path / "does-not-exist.npz"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "env.run_stage_b",
            "--stage",
            "b1",
            "--demonstrations",
            str(missing),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-updates",
            "1",
            "--rollout-count",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "output" / "sfpd_best.pt").exists()


def test_cli_enforce_acceptance_exits_nonzero_for_failed_gate(
    tmp_path,
    monkeypatch,
):
    bank = _small_bank(seed=54)
    monkeypatch.setattr(run_stage_b_module, "load_demonstration_bank", lambda _: bank)
    monkeypatch.setattr(
        run_stage_b_module,
        "run_stage_b1",
        lambda *args, **kwargs: {"accepted": False, "failures": ["gate failed"]},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage_b",
            "--demonstrations",
            str(tmp_path / "demonstrations.npz"),
            "--output-dir",
            str(tmp_path / "output"),
            "--enforce-acceptance",
        ],
    )

    with pytest.raises(SystemExit) as error:
        run_stage_b_module.main()

    assert error.value.code == 1
