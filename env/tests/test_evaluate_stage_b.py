from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from env.chunk_data import fit_pusht_stats, normalize_actions
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.evaluate_stage_b import (
    SFPDRollout,
    SFPDRolloutBatch,
    SFPSRollout,
    SFPSRolloutBatch,
    aggregate_sfpd_metrics,
    evaluate_sfpd,
    evaluate_sfps,
    rollout_sfpd,
    rollout_sfps,
    stage_b1_acceptance_failures,
)
from env.sfp_policies import PolicyNumericalError


class ScriptedNormalizedPolicy:
    def __init__(self, normalized_positions: np.ndarray) -> None:
        self.normalized_positions = normalized_positions
        self.calls = 0
        self.conditions: list[np.ndarray] = []

    def predict(self, nobs, num_actions, integration_steps_per_action):
        assert nobs.dtype == torch.float32
        assert num_actions == 9
        self.conditions.append(nobs.detach().cpu().numpy().copy())
        start = self.calls * 8
        self.calls += 1
        chunk = self.normalized_positions[start : start + 9]
        return torch.from_numpy(chunk[None]).to(dtype=torch.float32)


class SingleChunkPolicy:
    def __init__(self, normalized_chunk: np.ndarray) -> None:
        self.normalized_chunk = normalized_chunk
        self.calls = 0

    def predict(self, nobs, num_actions, integration_steps_per_action):
        self.calls += 1
        return torch.from_numpy(self.normalized_chunk[None])


class NumericallyFailingPolicy(ScriptedNormalizedPolicy):
    def __init__(self, normalized_positions: np.ndarray, fail_call: int) -> None:
        super().__init__(normalized_positions)
        self.fail_call = fail_call

    def predict(self, nobs, num_actions, integration_steps_per_action):
        if self.calls == self.fail_call:
            self.calls += 1
            raise PolicyNumericalError("adaptive solver produced non-finite state")
        return super().predict(nobs, num_actions, integration_steps_per_action)


class AlwaysNumericallyFailingPolicy:
    def predict(self, nobs, num_actions, integration_steps_per_action):
        raise PolicyNumericalError("adaptive solver produced non-finite state")


class ScriptedStochasticPolicy:
    device = torch.device("cpu")

    def __init__(self, normalized_positions: np.ndarray) -> None:
        self.normalized_positions = normalized_positions
        self.calls = 0
        self.latents: list[np.ndarray] = []

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
        assert num_actions == 9
        self.latents.append(latent.detach().cpu().numpy().copy())
        start = self.calls * 8
        self.calls += 1
        return torch.from_numpy(self.normalized_positions[start : start + 9][None])


class LatentSensitiveStochasticPolicy:
    device = torch.device("cpu")

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
        anchor = nobs[-1, :2]
        fractions = torch.linspace(0.0, 1.0, 9, dtype=torch.float32)[:, None]
        delta = 0.02 * torch.tanh(latent)[None, :]
        return (anchor[None, :] + fractions * delta).unsqueeze(0)


def test_sfps_rollout_uses_fresh_seeded_latent_at_each_chunk():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=301)
    stats = fit_pusht_stats(bank)
    normalized_expert = normalize_actions(bank.select("test").positions[0], stats)
    first_policy = ScriptedStochasticPolicy(normalized_expert)
    second_policy = ScriptedStochasticPolicy(normalized_expert)

    first = rollout_sfps(
        first_policy,
        stats,
        DEFAULT_CONFIG.environment,
        environment_seed=302,
        latent_seed=303,
        center_init=True,
        integration_steps_per_action=1,
    )
    second = rollout_sfps(
        second_policy,
        stats,
        DEFAULT_CONFIG.environment,
        environment_seed=302,
        latent_seed=303,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert isinstance(first, SFPSRollout)
    assert first_policy.calls == 8
    assert first.latent_seed == 303
    assert first.chunk_latents.shape == (8, 2)
    assert first.chunk_latents.dtype == np.float32
    assert len({row.tobytes() for row in first.chunk_latents}) == 8
    np.testing.assert_array_equal(first.chunk_latents, second.chunk_latents)
    np.testing.assert_array_equal(first.executed_positions, second.executed_positions)
    np.testing.assert_array_equal(first.raw_predicted_chunks, second.raw_predicted_chunks)


def test_sfps_centered_evaluation_changes_only_latent_streams():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=304)
    stats = fit_pusht_stats(bank)

    metrics, batch = evaluate_sfps(
        LatentSensitiveStochasticPolicy(),
        stats,
        DEFAULT_CONFIG.environment,
        environment_seeds=[305, 305, 305],
        latent_seeds=[306, 307, 308],
        center_init=True,
        integration_steps_per_action=1,
    )

    assert isinstance(batch, SFPSRolloutBatch)
    assert all(
        np.array_equal(batch.rollouts[0].initial_observation, rollout.initial_observation)
        for rollout in batch.rollouts[1:]
    )
    assert metrics["same_state_diversity_applicable"] is True
    assert metrics["same_state_unique_raw_trajectory_count"] > 1
    assert metrics["same_state_stochastic_diversity_observed"] is True
    assert metrics["latent_seed_count"] == 3


def test_rollout_replans_eight_times_and_executes_64_actions():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=31)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    scripted = ScriptedNormalizedPolicy(normalize_actions(expert, stats))

    rollout = rollout_sfpd(
        scripted,
        stats,
        DEFAULT_CONFIG.environment,
        seed=32,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert scripted.calls == 8
    assert rollout.executed_action_count == 64
    assert rollout.success
    assert rollout.initial_observation.dtype == np.float32
    assert rollout.executed_positions.dtype == np.float32
    assert rollout.requested_actions.dtype == np.float32
    assert rollout.raw_predicted_chunks.dtype == np.float32
    assert rollout.raw_predicted_chunks.shape == (8, 9, 2)
    np.testing.assert_allclose(rollout.executed_positions, expert, atol=2e-6)
    np.testing.assert_array_equal(
        scripted.conditions[0][0], scripted.conditions[0][1]
    )
    assert scripted.conditions[1][0, 2] < scripted.conditions[1][1, 2]


def test_action_limit_failure_preserves_raw_request_and_unchanged_state():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=33)
    stats = fit_pusht_stats(bank)
    anchor = DEFAULT_CONFIG.environment.start_array()
    invalid = anchor + np.array([0.2, 0.0], dtype=np.float32)
    physical_chunk = np.repeat(anchor[None], 9, axis=0)
    physical_chunk[1] = invalid
    policy = SingleChunkPolicy(normalize_actions(physical_chunk, stats))

    rollout = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=34,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert policy.calls == 1
    assert rollout.executed_action_count == 1
    assert rollout.action_limit_failure
    assert not rollout.numerical_failure
    assert rollout.info["action_limit_activation_count"] == 1
    assert rollout.info["action_attempt_count"] == 1
    assert rollout.info["max_requested_step_distance"] == pytest.approx(0.2)
    np.testing.assert_allclose(rollout.requested_actions[0], invalid, atol=1e-7)
    np.testing.assert_allclose(rollout.raw_predicted_chunks[0, 1], invalid, atol=1e-7)
    np.testing.assert_array_equal(
        rollout.executed_positions,
        np.stack((anchor, anchor)),
    )


def test_nonfinite_action_is_preserved_and_counted_as_numerical_failure():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=35)
    stats = fit_pusht_stats(bank)
    physical_chunk = np.repeat(
        DEFAULT_CONFIG.environment.start_array()[None],
        9,
        axis=0,
    )
    physical_chunk[1, 0] = np.nan
    policy = SingleChunkPolicy(normalize_actions(physical_chunk, stats))

    rollout = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=36,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert rollout.numerical_failure
    assert not rollout.action_limit_failure
    assert np.isnan(rollout.requested_actions[0, 0])
    assert np.isnan(rollout.raw_predicted_chunks[0, 1, 0])
    np.testing.assert_array_equal(
        rollout.executed_positions[0], rollout.executed_positions[1]
    )


def test_nonfinite_discarded_anchor_marks_generation_failure_and_false_success():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=135)
    stats = fit_pusht_stats(bank)
    normalized_expert = normalize_actions(bank.select("test").positions[0], stats)
    normalized_expert[0, 0] = np.nan
    policy = ScriptedNormalizedPolicy(normalized_expert)

    rollout = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=136,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert policy.calls == 8
    assert rollout.executed_action_count == 64
    assert not rollout.success
    assert rollout.numerical_failure
    assert not rollout.action_limit_failure
    assert np.isnan(rollout.raw_predicted_chunks[0, 0, 0])
    assert rollout.info["policy_generation_numerical_failure"] is True
    assert rollout.info["policy_generation_nonfinite_chunk_indices"] == [0]
    assert rollout.info["gym_numerical_failure"] is False
    assert rollout.info["numerical_failure"] is True
    assert rollout.info["success"] is False


def test_nonfinite_unexecuted_suffix_is_counted_after_earlier_limit_failure():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=137)
    stats = fit_pusht_stats(bank)
    anchor = DEFAULT_CONFIG.environment.start_array()
    physical_chunk = np.repeat(anchor[None], 9, axis=0)
    physical_chunk[1] = anchor + np.array([0.2, 0.0], dtype=np.float32)
    physical_chunk[8, 1] = np.nan

    rollout = rollout_sfpd(
        SingleChunkPolicy(normalize_actions(physical_chunk, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=138,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert rollout.executed_action_count == 1
    assert rollout.action_limit_failure
    assert rollout.numerical_failure
    assert not rollout.success
    assert np.isfinite(rollout.requested_actions).all()
    assert np.isnan(rollout.raw_predicted_chunks[0, 8, 1])
    assert rollout.info["action_limit_activation_count"] == 1
    assert rollout.info["policy_generation_numerical_failure"] is True
    assert rollout.info["policy_generation_nonfinite_chunk_indices"] == [0]
    assert rollout.info["gym_numerical_failure"] is False


def test_rollout_is_byte_deterministic_for_same_gaussian_seed():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=37)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]

    first = rollout_sfpd(
        ScriptedNormalizedPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=38,
        center_init=False,
        integration_steps_per_action=2,
    )
    second = rollout_sfpd(
        ScriptedNormalizedPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=38,
        center_init=False,
        integration_steps_per_action=2,
    )

    assert first.initial_observation.tobytes() == second.initial_observation.tobytes()
    assert first.executed_positions.tobytes() == second.executed_positions.tobytes()
    assert first.requested_actions.tobytes() == second.requested_actions.tobytes()
    assert first.raw_predicted_chunks.tobytes() == second.raw_predicted_chunks.tobytes()


def _recorded_rollout(
    *,
    seed: int,
    positions: np.ndarray,
    success: bool,
    action_limit_failure: bool = False,
    numerical_failure: bool = False,
    action_limit_activation_count: int = 0,
    requested_actions: np.ndarray | None = None,
) -> SFPDRollout:
    positions = positions.astype(np.float32, copy=False)
    if requested_actions is None:
        requested_actions = positions[1:].copy()
    requested_actions = requested_actions.astype(np.float32, copy=False)
    attempts = len(requested_actions)
    chunk_count = (attempts + 7) // 8 if attempts else 1
    raw_chunks = np.repeat(
        positions[[0]][None],
        chunk_count * 9,
        axis=1,
    ).reshape(chunk_count, 9, 2).astype(np.float32)
    for chunk_index in range(chunk_count):
        start = chunk_index * 8
        stop = min(start + 8, attempts)
        raw_chunks[chunk_index, 0] = positions[min(start, len(positions) - 1)]
        raw_chunks[chunk_index, 1 : 1 + stop - start] = requested_actions[start:stop]
    return SFPDRollout(
        seed=seed,
        initial_observation=np.array(
            [positions[0, 0], positions[0, 1], 0.0], dtype=np.float32
        ),
        executed_positions=positions,
        requested_actions=requested_actions,
        raw_predicted_chunks=raw_chunks,
        executed_action_count=attempts,
        success=success,
        numerical_failure=numerical_failure,
        action_limit_failure=action_limit_failure,
        info={
            "step_index": min(attempts, 64) if not action_limit_failure else 0,
            "goal_error": float(
                np.linalg.norm(positions[-1] - DEFAULT_CONFIG.environment.goal_array())
            ),
            "action_attempt_count": attempts,
            "action_limit_activation_count": action_limit_activation_count,
            "action_limit_failure": action_limit_failure,
            "numerical_failure": numerical_failure,
            "success": success,
            "gym_numerical_failure": numerical_failure,
            "policy_generation_numerical_failure": False,
            "policy_generation_nonfinite_chunk_indices": [],
            "policy_generation_exception_chunk_indices": [],
            "max_requested_step_distance": (
                float(
                    np.nanmax(
                        np.linalg.norm(
                            requested_actions - positions[:attempts], axis=1
                        )
                    )
                )
                if attempts
                else 0.0
            ),
            "normalized_anchor_discrepancies": np.zeros(
                chunk_count, dtype=np.float32
            ),
            "physical_anchor_discrepancies": np.full(
                chunk_count,
                np.float32(seed) / np.float32(1000.0),
                dtype=np.float32,
            ),
        },
    )


def test_aggregate_uses_state_32_and_separates_failed_before_midpoint():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=39).select("test")
    mode_rollouts = []
    for seed, mode in enumerate(
        ("upper-narrow", "upper-wide", "lower-narrow", "lower-wide"),
        start=1,
    ):
        index = int(np.flatnonzero(bank.modes == mode)[0])
        mode_rollouts.append(
            _recorded_rollout(
                seed=seed,
                positions=bank.positions[index],
                success=True,
            )
        )
    anchor = DEFAULT_CONFIG.environment.start_array()
    failed_request = anchor + np.array([0.2, 0.0], dtype=np.float32)
    failed = _recorded_rollout(
        seed=5,
        positions=np.stack((anchor, anchor)),
        requested_actions=failed_request[None],
        success=False,
        action_limit_failure=True,
        action_limit_activation_count=1,
    )

    metrics = aggregate_sfpd_metrics(
        SFPDRolloutBatch.from_rollouts((*mode_rollouts, failed)),
        DEFAULT_CONFIG.environment,
    )

    assert metrics["rollout_count"] == 5
    assert metrics["goal_success_count"] == 4
    assert metrics["goal_success_rate"] == pytest.approx(0.8)
    assert len(metrics["final_goal_errors"]) == 5
    assert metrics["numerical_failure_count"] == 0
    assert metrics["action_limit_failure_count"] == 1
    assert metrics["action_limit_failure_indices"] == [4]
    assert metrics["action_limit_activation_count"] == 1
    assert metrics["action_limit_activation_rate"] == pytest.approx(1.0 / 257.0)
    assert metrics["max_requested_step_distance"] == pytest.approx(0.2)
    assert metrics["failed_before_midpoint_count"] == 1
    assert metrics["failed_before_midpoint_indices"] == [4]
    assert metrics["midpoint_classified_count"] == 4
    assert metrics["midpoint_occupancy"] == pytest.approx(
        {
            "upper-narrow": 0.25,
            "upper-wide": 0.25,
            "lower-narrow": 0.25,
            "lower-wide": 0.25,
            "other": 0.0,
            "nonfinite": 0.0,
        }
    )
    assert metrics["physical_anchor_discrepancy_max"] == pytest.approx(0.005)
    assert metrics["normalized_anchor_discrepancy_max"] == 0.0


def test_acceptance_failures_return_exact_development_gate_reasons():
    passing = {
        "goal_success_rate": 0.95,
        "numerical_failure_count": 0,
        "midpoint_occupancy": {
            "upper-narrow": 0.25,
            "upper-wide": 0.25,
            "lower-narrow": 0.25,
            "lower-wide": 0.25,
            "other": 0.0,
        },
    }
    assert stage_b1_acceptance_failures(passing) == []

    failing = {
        **passing,
        "goal_success_rate": 0.94,
        "numerical_failure_count": 1,
        "midpoint_occupancy": {
            "upper-narrow": 0.04,
            "upper-wide": 0.36,
            "lower-narrow": 0.25,
            "lower-wide": 0.25,
            "other": 0.06,
        },
    }
    assert stage_b1_acceptance_failures(failing) == [
        "goal success rate is below 0.95",
        "numerical failures are nonzero",
        "upper-narrow occupancy is below 0.05",
        "upper-narrow occupancy differs from 0.25 by more than 0.10",
        "upper-wide occupancy differs from 0.25 by more than 0.10",
        "other occupancy is above 0.05",
    ]


def test_evaluate_returns_batch_and_attaches_acceptance_failures():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=40)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    metrics, batch = evaluate_sfpd(
        ScriptedNormalizedPolicy(normalize_actions(expert, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        [41],
        center_init=True,
        integration_steps_per_action=1,
    )

    assert len(batch.rollouts) == 1
    assert metrics["goal_success_rate"] == 1.0
    assert metrics["acceptance_failures"] == [
        "upper-narrow occupancy differs from 0.25 by more than 0.10",
        "upper-wide occupancy is below 0.05",
        "upper-wide occupancy differs from 0.25 by more than 0.10",
        "lower-narrow occupancy is below 0.05",
        "lower-narrow occupancy differs from 0.25 by more than 0.10",
        "lower-wide occupancy is below 0.05",
        "lower-wide occupancy differs from 0.25 by more than 0.10",
    ]


def test_empty_rollout_batch_is_rejected():
    with pytest.raises(ValueError, match="at least one rollout"):
        SFPDRolloutBatch.from_rollouts([])


def test_rollout_rejects_float64_and_shape_mismatches():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=139).positions[0]
    valid = _recorded_rollout(seed=1, positions=path, success=True)

    with pytest.raises(ValueError, match="initial_observation.*float32"):
        replace(valid, initial_observation=valid.initial_observation.astype(np.float64))
    with pytest.raises(ValueError, match=r"raw_predicted_chunks.*\[C, 9, 2\]"):
        replace(valid, raw_predicted_chunks=np.zeros((8, 8, 2), dtype=np.float32))


def test_rollout_rejects_action_and_chunk_relationship_mismatches():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=140).positions[0]
    valid = _recorded_rollout(seed=2, positions=path, success=True)

    with pytest.raises(ValueError, match="executed_action_count.*requested_actions"):
        replace(valid, requested_actions=valid.requested_actions[:-1])
    with pytest.raises(ValueError, match=r"executed_positions.*K \+ 1"):
        replace(valid, executed_positions=valid.executed_positions[:-1])
    with pytest.raises(ValueError, match="chunk count.*action count"):
        replace(valid, raw_predicted_chunks=valid.raw_predicted_chunks[:1])


def test_rollout_rejects_requested_values_not_copied_from_chunks_in_order():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=147).positions[0]
    valid = _recorded_rollout(seed=9, positions=path, success=True)
    chunks = valid.raw_predicted_chunks.copy()
    chunks[0, 1, 0] += np.float32(0.01)

    with pytest.raises(ValueError, match="requested_actions.*predicted chunk"):
        replace(valid, raw_predicted_chunks=chunks)


def test_rollout_rejects_trailing_chunk_without_attempted_actions():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=148).positions[0]
    valid = _recorded_rollout(seed=10, positions=path, success=True)
    trailing = np.concatenate(
        (valid.raw_predicted_chunks, valid.raw_predicted_chunks[[-1]]),
        axis=0,
    )
    info = {
        **valid.info,
        "normalized_anchor_discrepancies": np.append(
            valid.info["normalized_anchor_discrepancies"], np.float32(0.0)
        ).astype(np.float32),
        "physical_anchor_discrepancies": np.append(
            valid.info["physical_anchor_discrepancies"], np.float32(0.0)
        ).astype(np.float32),
    }

    with pytest.raises(ValueError, match="chunk count.*ceil"):
        replace(valid, raw_predicted_chunks=trailing, info=info)


def test_rollout_rejects_inconsistent_info_and_success_flags():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=141).positions[0]
    valid = _recorded_rollout(seed=3, positions=path, success=True)

    with pytest.raises(ValueError, match="action_attempt_count"):
        replace(valid, info={**valid.info, "action_attempt_count": 63})
    with pytest.raises(ValueError, match="step_index.*accepted transitions"):
        replace(valid, info={**valid.info, "step_index": 63})
    with pytest.raises(ValueError, match="success.*failure"):
        replace(
            valid,
            numerical_failure=True,
            info={**valid.info, "numerical_failure": True},
        )


def test_rollout_rejects_inconsistent_generation_failure_metadata():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=146).positions[0]
    valid = _recorded_rollout(seed=8, positions=path, success=True)

    with pytest.raises(ValueError, match="combined numerical_failure"):
        replace(
            valid,
            success=False,
            numerical_failure=True,
            info={
                **valid.info,
                "success": False,
                "numerical_failure": True,
            },
        )
    with pytest.raises(ValueError, match="nonfinite chunk indices"):
        replace(
            valid,
            success=False,
            numerical_failure=True,
            info={
                **valid.info,
                "success": False,
                "numerical_failure": True,
                "policy_generation_numerical_failure": True,
                "policy_generation_nonfinite_chunk_indices": [],
            },
        )


def test_rollout_rejects_missing_gym_flag_for_attempted_nonfinite_action():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=149)
    stats = fit_pusht_stats(bank)
    anchor = DEFAULT_CONFIG.environment.start_array()
    physical_chunk = np.repeat(anchor[None], 9, axis=0)
    physical_chunk[1, 0] = np.nan
    failed = rollout_sfpd(
        SingleChunkPolicy(normalize_actions(physical_chunk, stats)),
        stats,
        DEFAULT_CONFIG.environment,
        seed=150,
        center_init=True,
        integration_steps_per_action=1,
    )

    with pytest.raises(ValueError, match="gym_numerical_failure.*attempted"):
        replace(failed, info={**failed.info, "gym_numerical_failure": False})


def test_rollout_rejects_nonfinite_initial_and_executed_states():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=151).positions[0]
    valid = _recorded_rollout(seed=11, positions=path, success=True)

    initial = valid.initial_observation.copy()
    executed = valid.executed_positions.copy()
    chunks = valid.raw_predicted_chunks.copy()
    initial[0] = np.nan
    executed[0, 0] = np.nan
    chunks[0, 0, 0] = np.nan
    generation_info = {
        **valid.info,
        "success": False,
        "numerical_failure": True,
        "policy_generation_numerical_failure": True,
        "policy_generation_nonfinite_chunk_indices": [0],
    }
    with pytest.raises(ValueError, match="initial_observation.*finite"):
        replace(
            valid,
            initial_observation=initial,
            executed_positions=executed,
            raw_predicted_chunks=chunks,
            success=False,
            numerical_failure=True,
            info=generation_info,
        )

    requested = valid.requested_actions.copy()
    executed = valid.executed_positions.copy()
    chunks = valid.raw_predicted_chunks.copy()
    requested[9, 0] = np.inf
    executed[10, 0] = np.inf
    chunks[1, 2, 0] = np.inf
    execution_info = {
        **valid.info,
        "success": False,
        "numerical_failure": True,
        "gym_numerical_failure": True,
        "policy_generation_numerical_failure": True,
        "policy_generation_nonfinite_chunk_indices": [1],
    }
    with pytest.raises(ValueError, match="executed_positions.*finite"):
        replace(
            valid,
            executed_positions=executed,
            requested_actions=requested,
            raw_predicted_chunks=chunks,
            success=False,
            numerical_failure=True,
            info=execution_info,
        )


def test_rollout_allows_one_zero_attempt_nonfinite_generation_record():
    anchor = DEFAULT_CONFIG.environment.start_array()
    chunks = np.full((1, 9, 2), np.nan, dtype=np.float32)

    rollout = SFPDRollout(
        seed=12,
        initial_observation=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        executed_positions=anchor[None],
        requested_actions=np.empty((0, 2), dtype=np.float32),
        raw_predicted_chunks=chunks,
        executed_action_count=0,
        success=False,
        numerical_failure=True,
        action_limit_failure=False,
        info={
            "action_attempt_count": 0,
            "step_index": 0,
            "action_limit_activation_count": 0,
            "success": False,
            "numerical_failure": True,
            "action_limit_failure": False,
            "gym_numerical_failure": False,
            "policy_generation_numerical_failure": True,
            "policy_generation_nonfinite_chunk_indices": [0],
            "policy_generation_exception_chunk_indices": [0],
            "normalized_anchor_discrepancies": np.array(
                [np.nan], dtype=np.float32
            ),
            "physical_anchor_discrepancies": np.array(
                [np.nan], dtype=np.float32
            ),
        },
    )

    assert rollout.executed_action_count == 0


def test_first_policy_numerical_exception_becomes_terminal_sentinel_rollout():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=152)
    stats = fit_pusht_stats(bank)
    policy = NumericallyFailingPolicy(
        normalize_actions(bank.select("test").positions[0], stats),
        fail_call=0,
    )

    rollout = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=153,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert rollout.executed_action_count == 0
    assert rollout.executed_positions.shape == (1, 2)
    assert rollout.requested_actions.shape == (0, 2)
    assert rollout.raw_predicted_chunks.shape == (1, 9, 2)
    assert np.isnan(rollout.raw_predicted_chunks).all()
    assert np.isnan(rollout.info["normalized_anchor_discrepancies"]).all()
    assert np.isnan(rollout.info["physical_anchor_discrepancies"]).all()
    assert rollout.info["policy_generation_exception_chunk_indices"] == [0]
    assert rollout.info["policy_generation_nonfinite_chunk_indices"] == [0]
    assert rollout.info["policy_generation_numerical_failure"] is True
    assert rollout.info["gym_numerical_failure"] is False
    assert rollout.info["terminated"] is True
    assert rollout.numerical_failure
    assert not rollout.success


def test_later_policy_numerical_exception_preserves_completed_chunks_and_actions():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=154)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    policy = NumericallyFailingPolicy(
        normalize_actions(expert, stats),
        fail_call=1,
    )

    rollout = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=155,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert rollout.executed_action_count == 8
    np.testing.assert_allclose(rollout.executed_positions, expert[:9], atol=2e-6)
    np.testing.assert_allclose(rollout.requested_actions, expert[1:9], atol=2e-6)
    assert rollout.raw_predicted_chunks.shape == (2, 9, 2)
    assert np.isfinite(rollout.raw_predicted_chunks[0]).all()
    assert np.isnan(rollout.raw_predicted_chunks[1]).all()
    assert rollout.info["step_index"] == 8
    assert rollout.info["policy_generation_exception_chunk_indices"] == [1]
    assert rollout.info["policy_generation_nonfinite_chunk_indices"] == [1]
    assert np.isnan(rollout.info["normalized_anchor_discrepancies"][1])
    assert np.isnan(rollout.info["physical_anchor_discrepancies"][1])
    assert rollout.numerical_failure
    assert not rollout.success


def test_exception_sentinel_keeps_prior_returned_nonfinite_chunk_semantics():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=162)
    stats = fit_pusht_stats(bank)
    normalized = normalize_actions(bank.select("test").positions[0], stats)
    normalized[0, 0] = np.nan
    policy = NumericallyFailingPolicy(normalized, fail_call=1)

    rollout = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=163,
        center_init=True,
        integration_steps_per_action=1,
    )

    assert rollout.executed_action_count == 8
    assert rollout.info["policy_generation_nonfinite_chunk_indices"] == [0, 1]
    assert rollout.info["policy_generation_exception_chunk_indices"] == [1]
    assert rollout.numerical_failure


def test_evaluation_continues_after_each_policy_numerical_exception():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=156)
    stats = fit_pusht_stats(bank)

    metrics, batch = evaluate_sfpd(
        AlwaysNumericallyFailingPolicy(),
        stats,
        DEFAULT_CONFIG.environment,
        [157, 158, 159],
        center_init=True,
        integration_steps_per_action=1,
    )

    assert len(batch.rollouts) == 3
    assert metrics["numerical_failure_count"] == 3
    assert metrics["numerical_failure_indices"] == [0, 1, 2]
    assert metrics["goal_success_count"] == 0


def test_rollout_does_not_swallow_programming_value_error():
    class InvalidPolicy:
        def predict(self, nobs, num_actions, integration_steps_per_action):
            raise ValueError("bad policy configuration")

    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=160)
    stats = fit_pusht_stats(bank)

    with pytest.raises(ValueError, match="bad policy configuration"):
        rollout_sfpd(
            InvalidPolicy(),
            stats,
            DEFAULT_CONFIG.environment,
            seed=161,
            center_init=True,
            integration_steps_per_action=1,
        )


def test_rollout_rejects_invalid_scalar_and_info_types():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=143).positions[0]
    valid = _recorded_rollout(seed=5, positions=path, success=True)

    with pytest.raises(ValueError, match="seed.*nonnegative integer"):
        replace(valid, seed=True)
    with pytest.raises(ValueError, match="executed_action_count.*integer"):
        replace(valid, executed_action_count=True)
    with pytest.raises(ValueError, match="success.*bool"):
        replace(valid, success=np.bool_(True))
    with pytest.raises(ValueError, match="info.*dict"):
        replace(valid, info=None)


def test_rollout_rejects_step_index_that_disagrees_with_returned_states():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=144).positions[0]
    valid = _recorded_rollout(seed=6, positions=path, success=True)
    returned = valid.executed_positions.copy()
    returned[-1] = returned[-2]

    with pytest.raises(ValueError, match="step_index.*accepted transitions"):
        replace(valid, executed_positions=returned)


def test_rollout_allows_nonfinite_raw_chunk_only_for_numerical_failure():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=142).positions[0]
    valid = _recorded_rollout(seed=4, positions=path, success=True)
    chunks = valid.raw_predicted_chunks.copy()
    chunks[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite.*numerical_failure"):
        replace(valid, raw_predicted_chunks=chunks)

    failed = replace(
        valid,
        raw_predicted_chunks=chunks,
        success=False,
        numerical_failure=True,
        info={
            **valid.info,
            "success": False,
            "numerical_failure": True,
            "policy_generation_numerical_failure": True,
            "policy_generation_nonfinite_chunk_indices": [0],
        },
    )
    assert np.isnan(failed.raw_predicted_chunks[0, 0, 0])


def test_aggregate_counts_nonfinite_anchor_discrepancies_without_nan_json():
    path = generate_demonstration_bank(DEFAULT_CONFIG, seed=145).positions[0]
    valid = _recorded_rollout(seed=7, positions=path, success=True)
    chunks = valid.raw_predicted_chunks.copy()
    chunks[0, 0, 0] = np.nan
    normalized_anchor = valid.info["normalized_anchor_discrepancies"].copy()
    physical_anchor = valid.info["physical_anchor_discrepancies"].copy()
    normalized_anchor[0] = np.nan
    physical_anchor[0] = np.nan
    failed = replace(
        valid,
        raw_predicted_chunks=chunks,
        success=False,
        numerical_failure=True,
        info={
            **valid.info,
            "success": False,
            "numerical_failure": True,
            "policy_generation_numerical_failure": True,
            "policy_generation_nonfinite_chunk_indices": [0],
            "normalized_anchor_discrepancies": normalized_anchor,
            "physical_anchor_discrepancies": physical_anchor,
        },
    )

    metrics = aggregate_sfpd_metrics(
        SFPDRolloutBatch.from_rollouts([failed]),
        DEFAULT_CONFIG.environment,
    )

    assert metrics["normalized_anchor_discrepancy_nonfinite_count"] == 1
    assert metrics["physical_anchor_discrepancy_nonfinite_count"] == 1
    assert np.isfinite(metrics["normalized_anchor_discrepancy_max"])
    assert np.isfinite(metrics["physical_anchor_discrepancy_max"])
    json.dumps(metrics, allow_nan=False)
