"""Closed-loop Gym evaluation for deterministic streaming flow policies."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from env.chunk_data import (
    PushTStats,
    normalize_observations,
    unnormalize_actions,
)
from env.config import EnvironmentConfig
from env.environment import PointReach2DPreferenceEnv
from env.run_stage_a import classify_midpoint_modes


@dataclass(frozen=True)
class SFPDRollout:
    seed: int
    initial_observation: np.ndarray
    executed_positions: np.ndarray
    requested_actions: np.ndarray
    raw_predicted_chunks: np.ndarray
    executed_action_count: int
    success: bool
    numerical_failure: bool
    action_limit_failure: bool
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            type(self.executed_action_count) is not int
            or self.executed_action_count < 0
        ):
            raise ValueError("executed_action_count must be a nonnegative integer")
        for name in (
            "success",
            "numerical_failure",
            "action_limit_failure",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if type(self.info) is not dict:
            raise ValueError("info must be a dict")

        expected_arrays = {
            "initial_observation": ((3,), "[3]"),
            "executed_positions": (None, "[K + 1, 2]"),
            "requested_actions": (None, "[K, 2]"),
            "raw_predicted_chunks": (None, "[C, 9, 2]"),
        }
        for name, (shape, shape_text) in expected_arrays.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise ValueError(
                    f"{name} must be a float32 NumPy array with shape {shape_text}"
                )
            valid_shape = value.shape == shape if shape is not None else True
            if name == "executed_positions":
                valid_shape = value.ndim == 2 and value.shape[1:] == (2,)
            elif name == "requested_actions":
                valid_shape = value.ndim == 2 and value.shape[1:] == (2,)
            elif name == "raw_predicted_chunks":
                valid_shape = value.ndim == 3 and value.shape[1:] == (9, 2)
            if (
                value.dtype != np.float32
                or not valid_shape
            ):
                raise ValueError(
                    f"{name} must be a float32 NumPy array with shape {shape_text}"
                )

        action_count = self.executed_action_count
        if len(self.requested_actions) != action_count:
            raise ValueError(
                "executed_action_count must match requested_actions length"
            )
        if len(self.executed_positions) != action_count + 1:
            raise ValueError("executed_positions length must equal K + 1")
        if not np.array_equal(
            self.initial_observation[:2],
            self.executed_positions[0],
            equal_nan=True,
        ):
            raise ValueError(
                "initial_observation position must match executed_positions[0]"
            )

        chunk_count = len(self.raw_predicted_chunks)
        if chunk_count == 0:
            valid_chunk_relationship = action_count == 0
        else:
            valid_chunk_relationship = (
                8 * (chunk_count - 1) <= action_count <= 8 * chunk_count
            )
        if not valid_chunk_relationship:
            raise ValueError("chunk count is inconsistent with action count")

        required_info_counts = (
            "action_attempt_count",
            "step_index",
            "action_limit_activation_count",
        )
        for key in required_info_counts:
            value = self.info.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"info {key} must be a nonnegative integer")
        if self.info["action_attempt_count"] != action_count:
            raise ValueError(
                "info action_attempt_count must match executed_action_count"
            )
        if self.info["action_limit_activation_count"] > action_count:
            raise ValueError("action_limit_activation_count cannot exceed K")

        for key, expected in (
            ("success", self.success),
            ("numerical_failure", self.numerical_failure),
            ("action_limit_failure", self.action_limit_failure),
        ):
            value = self.info.get(key)
            if type(value) is not bool:
                raise ValueError(f"info {key} must be a bool")
            if value is not expected:
                raise ValueError(f"info {key} must match rollout {key}")
        if self.success and (
            self.numerical_failure or self.action_limit_failure
        ):
            raise ValueError("success cannot be true when a failure flag is set")
        if self.action_limit_failure != (
            self.info["action_limit_activation_count"] > 0
        ):
            raise ValueError(
                "action_limit_failure must match action_limit_activation_count"
            )

        accepted_transitions = sum(
            np.array_equal(position, request)
            for position, request in zip(
                self.executed_positions[1:],
                self.requested_actions,
            )
        )
        if self.info["step_index"] != accepted_transitions:
            raise ValueError("info step_index must match accepted transitions")
        if action_count - accepted_transitions not in (0, 1):
            raise ValueError("at most one terminal failed action is allowed")
        if action_count > accepted_transitions:
            failed_index = action_count - 1
            if not all(
                np.array_equal(position, request)
                for position, request in zip(
                    self.executed_positions[1:failed_index + 1],
                    self.requested_actions[:failed_index],
                )
            ):
                raise ValueError("accepted transitions must precede a failed action")
            if not np.array_equal(
                self.executed_positions[-1],
                self.executed_positions[-2],
                equal_nan=True,
            ):
                raise ValueError("a failed action must leave the state unchanged")

        for key in (
            "normalized_anchor_discrepancies",
            "physical_anchor_discrepancies",
        ):
            if key not in self.info:
                continue
            value = self.info[key]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.shape != (chunk_count,)
            ):
                raise ValueError(
                    f"info {key} must be a float32 array with shape [C]"
                )

        has_nonfinite = not all(
            np.isfinite(value).all()
            for value in (
                self.initial_observation,
                self.executed_positions,
                self.requested_actions,
                self.raw_predicted_chunks,
            )
        )
        if has_nonfinite and not self.numerical_failure:
            raise ValueError(
                "non-finite rollout arrays require numerical_failure=True"
            )

        gym_numerical_failure = self.info.get("gym_numerical_failure")
        if type(gym_numerical_failure) is not bool:
            raise ValueError("info gym_numerical_failure must be a bool")
        policy_generation_failure = self.info.get(
            "policy_generation_numerical_failure"
        )
        if type(policy_generation_failure) is not bool:
            raise ValueError(
                "info policy_generation_numerical_failure must be a bool"
            )
        nonfinite_chunk_indices = self.info.get(
            "policy_generation_nonfinite_chunk_indices"
        )
        if (
            type(nonfinite_chunk_indices) is not list
            or any(
                type(index) is not int or not 0 <= index < chunk_count
                for index in nonfinite_chunk_indices
            )
            or nonfinite_chunk_indices != sorted(set(nonfinite_chunk_indices))
        ):
            raise ValueError(
                "info policy generation nonfinite chunk indices are invalid"
            )
        observed_nonfinite_chunk_indices = [
            index
            for index, chunk in enumerate(self.raw_predicted_chunks)
            if not np.isfinite(chunk).all()
        ]
        if (
            nonfinite_chunk_indices != observed_nonfinite_chunk_indices
            or policy_generation_failure != bool(nonfinite_chunk_indices)
        ):
            raise ValueError(
                "policy generation nonfinite chunk indices must match "
                "raw_predicted_chunks"
            )
        if self.numerical_failure != (
            gym_numerical_failure or policy_generation_failure
        ):
            raise ValueError(
                "rollout numerical_failure must match combined numerical_failure"
            )


@dataclass(frozen=True)
class SFPDRolloutBatch:
    rollouts: tuple[SFPDRollout, ...]

    @classmethod
    def from_rollouts(
        cls,
        rollouts: Sequence[SFPDRollout],
    ) -> "SFPDRolloutBatch":
        if len(rollouts) == 0:
            raise ValueError("at least one rollout is required")
        if not all(isinstance(rollout, SFPDRollout) for rollout in rollouts):
            raise ValueError("rollouts must contain SFPDRollout values")
        return cls(tuple(rollouts))


def _policy_device(policy: object) -> torch.device:
    return torch.device(getattr(policy, "device", "cpu"))


def rollout_sfpd(
    policy: object,
    stats: PushTStats,
    environment_config: EnvironmentConfig,
    *,
    seed: int,
    center_init: bool,
    integration_steps_per_action: int,
) -> SFPDRollout:
    """Run one 8-action-per-chunk rollout through the real Gym environment."""
    environment = PointReach2DPreferenceEnv(config=environment_config)
    try:
        observation, info = environment.reset(
            seed=seed,
            options={"center_init": center_init},
        )
        initial_observation = observation.copy()
        observation_history: deque[np.ndarray] = deque(
            (observation.copy(), observation.copy()),
            maxlen=2,
        )
        executed_positions = [observation[:2].copy()]
        requested_actions: list[np.ndarray] = []
        predicted_chunks: list[np.ndarray] = []
        normalized_anchor_discrepancies: list[np.float32] = []
        physical_anchor_discrepancies: list[np.float32] = []
        policy_generation_nonfinite_chunk_indices: list[int] = []
        terminated = False
        truncated = False
        device = _policy_device(policy)

        while not (terminated or truncated):
            raw_observations = np.stack(observation_history).astype(
                np.float32,
                copy=False,
            )
            normalized_observations = normalize_observations(
                raw_observations,
                stats,
            )
            nobs = torch.from_numpy(normalized_observations).to(device=device)
            normalized_prediction = policy.predict(
                nobs,
                num_actions=9,
                integration_steps_per_action=integration_steps_per_action,
            )
            if not isinstance(normalized_prediction, torch.Tensor):
                raise ValueError("policy prediction must be a torch.Tensor")
            if normalized_prediction.shape != (1, 9, 2):
                raise ValueError("policy prediction must have shape [1, 9, 2]")
            if normalized_prediction.dtype != torch.float32:
                raise ValueError("policy prediction must have dtype float32")
            normalized_chunk = (
                normalized_prediction[0].detach().to(device="cpu").numpy().copy()
            )
            physical_chunk = unnormalize_actions(normalized_chunk, stats)
            predicted_chunks.append(physical_chunk.copy())
            if not (
                np.isfinite(normalized_chunk).all()
                and np.isfinite(physical_chunk).all()
            ):
                policy_generation_nonfinite_chunk_indices.append(
                    len(predicted_chunks) - 1
                )
            normalized_anchor_discrepancies.append(
                np.float32(
                    np.linalg.norm(
                        normalized_chunk[0] - normalized_observations[-1, :2]
                    )
                )
            )
            physical_anchor_discrepancies.append(
                np.float32(
                    np.linalg.norm(physical_chunk[0] - raw_observations[-1, :2])
                )
            )

            for action in physical_chunk[1:]:
                requested_actions.append(action.copy())
                observation, _, terminated, truncated, info = environment.step(action)
                executed_positions.append(observation[:2].copy())
                observation_history.append(observation.copy())
                if terminated or truncated:
                    break

        gym_numerical_failure = bool(info["numerical_failure"])
        policy_generation_numerical_failure = bool(
            policy_generation_nonfinite_chunk_indices
        )
        numerical_failure = (
            gym_numerical_failure or policy_generation_numerical_failure
        )
        success = bool(info["success"]) and not numerical_failure
        final_info = dict(info)
        final_info.update(
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
            numerical_failure=numerical_failure,
            gym_numerical_failure=gym_numerical_failure,
            policy_generation_numerical_failure=(
                policy_generation_numerical_failure
            ),
            policy_generation_nonfinite_chunk_indices=list(
                policy_generation_nonfinite_chunk_indices
            ),
            normalized_anchor_discrepancies=np.asarray(
                normalized_anchor_discrepancies,
                dtype=np.float32,
            ),
            physical_anchor_discrepancies=np.asarray(
                physical_anchor_discrepancies,
                dtype=np.float32,
            ),
        )
        requested_array = np.asarray(requested_actions, dtype=np.float32).reshape(-1, 2)
        return SFPDRollout(
            seed=seed,
            initial_observation=initial_observation.astype(np.float32, copy=False),
            executed_positions=np.asarray(
                executed_positions,
                dtype=np.float32,
            ).reshape(-1, 2),
            requested_actions=requested_array,
            raw_predicted_chunks=np.asarray(
                predicted_chunks,
                dtype=np.float32,
            ).reshape(-1, 9, 2),
            executed_action_count=len(requested_actions),
            success=success,
            numerical_failure=numerical_failure,
            action_limit_failure=bool(info["action_limit_failure"]),
            info=final_info,
        )
    finally:
        environment.close()


def _discrepancy_values(
    batch: SFPDRolloutBatch,
    key: str,
) -> np.ndarray:
    values = [
        np.asarray(rollout.info.get(key, ()), dtype=np.float32).reshape(-1)
        for rollout in batch.rollouts
    ]
    nonempty = [value for value in values if len(value)]
    if not nonempty:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(nonempty).astype(np.float32, copy=False)


def aggregate_sfpd_metrics(
    batch: SFPDRolloutBatch,
    environment_config: EnvironmentConfig,
) -> dict[str, Any]:
    """Aggregate Gym failures, geometry, and midpoint mode occupancy."""
    rollout_count = len(batch.rollouts)
    successes = [rollout.success for rollout in batch.rollouts]
    final_goal_errors = np.asarray(
        [
            np.linalg.norm(
                rollout.executed_positions[-1] - environment_config.goal_array()
            )
            for rollout in batch.rollouts
        ],
        dtype=np.float32,
    )
    numerical_failure_indices = [
        index
        for index, rollout in enumerate(batch.rollouts)
        if rollout.numerical_failure
    ]
    action_limit_failure_indices = [
        index
        for index, rollout in enumerate(batch.rollouts)
        if rollout.action_limit_failure
    ]
    action_attempt_count = sum(
        int(rollout.info.get("action_attempt_count", rollout.executed_action_count))
        for rollout in batch.rollouts
    )
    action_limit_activation_count = sum(
        int(rollout.info.get("action_limit_activation_count", 0))
        for rollout in batch.rollouts
    )
    requested_step_maxima = [
        float(value)
        for rollout in batch.rollouts
        if (value := rollout.info.get("max_requested_step_distance")) is not None
        and math.isfinite(float(value))
    ]

    midpoint_positions: list[np.ndarray] = []
    failed_before_midpoint_indices: list[int] = []
    for index, rollout in enumerate(batch.rollouts):
        reached_state_index = int(
            rollout.info.get("step_index", len(rollout.executed_positions) - 1)
        )
        if reached_state_index < 32 or len(rollout.executed_positions) <= 32:
            failed_before_midpoint_indices.append(index)
            continue
        midpoint_positions.append(rollout.executed_positions[32])
    if midpoint_positions:
        midpoint_array = np.asarray(midpoint_positions, dtype=np.float32)
        midpoint_occupancy = classify_midpoint_modes(
            torch.from_numpy(midpoint_array)
        )
    else:
        midpoint_occupancy = {
            "upper-narrow": 0.0,
            "upper-wide": 0.0,
            "lower-narrow": 0.0,
            "lower-wide": 0.0,
            "other": 0.0,
            "nonfinite": 0.0,
        }

    normalized_anchor = _discrepancy_values(
        batch,
        "normalized_anchor_discrepancies",
    )
    physical_anchor = _discrepancy_values(
        batch,
        "physical_anchor_discrepancies",
    )
    finite_normalized_anchor = normalized_anchor[np.isfinite(normalized_anchor)]
    finite_physical_anchor = physical_anchor[np.isfinite(physical_anchor)]
    return {
        "rollout_count": rollout_count,
        "goal_success_count": int(sum(successes)),
        "goal_success_rate": float(sum(successes) / rollout_count),
        "final_goal_errors": final_goal_errors.tolist(),
        "final_goal_error_mean": float(final_goal_errors.mean()),
        "final_goal_error_median": float(np.median(final_goal_errors)),
        "final_goal_error_max": float(final_goal_errors.max()),
        "numerical_failure_count": len(numerical_failure_indices),
        "numerical_failure_indices": numerical_failure_indices,
        "action_limit_failure_count": len(action_limit_failure_indices),
        "action_limit_failure_indices": action_limit_failure_indices,
        "action_limit_activation_count": action_limit_activation_count,
        "action_limit_activation_rate": (
            action_limit_activation_count / action_attempt_count
            if action_attempt_count
            else 0.0
        ),
        "action_attempt_count": action_attempt_count,
        "max_requested_step_distance": (
            max(requested_step_maxima) if requested_step_maxima else None
        ),
        "normalized_anchor_discrepancy_mean": (
            float(finite_normalized_anchor.mean())
            if len(finite_normalized_anchor)
            else None
        ),
        "normalized_anchor_discrepancy_max": (
            float(finite_normalized_anchor.max())
            if len(finite_normalized_anchor)
            else None
        ),
        "normalized_anchor_discrepancy_nonfinite_count": int(
            len(normalized_anchor) - len(finite_normalized_anchor)
        ),
        "physical_anchor_discrepancy_mean": (
            float(finite_physical_anchor.mean())
            if len(finite_physical_anchor)
            else None
        ),
        "physical_anchor_discrepancy_max": (
            float(finite_physical_anchor.max())
            if len(finite_physical_anchor)
            else None
        ),
        "physical_anchor_discrepancy_nonfinite_count": int(
            len(physical_anchor) - len(finite_physical_anchor)
        ),
        "midpoint_classified_count": len(midpoint_positions),
        "failed_before_midpoint_count": len(failed_before_midpoint_indices),
        "failed_before_midpoint_indices": failed_before_midpoint_indices,
        "midpoint_occupancy": midpoint_occupancy,
    }


def stage_b1_acceptance_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Return the fixed B1 development-gate failures without changing thresholds."""
    failures: list[str] = []
    if metrics["goal_success_rate"] < 0.95:
        failures.append("goal success rate is below 0.95")
    if metrics["numerical_failure_count"] != 0:
        failures.append("numerical failures are nonzero")
    for mode in (
        "upper-narrow",
        "upper-wide",
        "lower-narrow",
        "lower-wide",
    ):
        occupancy = metrics["midpoint_occupancy"][mode]
        if occupancy < 0.05:
            failures.append(f"{mode} occupancy is below 0.05")
        if abs(occupancy - 0.25) > 0.10:
            failures.append(
                f"{mode} occupancy differs from 0.25 by more than 0.10"
            )
    if metrics["midpoint_occupancy"]["other"] > 0.05:
        failures.append("other occupancy is above 0.05")
    return failures


def evaluate_sfpd(
    policy: object,
    stats: PushTStats,
    environment_config: EnvironmentConfig,
    rollout_seeds: Sequence[int],
    *,
    center_init: bool,
    integration_steps_per_action: int,
) -> tuple[dict[str, Any], SFPDRolloutBatch]:
    """Evaluate explicit rollout seeds and attach the fixed development gates."""
    batch = SFPDRolloutBatch.from_rollouts(
        [
            rollout_sfpd(
                policy,
                stats,
                environment_config,
                seed=seed,
                center_init=center_init,
                integration_steps_per_action=integration_steps_per_action,
            )
            for seed in rollout_seeds
        ]
    )
    metrics = aggregate_sfpd_metrics(batch, environment_config)
    metrics["acceptance_failures"] = stage_b1_acceptance_failures(metrics)
    return metrics, batch
