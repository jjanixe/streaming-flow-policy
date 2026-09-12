"""Explicit-latent full continuations and selection for frozen SFPS policies."""

from dataclasses import dataclass

import numpy as np
import torch

from env.chunk_data import PushTStats, normalize_observations, unnormalize_actions
from env.config import EnvironmentConfig
from env.sfp_policies import PolicyNumericalError


@dataclass(frozen=True)
class ContinuationBatch:
    """Full paths; lengths count accepted states, excluding rejected commands.

    Requested actions include the accepted prefix and the first rejected command.
    Unattempted commands and unexecuted states are NaN padded. A policy exception
    leaves the next requested action NaN because no physical command was produced.
    model_anchors stores the discarded point zero in physical action coordinates
    at absolute chunk indices [B,8,2]; ungenerated anchors are NaN.
    """

    positions: np.ndarray
    lengths: np.ndarray
    requested_actions: np.ndarray
    numerical_failure: np.ndarray
    action_limit_failure: np.ndarray
    success: np.ndarray
    goal_errors: np.ndarray
    model_anchors: np.ndarray | None = None


def _validate_inputs(prefixes, latents, environment_config, integration_steps_per_action):
    if environment_config.horizon_steps != 64:
        raise ValueError("continuations require a 64-action horizon")
    if (
        not isinstance(integration_steps_per_action, int)
        or isinstance(integration_steps_per_action, bool)
        or integration_steps_per_action <= 0
    ):
        raise ValueError("integration_steps_per_action must be a positive integer")
    for name, value in (("prefixes", prefixes), ("latents", latents)):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise ValueError(f"{name} must be a float32 NumPy array")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if prefixes.ndim != 3 or prefixes.shape[0] == 0 or prefixes.shape[2] != 2:
        raise ValueError("prefixes must have shape [B, q+1, 2] with positive B")
    q = prefixes.shape[1] - 1
    if not 0 <= q <= 64 or q % 8:
        raise ValueError("prefix must end at an eight-action boundary, including 64")
    if latents.shape != (len(prefixes), (64 - q) // 8, 2):
        raise ValueError("latents must have shape [B, (64-q)/8, 2]")
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(np.diff(prefixes, axis=1), axis=2)
    if np.any(distances > np.float32(environment_config.max_step_distance)):
        raise ValueError("prefix contains a rejected over-limit command")
    return q


def _predict(policy, nobs, latents, integration_steps_per_action):
    """Retry individual rows only when a batched numerical exception hides them."""
    def call(observations, latent_rows):
        prediction = policy.predict_batch(
            observations,
            num_actions=9,
            integration_steps_per_action=integration_steps_per_action,
            latents=latent_rows,
        )
        if (
            not isinstance(prediction, torch.Tensor)
            or prediction.shape != (len(observations), 9, 2)
            or prediction.dtype != torch.float32
            or prediction.device != observations.device
        ):
            raise ValueError("policy prediction must be a float32 tensor [B, 9, 2] on the input device")
        return prediction.detach().cpu().numpy()

    failed = np.zeros(len(nobs), dtype=bool)
    try:
        return call(nobs, latents), failed
    except PolicyNumericalError:
        predictions = np.full((len(nobs), 9, 2), np.nan, dtype=np.float32)
        for row in range(len(nobs)):
            try:
                predictions[row] = call(nobs[row:row + 1], latents[row:row + 1])[0]
            except PolicyNumericalError:
                failed[row] = True
        return predictions, failed


@torch.inference_mode()
def simulate_continuations(
    policy: object,
    stats: PushTStats,
    prefixes: np.ndarray,
    latents: np.ndarray,
    *,
    environment_config: EnvironmentConfig,
    integration_steps_per_action: int = 6,
    stop_at: int = 64,
) -> ContinuationBatch:
    """Execute remaining latent chunks using the position-command Gym contract.

    Each chunk predicts nine positions and executes the eight after its anchor.
    Inputs are finite float32 arrays, and prefixes end on a replan boundary.
    stop_at optionally ends on an earlier boundary, preserving full-horizon
    padding and success semantics. Latents still cover the entire remainder.
    Policy and random-generator state are not updated by this helper.
    """
    q = _validate_inputs(prefixes, latents, environment_config, integration_steps_per_action)
    if type(stop_at) is not int or not q <= stop_at <= 64 or stop_at % 8:
        raise ValueError("stop_at must be an eight-action boundary between q and 64")
    batch_size = len(prefixes)
    positions = np.full((batch_size, 65, 2), np.nan, dtype=np.float32)
    requested_actions = np.full((batch_size, 64, 2), np.nan, dtype=np.float32)
    positions[:, :q + 1] = prefixes
    requested_actions[:, :q] = prefixes[:, 1:]
    lengths = np.full(batch_size, q + 1, dtype=np.int64)
    numerical_failure = np.zeros(batch_size, dtype=bool)
    action_limit_failure = np.zeros(batch_size, dtype=bool)
    model_anchors = np.full((batch_size, 8, 2), np.nan, dtype=np.float32)
    device = torch.device(getattr(policy, "device", "cpu"))

    for chunk, start in enumerate(range(q, stop_at, 8)):
        active = np.flatnonzero(~(numerical_failure | action_limit_failure))
        if len(active) == 0:
            break
        history_indices = np.array([max(0, start - 1), start])
        observations = np.empty((len(active), 2, 3), dtype=np.float32)
        observations[:, :, :2] = positions[active[:, None], history_indices]
        observations[:, :, 2] = history_indices.astype(np.float32) / np.float32(64)
        nobs = torch.from_numpy(normalize_observations(observations, stats)).to(device)
        latent_rows = torch.from_numpy(latents[active, chunk]).to(device)
        predictions, generation_failed = _predict(policy, nobs, latent_rows, integration_steps_per_action)
        with np.errstate(over="ignore", invalid="ignore"):
            commands = unnormalize_actions(predictions, stats)
        model_anchors[active, start // 8] = commands[:, 0]
        # The anchor is skipped, but a nonfinite anchor still marks bad generation.
        generation_failed |= ~np.isfinite(commands[:, 0]).all(axis=1)
        numerical_failure[active[generation_failed]] = True

        for offset in range(8):
            eligible = ~(numerical_failure[active] | action_limit_failure[active])
            rows = active[eligible]
            if len(rows) == 0:
                break
            actions = commands[eligible, offset + 1]
            index = start + offset
            requested_actions[rows, index] = actions
            finite = np.isfinite(actions).all(axis=1)
            numerical_failure[rows[~finite]] = True
            with np.errstate(over="ignore", invalid="ignore"):
                distances = np.linalg.norm(actions - positions[rows, index], axis=1)
            limited = finite & (distances > np.float32(environment_config.max_step_distance))
            action_limit_failure[rows[limited]] = True
            accepted = finite & ~limited
            positions[rows[accepted], index + 1] = actions[accepted]
            lengths[rows[accepted]] += 1

    final_positions = positions[np.arange(batch_size), lengths - 1]
    with np.errstate(over="ignore", invalid="ignore"):
        goal_errors = np.linalg.norm(final_positions - environment_config.goal_array(), axis=1)
    success = (
        (lengths == 65)
        & ~(numerical_failure | action_limit_failure)
        # Gym converts the float32 norm to a Python float before comparing.
        & (goal_errors.astype(np.float64) <= environment_config.goal_tolerance)
    )
    return ContinuationBatch(positions, lengths, requested_actions, numerical_failure,
                             action_limit_failure, success, goal_errors, model_anchors)


def select_candidates(scores, valid, *, method, rng):
    """Select pre-scored candidates; scores already include beta and task cost.

    Base always selects zero without filtering. Best/soft fall back to zero only
    when no valid candidate exists. Invalid rows may have NaN or infinite scores.
    ESS is one for deterministic selection and zero for an all-invalid fallback.
    """
    scores = np.asarray(scores)
    valid = np.asarray(valid)
    if scores.ndim != 1 or scores.size == 0 or valid.shape != scores.shape:
        raise ValueError("scores and valid must have matching nonempty vector shapes")
    if valid.dtype != np.bool_ or not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise ValueError("scores must be real numeric values and valid must be boolean")
    if method not in ("base", "best", "soft"):
        raise ValueError("method must be base, best, or soft")
    if method == "base":
        return 0, 1.0, False
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return 0, 0.0, True
    if not np.isfinite(scores[indices]).all():
        raise ValueError("valid candidate scores must be finite")
    if method == "best":
        return int(indices[np.argmax(scores[indices])]), 1.0, False
    # float64 accumulation avoids overflow when subtracting extreme float32
    # scores and gives NumPy's categorical sampler accurately normalized mass.
    logits = scores[indices].astype(np.float64)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(logits - np.max(logits))
    probabilities = weights / weights.sum()
    ess = float(1 / np.square(probabilities).sum())
    return int(rng.choice(indices, p=probabilities)), ess, False
