"""Conditional base-policy futures of fixed current SFPS latent candidates."""

from dataclasses import dataclass, fields

import numpy as np

from env.config import EnvironmentConfig
from env.grouped_preference import GroupedPreference, grouped_features, grouped_utility
from env.preference_rollout import ContinuationBatch, simulate_continuations


@dataclass(frozen=True)
class ConditionalContinuationBatch:
    """Candidate arrays use [B,M]; continuation rows flatten [B,M,L] in C order.

    Observed anchors are physical positions and never commands. Model anchors
    are the actual returned point zero unnormalized with frozen action stats.
    Failed current candidates have their partial path repeated L times without
    further model evaluation. All incomplete paths contribute zero features.
    Future failure fractions count all task failures, including goal misses.
    """

    observed_anchors: np.ndarray       # [B,M,2]
    model_anchors: np.ndarray          # [B,M,2]
    current_commands: np.ndarray       # [B,M,8,2], NaN after rejected command
    current_valid: np.ndarray          # [B,M]
    continuations: ContinuationBatch
    continuation_features: np.ndarray  # [B,M,L,2,2]
    expected_features: np.ndarray      # [B,M,2,2]
    goal_costs: np.ndarray             # [B,M]
    future_failure_fractions: np.ndarray  # [B,M]


def evaluate_conditional_continuations(
    policy, stats, prefixes, current_latents, future_latents, *,
    environment_config: EnvironmentConfig, integration_steps_per_action: int = 6,
    feature_kind: str = "continuous",
) -> ConditionalContinuationBatch:
    """Integrate each current chunk once, then branch into independent futures.

    Inputs are finite float32 prefixes [B,q+1,2], current latents [B,M,2],
    and future latents [B,M,L,(64-q)/8-1,2]. q is in {0,8,...,56}.
    Costs are mean((goal_error/.1)^2 + 25*(not success)); their float64
    arithmetic uses no clipping or cap. Unrepresentable costs raise explicitly.
    """
    if feature_kind not in ("continuous", "mode"):
        raise ValueError("feature_kind must be continuous or mode")
    for name, values in (("prefixes", prefixes), ("current_latents", current_latents),
                         ("future_latents", future_latents)):
        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite float32 NumPy array")
    if prefixes.ndim != 3 or not len(prefixes) or prefixes.shape[2] != 2:
        raise ValueError("prefixes must have shape [B,q+1,2]")
    q = prefixes.shape[1] - 1
    if q < 0 or q > 56 or q % 8:
        raise ValueError("q must be an eight-action boundary from 0 through 56")
    b = len(prefixes)
    if current_latents.ndim != 3 or current_latents.shape[0] != b or current_latents.shape[2] != 2 or not current_latents.shape[1]:
        raise ValueError("current_latents must have shape [B,M,2] with M positive")
    m, remaining = current_latents.shape[1], (64-q)//8
    if future_latents.ndim != 5 or future_latents.shape[:2] != (b, m) or future_latents.shape[3:] != (remaining-1, 2) or not future_latents.shape[2]:
        raise ValueError("future_latents must have shape [B,M,L,R-1,2] with L positive")
    if getattr(policy, "pred_horizon", 16) != 16:
        raise ValueError("conditional SFPS evaluation requires model horizon 16")
    l = future_latents.shape[2]
    schedule = np.zeros((b*m, remaining, 2), np.float32)
    schedule[:, 0] = current_latents.reshape(-1, 2)
    current = simulate_continuations(
        policy, stats, np.repeat(prefixes, m, axis=0), schedule,
        environment_config=environment_config,
        integration_steps_per_action=integration_steps_per_action, stop_at=q+8)
    valid = ((current.lengths == q+9) & ~current.numerical_failure & ~current.action_limit_failure)
    # Retain every failure in the estimator denominator; branch only valid rows.
    combined = {field.name: np.repeat(getattr(current, field.name), l, axis=0)
                for field in fields(ContinuationBatch)}
    rows = np.flatnonzero(np.repeat(valid, l))
    if len(rows):
        futures = simulate_continuations(
            policy, stats, combined["positions"][rows, :q+9],
            future_latents.reshape(b*m*l, remaining-1, 2)[rows],
            environment_config=environment_config,
            integration_steps_per_action=integration_steps_per_action)
        for field in fields(ContinuationBatch):
            if field.name == "model_anchors":
                combined[field.name][rows, (q+8)//8:] = futures.model_anchors[:, (q+8)//8:]
            else:
                combined[field.name][rows] = getattr(futures, field.name)
    full = ContinuationBatch(**combined)
    complete = ((full.lengths == 65) & ~full.numerical_failure & ~full.action_limit_failure)
    features = np.zeros((b*m*l, 2, 2), np.float64)
    features[complete] = grouped_features(
        full.positions[complete], feature_kind=feature_kind)
    features = features.reshape(b, m, l, 2, 2)
    # Recompute the cost distance in float64: float32 norms can overflow for
    # arbitrary finite fixtures even when the distance squared fits float64.
    final = full.positions[np.arange(b*m*l), full.lengths-1].astype(np.float64)
    errors = np.linalg.norm(final-environment_config.goal_array().astype(np.float64), axis=-1)
    costs = (errors/.1)**2 + 25*(~full.success)
    if not np.isfinite(costs).all():
        raise ValueError("goal costs are not representable in float64")
    return ConditionalContinuationBatch(
        np.repeat(prefixes[:, -1:, :], m, axis=1),
        current.model_anchors[:, q//8].reshape(b, m, 2),
        current.requested_actions[:, q:q+8].reshape(b, m, 8, 2),
        valid.reshape(b, m), full, features, features.mean(axis=2),
        costs.reshape(b, m, l).mean(axis=2),
        (~full.success).reshape(b, m, l).mean(axis=2))


def conditional_scores(batch: ConditionalContinuationBatch,
                       preference: GroupedPreference | None, *, beta: float = 16.) -> np.ndarray:
    """Task-conditioned logits: beta * U(E[features]) - E[task cost]."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    utility = 0. if preference is None else grouped_utility(batch.expected_features, preference)
    scores = beta*utility-batch.goal_costs
    if not np.isfinite(scores[batch.current_valid]).all():
        raise ValueError("valid conditional scores must be finite")
    return scores
