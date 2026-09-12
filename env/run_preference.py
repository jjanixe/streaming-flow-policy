"""Synthetic-user pilot: frozen SFPS, full-future candidates, and few-shot BT."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from env.artifacts import (load_demonstration_bank, save_feature_normalizer,
                           save_json, train_data_digest)
from env.config import DEFAULT_CONFIG
from env.demonstrations import evaluate_path
from env.environment import PointReach2DPreferenceEnv
from env.features import fit_feature_normalizer, trajectory_features
from env.preference import (fit_bradley_terry, preference_metrics,
                            preference_probabilities)
from env.preference_rollout import select_candidates, simulate_continuations
from env.run_stage_b import load_trained_sfps


USERS = {"upper_narrow": (2.0, -1.0), "upper_wide": (2.0, 1.0),
         "lower_narrow": (-2.0, -1.0), "lower_wide": (-2.0, 1.0)}
HORIZON = 64
EXECUTION_HORIZON = 8


def _seed(root: int, *parts: int) -> int:
    return int(np.random.SeedSequence([root, *parts]).generate_state(1)[0])


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_comparison_queries(*, seed: int, count: int = 40) -> dict:
    """Generate paired clean full paths without consulting any user weights."""
    if type(seed) is not int or seed < 0 or type(count) is not int or count <= 0:
        raise ValueError("seed must be nonnegative and count positive integers")
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 1, HORIZON + 1, dtype=np.float32)
    trajectories = np.empty((count, 2, HORIZON + 1, 2), dtype=np.float32)
    kinds = []
    for i in range(count):
        sign = float(rng.choice([-1, 1]))
        narrow, wide = rng.uniform(0.22, 0.32), rng.uniform(0.48, 0.58)
        if i % 3 == 0:
            amplitude = rng.choice([narrow, wide])
            pair = ((sign, amplitude), (-sign, amplitude))
            kinds.append("direction")
        elif i % 3 == 1:
            pair = ((sign, narrow), (sign, wide))
            kinds.append("width")
        else:
            pair = ((sign, narrow), (-sign, wide))
            kinds.append("tradeoff")
        if rng.integers(2):
            pair = pair[::-1]
        for j, (direction, amplitude) in enumerate(pair):
            trajectories[i, j] = evaluate_path(direction, amplitude, times)[0]
    ids = np.array([[f"query-{seed}-{i}-{j}" for j in range(2)]
                    for i in range(count)])
    return {"trajectories": trajectories, "ids": ids, "types": np.array(kinds)}


def _labels(delta: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    probabilities = preference_probabilities(delta, weights)
    draws = np.random.default_rng(seed).random(len(delta))
    return np.where(draws < probabilities, 1, -1).astype(np.int8)


def _initial_positions(seeds: list[int], centered: bool) -> np.ndarray:
    positions = []
    for seed in seeds:
        instance = PointReach2DPreferenceEnv()
        try:
            obs, _ = instance.reset(seed=seed, options={"center_init": centered})
            positions.append(obs[:2])
        finally:
            instance.close()
    return np.asarray(positions, dtype=np.float32)


def _path_metrics(positions, lengths, success, numerical, limits, goal_errors,
                  normalizer, users) -> dict:
    complete = (lengths == HORIZON + 1) & ~numerical & ~limits
    features = np.full((len(positions), 2), np.nan, dtype=np.float32)
    if complete.any():
        features[complete] = trajectory_features(positions[complete], normalizer, 1 / HORIZON)
    reached = lengths > HORIZON // 2
    midpoints = positions[reached, HORIZON // 2, 1]
    modes = {"upper_narrow": (midpoints >= 0.12) & (midpoints < 0.4),
             "upper_wide": midpoints >= 0.4,
             "lower_narrow": (midpoints <= -0.12) & (midpoints > -0.4),
             "lower_wide": midpoints <= -0.4,
             "other": np.abs(midpoints) < 0.12}
    count, successes = len(positions), int(success.sum())
    # Wilson interval includes every episode, including failures.
    rate = successes / count
    center = (rate + 1.96**2 / (2 * count)) / (1 + 1.96**2 / count)
    half = 1.96 * np.sqrt(rate * (1 - rate) / count + 1.96**2 / (4 * count**2)) / (
        1 + 1.96**2 / count)
    result = {"count": count, "success_count": successes, "success_rate": rate,
              "success_wilson_95": [center - half, center + half],
              "complete_count": int(complete.sum()),
              "incomplete_count": int((~complete).sum()),
              "numerical_failure_count": int(numerical.sum()),
              "action_limit_failure_count": int(limits.sum()),
              "midpoint_denominator": int(reached.sum()),
              "midpoint_counts": {k: int(v.sum()) for k, v in modes.items()},
              "mean_goal_error": float(np.mean(goal_errors)), "utility_by_user": {}}
    for name, weight in users.items():
        utility = features @ np.asarray(weight, dtype=np.float32)
        successful = utility[success & complete]
        result["utility_by_user"][name] = {
            "complete_mean": float(np.mean(utility[complete])) if complete.any() else None,
            "successful_mean": float(np.mean(successful)) if len(successful) else None,
            "successful_standard_error": float(np.std(successful, ddof=1) / np.sqrt(len(successful)))
            if len(successful) > 1 else None}
    return result


def _closed_loop(policy, stats, normalizer, weights, *, environment_seeds,
                 rollout_seeds, centered, candidates, method, beta, integration_steps):
    """Replan in batches, but execute selected commands through actual Gym."""
    count = len(environment_seeds)
    envs = [PointReach2DPreferenceEnv() for _ in range(count)]
    positions = np.full((count, 65, 2), np.nan, dtype=np.float32)
    requests = np.full((count, 64, 2), np.nan, dtype=np.float32)
    lengths = np.ones(count, dtype=np.int64)
    active = np.ones(count, dtype=bool)
    success = np.zeros(count, dtype=bool)
    numerical = np.zeros(count, dtype=bool)
    limits = np.zeros(count, dtype=bool)
    goal_errors = np.zeros(count, dtype=np.float32)
    candidate_count = 1 if method == "base" else candidates
    selected_indices = np.full((count, 8), -1, dtype=np.int64)
    candidate_scores = np.full((count, 8, candidate_count), np.nan, dtype=np.float32)
    candidate_valid = np.zeros((count, 8, candidate_count), dtype=bool)
    selected_futures = np.full((count, 8, 65, 2), np.nan, dtype=np.float32)
    selected_latents = np.full((count, 8, 8, 2), np.nan, dtype=np.float32)
    ess = np.full((count, 8), np.nan, dtype=np.float32)
    fallback = np.zeros((count, 8), dtype=bool)
    decision_seconds = []
    try:
        for i, (instance, seed) in enumerate(zip(envs, environment_seeds)):
            observation, info = instance.reset(seed=seed, options={"center_init": centered})
            positions[i, 0] = observation[:2]
            goal_errors[i] = info["goal_error"]
        for chunk in range(8):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            started = time.perf_counter()
            q, remaining = chunk * 8, 8 - chunk
            latent_sequences = np.stack([
                np.random.default_rng(_seed(rollout_seeds[i], 10, chunk)).standard_normal(
                    (candidate_count, remaining, 2)).astype(np.float32) for i in indices])
            prefixes = np.repeat(positions[indices, :q + 1], candidate_count, axis=0)
            futures = simulate_continuations(policy, stats, prefixes,
                latent_sequences.reshape(-1, remaining, 2),
                environment_config=DEFAULT_CONFIG.environment,
                integration_steps_per_action=integration_steps)
            valid = (futures.lengths == 65) & ~futures.numerical_failure & ~futures.action_limit_failure
            scores = np.full(len(valid), -np.inf, dtype=np.float32)
            if valid.any():
                features = trajectory_features(futures.positions[valid], normalizer, 1 / 64)
                scores[valid] = beta * (features @ weights) - (
                    futures.goal_errors[valid] / DEFAULT_CONFIG.environment.goal_tolerance)**2
            for row, i in enumerate(indices):
                section = slice(row * candidate_count, (row + 1) * candidate_count)
                rng = np.random.default_rng(_seed(rollout_seeds[i], 20, chunk))
                selected, effective, failed_selection = select_candidates(
                    scores[section], valid[section], method=method, rng=rng)
                chosen = row * candidate_count + selected
                selected_indices[i, chunk] = selected
                candidate_scores[i, chunk] = scores[section]
                candidate_valid[i, chunk] = valid[section]
                selected_futures[i, chunk] = futures.positions[chosen]
                selected_latents[i, chunk, :remaining] = latent_sequences[row, selected]
                ess[i, chunk], fallback[i, chunk] = effective, failed_selection
                for step, action in enumerate(futures.requested_actions[chosen, q:q + 8]):
                    requests[i, q + step] = action
                    observation, _, done, truncated, info = envs[i].step(action)
                    numerical[i] = info["numerical_failure"]
                    limits[i] = info["action_limit_failure"]
                    if not numerical[i] and not limits[i]:
                        positions[i, lengths[i]] = observation[:2]
                        lengths[i] += 1
                    goal_errors[i] = info["goal_error"]
                    success[i] = info["success"]
                    if done or truncated:
                        active[i] = False
                        break
            decision_seconds.append(time.perf_counter() - started)
    finally:
        for instance in envs:
            instance.close()
    return dict(positions=positions, requested_actions=requests, lengths=lengths,
                success=success, numerical_failure=numerical, action_limit_failure=limits,
                goal_errors=goal_errors, selected_indices=selected_indices,
                candidate_scores=candidate_scores, candidate_valid=candidate_valid,
                selected_futures=selected_futures, selected_latents=selected_latents,
                ess=ess, fallback=fallback, environment_seeds=np.array(environment_seeds, dtype=np.uint32),
                rollout_seeds=np.array(rollout_seeds, dtype=np.uint32),
                batch_replan_seconds=np.array(decision_seconds, dtype=np.float32))


def run_preference(output_dir, *, checkpoint_path, stats_path, demonstrations_path,
                   seed=0, device="cpu", candidates=32, rollout_count=16,
                   coverage_count=512, integration_steps_per_action=6,
                   budgets=(5, 10, 20, 40), user_weights=None, make_plots=True,
                   progress: Callable[[str], None] | None = None) -> dict:
    for name, value in (("candidates", candidates), ("rollout_count", rollout_count),
                        ("coverage_count", coverage_count), ("integration_steps_per_action", integration_steps_per_action)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not budgets or any(type(k) is not int or k <= 0 for k in budgets):
        raise ValueError("budgets must contain positive integers")
    users = dict(USERS if user_weights is None else user_weights)
    if not users or any(np.asarray(w).shape != (2,) or not np.isfinite(w).all()
                        for w in users.values()):
        raise ValueError("users must have finite two-dimensional weights")
    if any(not name.replace("_", "").isalnum() for name in users):
        raise ValueError("user names must contain only letters, digits and underscores")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory must be empty; previous experiments are preserved")
    destination.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda message: None)
    started = time.perf_counter()
    bank = load_demonstration_bank(demonstrations_path)
    train_digest = train_data_digest(bank)
    checkpoint_digest = _digest(checkpoint_path)
    policy, stats, metadata = load_trained_sfps(checkpoint_path, stats_path,
        expected_train_data_digest=train_digest, device=device)
    policy.requires_grad_(False)
    before_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    save_feature_normalizer(destination / "feature_normalizer.npz", normalizer, train_digest)
    config = dict(seed=seed, device=str(device), candidates=candidates,
                  rollout_count=rollout_count, coverage_count=coverage_count,
                  integration_steps_per_action=integration_steps_per_action,
                  budgets=list(budgets), users=users, beta=1.0, l2=0.1,
                  checkpoint_path=str(Path(checkpoint_path).resolve()),
                  stats_path=str(Path(stats_path).resolve()),
                  demonstrations_path=str(Path(demonstrations_path).resolve()),
                  torch_threads=torch.get_num_threads(),
                  environment=asdict(DEFAULT_CONFIG.environment))
    save_json(destination / "resolved_config.json", _json_ready(config))
    result = dict(synthetic_users=True, checkpoint_digest=checkpoint_digest,
                  train_data_digest=train_digest, policy_unchanged=False,
                  checkpoint_architecture=metadata["architecture"], config=config,
                  coverage={}, preference_fits={}, conditions={},
                  limitations=["Synthetic labels, fixed two-feature utility, small pilot.",
                    "Utility excludes incomplete paths; task success includes every episode.",
                    "Soft selection is finite-candidate resampling, not exact path sampling.",
                    "Batch replan latency is not single-robot control latency."])
    queries = build_comparison_queries(seed=_seed(seed, 1), count=max(budgets))
    heldout = build_comparison_queries(seed=_seed(seed, 2), count=256)
    q_features = trajectory_features(queries["trajectories"], normalizer, 1 / 64)
    h_features = trajectory_features(heldout["trajectories"], normalizer, 1 / 64)
    delta, held_delta = q_features[:, 0] - q_features[:, 1], h_features[:, 0] - h_features[:, 1]
    preference_data = {"query_trajectories": queries["trajectories"], "query_ids": queries["ids"],
                       "query_types": queries["types"], "query_delta_features": delta,
                       "heldout_trajectories": heldout["trajectories"], "heldout_ids": heldout["ids"],
                       "heldout_delta_features": held_delta}
    fitted = {}
    for index, (name, weight) in enumerate(users.items()):
        w = np.asarray(weight, dtype=np.float32)
        labels = _labels(delta, w, _seed(seed, 3, index))
        held_labels = _labels(held_delta, w, _seed(seed, 4, index))
        preference_data[f"{name}_labels"] = labels
        preference_data[f"{name}_heldout_labels"] = held_labels
        preference_data[f"{name}_oracle_weights"] = w
        fitted[name] = {}
        result["preference_fits"][name] = {}
        for budget in budgets:
            fit = fit_bradley_terry(delta[:budget], labels[:budget])
            fitted[name][budget] = fit.weights
            record = asdict(fit)
            record["heldout"] = preference_metrics(held_delta, held_labels, fit.weights)
            oracle_labels = np.where(held_delta @ w >= 0, 1, -1).astype(np.int8)
            record["heldout"]["oracle_ordering_accuracy"] = preference_metrics(
                held_delta, oracle_labels, fit.weights)["accuracy"]
            result["preference_fits"][name][str(budget)] = record
            preference_data[f"{name}_weights_{budget}"] = fit.weights
    np.savez_compressed(destination / "preferences.npz", **preference_data)
    for context_index, context in enumerate(("centered", "gaussian_0", "gaussian_1")):
        initial_seed = _seed(seed, 5, context_index)
        initial = _initial_positions([initial_seed], context == "centered")
        latents = np.random.default_rng(_seed(seed, 6, context_index)).standard_normal(
            (coverage_count, 8, 2)).astype(np.float32)
        batch = simulate_continuations(policy, stats, np.repeat(initial[:, None], coverage_count, axis=0),
            latents, environment_config=DEFAULT_CONFIG.environment,
            integration_steps_per_action=integration_steps_per_action)
        coverage_path = f"coverage_{context}.npz"
        np.savez_compressed(destination / coverage_path, **asdict(batch), latents=latents)
        record = _path_metrics(batch.positions, batch.lengths, batch.success,
            batch.numerical_failure, batch.action_limit_failure, batch.goal_errors, normalizer, users)
        record.update(artifact=coverage_path, initial_position=initial[0], environment_seed=initial_seed)
        result["coverage"][context] = record
        complete_success = batch.success & (batch.lengths == 65)
        if context == "centered" and complete_success.sum() >= 2:
            generated_features = trajectory_features(batch.positions[complete_success], normalizer, 1 / 64)
            rng = np.random.default_rng(_seed(seed, 7))
            pairs = np.array([rng.choice(len(generated_features), 2, replace=False) for _ in range(256)])
            generated_delta = generated_features[pairs[:, 0]] - generated_features[pairs[:, 1]]
            generated_payload = dict(delta_features=generated_delta,
                coverage_pair_indices=np.flatnonzero(complete_success)[pairs])
            for index, (name, weight) in enumerate(users.items()):
                w = np.asarray(weight, dtype=np.float32)
                labels = _labels(generated_delta, w, _seed(seed, 8, index))
                generated_payload[f"{name}_labels"] = labels
                for budget in budgets:
                    metrics = preference_metrics(generated_delta, labels, fitted[name][budget])
                    metrics["oracle_ordering_accuracy"] = preference_metrics(generated_delta,
                        np.where(generated_delta @ w >= 0, 1, -1).astype(np.int8),
                        fitted[name][budget])["accuracy"]
                    result["preference_fits"][name][str(budget)]["generated_heldout"] = metrics
            np.savez_compressed(destination / "generated_preferences.npz", **generated_payload)
        notify(f"coverage {context}: success {record['success_count']}/{coverage_count}, {record['midpoint_counts']}")
    conditions = [("base", "base", np.zeros(2, dtype=np.float32)),
                  ("task_only", "soft", np.zeros(2, dtype=np.float32))]
    for name, weight in users.items():
        w = np.asarray(weight, dtype=np.float32)
        conditions.extend([(f"{name}/oracle_soft", "soft", w), (f"{name}/oracle_best", "best", w)])
        conditions.extend((f"{name}/learned_{budget}", "soft", fitted[name][budget]) for budget in budgets)
    for regime in ("centered", "gaussian"):
        environment_seeds = [_seed(seed, 9, i) for i in range(rollout_count)]
        rollout_seeds = [_seed(seed, 10, i) for i in range(rollout_count)]
        for name, method, weights in conditions:
            key = f"{regime}/{name}"
            output = _closed_loop(policy, stats, normalizer, weights,
                environment_seeds=environment_seeds, rollout_seeds=rollout_seeds,
                centered=regime == "centered", candidates=candidates, method=method,
                beta=1.0, integration_steps=integration_steps_per_action)
            artifact = key.replace("/", "--") + ".npz"
            np.savez_compressed(destination / artifact, **output)
            record = _path_metrics(output["positions"], output["lengths"], output["success"],
                output["numerical_failure"], output["action_limit_failure"], output["goal_errors"], normalizer, users)
            valid_decisions = output["selected_indices"] >= 0
            latency = output["batch_replan_seconds"]
            record.update(artifact=artifact, method=method, weights=weights,
                decision_count=int(valid_decisions.sum()), fallback_count=int(output["fallback"].sum()),
                mean_ess=float(np.mean(output["ess"][valid_decisions])),
                mean_valid_candidate_fraction=float(np.mean(output["candidate_valid"][valid_decisions])),
                batch_replan_latency_p50_seconds=float(np.median(latency)),
                batch_replan_latency_p95_seconds=float(np.quantile(latency, .95)))
            result["conditions"][key] = record
            notify(f"{key}: success {record['success_count']}/{rollout_count}, fallback {record['fallback_count']}")
    result["policy_unchanged"] = checkpoint_digest == _digest(checkpoint_path) and all(
        torch.equal(before_state[k], value.detach().cpu()) for k, value in policy.state_dict().items())
    if not result["policy_unchanged"]:
        raise RuntimeError("frozen policy or checkpoint changed during evaluation")
    result["elapsed_seconds"] = time.perf_counter() - started
    result = _json_ready(result)
    save_json(destination / "diagnostics.json", result)
    if make_plots:
        from env.preference_plots import plot_preference_results
        plot_preference_results(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--demonstrations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidates", default=32, type=int)
    parser.add_argument("--rollout-count", default=16, type=int)
    parser.add_argument("--coverage-count", default=512, type=int)
    parser.add_argument("--integration-steps-per-action", default=6, type=int)
    parser.add_argument("--torch-threads", default=1, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    run_preference(args.output_dir, checkpoint_path=args.checkpoint, stats_path=args.stats,
        demonstrations_path=args.demonstrations, seed=args.seed, device=args.device,
        candidates=args.candidates, rollout_count=args.rollout_count,
        coverage_count=args.coverage_count, integration_steps_per_action=args.integration_steps_per_action,
        progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
