"""Scoped synthetic feedback and conditional preference steering of frozen SFPS."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from env.artifacts import load_demonstration_bank, train_data_digest
from env.config import DEFAULT_CONFIG
from env.environment import PointReach2DPreferenceEnv
from env.grouped_preference import (GroupedPreference, build_grouped_queries,
    classify_modes, fit_grouped_preference, grouped_features, grouped_metrics,
    grouped_probabilities, grouped_utility, synthetic_grouped_users)
from env.grouped_rollout import conditional_scores, evaluate_conditional_continuations
from env.preference_rollout import select_candidates
from env.run_preference import _digest, _seed
from env.run_stage_b import load_trained_sfps


def draw_latents(rollout_seeds, *, chunk, candidates, continuations):
    """Independent current/future/selection streams, invariant to batch and L.

    Future streams are seeded per candidate, so changing M or L preserves all
    overlapping draws. Base candidate zero matches guided candidate zero.
    """
    if type(chunk) is not int or not 0 <= chunk < 8:
        raise ValueError("chunk must be an integer from 0 through 7")
    if any(type(v) is not int or v <= 0 for v in (candidates, continuations)):
        raise ValueError("candidates and continuations must be positive integers")
    if not len(rollout_seeds) or any(not isinstance(v, (int, np.integer)) or v < 0 for v in rollout_seeds):
        raise ValueError("rollout_seeds must be nonempty nonnegative integers")
    current_seeds = np.array([_seed(int(s), 10, chunk) for s in rollout_seeds], np.uint32)
    future_seeds = np.array([[_seed(int(s), 11, chunk, m) for m in range(candidates)]
                            for s in rollout_seeds], np.uint32)
    selection_seeds = np.array([_seed(int(s), 12, chunk) for s in rollout_seeds], np.uint32)
    current = np.stack([np.random.default_rng(s).standard_normal((candidates, 2)).astype(np.float32)
                        for s in current_seeds])
    future = np.stack([np.stack([np.random.default_rng(s).standard_normal(
        (continuations, 7-chunk, 2)).astype(np.float32) for s in row]) for row in future_seeds])
    return dict(current_latents=current, future_latents=future,
                current_seeds=current_seeds, future_seeds=future_seeds, selection_seeds=selection_seeds)


def _closed_loop(policy, stats, preference, *, environment_seeds, rollout_seeds,
                 centered, candidates, continuations, method, beta, integration_steps,
                 feature_kind='continuous', proposal_std=1.0):
    """Evaluate hypothetical futures, execute only chosen current commands in Gym."""
    if method not in ('base', 'soft', 'best'):
        raise ValueError("method must be base, soft, or best")
    if feature_kind not in ('continuous', 'mode'):
        raise ValueError("feature_kind must be continuous or mode")
    if (isinstance(proposal_std, (bool, np.bool_)) or
            not isinstance(proposal_std, (int, float, np.integer, np.floating)) or
            not np.isfinite(proposal_std) or float(proposal_std) <= 0):
        raise ValueError("proposal_std must be finite and positive")
    proposal_std = float(proposal_std)
    n, m, l = len(environment_seeds), 1 if method == 'base' else candidates, continuations
    if n != len(rollout_seeds) or not n:
        raise ValueError("environment and rollout seeds must have matching nonzero lengths")
    def floats(shape, dtype=np.float32):
        return np.full(shape, np.nan, dtype=dtype)
    candidate_shape = (n, 8, m)
    future_shape = (*candidate_shape, l)
    output = dict(
        positions=floats((n, 65, 2)), requested_actions=floats((n, 64, 2)),
        lengths=np.ones(n, np.int64), success=np.zeros(n, bool),
        numerical_failure=np.zeros(n, bool), action_limit_failure=np.zeros(n, bool),
        goal_errors=np.zeros(n, np.float32),
        decision_mask=np.zeros((n, 8), bool), selected_indices=np.full((n, 8), -1, np.int64),
        selected_current_valid=np.zeros((n, 8), bool),
        selected_current_latents=floats((n, 8, 2)),
        ess=floats((n, 8), np.float64), fallback=np.zeros((n, 8), bool),
        current_latents=floats((*candidate_shape, 2)),
        future_latents=floats((*future_shape, 7, 2)),
        current_seeds=np.zeros((n, 8), np.uint32),
        future_seeds=np.zeros(candidate_shape, np.uint32),
        selection_seeds=np.zeros((n, 8), np.uint32),
        observed_anchors=floats((*candidate_shape, 2)), model_anchors=floats((*candidate_shape, 2)),
        current_commands=floats((*candidate_shape, 8, 2)),
        current_valid=np.zeros(candidate_shape, bool),
        expected_features=floats((*candidate_shape, 2, 2), np.float64),
        expected_utility=floats(candidate_shape, np.float64),
        goal_costs=floats(candidate_shape, np.float64),
        candidate_scores=floats(candidate_shape, np.float64),
        future_failure_fractions=floats(candidate_shape, np.float64),
        continuation_features=floats((*future_shape, 2, 2), np.float64),
        continuation_positions=floats((*future_shape, 65, 2)),
        continuation_requested_actions=floats((*future_shape, 64, 2)),
        continuation_lengths=np.zeros(future_shape, np.int64),
        continuation_success=np.zeros(future_shape, bool),
        continuation_numerical_failure=np.zeros(future_shape, bool),
        continuation_action_limit_failure=np.zeros(future_shape, bool),
        continuation_goal_errors=floats(future_shape),
        continuation_model_anchors=floats((*future_shape, 8, 2)),
        environment_seeds=np.asarray(environment_seeds, np.uint32),
        rollout_seeds=np.asarray(rollout_seeds, np.uint32))
    environments = [PointReach2DPreferenceEnv() for _ in range(n)]
    active = np.ones(n, bool)
    timings = []
    device = torch.device(getattr(policy, 'device', 'cpu'))
    try:
        for i, (instance, seed) in enumerate(zip(environments, environment_seeds)):
            observation, info = instance.reset(seed=int(seed), options={'center_init': centered})
            output['positions'][i, 0] = observation[:2]
            output['goal_errors'][i] = info['goal_error']
        for chunk in range(8):
            rows = np.flatnonzero(active)
            if not len(rows):
                break
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            q = chunk*8
            streams = draw_latents([rollout_seeds[i] for i in rows], chunk=chunk,
                                   candidates=m, continuations=l)
            if method != 'base' and proposal_std != 1.0:
                streams['current_latents'] = streams['current_latents'] * proposal_std
            batch = evaluate_conditional_continuations(
                policy, stats, output['positions'][rows, :q+1],
                streams['current_latents'], streams['future_latents'],
                environment_config=DEFAULT_CONFIG.environment,
                integration_steps_per_action=integration_steps,
                feature_kind=feature_kind)
            scores = conditional_scores(batch, preference, beta=beta)
            output['decision_mask'][rows, chunk] = True
            for name in ('current_latents', 'current_seeds', 'future_seeds', 'selection_seeds'):
                output[name][rows, chunk] = streams[name]
            output['future_latents'][rows, chunk, :, :, :7-chunk] = streams['future_latents']
            for name in ('observed_anchors', 'model_anchors', 'current_commands', 'current_valid',
                         'expected_features', 'goal_costs', 'future_failure_fractions', 'continuation_features'):
                output[name][rows, chunk] = getattr(batch, name)
            output['candidate_scores'][rows, chunk] = scores
            output['expected_utility'][rows, chunk] = (
                0. if preference is None else grouped_utility(batch.expected_features, preference))
            for field in fields(batch.continuations):
                values = getattr(batch.continuations, field.name)
                output['continuation_'+field.name][rows, chunk] = values.reshape(len(rows), m, l, *values.shape[1:])
            for local, i in enumerate(rows):
                selected, ess, fallback = select_candidates(
                    scores[local], batch.current_valid[local], method=method,
                    rng=np.random.default_rng(streams['selection_seeds'][local]))
                output['ess'][i, chunk], output['fallback'][i, chunk] = ess, fallback
                if fallback:
                    commands = np.repeat(output['positions'][i, q:q+1], 8, axis=0)
                else:
                    output['selected_indices'][i, chunk] = selected
                    output['selected_current_valid'][i, chunk] = batch.current_valid[local, selected]
                    output['selected_current_latents'][i, chunk] = streams['current_latents'][local, selected]
                    commands = batch.current_commands[local, selected]
                for step, command in enumerate(commands):
                    output['requested_actions'][i, q+step] = command
                    observation, _, done, truncated, info = environments[i].step(command)
                    for name in ('success', 'numerical_failure', 'action_limit_failure', 'goal_errors'):
                        output[name][i] = info['goal_error' if name == 'goal_errors' else name]
                    if not info['numerical_failure'] and not info['action_limit_failure']:
                        output['positions'][i, output['lengths'][i]] = observation[:2]
                        output['lengths'][i] += 1
                    if done or truncated:
                        active[i] = False
                        break
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter()-started)
    finally:
        for instance in environments:
            instance.close()
    output['batch_replan_seconds'] = np.asarray(timings, np.float64)
    return output


def _path_metrics(output, users, *, feature_kind='continuous'):
    positions, lengths, success = output['positions'], output['lengths'], output['success']
    n, succeeded = len(positions), int(success.sum())
    complete = (lengths == 65) & ~output['numerical_failure'] & ~output['action_limit_failure']
    features = np.zeros((n, 2, 2), np.float64)
    features[complete] = grouped_features(
        positions[complete], feature_kind=feature_kind)
    reached = lengths > 32
    mode_names = ('upper_narrow', 'upper_wide', 'lower_narrow', 'lower_wide', 'other')
    reached_modes = classify_modes(positions[reached, 32, 1])
    successful_midpoint = complete & success
    successful_modes = classify_modes(positions[successful_midpoint, 32, 1])
    rate = succeeded/n
    center = (rate+1.96**2/(2*n))/(1+1.96**2/n)
    half = 1.96*np.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n))/(1+1.96**2/n)
    decisions = output['decision_mask']
    candidate_valid = output['current_valid'][decisions]
    failures = output['future_failure_fractions'][decisions]
    selected = decisions & (output['selected_indices'] >= 0)
    selected_failures = output['future_failure_fractions'][
        (*np.where(selected), output['selected_indices'][selected])]
    latency = output['batch_replan_seconds']
    record = dict(count=n, success_count=succeeded, success_rate=rate,
        success_wilson_95=[center-half, center+half], complete_count=int(complete.sum()),
        incomplete_count=int((~complete).sum()),
        numerical_failure_count=int(output['numerical_failure'].sum()),
        action_limit_failure_count=int(output['action_limit_failure'].sum()),
        mean_goal_error=float(output['goal_errors'].mean(dtype=np.float64)),
        midpoint_denominator=int(reached.sum()),
        midpoint_counts={name: int(np.sum(reached_modes == mode))
                         for mode, name in enumerate(mode_names)},
        successful_midpoint_counts={name: int(np.sum(successful_modes == mode))
                                    for mode, name in enumerate(mode_names)},
        decision_count=int(decisions.sum()), fallback_count=int(output['fallback'].sum()),
        mean_ess=float(output['ess'][decisions].mean()),
        mean_valid_candidate_fraction=float(candidate_valid.mean()),
        mean_expected_continuation_failure_fraction=float(failures.mean()),
        mean_valid_current_continuation_failure_fraction=float(failures[candidate_valid].mean()) if candidate_valid.any() else None,
        mean_selected_continuation_failure_fraction=float(selected_failures.mean()) if len(selected_failures) else None,
        selected_continuation_denominator=int(selected.sum()),
        batch_replan_latency_p50_seconds=float(np.median(latency)),
        batch_replan_latency_p95_seconds=float(np.quantile(latency, .95)), utility_by_user={})
    for name, user in users.items():
        utility = grouped_utility(features, user)
        passed = utility[success]
        record['utility_by_user'][name] = dict(
            all_count=n, all_mean=float(utility.mean()),
            all_standard_error=float(utility.std(ddof=1)/np.sqrt(n)) if n > 1 else None,
            successful_count=succeeded, successful_mean=float(passed.mean()) if succeeded else None,
            successful_standard_error=float(passed.std(ddof=1)/np.sqrt(succeeded)) if succeeded > 1 else None)
    return record


def _json_values(value):
    """Keep absent quantities as explicit None; never silently sanitize NaN."""
    if isinstance(value, dict):
        return {str(k): _json_values(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite diagnostic cannot be saved as JSON')
    return value


def _save_json(path, value):
    path.write_text(json.dumps(_json_values(value), indent=2, allow_nan=False)+'\n')


def _ordering_metrics(delta, scopes, oracle, learned):
    # Compute unscaled signed margins directly to avoid probability saturation.
    def margins(preference):
        within = (delta*preference.weights).sum(-1)
        values = (within*preference.alpha).sum(-1)
        values[scopes == 'direction'] = within[scopes == 'direction', 0]
        values[scopes == 'width'] = within[scopes == 'width', 1]
        return values
    truth, prediction = margins(oracle), margins(learned)
    output = {}
    for scope in ('aggregate', 'direction', 'width', 'overall'):
        mask = np.ones(len(delta), bool) if scope == 'aggregate' else scopes == scope
        eligible = mask & (np.abs(truth) > 1e-12)
        output[scope] = dict(count=int(eligible.sum()), excluded_oracle_ties=int((mask & ~eligible).sum()),
            accuracy=float(np.mean(np.sign(prediction[eligible]) == np.sign(truth[eligible]))) if eligible.any() else None)
    return output


def run_grouped_preference(output_dir, *, checkpoint_path, stats_path, demonstrations_path,
        seed=0, device='cpu', candidates=8, continuations=4, rollout_count=16,
        integration_steps_per_action=6, budgets=(5, 10, 20, 40), rollout_budget=20,
        users=None, beta=16., rho=16., feature_kind='continuous',
        selection_method='soft', proposal_std=1.0, progress=None):
    for name, value in (('candidates', candidates), ('continuations', continuations),
            ('rollout_count', rollout_count), ('integration_steps_per_action', integration_steps_per_action)):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not budgets or any(type(k) is not int or k <= 0 for k in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError('budgets must contain distinct positive integers')
    if type(rollout_budget) is not int or rollout_budget not in budgets:
        raise ValueError('rollout_budget must occur in budgets')
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(rho) or rho <= 0:
        raise ValueError('beta must be finite nonnegative and rho finite positive')
    if feature_kind not in ('continuous', 'mode'):
        raise ValueError('feature_kind must be continuous or mode')
    if selection_method not in ('soft', 'best'):
        raise ValueError('selection_method must be soft or best')
    if (isinstance(proposal_std, (bool, np.bool_)) or
            not isinstance(proposal_std, (int, float, np.integer, np.floating)) or
            not np.isfinite(proposal_std) or float(proposal_std) <= 0):
        raise ValueError('proposal_std must be finite and positive')
    proposal_std = float(proposal_std)
    users = dict(synthetic_grouped_users() if users is None else users)
    if not users or any(not isinstance(v, GroupedPreference) for v in users.values()):
        raise ValueError('users must be a nonempty mapping of GroupedPreference profiles')
    if any(not isinstance(k, str) or not k or not k.replace('_', '').isalnum() or '__' in k for k in users):
        raise ValueError('user names must contain letters/digits/single underscores')
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('output directory must be empty; previous experiments are preserved')
    destination.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda message: None)
    started = time.perf_counter()
    bank = load_demonstration_bank(demonstrations_path)
    train_digest = train_data_digest(bank)
    checkpoint_digest, stats_digest = _digest(checkpoint_path), _digest(stats_path)
    policy, stats, metadata = load_trained_sfps(checkpoint_path, stats_path,
        expected_train_data_digest=train_digest, device=device)
    policy.requires_grad_(False)
    state_before = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    stats_before = {field.name: getattr(stats, field.name).copy() for field in fields(stats)}
    config = dict(seed=seed, device=str(device), candidates=candidates, continuations=continuations,
        rollout_count=rollout_count, integration_steps_per_action=integration_steps_per_action,
        budgets=list(budgets), rollout_budget=rollout_budget,
        users={name: asdict(user) for name, user in users.items()}, beta=beta, rho=rho, l2=.1,
        feature_kind=feature_kind, selection_method=selection_method,
        proposal_std=proposal_std,
        proposal_distribution=('guided current latents use proposal_std * N(0,I); '
                               'base current and all future latents use N(0,I)'),
        checkpoint_path=str(Path(checkpoint_path).resolve()), stats_path=str(Path(stats_path).resolve()),
        demonstrations_path=str(Path(demonstrations_path).resolve()), torch_threads=torch.get_num_threads(),
        environment=asdict(DEFAULT_CONFIG.environment), model_horizon=16, prediction_points=9,
        executed_commands=8, goal_cost='(error/.1)^2 + 25*(not success)',
        feature_failure_policy='incomplete paths have zero features; complete goal misses retain geometry')
    _save_json(destination/'resolved_config.json', config)
    result = dict(synthetic_users=True, checkpoint_digest=checkpoint_digest, stats_digest=stats_digest,
        train_data_digest=train_digest, policy_unchanged=False, stats_unchanged=False,
        checkpoint_architecture=metadata['architecture'], config=config,
        preference_fits={}, conditions={}, limitations=[
            'Synthetic labels, eight designed users, and a small frozen-policy pilot are not human evidence.',
            'Finite candidate coverage can prevent desired steering; selection cannot invent missing options.',
            'Finite-L feature averages inside an exponential induce bias relative to the infinite-future target.',
            'Task-conditioned logits include the goal cost separately from the preference tilt.',
            'Full-batch timing includes candidate simulation and Gym execution, not single-robot latency.',
            'Optimizer convergence and local Jacobian rank do not certify a global optimum or global identifiability.'])
    query_seed, heldout_seed = _seed(seed, 1), _seed(seed, 2)
    query = build_grouped_queries(query_seed, max(budgets))
    heldout = build_grouped_queries(heldout_seed, 256)
    q_features = grouped_features(query['trajectories'], feature_kind=feature_kind)
    h_features = grouped_features(heldout['trajectories'], feature_kind=feature_kind)
    delta, held_delta = q_features[:, 0]-q_features[:, 1], h_features[:, 0]-h_features[:, 1]
    payload = dict(query_seed=np.array(query_seed, np.uint32), heldout_seed=np.array(heldout_seed, np.uint32),
        query_trajectories=query['trajectories'], query_ids=query['ids'], query_scopes=query['scopes'],
        query_features=q_features, query_delta_features=delta,
        heldout_trajectories=heldout['trajectories'], heldout_ids=heldout['ids'], heldout_scopes=heldout['scopes'],
        heldout_features=h_features, heldout_delta_features=held_delta)
    fitted = {}
    for index, (name, user) in enumerate(users.items()):
        label_seed, held_label_seed = _seed(seed, 3, index), _seed(seed, 4, index)
        probabilities = grouped_probabilities(delta, query['scopes'], user, rho=rho)
        held_probabilities = grouped_probabilities(held_delta, heldout['scopes'], user, rho=rho)
        labels = np.where(np.random.default_rng(label_seed).random(len(delta)) < probabilities, 1, -1).astype(np.int8)
        held_labels = np.where(np.random.default_rng(held_label_seed).random(len(held_delta)) < held_probabilities, 1, -1).astype(np.int8)
        for field, values in dict(labels=labels, heldout_labels=held_labels, probabilities=probabilities,
                heldout_probabilities=held_probabilities, label_seed=np.array(label_seed, np.uint32),
                heldout_label_seed=np.array(held_label_seed, np.uint32), oracle_weights=user.weights,
                oracle_alpha=user.alpha).items():
            payload[f'{name}_{field}'] = values
        result['preference_fits'][name], fitted[name] = {}, {}
        for budget in budgets:
            fit = fit_grouped_preference(delta[:budget], labels[:budget], query['scopes'][:budget], rho=rho)
            fitted[name][budget] = fit.preference
            record = asdict(fit)
            record['heldout'] = grouped_metrics(held_delta, held_labels, heldout['scopes'], fit.preference, rho=rho)
            record['oracle_ordering'] = _ordering_metrics(held_delta, heldout['scopes'], user, fit.preference)
            result['preference_fits'][name][str(budget)] = record
            payload[f'{name}_weights_{budget}'], payload[f'{name}_alpha_{budget}'] = fit.preference.weights, fit.preference.alpha
        notify(f'Fitted scoped feedback for {name}: budgets {list(budgets)}')
    np.savez_compressed(destination/'preferences.npz', **payload)
    conditions = [('base', 'base', None), ('task_only', selection_method, None)]
    for name, user in users.items():
        conditions.extend([(f'{name}__oracle_{selection_method}', selection_method, user),
                           (f'{name}__learned_{rollout_budget}', selection_method,
                            fitted[name][rollout_budget])])
    environment_seeds = [_seed(seed, 9, i) for i in range(rollout_count)]
    rollout_seeds = [_seed(seed, 10, i) for i in range(rollout_count)]
    for regime in ('centered', 'gaussian'):
        for name, method, preference in conditions:
            key = f'{regime}__{name}'
            output = _closed_loop(policy, stats, preference, environment_seeds=environment_seeds,
                rollout_seeds=rollout_seeds, centered=regime == 'centered', candidates=candidates,
                continuations=continuations, method=method, beta=beta,
                integration_steps=integration_steps_per_action,
                feature_kind=feature_kind, proposal_std=proposal_std)
            artifact = key+'.npz'
            np.savez_compressed(destination/artifact, **output)
            record = _path_metrics(output, users, feature_kind=feature_kind)
            record.update(artifact=artifact, method=method, preference=None if preference is None else asdict(preference))
            result['conditions'][key] = record
            notify(f"{key}: success {record['success_count']}/{rollout_count}, fallback {record['fallback_count']}")
    result['policy_unchanged'] = checkpoint_digest == _digest(checkpoint_path) and all(
        torch.equal(state_before[k], v.detach().cpu()) for k, v in policy.state_dict().items())
    result['stats_unchanged'] = stats_digest == _digest(stats_path) and all(
        np.array_equal(before, getattr(stats, name)) for name, before in stats_before.items())
    if not result['policy_unchanged'] or not result['stats_unchanged']:
        raise RuntimeError('frozen checkpoint, policy, or statistics changed during evaluation')
    result['elapsed_seconds'] = time.perf_counter()-started
    result = _json_values(result)
    _save_json(destination/'diagnostics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir', 'checkpoint-path', 'stats-path', 'demonstrations-path'):
        parser.add_argument('--'+name, required=True, type=Path)
    for name, default in (('seed', 0), ('candidates', 8), ('continuations', 4), ('rollout-count', 16),
                          ('integration-steps-per-action', 6), ('rollout-budget', 20), ('torch-threads', 1)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--budgets', nargs='+', type=int, default=[5, 10, 20, 40])
    parser.add_argument('--users', nargs='+', choices=list(synthetic_grouped_users()))
    parser.add_argument('--beta', type=float, default=16.)
    parser.add_argument('--rho', type=float, default=16.)
    parser.add_argument('--feature-kind', choices=('continuous', 'mode'), default='continuous')
    parser.add_argument('--selection-method', choices=('soft', 'best'), default='soft')
    parser.add_argument('--proposal-std', type=float, default=1.)
    args = vars(parser.parse_args())
    threads = args.pop('torch_threads')
    if threads <= 0:
        parser.error('--torch-threads must be positive')
    torch.set_num_threads(threads)
    if args['users'] is not None:
        all_users = synthetic_grouped_users()
        args['users'] = {name: all_users[name] for name in args['users']}
    run_grouped_preference(**args, progress=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
