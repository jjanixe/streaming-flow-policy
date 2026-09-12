import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from env.artifacts import save_demonstration_bank
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.grouped_preference import (
    grouped_features, grouped_probabilities, synthetic_grouped_users)
from env.stage_b_config import DEFAULT_STAGE_B2_CONFIG
from env.train_stage_b import train_sfps
from env.tests.test_grouped_rollout import LinearPolicy, stats


def test_latent_streams_are_shared_across_conditions_and_independent_of_l():
    from env.run_grouped_preference import draw_latents
    a = draw_latents([30, 31], chunk=2, candidates=3, continuations=2)
    b = draw_latents([30, 31], chunk=2, candidates=3, continuations=5)
    base = draw_latents([30, 31], chunk=2, candidates=1, continuations=2)
    np.testing.assert_array_equal(a['current_latents'], b['current_latents'])
    np.testing.assert_array_equal(a['current_latents'][:, :1], base['current_latents'])
    np.testing.assert_array_equal(a['future_latents'], b['future_latents'][:, :, :2])
    assert not np.array_equal(a['future_latents'][:, 0], a['future_latents'][:, 1])
    assert np.all(a['current_seeds'][:, None] != a['future_seeds'])
    assert np.all(a['current_seeds'] != a['selection_seeds'])


@pytest.mark.parametrize('method', ['soft', 'best'])
def test_all_invalid_guided_holds_in_actual_gym_and_raw_base_exposes_failure(stats, method):
    from env.run_grouped_preference import _closed_loop
    class RejectingPolicy(LinearPolicy):
        def predict_batch(self, *args, **kwargs):
            points = super().predict_batch(*args, **kwargs)
            points[:, 1, 0] = 100
            return points

    kwargs = dict(environment_seeds=[2, 3], rollout_seeds=[4, 5], centered=True,
                  candidates=2, continuations=2, beta=16., integration_steps=2)
    guided = _closed_loop(RejectingPolicy(), stats, None, method=method, **kwargs)
    assert guided['lengths'].tolist() == [65, 65]
    assert guided['fallback'].all() and guided['decision_mask'].all()
    assert np.all(guided['selected_indices'] == -1)
    assert not guided['selected_current_valid'].any()
    assert not guided['action_limit_failure'].any()
    np.testing.assert_array_equal(guided['positions'], np.tile([-1, 0], (2, 65, 1)))
    np.testing.assert_array_equal(guided['requested_actions'], np.tile([-1, 0], (2, 64, 1)))
    base = _closed_loop(RejectingPolicy(), stats, None, method='base', **kwargs)
    assert base['lengths'].tolist() == [1, 1]
    assert base['action_limit_failure'].all()
    assert not base['fallback'].any()
    assert base['decision_mask'].sum() == 2


def test_best_selection_executes_each_valid_argmax_in_actual_gym(stats):
    from env.run_grouped_preference import _closed_loop

    output = _closed_loop(
        LinearPolicy(), stats, synthetic_grouped_users()['upper_narrow_direction'],
        environment_seeds=[2, 3], rollout_seeds=[4, 5], centered=True,
        candidates=3, continuations=2, method='best', beta=16., integration_steps=2,
        feature_kind='mode')

    for episode, chunk in zip(*np.where(output['decision_mask'])):
        valid = output['current_valid'][episode, chunk]
        if not valid.any():
            assert output['fallback'][episode, chunk]
            assert output['selected_indices'][episode, chunk] == -1
            continue
        scores = output['candidate_scores'][episode, chunk].copy()
        scores[~valid] = -np.inf
        selected = int(np.argmax(scores))
        assert output['selected_indices'][episode, chunk] == selected
        start = chunk * 8
        np.testing.assert_array_equal(
            output['requested_actions'][episode, start:start + 8],
            output['current_commands'][episode, chunk, selected],
        )


def test_path_metrics_use_selected_features_and_exclude_goal_misses_from_successful_modes():
    from env.run_grouped_preference import _path_metrics

    midpoint_y = np.array([.2, .5, -.2, -.5, 0.])
    positions = np.zeros((5, 65, 2), np.float32)
    positions[:, 32, 1] = midpoint_y
    output = dict(
        positions=positions, lengths=np.full(5, 65),
        success=np.array([True, False, True, False, True]),
        numerical_failure=np.zeros(5, bool), action_limit_failure=np.zeros(5, bool),
        goal_errors=np.zeros(5, np.float32), decision_mask=np.ones((5, 1), bool),
        current_valid=np.ones((5, 1, 1), bool),
        future_failure_fractions=np.zeros((5, 1, 1), np.float64),
        selected_indices=np.zeros((5, 1), np.int64), fallback=np.zeros((5, 1), bool),
        ess=np.ones((5, 1), np.float64), batch_replan_seconds=np.array([1.]),
    )

    metrics = _path_metrics(
        output, {'upper_wide': synthetic_grouped_users()['upper_wide_direction']},
        feature_kind='mode')

    assert metrics['midpoint_counts'] == {
        'upper_narrow': 1, 'upper_wide': 1, 'lower_narrow': 1,
        'lower_wide': 1, 'other': 1,
    }
    assert metrics['successful_midpoint_counts'] == {
        'upper_narrow': 1, 'upper_wide': 0, 'lower_narrow': 1,
        'lower_wide': 0, 'other': 1,
    }
    assert metrics['utility_by_user']['upper_wide']['all_mean'] == pytest.approx(.4)
    assert metrics['utility_by_user']['upper_wide']['successful_mean'] == pytest.approx(.28)


def test_small_actual_checkpoint_experiment_reproduces_all_non_timing_arrays(tmp_path):
    from env.run_grouped_preference import run_grouped_preference
    config = replace(DEFAULT_CONFIG, demonstrations=replace(
        DEFAULT_CONFIG.demonstrations, train_per_mode=2, validation_per_mode=1, test_per_mode=1))
    bank = generate_demonstration_bank(config, seed=42)
    bank_path = tmp_path/'demonstrations.npz'
    save_demonstration_bank(bank_path, bank)
    training_config = replace(DEFAULT_STAGE_B2_CONFIG, hidden_dim=8, hidden_layers=1,
        batch_size=16, max_updates=1, validation_interval=1, warmup_updates=1, integration_steps_per_action=1)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = train_sfps(bank, tmp_path/'training', config=training_config, root_seed=43, device='cpu')
        before = hashlib.sha256(training.checkpoint_path.read_bytes()).hexdigest()
        user = synthetic_grouped_users()['upper_narrow_direction']
        kwargs = dict(checkpoint_path=training.checkpoint_path, stats_path=training.stats_path,
            demonstrations_path=bank_path, seed=44, candidates=2, continuations=2,
            rollout_count=2, integration_steps_per_action=1, budgets=(5,), rollout_budget=5,
            users={'test_user': user})
        first = run_grouped_preference(tmp_path/'first', **kwargs)
        second = run_grouped_preference(tmp_path/'second', **kwargs)
        with pytest.raises(ValueError, match='empty'):
            run_grouped_preference(tmp_path/'first', **kwargs)
    finally:
        torch.set_num_threads(previous)
    assert before == hashlib.sha256(training.checkpoint_path.read_bytes()).hexdigest()
    assert first['policy_unchanged'] and first['stats_unchanged']
    assert first['checkpoint_digest'] == before
    assert len(first['conditions']) == 8
    assert first['synthetic_users']
    for key, metrics in first['conditions'].items():
        assert metrics['count'] == 2
        assert metrics['complete_count']+metrics['incomplete_count'] == 2
        assert metrics['utility_by_user']['test_user']['all_count'] == 2
        assert metrics['utility_by_user']['test_user']['successful_count'] == metrics['success_count']
        with np.load(tmp_path/'first'/metrics['artifact'], allow_pickle=False) as a:
            with np.load(tmp_path/'second'/second['conditions'][key]['artifact'], allow_pickle=False) as b:
                for field in a.files:
                    if field != 'batch_replan_seconds':
                        np.testing.assert_array_equal(a[field], b[field], err_msg=field)
                assert a['positions'].dtype == np.float32
                assert a['expected_features'].shape[-2:] == (2, 2)
                assert a['selected_current_latents'].shape == (2, 8, 2)
                assert a['decision_mask'].any()
    raw = first['conditions']['centered__base']
    assert raw['success_count'] < 2  # Completed goal misses are task failures too.
    with np.load(tmp_path/'first'/'preferences.npz', allow_pickle=False) as prefs:
        assert prefs['heldout_scopes'].shape == (256,)
        assert prefs['query_scopes'].tolist() == ['direction', 'width', 'overall', 'direction', 'width']
        probability = grouped_probabilities(prefs['query_delta_features'], prefs['query_scopes'], user)
        np.testing.assert_array_equal(prefs['test_user_probabilities'], probability)
    for filename in ('diagnostics.json', 'resolved_config.json'):
        with (tmp_path/'first'/filename).open() as stream:
            json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_small_actual_checkpoint_saves_mode_feedback_and_best_configuration(tmp_path):
    from env.run_grouped_preference import run_grouped_preference

    config = replace(DEFAULT_CONFIG, demonstrations=replace(
        DEFAULT_CONFIG.demonstrations, train_per_mode=2, validation_per_mode=1,
        test_per_mode=1))
    bank = generate_demonstration_bank(config, seed=52)
    bank_path = tmp_path/'demonstrations.npz'
    save_demonstration_bank(bank_path, bank)
    training_config = replace(
        DEFAULT_STAGE_B2_CONFIG, hidden_dim=8, hidden_layers=1, batch_size=16,
        max_updates=1, validation_interval=1, warmup_updates=1,
        integration_steps_per_action=1)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = train_sfps(
            bank, tmp_path/'training', config=training_config, root_seed=53,
            device='cpu')
        user = synthetic_grouped_users()['upper_wide_direction']
        result = run_grouped_preference(
            tmp_path/'mode-best', checkpoint_path=training.checkpoint_path,
            stats_path=training.stats_path, demonstrations_path=bank_path, seed=54,
            candidates=2, continuations=2, rollout_count=1,
            integration_steps_per_action=1, budgets=(5,), rollout_budget=5,
            users={'test_user': user}, feature_kind='mode', selection_method='best')
    finally:
        torch.set_num_threads(previous)

    assert result['config']['feature_kind'] == 'mode'
    assert result['config']['selection_method'] == 'best'
    assert 'centered__test_user__oracle_best' in result['conditions']
    assert 'centered__test_user__oracle_soft' not in result['conditions']
    assert result['conditions']['centered__base']['method'] == 'base'
    assert all(
        condition['method'] == 'best'
        for key, condition in result['conditions'].items() if not key.endswith('__base'))
    assert all('successful_midpoint_counts' in condition
               for condition in result['conditions'].values())
    with np.load(tmp_path/'mode-best'/'preferences.npz', allow_pickle=False) as prefs:
        expected = grouped_features(prefs['query_trajectories'], feature_kind='mode')
        np.testing.assert_array_equal(prefs['query_features'], expected)
        np.testing.assert_array_equal(
            prefs['query_delta_features'], expected[:, 0]-expected[:, 1])
        probability = grouped_probabilities(
            prefs['query_delta_features'], prefs['query_scopes'], user)
        np.testing.assert_array_equal(prefs['test_user_probabilities'], probability)
