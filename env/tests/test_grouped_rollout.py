import numpy as np
import pytest
import torch

from env.chunk_data import PushTStats
from env.config import EnvironmentConfig
from env.environment import PointReach2DPreferenceEnv


@pytest.fixture
def stats():
    return PushTStats(obs_min=np.array([-2, -4, 0], np.float32),
                     obs_max=np.array([2, 4, 1], np.float32),
                     action_min=np.array([-3, -2], np.float32),
                     action_max=np.array([3, 2], np.float32))


class LinearPolicy:
    device = torch.device('cpu')

    def __init__(self):
        self.rows_per_call = []

    def predict_batch(self, nobs, num_actions, integration_steps_per_action, *, latents):
        assert num_actions == 9
        self.rows_per_call.append(len(nobs))
        physical = nobs[:, -1, :2] * torch.tensor([2, 4])
        delta = torch.stack(((1 + latents[:, 0]) / 32, latents[:, 1] / 64), -1)
        points = physical[:, None] + torch.arange(9)[None, :, None] * delta[:, None]
        # Actual SFPS anchor is in observation-normalized coordinates.
        points[:, 0] = nobs[:, -1, :2] * torch.tensor([3, 2])
        return points / torch.tensor([3, 2])


def evaluate(stats, current=None, futures=None, prefix=None, policy=None,
             feature_kind='continuous'):
    from env.grouped_rollout import evaluate_conditional_continuations
    if prefix is None:
        prefix = np.array([[[-1, 0]]], np.float32)
    if current is None:
        current = np.zeros((len(prefix), 2, 2), np.float32)
    if futures is None:
        futures = np.zeros((*current.shape[:2], 3, (65-prefix.shape[1])//8-1, 2), np.float32)
    return evaluate_conditional_continuations(
        policy or LinearPolicy(), stats, prefix, current, futures,
        environment_config=EnvironmentConfig(), integration_steps_per_action=2,
        feature_kind=feature_kind)


def test_current_chunk_is_evaluated_once_and_shared_across_independent_futures(stats):
    policy = LinearPolicy()
    futures = np.zeros((1, 2, 3, 7, 2), np.float32)
    futures[0, 0, 1, :, 1] = .5
    futures[0, 0, 2, :, 1] = -.5
    result = evaluate(stats, futures=futures, policy=policy)
    assert policy.rows_per_call == [2] + [6]*7
    paths = result.continuations.positions.reshape(1, 2, 3, 65, 2)
    np.testing.assert_array_equal(paths[:, :, 0, :9], paths[:, :, 1, :9])
    np.testing.assert_array_equal(paths[:, :, 0, :9], paths[:, :, 2, :9])
    np.testing.assert_allclose(paths[0, 0, :, -1], [[1, 0], [1, .4375], [1, -.4375]], atol=4e-6)
    one = evaluate(stats, futures=futures[:, :, :1])
    np.testing.assert_array_equal(one.current_commands, result.current_commands)
    np.testing.assert_array_equal(result.observed_anchors, [[[-1, 0], [-1, 0]]])
    np.testing.assert_allclose(result.model_anchors, [[[-1.5, 0], [-1.5, 0]]])
    np.testing.assert_allclose(result.current_commands[0, :, 0, 0], [-.96875, -.96875], atol=1e-6)
    assert result.current_valid.all()


def test_features_average_before_exponent_and_incomplete_futures_stay_in_denominator(stats):
    from env.grouped_preference import GroupedPreference
    from env.grouped_rollout import conditional_scores
    futures = np.zeros((1, 2, 2, 7, 2), np.float32)
    futures[0, 0, 1, 0, 0] = 9  # Rejected future, not an invalid current candidate.
    result = evaluate(stats, futures=futures)
    assert result.current_valid.all()
    np.testing.assert_allclose(result.expected_features[0], [[[.25, .25], [0, .5]], [[.5, .5], [0, 1]]], atol=1e-6)
    np.testing.assert_array_equal(result.future_failure_fractions, [[.5, 0]])
    assert result.continuations.lengths.tolist() == [65, 9, 65, 65]
    np.testing.assert_allclose(result.goal_costs, [[((1.75/.1)**2+25)/2, 0]], atol=.002)
    preference = GroupedPreference(np.array([[1., 0.], [0., 1.]]), np.array([.5, .5]))
    scores = conditional_scores(result, preference, beta=16)
    np.testing.assert_allclose(scores, 16*np.array([[.375, .75]])-result.goal_costs)
    # E[exp(16 U)] differs from exp(16 E[U]); catches the wrong target.
    assert not np.isclose(np.exp(scores[0, 0]+result.goal_costs[0, 0]), (.5*np.exp(12)+.5))


def test_completed_goal_misses_keep_geometry_and_final_replan_accepts_empty_futures(stats):
    prefix = np.zeros((1, 57, 2), np.float32)
    prefix[0, :, 0] = np.linspace(-1, .75, 57)
    current = np.zeros((1, 2, 2), np.float32)
    current[0, 1, 1] = 1
    result = evaluate(stats, current=current, futures=np.empty((1, 2, 2, 0, 2), np.float32), prefix=prefix)
    assert result.continuations.lengths.tolist() == [65]*4
    assert result.continuations.success.tolist() == [True, True, False, False]
    np.testing.assert_allclose(result.expected_features.sum(-1), 1)
    np.testing.assert_allclose(result.goal_costs[0, 1], 26.5625, atol=.002)
    assert result.future_failure_fractions[0, 1] == 1


def test_mode_features_propagate_to_continuation_and_conditional_means(stats):
    current = np.array([[[0, .4], [0, -.4]]], np.float32)
    futures = np.zeros((1, 2, 2, 7, 2), np.float32)
    futures[0, 0, :, :, 1] = .4
    futures[0, 1, :, :, 1] = -.4
    futures[0, 0, 1, 0, 0] = 9  # Incomplete continuation remains a zero feature.

    result = evaluate(stats, current=current, futures=futures, feature_kind='mode')

    np.testing.assert_array_equal(
        result.continuation_features[0],
        [[[[1, 0], [0, 1]], [[0, 0], [0, 0]]],
         [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]],
    )
    np.testing.assert_array_equal(
        result.expected_features[0],
        [[[.5, 0], [0, .5]], [[0, 1], [0, 1]]],
    )


def test_current_failure_is_ineligible_and_accepted_commands_match_actual_gym(stats):
    current = np.array([[[0, 0], [9, 0]]], np.float32)
    result = evaluate(stats, current=current)
    assert result.current_valid.tolist() == [[True, False]]
    np.testing.assert_array_equal(result.expected_features[0, 1], 0)
    instance = PointReach2DPreferenceEnv()
    instance.reset(options={'center_init': True})
    accepted = []
    for action in result.current_commands[0, 0]:
        obs, _, done, _, _ = instance.step(action)
        accepted.append(obs[:2])
        assert not done
    instance.close()
    np.testing.assert_array_equal(accepted, result.continuations.positions[0, 1:9])


@pytest.mark.parametrize('case', ['dtype', 'shape', 'nonfinite', 'boundary', 'zero_l', 'horizon'])
def test_rejects_invalid_conditional_inputs(stats, case):
    from env.grouped_rollout import evaluate_conditional_continuations
    prefix = np.array([[[-1, 0]]], np.float32)
    current = np.zeros((1, 2, 2), np.float32)
    futures = np.zeros((1, 2, 3, 7, 2), np.float32)
    config = EnvironmentConfig()
    if case == 'dtype': current = current.astype(np.float64)
    if case == 'shape': futures = futures[..., :1]
    if case == 'nonfinite': futures[0, 0, 0, 0, 0] = np.nan
    if case == 'boundary': prefix = np.zeros((1, 2, 2), np.float32)
    if case == 'zero_l': futures = futures[:, :, :0]
    if case == 'horizon': config = EnvironmentConfig(horizon_steps=32)
    with pytest.raises(ValueError):
        evaluate_conditional_continuations(LinearPolicy(), stats, prefix, current, futures, environment_config=config)
