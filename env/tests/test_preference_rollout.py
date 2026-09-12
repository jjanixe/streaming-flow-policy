import numpy as np
import pytest
import torch

from env.chunk_data import PushTStats
from env.config import EnvironmentConfig
from env.environment import PointReach2DPreferenceEnv
from env.sfp_policies import PolicyNumericalError


def rollout_module():
    from env import preference_rollout

    return preference_rollout


@pytest.fixture
def stats():
    return PushTStats(
        obs_min=np.array([-2, -4, 0], dtype=np.float32),
        obs_max=np.array([2, 4, 1], dtype=np.float32),
        action_min=np.array([-3, -2], dtype=np.float32),
        action_max=np.array([3, 2], dtype=np.float32),
    )


class LinearPredictor:
    """Physical positions with explicit per-chunk latent-dependent velocity."""

    device = torch.device("cpu")

    def __init__(self):
        self.histories = []
        self.latents = []

    def predict_batch(self, nobs, num_actions, integration_steps_per_action, *, latents):
        assert num_actions == 9
        assert integration_steps_per_action == 2
        assert nobs.dtype == latents.dtype == torch.float32
        self.histories.append(nobs.clone())
        self.latents.append(latents.clone())
        physical = nobs[:, -1, :2] * torch.tensor([2, 4])
        delta = torch.stack(((1 + latents[:, 0]) / 32, latents[:, 1] / 64), dim=1)
        positions = physical[:, None] + torch.arange(9)[None, :, None] * delta[:, None]
        positions[:, 0] = 100  # A deliberately wrong anchor must never execute.
        return positions / torch.tensor([3, 2])


def run(policy, stats, prefixes=None, latents=None, **kwargs):
    if prefixes is None:
        prefixes = np.array([[[-1, 0]]], dtype=np.float32)
    if latents is None:
        latents = np.zeros((len(prefixes), (65 - prefixes.shape[1]) // 8, 2), dtype=np.float32)
    return rollout_module().simulate_continuations(
        policy, stats, prefixes, latents,
        environment_config=kwargs.pop("environment_config", EnvironmentConfig()),
        integration_steps_per_action=2, **kwargs,
    )


def test_executes_exactly_64_commands_skips_anchor_and_updates_history_time(stats):
    policy = LinearPredictor()
    result = run(policy, stats)
    assert result.lengths.tolist() == [65]
    expected_x = np.linspace(-1, 1, 65, dtype=np.float32)
    np.testing.assert_allclose(result.positions[0, :, 0], expected_x, atol=3e-6)
    np.testing.assert_allclose(result.positions[0, :, 1], 0, atol=3e-6)
    np.testing.assert_array_equal(result.requested_actions[0], result.positions[0, 1:])
    assert result.positions.dtype == result.requested_actions.dtype == np.float32
    assert result.success.tolist() == [True]
    assert result.goal_errors[0] < 3e-6
    assert not result.numerical_failure.any() and not result.action_limit_failure.any()
    torch.testing.assert_close(policy.histories[0], torch.tensor([[[-.5, 0, -1], [-.5, 0, -1]]]))
    torch.testing.assert_close(policy.histories[1], torch.tensor([[[-.390625, 0, -.78125], [-.375, 0, -.75]]]))
    assert len(policy.histories) == 8


def test_preserves_realized_prefix_and_consumes_each_future_latent(stats):
    prefix = np.zeros((2, 9, 2), dtype=np.float32)
    prefix[:, :, 0] = np.linspace(-1, -.75, 9, dtype=np.float32)
    latents = np.zeros((2, 7, 2), dtype=np.float32)
    latents[1, :, 1] = np.arange(1, 8, dtype=np.float32) / 8
    before = prefix.copy()
    policy = LinearPredictor()
    result = run(policy, stats, prefix, latents)
    np.testing.assert_array_equal(result.positions[:, :9], before)
    np.testing.assert_array_equal(prefix, before)
    np.testing.assert_allclose(result.positions[:, -1], [[1, 0], [1, .4375]], atol=3e-6)
    np.testing.assert_array_equal(torch.stack(policy.latents, dim=1), latents)
    assert result.success.tolist() == [True, False]  # Goal miss is not hard invalid.
    assert not result.numerical_failure.any() and not result.action_limit_failure.any()


@pytest.mark.parametrize("bad_action,numerical,limited", [([5, 0], False, True), ([np.nan, .2], True, False), ([np.inf, 0], True, False)])
def test_invalid_command_matches_gym_without_counting_terminal_padding(stats, bad_action, numerical, limited):
    class FailingPredictor(LinearPredictor):
        def predict_batch(self, *args, **kwargs):
            values = super().predict_batch(*args, **kwargs)
            if len(self.histories) == 1:
                values[0, 3] = torch.tensor(bad_action) / torch.tensor([3, 2])
            return values

    result = run(FailingPredictor(), stats, np.array([[[-1, 0]], [[-1, 0]]], dtype=np.float32))
    environment = PointReach2DPreferenceEnv()
    environment.reset(options={"center_init": True})
    for action in result.requested_actions[0, :3]:
        observation, _, done, _, info = environment.step(action)
    assert done
    assert result.lengths.tolist() == [3, 65]
    np.testing.assert_array_equal(result.positions[0, 2], observation[:2])
    np.testing.assert_allclose(result.requested_actions[0, 2], bad_action, atol=1e-6)
    assert np.isnan(result.positions[0, 3:]).all()
    assert np.isnan(result.requested_actions[0, 3:]).all()
    assert result.numerical_failure.tolist() == [numerical, False]
    assert result.action_limit_failure.tolist() == [limited, False]
    assert bool(result.success[0]) == info["success"]
    assert result.goal_errors[0] == pytest.approx(info["goal_error"])


def test_numerical_batch_exception_isolates_failed_row(stats):
    class SometimesNumerical(LinearPredictor):
        def predict_batch(self, *args, **kwargs):
            if (kwargs["latents"][:, 0] > 0).any():
                raise PolicyNumericalError("one row failed integration")
            return super().predict_batch(*args, **kwargs)

    latents = np.zeros((2, 8, 2), dtype=np.float32)
    latents[0, 1, 0] = 1
    result = run(SometimesNumerical(), stats, np.array([[[-1, 0]], [[-1, 0]]], dtype=np.float32), latents)
    assert result.lengths.tolist() == [9, 65]
    assert result.numerical_failure.tolist() == [True, False]
    assert not result.action_limit_failure.any()
    assert np.isnan(result.requested_actions[0, 8:]).all()
    assert result.success.tolist() == [False, True]


def test_full_horizon_prefix_is_terminal_without_policy_call(stats):
    prefix = np.zeros((1, 65, 2), dtype=np.float32)
    prefix[0, :, 0] = np.linspace(-1, 1, 65, dtype=np.float32)
    policy = LinearPredictor()
    result = run(policy, stats, prefix)
    assert result.lengths.tolist() == [65]
    assert result.success.tolist() == [True]
    np.testing.assert_array_equal(result.positions, prefix)
    assert policy.histories == []


@pytest.mark.parametrize("prefix,latents", [
    (np.zeros((1, 2, 2), dtype=np.float32), np.zeros((1, 8, 2), dtype=np.float32)),
    (np.zeros((1, 1, 2), dtype=np.float64), np.zeros((1, 8, 2), dtype=np.float32)),
    (np.full((1, 1, 2), np.nan, dtype=np.float32), np.zeros((1, 8, 2), dtype=np.float32)),
    (np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 7, 2), dtype=np.float32)),
    (np.zeros((1, 1, 2), dtype=np.float32), np.full((1, 8, 2), np.inf, dtype=np.float32)),
])
def test_rejects_malformed_continuation_inputs(stats, prefix, latents):
    with pytest.raises(ValueError):
        run(LinearPredictor(), stats, prefix, latents)


def test_rejects_non64_horizon_and_impossible_prefix(stats):
    with pytest.raises(ValueError, match="64"):
        run(LinearPredictor(), stats, environment_config=EnvironmentConfig(horizon_steps=32))
    prefix = np.zeros((1, 9, 2), dtype=np.float32)
    prefix[0, 3, 0] = 1
    with pytest.raises(ValueError, match="prefix"):
        run(LinearPredictor(), stats, prefix)


def test_selection_base_best_all_invalid_and_stable_softmax():
    select = rollout_module().select_candidates
    scores = np.array([-10, 1e30, 1e30], dtype=np.float32)
    valid = np.array([False, True, True])
    rng = np.random.default_rng(1)
    assert select(scores, valid, method="base", rng=rng) == (0, 1.0, False)
    assert select(scores, valid, method="best", rng=rng) == (1, 1.0, False)
    index, ess, fallback = select(scores, valid, method="soft", rng=rng)
    assert index in (1, 2) and ess == pytest.approx(2) and not fallback
    assert select(scores, np.zeros(3, dtype=bool), method="base", rng=rng) == (0, 1.0, False)
    for method in ("best", "soft"):
        assert select(scores, np.zeros(3, dtype=bool), method=method, rng=rng) == (0, 0.0, True)


def test_zero_scores_are_uniform_over_valid_rows_and_seeded_replay():
    select = rollout_module().select_candidates
    scores = np.zeros(3, dtype=np.float32)
    valid = np.array([True, False, True])
    first_rng, second_rng = np.random.default_rng(25), np.random.default_rng(25)
    first = [select(scores, valid, method="soft", rng=first_rng) for _ in range(400)]
    second = [select(scores, valid, method="soft", rng=second_rng) for _ in range(400)]
    assert first == second
    assert {i for i, _, _ in first} == {0, 2}
    assert all(ess == pytest.approx(2) and not fallback for _, ess, fallback in first)
    assert 150 < sum(i == 0 for i, _, _ in first) < 250


def test_best_selection_depends_on_full_future_outcome(stats):
    latents = np.zeros((2, 8, 2), dtype=np.float32)
    latents[1, -1, 1] = 1
    result = run(LinearPredictor(), stats, np.array([[[-1, 0]], [[-1, 0]]], dtype=np.float32), latents)
    np.testing.assert_array_equal(result.positions[0, :57], result.positions[1, :57])
    assert rollout_module().select_candidates(result.positions[:, -1, 1], np.ones(2, dtype=bool), method="best", rng=np.random.default_rng(0))[0] == 1


def test_success_at_float32_tolerance_boundary_matches_gym(stats):
    config = EnvironmentConfig(start=(0, 0), goal=(.1, 0))
    environment = PointReach2DPreferenceEnv(config)
    environment.reset(options={"center_init": True})
    for _ in range(64):
        _, _, _, _, info = environment.step(np.zeros(2, dtype=np.float32))
    result = run(LinearPredictor(), stats, np.zeros((1, 65, 2), dtype=np.float32), environment_config=config)
    assert bool(result.success[0]) == info["success"]


def test_selection_ignores_invalid_nan_scores_but_rejects_nonfinite_valid_scores():
    select = rollout_module().select_candidates
    rng = np.random.default_rng(5)
    for method in ("best", "soft"):
        assert select(np.array([np.nan, 2, -np.inf], dtype=np.float32), np.array([False, True, False]), method=method, rng=rng) == (1, 1.0, False)
        with pytest.raises(ValueError, match="finite"):
            select(np.array([np.nan], dtype=np.float32), np.array([True]), method=method, rng=rng)


def test_programming_errors_from_predictor_are_not_silently_numerical_failures(stats):
    class BrokenPredictor:
        def predict_batch(self, *args, **kwargs):
            raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        run(BrokenPredictor(), stats)


def test_real_sfps_solver_matches_existing_gym_rollout_with_identical_latent_schedule():
    from env.evaluate_stage_b import rollout_sfps
    from env.models import SFPSVelocityMLP
    from env.sfp_policies import StreamingFlowPolicyStochastic

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(893)
            model = SFPSVelocityMLP(hidden_dim=16, hidden_layers=1)
        policy = StreamingFlowPolicyStochastic(model).eval()
        identity_stats = PushTStats(
            obs_min=np.array([-1, -1, 0], dtype=np.float32),
            obs_max=np.array([1, 1, 1], dtype=np.float32),
            action_min=np.array([-1, -1], dtype=np.float32),
            action_max=np.array([1, 1], dtype=np.float32),
        )
        state_before = {name: value.clone() for name, value in policy.state_dict().items()}
        original = rollout_sfps(policy, identity_stats, EnvironmentConfig(), environment_seed=4,
                                latent_seed=90, center_init=True, integration_steps_per_action=2)
        assert original.executed_positions.shape == (65, 2)
        result = run(policy, identity_stats, original.executed_positions[None, :1], original.chunk_latents[None])
        np.testing.assert_allclose(result.positions[0], original.executed_positions, atol=2e-6, rtol=0)
        np.testing.assert_allclose(result.requested_actions[0], original.requested_actions, atol=2e-6, rtol=0)
        assert result.lengths.tolist() == [65]
        assert not result.numerical_failure.any() and not result.action_limit_failure.any()
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, state_before[name], atol=0, rtol=0)
    finally:
        torch.set_num_threads(previous_threads)
