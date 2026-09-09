import pytest
import torch

from env.analytic_fields import AnalyticField
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank


@pytest.fixture
def analytic_field():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=7)
    return AnalyticField(bank.select("train"), DEFAULT_CONFIG)


def test_score_matches_gradient_of_log_density(analytic_field):
    actions = torch.tensor(
        [[-0.45, 0.18], [0.0, -0.48], [0.61, 0.21]],
        dtype=torch.float32,
        requires_grad=True,
    )
    times = torch.tensor([0.27, 0.50, 0.79], dtype=torch.float32)

    gradient = torch.autograd.grad(
        analytic_field.log_density(actions, times).sum(),
        actions,
    )[0]

    torch.testing.assert_close(
        analytic_field.score(actions, times),
        gradient,
        rtol=1e-3,
        atol=2e-4,
    )


def test_field_shapes_dtype_responsibilities_and_drift(analytic_field):
    actions = torch.zeros((5, 2), dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)

    weights = analytic_field.responsibilities(actions, times)
    velocity = analytic_field.velocity_pf(actions, times)
    score = analytic_field.score(actions, times)
    expected_drift = (
        velocity + analytic_field.epsilon(times)[:, None] * score
    )

    assert weights.shape == (5, 512)
    assert velocity.shape == (5, 2)
    assert score.dtype == torch.float32
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(5),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        analytic_field.base_sde_drift(actions, times),
        expected_drift,
    )


def test_marginal_sampling_uses_explicit_generator(analytic_field):
    times = torch.full((32,), 0.5, dtype=torch.float32)

    first = analytic_field.sample_marginal(
        times,
        generator=torch.Generator().manual_seed(41),
    )
    second = analytic_field.sample_marginal(
        times,
        generator=torch.Generator().manual_seed(41),
    )

    assert first.shape == (32, 2)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_field_requires_train_only_bank():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=7)

    with pytest.raises(ValueError, match="train"):
        AnalyticField(bank, DEFAULT_CONFIG)


def test_input_contract_is_enforced(analytic_field):
    with pytest.raises(ValueError, match="actions"):
        analytic_field.score(torch.zeros(2), torch.zeros(1))
    with pytest.raises(ValueError, match="times"):
        analytic_field.score(torch.zeros((1, 2)), torch.tensor([1.1]))
    with pytest.raises(ValueError, match="float32"):
        analytic_field.score(
            torch.zeros((1, 2), dtype=torch.float64),
            torch.zeros(1),
        )


def test_finite_extreme_actions_produce_finite_fields(analytic_field):
    actions = torch.tensor(
        [[1e18, 0.0], [-1e18, 1e18]],
        dtype=torch.float32,
    )
    times = torch.tensor([1.0, 0.5], dtype=torch.float32)

    weights = analytic_field.responsibilities(actions, times)
    outputs = (
        analytic_field.log_density(actions, times),
        analytic_field.velocity_pf(actions, times),
        analytic_field.score(actions, times),
        analytic_field.base_sde_drift(actions, times),
    )

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(2, dtype=torch.float32),
    )
    assert all(torch.isfinite(output).all() for output in outputs)


def test_extreme_log_density_gradient_matches_finite_score(analytic_field):
    actions = torch.tensor(
        [[1e18, 1e18]],
        dtype=torch.float32,
        requires_grad=True,
    )
    times = torch.tensor([0.5], dtype=torch.float32)

    gradient = torch.autograd.grad(
        analytic_field.log_density(actions, times).sum(),
        actions,
    )[0]
    score = analytic_field.score(actions.detach(), times)

    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, score, rtol=1e-5, atol=0.0)


def test_near_maximum_action_preserves_component_ordering(analytic_field):
    actions = torch.tensor([[3e38, 3e38]], dtype=torch.float32)
    times = torch.tensor([0.5], dtype=torch.float32)
    centers, _ = analytic_field.centers_and_derivatives(times)
    expected_component = int(centers[0, :, 1].argmax().item())

    weights = analytic_field.responsibilities(actions, times)

    assert int(weights.argmax(dim=1).item()) == expected_component
    assert int((weights == weights.max()).sum().item()) == 1
