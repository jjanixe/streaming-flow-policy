import pytest
import torch

from env.models import SFPDVelocityMLP
from env.sfp_policies import StreamingFlowPolicyDeterministic


def test_sfpd_loss_is_scalar_finite_float32():
    policy = StreamingFlowPolicyDeterministic(SFPDVelocityMLP(), pred_horizon=16)
    batch = {
        "obs": torch.zeros((4, 2, 3), dtype=torch.float32),
        "x": torch.zeros((4, 1, 2), dtype=torch.float32),
        "v": torch.ones((4, 1, 2), dtype=torch.float32),
        "t": torch.full((4,), 0.5, dtype=torch.float32),
    }
    loss = policy.loss(batch)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_sfpd_loss_rejects_non_float32_source_batch_tensors():
    policy = StreamingFlowPolicyDeterministic(SFPDVelocityMLP())
    batch = {
        "obs": torch.zeros((1, 2, 3), dtype=torch.float32),
        "x": torch.zeros((1, 1, 2), dtype=torch.float64),
        "v": torch.ones((1, 1, 2), dtype=torch.float32),
        "t": torch.full((1,), 0.5, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="float32"):
        policy.loss(batch)


class ConstantVelocity(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        velocity = torch.tensor([0.3, -0.15], dtype=torch.float32, device=sample.device)
        return velocity.expand_as(sample)


def test_prediction_starts_at_anchor_and_samples_eight_future_actions():
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    nobs = torch.tensor(
        [[-0.5, 0.25, -1.0], [-0.4, 0.2, -0.75]],
        dtype=torch.float32,
    )
    actions = policy.predict(nobs, num_actions=9, integration_steps_per_action=2)
    assert actions.shape == (1, 9, 2)
    assert actions.dtype == torch.float32
    torch.testing.assert_close(actions[0, 0], nobs[-1, :2])
    torch.testing.assert_close(
        actions[0, -1],
        nobs[-1, :2] + torch.tensor([0.3, -0.15]) * (8.0 / 15.0),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "nobs",
    [
        torch.zeros((2, 3), dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]], dtype=torch.float32),
    ],
)
def test_prediction_rejects_invalid_normalized_observations(nobs):
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    with pytest.raises(ValueError):
        policy.predict(nobs, num_actions=1, integration_steps_per_action=1)


@pytest.mark.parametrize("num_actions, steps", [(0, 1), (17, 1), (1, 0)])
def test_prediction_rejects_invalid_action_or_integration_counts(num_actions, steps):
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    nobs = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        policy.predict(nobs, num_actions=num_actions, integration_steps_per_action=steps)
