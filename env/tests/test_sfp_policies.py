import subprocess
import sys

import pytest
import torch

from env.models import SFPDVelocityMLP, SFPSVelocityMLP
import env.sfp_policies as sfp_policies
from env.sfp_policies import (
    PolicyNumericalError,
    StreamingFlowPolicyDeterministic,
    StreamingFlowPolicyStochastic,
    _ConditionedVectorField,
)


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


def test_sfps_loss_is_scalar_finite_float32():
    policy = StreamingFlowPolicyStochastic(
        SFPSVelocityMLP(hidden_dim=16, hidden_layers=1)
    )
    batch = {
        "obs": torch.zeros((4, 2, 3), dtype=torch.float32),
        "a": torch.zeros((4, 1, 2), dtype=torch.float32),
        "z": torch.zeros((4, 1, 2), dtype=torch.float32),
        "va": torch.ones((4, 1, 2), dtype=torch.float32),
        "vz": torch.ones((4, 1, 2), dtype=torch.float32),
        "t": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
    }

    loss = policy.loss(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_sfps_loss_rejects_invalid_joint_batch():
    policy = StreamingFlowPolicyStochastic(
        SFPSVelocityMLP(hidden_dim=16, hidden_layers=1)
    )
    batch = {
        "obs": torch.zeros((1, 2, 3), dtype=torch.float32),
        "a": torch.zeros((1, 1, 2), dtype=torch.float32),
        "z": torch.zeros((1, 1, 2), dtype=torch.float64),
        "va": torch.ones((1, 1, 2), dtype=torch.float32),
        "vz": torch.ones((1, 1, 2), dtype=torch.float32),
        "t": torch.zeros((1,), dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="float32"):
        policy.loss(batch)


class ConstantVelocity(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        velocity = torch.tensor([0.3, -0.15], dtype=torch.float32, device=sample.device)
        return velocity.expand_as(sample)


class LatentSensitiveJointVelocity(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        velocity = torch.zeros_like(sample)
        velocity[:, 0, :] = sample[:, 1, :]
        return velocity


def test_sfps_prediction_replays_with_explicit_latent_generator():
    policy = StreamingFlowPolicyStochastic(LatentSensitiveJointVelocity())
    nobs = torch.tensor(
        [[-0.5, 0.25, -1.0], [-0.4, 0.2, -0.75]],
        dtype=torch.float32,
    )
    global_state = torch.random.get_rng_state().clone()

    first = policy.predict(
        nobs,
        num_actions=9,
        integration_steps_per_action=2,
        generator=torch.Generator().manual_seed(7),
    )
    second = policy.predict(
        nobs,
        num_actions=9,
        integration_steps_per_action=2,
        generator=torch.Generator().manual_seed(7),
    )
    third = policy.predict(
        nobs,
        num_actions=9,
        integration_steps_per_action=2,
        generator=torch.Generator().manual_seed(8),
    )

    assert first.shape == (1, 9, 2)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first[0, 0], nobs[-1, :2], rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, third)
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_sfps_prediction_can_use_an_explicit_latent_without_sampling():
    policy = StreamingFlowPolicyStochastic(LatentSensitiveJointVelocity())
    nobs = torch.zeros((2, 3), dtype=torch.float32)
    latent = torch.tensor([0.3, -0.15], dtype=torch.float32)

    actions = policy.predict(
        nobs,
        num_actions=9,
        integration_steps_per_action=2,
        latent=latent,
    )

    torch.testing.assert_close(actions[0, -1], latent * (8.0 / 15.0), atol=1e-4, rtol=1e-4)


def test_sfps_batched_prediction_integrates_multiple_explicit_latents_together():
    policy = StreamingFlowPolicyStochastic(LatentSensitiveJointVelocity())
    nobs = torch.tensor(
        [
            [[-0.5, 0.1, -1.0], [-0.4, 0.2, -0.75]],
            [[0.1, -0.2, -1.0], [0.2, -0.1, -0.75]],
            [[0.0, 0.0, -1.0], [0.3, 0.4, -0.75]],
        ],
        dtype=torch.float32,
    )
    latents = torch.tensor(
        [[0.3, -0.15], [-0.2, 0.25], [0.1, 0.2]],
        dtype=torch.float32,
    )

    actions = policy.predict_batch(
        nobs,
        num_actions=9,
        integration_steps_per_action=2,
        latents=latents,
    )

    assert actions.shape == (3, 9, 2)
    assert actions.dtype == torch.float32
    torch.testing.assert_close(actions[:, 0], nobs[:, -1, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actions[:, -1],
        nobs[:, -1, :2] + latents * (8.0 / 15.0),
        atol=1e-4,
        rtol=1e-4,
    )


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


def test_prediction_with_only_anchor_returns_normalized_anchor_without_ode(monkeypatch):
    def fail_if_ode_constructed(*args, **kwargs):
        raise AssertionError("one-action prediction must not construct an ODE")

    monkeypatch.setattr(sfp_policies, "NeuralODE", fail_if_ode_constructed)
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    nobs = torch.tensor(
        [[-0.5, 0.25, -1.0], [-0.4, 0.2, -0.75]],
        dtype=torch.float32,
    )
    actions = policy.predict(nobs, num_actions=1, integration_steps_per_action=1)
    assert actions.shape == (1, 1, 2)
    assert actions.dtype == torch.float32
    assert actions.device == nobs.device
    torch.testing.assert_close(actions[0, 0], torch.tensor([-0.4, 0.2]))


def test_policy_import_restores_missing_lzma_module_semantics():
    script = """
import sys

try:
    import lzma
except ModuleNotFoundError as error:
    if error.name != "_lzma":
        raise
    missing_lzma = True
else:
    missing_lzma = False
    real_lzma = lzma

import env.sfp_policies

if missing_lzma:
    assert "lzma" not in sys.modules
    try:
        import lzma
    except ModuleNotFoundError as error:
        if error.name != "_lzma":
            raise
    else:
        raise AssertionError("policy import installed a fake lzma module")
else:
    assert sys.modules["lzma"] is real_lzma
    assert real_lzma.decompress(real_lzma.compress(b"policy import")) == b"policy import"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_prediction_configures_torchdyn_dopri5_adjoint_tolerances(monkeypatch):
    captured = {}

    class CapturingNeuralODE:
        def __init__(self, vector_field, **kwargs):
            captured["vector_field"] = vector_field
            captured["kwargs"] = kwargs

        def trajectory(self, x, t_span):
            return torch.stack((x, x))

    monkeypatch.setattr(sfp_policies, "NeuralODE", CapturingNeuralODE)
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    nobs = torch.zeros((2, 3), dtype=torch.float32)
    policy.predict(nobs, num_actions=2, integration_steps_per_action=1)

    assert isinstance(captured["vector_field"], torch.nn.Module)
    assert captured["kwargs"] == {
        "solver": "dopri5",
        "sensitivity": "adjoint",
        "atol": 1e-4,
        "rtol": 1e-4,
    }


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


def test_conditioned_vector_field_classifies_only_nonfinite_values_as_numerical():
    condition = torch.zeros((1, 6), dtype=torch.float32)
    finite_field = _ConditionedVectorField(ConstantVelocity(), condition)
    with pytest.raises(PolicyNumericalError, match="ODE state"):
        finite_field(
            torch.tensor(0.25, dtype=torch.float32),
            torch.tensor([float("nan"), 0.0], dtype=torch.float32),
        )
    with pytest.raises(PolicyNumericalError, match="ODE time"):
        finite_field(
            torch.tensor(float("nan"), dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
        )

    class NonFiniteVelocity(torch.nn.Module):
        def forward(self, sample, timestep, global_cond):
            return torch.full_like(sample, float("inf"))

    nonfinite_field = _ConditionedVectorField(NonFiniteVelocity(), condition)
    with pytest.raises(PolicyNumericalError, match="velocity"):
        nonfinite_field(
            torch.tensor(0.25, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
        )


def test_prediction_validates_final_trajectory_shape_and_finiteness(monkeypatch):
    class InvalidTrajectoryODE:
        def __init__(self, vector_field, **kwargs):
            pass

        def trajectory(self, x, t_span):
            return torch.full(
                (len(t_span), 2),
                float("nan"),
                dtype=torch.float32,
            )

    monkeypatch.setattr(sfp_policies, "NeuralODE", InvalidTrajectoryODE)
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    nobs = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(PolicyNumericalError, match="trajectory"):
        policy.predict(nobs, num_actions=2, integration_steps_per_action=1)

    class WrongShapeODE(InvalidTrajectoryODE):
        def trajectory(self, x, t_span):
            return torch.zeros((len(t_span), 1), dtype=torch.float32)

    monkeypatch.setattr(sfp_policies, "NeuralODE", WrongShapeODE)
    with pytest.raises(ValueError, match="shape"):
        policy.predict(nobs, num_actions=2, integration_steps_per_action=1)


@pytest.mark.parametrize(
    "message,expected_exception",
    [
        ("adaptive solver produced non-finite state", PolicyNumericalError),
        ("banana-shaped internal tensor", RuntimeError),
        ("internal tensor shape mismatch", RuntimeError),
    ],
)
def test_prediction_translates_only_numerical_solver_runtime_errors(
    monkeypatch, message, expected_exception
):
    class FailingODE:
        def __init__(self, vector_field, **kwargs):
            pass

        def trajectory(self, x, t_span):
            raise RuntimeError(message)

    monkeypatch.setattr(sfp_policies, "NeuralODE", FailingODE)
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    with pytest.raises(expected_exception, match=message) as error:
        policy.predict(
            torch.zeros((2, 3), dtype=torch.float32),
            num_actions=2,
            integration_steps_per_action=1,
        )
    assert type(error.value) is expected_exception
