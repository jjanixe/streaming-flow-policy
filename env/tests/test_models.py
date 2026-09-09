import pytest
import torch

from env.models import LocalTimeFeatures, SFPDVelocityMLP


def valid_inputs(batch_size: int = 5):
    return (
        torch.zeros((batch_size, 1, 2), dtype=torch.float32),
        torch.linspace(0.0, 1.0, batch_size, dtype=torch.float32),
        torch.zeros((batch_size, 6), dtype=torch.float32),
    )


def test_sfpd_velocity_mlp_preserves_sample_shape_and_float32():
    model = SFPDVelocityMLP(hidden_dim=128, hidden_layers=3)
    sample, time, condition = valid_inputs()
    output = model(sample=sample, timestep=time, global_cond=condition)
    assert output.shape == sample.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_local_time_features_has_expected_nine_features():
    features = LocalTimeFeatures()(torch.tensor([0.0, 0.25], dtype=torch.float32))
    assert features.shape == (2, 9)
    assert torch.allclose(features[:, 0], torch.tensor([0.0, 0.25]))


def test_sfpd_velocity_mlp_rejects_float64():
    model = SFPDVelocityMLP()
    sample, time, condition = valid_inputs(1)
    with pytest.raises(ValueError, match="float32"):
        model(sample=sample.double(), timestep=time, global_cond=condition)


@pytest.mark.parametrize(
    "which,bad",
    [
        ("sample", torch.zeros((2, 2), dtype=torch.float32)),
        ("sample", torch.zeros((2, 1, 3), dtype=torch.float32)),
        ("timestep", torch.zeros((2, 1), dtype=torch.float32)),
        ("global_cond", torch.zeros((2, 5), dtype=torch.float32)),
    ],
)
def test_sfpd_velocity_mlp_rejects_invalid_shapes(which, bad):
    model = SFPDVelocityMLP()
    sample, time, condition = valid_inputs(2)
    values = {"sample": sample, "timestep": time, "global_cond": condition}
    values[which] = bad
    with pytest.raises(ValueError, match="shape"):
        model(**values)


@pytest.mark.parametrize("time_value", [-1e-6, 1.000001])
def test_sfpd_velocity_mlp_rejects_out_of_bounds_timestep(time_value):
    model = SFPDVelocityMLP()
    sample, time, condition = valid_inputs(1)
    time[0] = time_value
    with pytest.raises(ValueError, match="timestep"):
        model(sample=sample, timestep=time, global_cond=condition)


def test_sfpd_velocity_mlp_rejects_nonfinite_inputs():
    model = SFPDVelocityMLP()
    sample, time, condition = valid_inputs(1)
    sample[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(sample=sample, timestep=time, global_cond=condition)


def test_sfpd_velocity_mlp_rejects_mismatched_batch_sizes():
    model = SFPDVelocityMLP()
    sample, time, condition = valid_inputs(2)
    with pytest.raises(ValueError, match="batch"):
        model(sample=sample, timestep=time[:1], global_cond=condition)


def test_sfpd_velocity_mlp_rejects_mismatched_input_devices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = SFPDVelocityMLP()
    sample, time, condition = valid_inputs(1)
    with pytest.raises(ValueError, match="device"):
        model(sample=sample.cuda(), timestep=time, global_cond=condition)
