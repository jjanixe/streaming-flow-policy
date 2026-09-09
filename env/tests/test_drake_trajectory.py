import numpy as np

from env.drake_trajectory import SFPDDrakeTransform, evaluate_drake_foh


def test_drake_first_order_hold_matches_linear_segments():
    action = np.stack(
        [np.arange(16, dtype=np.float32), -np.arange(16, dtype=np.float32)],
        axis=-1,
    )
    position, derivative = evaluate_drake_foh(action, np.float32(0.1))
    np.testing.assert_allclose(position, [[1.5, -1.5]], atol=1e-6)
    np.testing.assert_allclose(derivative, [[15.0, -15.0]], atol=1e-6)
    assert position.dtype == derivative.dtype == np.float32


def test_drake_boundary_is_not_used_as_a_float64_training_output():
    action = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2)
    position, derivative = evaluate_drake_foh(action, np.float32(0.37))
    assert position.shape == derivative.shape == (1, 2)
    assert position.dtype == derivative.dtype == np.float32


def test_sfpd_transform_is_seeded_and_returns_only_float32_arrays():
    datum = {
        "obs": np.zeros((2, 3), dtype=np.float32),
        "action": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2),
        "trajectory_id": "validation:upper-narrow:000",
        "source_split": "validation",
    }
    first = SFPDDrakeTransform(0.1, np.random.default_rng(7))(datum)
    second = SFPDDrakeTransform(0.1, np.random.default_rng(7))(datum)
    for key in ("obs", "x", "v", "t"):
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].dtype == np.float32
    assert first["x"].shape == first["v"].shape == (1, 2)
    assert np.asarray(first["t"]).shape == ()
    assert first["trajectory_id"] == datum["trajectory_id"]
    assert first["source_split"] == "validation"
