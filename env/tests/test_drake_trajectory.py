import numpy as np

from env.drake_trajectory import (
    SFPDDrakeTransform,
    SFPSDrakeTransform,
    evaluate_drake_foh,
)


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


def test_sfps_transform_matches_equal_sigma_formula_and_is_seeded():
    datum = {
        "obs": np.zeros((2, 3), dtype=np.float32),
        "action": np.linspace(-0.75, 0.75, 32, dtype=np.float32).reshape(16, 2),
        "trajectory_id": "train:upper-wide:000",
        "source_split": "train",
    }
    first = SFPSDrakeTransform(0.1, 0.1, np.random.default_rng(17))(datum)
    second = SFPSDrakeTransform(0.1, 0.1, np.random.default_rng(17))(datum)
    replay_rng = np.random.default_rng(17)
    tau = np.float32(replay_rng.random())
    xi, xi_dot = evaluate_drake_foh(datum["action"], tau)
    z0 = replay_rng.standard_normal((1, 2), dtype=np.float32)
    epsilon_a0 = np.float32(0.1) * replay_rng.standard_normal(
        (1, 2), dtype=np.float32
    )
    expected_a = xi + epsilon_a0
    expected_z = (
        np.float32(1.0) - np.float32(0.9) * tau
    ) * z0 + tau * xi
    expected_va = xi_dot
    expected_vz = xi + tau * xi_dot - np.float32(0.9) * z0

    for key in ("obs", "a", "z", "va", "vz", "t"):
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].dtype == np.float32
    np.testing.assert_allclose(first["a"], expected_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first["z"], expected_z, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first["va"], expected_va, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first["vz"], expected_vz, rtol=1e-6, atol=1e-6)
    assert first["a"].shape == first["z"].shape == (1, 2)
    assert first["trajectory_id"] == datum["trajectory_id"]
    assert "action" not in first


def test_sfps_transform_matches_nonzero_sigma_r_formula():
    datum = {
        "obs": np.zeros((2, 3), dtype=np.float32),
        "action": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2),
    }
    actual = SFPSDrakeTransform(0.1, 0.2, np.random.default_rng(23))(datum)
    replay_rng = np.random.default_rng(23)
    tau = np.float32(replay_rng.random())
    xi, xi_dot = evaluate_drake_foh(datum["action"], tau)
    z0 = replay_rng.standard_normal((1, 2), dtype=np.float32)
    epsilon_a0 = np.float32(0.1) * replay_rng.standard_normal(
        (1, 2), dtype=np.float32
    )
    sigma_r = np.float32(np.sqrt(np.float32(0.2**2 - 0.1**2)))

    np.testing.assert_allclose(
        actual["a"], xi + epsilon_a0 + sigma_r * tau * z0, atol=1e-6
    )
    np.testing.assert_allclose(actual["va"], xi_dot + sigma_r * z0, atol=1e-6)
