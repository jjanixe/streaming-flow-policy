import numpy as np
import torch

from env.drake_trajectory import (
    SFPDDrakeTransform,
    SFPSDrakeTransform,
    evaluate_drake_foh,
    evaluate_uniform_foh_batch,
    sample_sfpd_training_targets,
    sample_sfps_training_targets,
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


def test_batched_uniform_foh_matches_drake_at_interior_and_boundaries():
    first = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2)
    second = np.stack(
        (
            np.linspace(0.5, -0.5, 16, dtype=np.float32),
            np.linspace(-0.25, 0.75, 16, dtype=np.float32),
        ),
        axis=-1,
    )
    actions = torch.from_numpy(np.stack((first, second, first, second)))
    times = torch.tensor([0.0, 0.1, 0.731, 1.0], dtype=torch.float32)

    positions, derivatives = evaluate_uniform_foh_batch(actions, times)

    expected = [
        evaluate_drake_foh(action, np.float32(time))
        for action, time in zip(actions.numpy(), times.numpy())
    ]
    expected_positions = np.concatenate([value[0] for value in expected])[:, None, :]
    expected_derivatives = np.concatenate([value[1] for value in expected])[:, None, :]
    torch.testing.assert_close(
        positions,
        torch.from_numpy(expected_positions),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        derivatives,
        torch.from_numpy(expected_derivatives),
        rtol=0.0,
        atol=1e-6,
    )
    assert positions.shape == derivatives.shape == (4, 1, 2)
    assert positions.dtype == derivatives.dtype == torch.float32


def test_batched_sfps_targets_are_seeded_and_match_the_drake_formula():
    actions = torch.from_numpy(
        np.stack(
            (
                np.linspace(-0.8, 0.8, 32, dtype=np.float32).reshape(16, 2),
                np.linspace(0.7, -0.7, 32, dtype=np.float32).reshape(16, 2),
                np.arange(32, dtype=np.float32).reshape(16, 2) / 16.0,
            )
        )
    )
    first_generator = torch.Generator().manual_seed(31415)
    second_generator = torch.Generator().manual_seed(31415)

    actual = sample_sfps_training_targets(
        actions,
        sigma0=0.1,
        sigma1=0.2,
        generator=first_generator,
    )
    replay = sample_sfps_training_targets(
        actions,
        sigma0=0.1,
        sigma1=0.2,
        generator=second_generator,
    )

    assert actual.keys() == {"a", "z", "va", "vz", "t"}
    for name in actual:
        torch.testing.assert_close(actual[name], replay[name], rtol=0.0, atol=0.0)
        assert actual[name].dtype == torch.float32

    expected_xi = []
    expected_xi_dot = []
    for action, time in zip(actions.numpy(), actual["t"].numpy()):
        xi, xi_dot = evaluate_drake_foh(action, np.float32(time))
        expected_xi.append(xi)
        expected_xi_dot.append(xi_dot)
    xi = torch.from_numpy(np.stack(expected_xi))
    xi_dot = torch.from_numpy(np.stack(expected_xi_dot))
    # Recover the two source noises from the closed-form outputs and verify
    # every target against the independent Drake evaluation.
    sigma_r = np.float32(np.sqrt(np.float32(0.2**2 - 0.1**2)))
    z0 = (actual["va"] - xi_dot) / sigma_r
    epsilon_a0 = actual["a"] - xi - sigma_r * actual["t"][:, None, None] * z0
    expected_z = (
        1.0 - 0.8 * actual["t"][:, None, None]
    ) * z0 + actual["t"][:, None, None] * xi
    expected_vz = (
        xi
        + actual["t"][:, None, None] * xi_dot
        - 0.8 * z0
    )
    torch.testing.assert_close(actual["z"], expected_z, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual["vz"], expected_vz, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(epsilon_a0).all()
    assert actual["a"].shape == actual["z"].shape == (3, 1, 2)
    assert actual["t"].shape == (3,)


def test_batched_sfpd_targets_are_seeded_and_match_the_drake_formula():
    actions = torch.from_numpy(
        np.stack(
            (
                np.linspace(-0.9, 0.9, 32, dtype=np.float32).reshape(16, 2),
                np.linspace(0.6, -0.6, 32, dtype=np.float32).reshape(16, 2),
            )
        )
    )
    actual = sample_sfpd_training_targets(
        actions,
        sigma=0.1,
        generator=torch.Generator().manual_seed(2718),
    )
    replay = sample_sfpd_training_targets(
        actions,
        sigma=0.1,
        generator=torch.Generator().manual_seed(2718),
    )

    for name in actual:
        torch.testing.assert_close(actual[name], replay[name], rtol=0.0, atol=0.0)
        assert actual[name].dtype == torch.float32
    replay_generator = torch.Generator().manual_seed(2718)
    times = torch.rand((2,), dtype=torch.float32, generator=replay_generator)
    noise = torch.randn((2, 1, 2), dtype=torch.float32, generator=replay_generator)
    expected = [
        evaluate_drake_foh(action, np.float32(time))
        for action, time in zip(actions.numpy(), times.numpy())
    ]
    xi = torch.from_numpy(np.stack([value[0] for value in expected]))
    xi_dot = torch.from_numpy(np.stack([value[1] for value in expected]))
    torch.testing.assert_close(actual["x"], xi + 0.1 * noise, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(actual["v"], xi_dot, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(actual["t"], times, rtol=0.0, atol=0.0)


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
