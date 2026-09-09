import math

import numpy as np
import pytest
import torch

from env.analytic_fields import AnalyticField
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.samplers import integrate_ode, integrate_sde, sample_initial_states


def make_field() -> AnalyticField:
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=8)
    return AnalyticField(bank.select("train"), DEFAULT_CONFIG)


def test_initial_and_sampler_shapes_and_reproducibility():
    analytic_field = make_field()
    initial = sample_initial_states(
        16,
        DEFAULT_CONFIG,
        torch.Generator().manual_seed(20),
    )

    ode = integrate_ode(analytic_field, initial, horizon_steps=64)
    first = integrate_sde(
        analytic_field,
        initial,
        horizon_steps=64,
        generator=torch.Generator().manual_seed(21),
    )
    second = integrate_sde(
        analytic_field,
        initial,
        horizon_steps=64,
        generator=torch.Generator().manual_seed(21),
    )

    assert ode.shape == first.shape == (16, 65, 2)
    assert ode.dtype == first.dtype == torch.float32
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_centered_initialization_and_explicit_rng_do_not_use_global_state():
    global_state = torch.random.get_rng_state()
    centered = sample_initial_states(
        4,
        DEFAULT_CONFIG,
        torch.Generator().manual_seed(30),
        center_init=True,
    )
    gaussian = sample_initial_states(
        4,
        DEFAULT_CONFIG,
        torch.Generator().manual_seed(31),
    )

    torch.testing.assert_close(
        centered,
        torch.tensor([[-1.0, 0.0]], dtype=torch.float32).expand(4, -1),
    )
    assert not torch.equal(gaussian, centered)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_single_tube_euler_contraction():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=9).select("train")
    one_trajectory = bank.subset(np.array([0]))
    analytic_field = AnalyticField(one_trajectory, DEFAULT_CONFIG)
    initial = torch.tensor([[-1.0, 0.2]], dtype=torch.float32)

    rollout = integrate_ode(analytic_field, initial, horizon_steps=64)
    center, _ = analytic_field.centers_and_derivatives(
        torch.tensor([1.0], dtype=torch.float32)
    )
    final_error = torch.linalg.vector_norm(
        rollout[:, -1] - center[:, 0],
        dim=-1,
    )
    expected = torch.tensor(
        [0.2 * math.exp(-1.0)],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        final_error,
        expected,
        rtol=2e-2,
        atol=1e-4,
    )


def test_sampler_rejects_non_float32_or_invalid_horizon():
    analytic_field = make_field()

    with pytest.raises(ValueError, match="float32"):
        integrate_ode(
            analytic_field,
            torch.zeros((2, 2), dtype=torch.float64),
            horizon_steps=64,
        )
    with pytest.raises(ValueError, match="positive"):
        integrate_ode(
            analytic_field,
            torch.zeros((2, 2), dtype=torch.float32),
            horizon_steps=0,
        )
