import torch

from env.analytic_fields import AnalyticField
from env.config import StageAConfig


def _validate_count(count: int) -> None:
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")


def _validate_horizon(horizon_steps: int) -> None:
    if not isinstance(horizon_steps, int) or horizon_steps <= 0:
        raise ValueError("horizon_steps must be a positive integer")


def _validate_initial(initial_states: torch.Tensor) -> torch.Tensor:
    if not isinstance(initial_states, torch.Tensor) or (
        initial_states.ndim != 2 or initial_states.shape[-1] != 2
    ):
        raise ValueError("initial_states must be a Torch tensor with shape [B, 2]")
    if initial_states.dtype != torch.float32:
        raise ValueError("initial_states must use float32")
    if initial_states.device.type != "cpu":
        raise ValueError("Stage A samplers run on CPU")
    if not torch.isfinite(initial_states).all():
        raise ValueError("initial_states must be finite")
    return initial_states


def sample_initial_states(
    count: int,
    config: StageAConfig,
    generator: torch.Generator,
    center_init: bool = False,
) -> torch.Tensor:
    _validate_count(count)
    start = torch.tensor(
        config.environment.start,
        dtype=torch.float32,
    ).expand(count, -1)
    if center_init:
        return start.clone()
    noise = torch.randn(
        (count, 2),
        dtype=torch.float32,
        generator=generator,
    )
    return start + config.environment.initial_sigma * noise


def integrate_ode(
    field: AnalyticField,
    initial_states: torch.Tensor,
    horizon_steps: int,
) -> torch.Tensor:
    _validate_horizon(horizon_steps)
    states = _validate_initial(initial_states).clone()
    dt = torch.tensor(1.0 / horizon_steps, dtype=torch.float32)
    trajectory = [states]

    for index in range(horizon_steps):
        times = torch.full(
            (states.shape[0],),
            index / horizon_steps,
            dtype=torch.float32,
        )
        states = states + dt * field.velocity_pf(states, times)
        trajectory.append(states)

    return torch.stack(trajectory, dim=1)

def integrate_sde(
    field: AnalyticField,
    initial_states: torch.Tensor,
    horizon_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    _validate_horizon(horizon_steps)
    states = _validate_initial(initial_states).clone()
    dt = torch.tensor(1.0 / horizon_steps, dtype=torch.float32)
    trajectory = [states]

    for index in range(horizon_steps):
        times = torch.full(
            (states.shape[0],),
            index / horizon_steps,
            dtype=torch.float32,
        )
        epsilon = field.epsilon(times)
        noise = torch.randn(
            states.shape,
            dtype=torch.float32,
            generator=generator,
        )
        states = (
            states
            + dt * field.base_sde_drift(states, times)
            + torch.sqrt(2.0 * epsilon * dt)[:, None] * noise
        )
        trajectory.append(states)

    return torch.stack(trajectory, dim=1)
