import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.analytic_fields import AnalyticField
from env.artifacts import (
    save_demonstration_bank,
    save_feature_normalizer,
    save_json,
    save_rollouts,
    train_data_digest,
)
from env.config import DEFAULT_CONFIG, config_to_dict
from env.demonstrations import generate_demonstration_bank
from env.features import fit_feature_normalizer
from env.samplers import integrate_ode, integrate_sde, sample_initial_states


DIAGNOSTIC_INDICES = {
    "0.25": 16,
    "0.5": 32,
    "0.75": 48,
    "1.0": 64,
}
MEAN_ERROR_LIMIT = 0.03
COVARIANCE_ERROR_LIMIT = 0.03
MODE_ERROR_LIMIT = 0.08
OTHER_LIMIT = 0.02
MODE_NAMES = (
    "upper-narrow",
    "upper-wide",
    "lower-narrow",
    "lower-wide",
)


def sample_errors(
    samples: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    if (
        samples.ndim != 2
        or samples.shape[-1] != 2
        or reference.shape != samples.shape
        or len(samples) < 2
    ):
        raise ValueError("samples and reference must share shape [N, 2], N >= 2")
    if not torch.isfinite(samples).all() or not torch.isfinite(reference).all():
        raise ValueError("samples and reference must be finite")
    mean_error = torch.linalg.vector_norm(
        samples.mean(dim=0) - reference.mean(dim=0)
    )
    sample_covariance = torch.cov(samples.T)
    reference_covariance = torch.cov(reference.T)
    covariance_error = torch.linalg.matrix_norm(
        sample_covariance - reference_covariance
    )
    return float(mean_error.item()), float(covariance_error.item())


def classify_midpoint_modes(samples: torch.Tensor) -> dict[str, float]:
    if samples.ndim != 2 or samples.shape[-1] != 2 or len(samples) == 0:
        raise ValueError("samples must have shape [N, 2], N >= 1")
    if samples.dtype != torch.float32:
        raise ValueError("samples must use float32")
    y = samples[:, 1]
    finite = torch.isfinite(samples).all(dim=1)
    other = finite & (y.abs() < 0.12)
    wide = finite & (y.abs() >= 0.40)
    occupancy = {
        "upper-narrow": (finite & ~other & ~wide & (y > 0.0)).float().mean(),
        "upper-wide": (finite & ~other & wide & (y > 0.0)).float().mean(),
        "lower-narrow": (finite & ~other & ~wide & (y < 0.0)).float().mean(),
        "lower-wide": (finite & ~other & wide & (y < 0.0)).float().mean(),
        "other": other.float().mean(),
        "nonfinite": (~finite).float().mean(),
    }
    return {
        name: float(probability.item())
        for name, probability in occupancy.items()
    }


def action_limit_diagnostics(
    trajectories: torch.Tensor,
    max_step_distance: float,
) -> dict[str, Any]:
    if (
        not isinstance(trajectories, torch.Tensor)
        or trajectories.ndim != 3
        or trajectories.shape[0] < 1
        or trajectories.shape[1] < 2
        or trajectories.shape[2] != 2
    ):
        raise ValueError(
            "trajectories must be a Torch tensor with shape [N, T, 2]"
        )
    if trajectories.dtype != torch.float32:
        raise ValueError("trajectories must use float32")
    if not math.isfinite(max_step_distance) or max_step_distance <= 0.0:
        raise ValueError("max_step_distance must be finite and positive")

    distances = torch.linalg.vector_norm(
        torch.diff(trajectories, dim=1),
        dim=-1,
    )
    limit = trajectories.new_tensor(max_step_distance)
    finite = torch.isfinite(distances)
    activated = finite & (distances > limit)
    activated_by_trajectory = activated.any(dim=1)
    activated_indices = torch.nonzero(
        activated_by_trajectory,
        as_tuple=False,
    ).flatten()
    action_count = distances.numel()
    activation_count = int(activated.sum().item())
    finite_distances = distances[finite]
    max_requested_step_distance = (
        float(finite_distances.max().item()) if len(finite_distances) else None
    )
    return {
        "max_step_distance": float(max_step_distance),
        "action_count": action_count,
        "activation_count": activation_count,
        "activation_rate": activation_count / action_count,
        "activated_trajectory_count": int(activated_by_trajectory.sum().item()),
        "activated_trajectory_rate": float(
            activated_by_trajectory.float().mean().item()
        ),
        "activated_trajectory_indices": activated_indices.tolist(),
        "max_requested_step_distance": max_requested_step_distance,
    }


def _seed_value(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sampler_diagnostics(
    rollout: torch.Tensor,
    references: dict[str, torch.Tensor],
    max_step_distance: float,
) -> dict[str, Any]:
    time_metrics: dict[str, dict[str, float | int | None]] = {}
    for label, index in DIAGNOSTIC_INDICES.items():
        samples = rollout[:, index]
        reference = references[label]
        finite_rows = torch.isfinite(samples).all(dim=1) & torch.isfinite(
            reference
        ).all(dim=1)
        nonfinite_count = int((~finite_rows).sum().item())
        if nonfinite_count:
            mean_error = None
            covariance_error = None
        else:
            mean_error, covariance_error = sample_errors(samples, reference)
        time_metrics[label] = {
            "mean_l2_error": mean_error,
            "covariance_frobenius_error": covariance_error,
            "nonfinite_count": nonfinite_count,
        }
    return {
        "times": time_metrics,
        "midpoint_occupancy": classify_midpoint_modes(
            rollout[:, DIAGNOSTIC_INDICES["0.5"]]
        ),
        "nonfinite_state_count": int(
            (~torch.isfinite(rollout).all(dim=-1)).sum().item()
        ),
        "action_limit": action_limit_diagnostics(
            rollout,
            max_step_distance=max_step_distance,
        ),
    }


def acceptance_failures(samplers: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for sampler_name, diagnostics in samplers.items():
        nonfinite_states = diagnostics.get("nonfinite_state_count", 0)
        if not isinstance(nonfinite_states, int) or nonfinite_states != 0:
            failures.append(
                f"{sampler_name} contains {nonfinite_states} non-finite states"
            )
        for time_label, metrics in diagnostics["times"].items():
            if metrics.get("nonfinite_count", 0) != 0:
                failures.append(
                    f"{sampler_name} has non-finite samples at {time_label}"
                )
            mean_error = metrics["mean_l2_error"]
            covariance_error = metrics["covariance_frobenius_error"]
            if mean_error is None or not math.isfinite(mean_error):
                failures.append(
                    f"{sampler_name} has non-finite mean error at {time_label}"
                )
            elif mean_error > MEAN_ERROR_LIMIT:
                failures.append(
                    f"{sampler_name} mean error at {time_label} exceeds "
                    f"{MEAN_ERROR_LIMIT}"
                )
            if covariance_error is None or not math.isfinite(covariance_error):
                failures.append(
                    f"{sampler_name} has non-finite covariance error at "
                    f"{time_label}"
                )
            elif covariance_error > COVARIANCE_ERROR_LIMIT:
                failures.append(
                    f"{sampler_name} covariance error at {time_label} exceeds "
                    f"{COVARIANCE_ERROR_LIMIT}"
                )
        occupancy = diagnostics["midpoint_occupancy"]
        occupancy_names = (*MODE_NAMES, "other", "nonfinite")
        invalid_occupancies = [
            name
            for name in occupancy_names
            if name not in occupancy or not math.isfinite(occupancy[name])
        ]
        for name in invalid_occupancies:
            failures.append(
                f"{sampler_name} has non-finite midpoint occupancy for {name}"
            )
        if not invalid_occupancies and not math.isclose(
            sum(occupancy[name] for name in occupancy_names),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            failures.append(
                f"{sampler_name} midpoint occupancies do not sum to 1"
            )
        for mode_name in MODE_NAMES:
            if mode_name not in invalid_occupancies and (
                abs(occupancy[mode_name] - 0.25) > MODE_ERROR_LIMIT
            ):
                failures.append(
                    f"{sampler_name} {mode_name} occupancy differs from 0.25 "
                    f"by more than {MODE_ERROR_LIMIT}"
                )
        if "other" not in invalid_occupancies and occupancy["other"] > OTHER_LIMIT:
            failures.append(
                f"{sampler_name} other occupancy exceeds {OTHER_LIMIT}"
            )
        if (
            "nonfinite" not in invalid_occupancies
            and occupancy["nonfinite"] > 0.0
        ):
            failures.append(
                f"{sampler_name} midpoint occupancy contains non-finite samples"
            )
    return failures


def run_stage_a(
    output_dir: str | Path,
    seed: int = 0,
    num_rollouts: int = 4096,
    enforce_acceptance: bool = True,
) -> dict[str, Any]:
    if num_rollouts < 2:
        raise ValueError("num_rollouts must be at least 2")
    config = replace(
        DEFAULT_CONFIG,
        root_seed=seed,
        diagnostic_rollouts=num_rollouts,
    )
    stream_sequences = np.random.SeedSequence(seed).spawn(4)
    stream_seeds = {
        "data": _seed_value(stream_sequences[0]),
        "initial_state": _seed_value(stream_sequences[1]),
        "sde_noise": _seed_value(stream_sequences[2]),
        "target_reference": _seed_value(stream_sequences[3]),
    }

    bank = generate_demonstration_bank(config, seed=stream_seeds["data"])
    normalizer = fit_feature_normalizer(bank, config)
    analytic_field = AnalyticField(bank.select("train"), config)
    initial_states = sample_initial_states(
        num_rollouts,
        config,
        torch.Generator().manual_seed(stream_seeds["initial_state"]),
    )
    ode_positions = integrate_ode(
        analytic_field,
        initial_states,
        horizon_steps=config.environment.horizon_steps,
    )
    sde_positions = integrate_sde(
        analytic_field,
        initial_states,
        horizon_steps=config.environment.horizon_steps,
        generator=torch.Generator().manual_seed(stream_seeds["sde_noise"]),
    )

    reference_generator = torch.Generator().manual_seed(
        stream_seeds["target_reference"]
    )
    references: dict[str, torch.Tensor] = {}
    for label in DIAGNOSTIC_INDICES:
        normalized_time = float(label)
        times = torch.full(
            (num_rollouts,),
            normalized_time,
            dtype=torch.float32,
        )
        references[label] = analytic_field.sample_marginal(
            times,
            generator=reference_generator,
        )

    sampler_diagnostics = {
        "ode": _sampler_diagnostics(
            ode_positions,
            references,
            max_step_distance=config.environment.max_step_distance,
        ),
        "sde": _sampler_diagnostics(
            sde_positions,
            references,
            max_step_distance=config.environment.max_step_distance,
        ),
    }
    failures = acceptance_failures(sampler_diagnostics)
    diagnostics: dict[str, Any] = {
        "version": "toy-v0.1-stage-a",
        "seed": seed,
        "num_rollouts": num_rollouts,
        "acceptance_enforced": enforce_acceptance,
        "accepted": not failures,
        "failures": failures,
        "thresholds": {
            "mean_l2_error": MEAN_ERROR_LIMIT,
            "covariance_frobenius_error": COVARIANCE_ERROR_LIMIT,
            "mode_probability_error": MODE_ERROR_LIMIT,
            "other_probability": OTHER_LIMIT,
        },
        "stream_seeds": stream_seeds,
        "samplers": sampler_diagnostics,
    }

    destination = Path(output_dir)
    save_demonstration_bank(destination / "demonstrations.npz", bank)
    save_feature_normalizer(
        destination / "feature_normalizer.npz",
        normalizer,
        train_data_digest(bank),
    )
    save_rollouts(
        destination / "rollouts.npz",
        initial_states.numpy(),
        ode_positions.numpy(),
        sde_positions.numpy(),
    )
    save_json(destination / "resolved_config.json", config_to_dict(config))
    save_json(destination / "diagnostics.json", diagnostics)
    return diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage A analytic toy-environment diagnostics."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("env/artifacts"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-rollouts", type=int, default=4096)
    parser.add_argument(
        "--no-enforce-acceptance",
        action="store_true",
        help="Write diagnostics without returning a failing exit status.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    enforce_acceptance = not args.no_enforce_acceptance
    diagnostics = run_stage_a(
        args.output_dir,
        seed=args.seed,
        num_rollouts=args.num_rollouts,
        enforce_acceptance=enforce_acceptance,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    if enforce_acceptance and not diagnostics["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
