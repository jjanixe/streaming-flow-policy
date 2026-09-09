import argparse
import json
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
    if samples.ndim != 2 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape [N, 2]")
    if samples.dtype != torch.float32:
        raise ValueError("samples must use float32")
    y = samples[:, 1]
    other = y.abs() < 0.12
    wide = y.abs() >= 0.40
    occupancy = {
        "upper-narrow": (~other & ~wide & (y > 0.0)).float().mean(),
        "upper-wide": (~other & wide & (y > 0.0)).float().mean(),
        "lower-narrow": (~other & ~wide & (y < 0.0)).float().mean(),
        "lower-wide": (~other & wide & (y < 0.0)).float().mean(),
        "other": other.float().mean(),
    }
    return {
        name: float(probability.item())
        for name, probability in occupancy.items()
    }


def _seed_value(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sampler_diagnostics(
    rollout: torch.Tensor,
    references: dict[str, torch.Tensor],
) -> dict[str, Any]:
    time_metrics: dict[str, dict[str, float]] = {}
    for label, index in DIAGNOSTIC_INDICES.items():
        mean_error, covariance_error = sample_errors(
            rollout[:, index],
            references[label],
        )
        time_metrics[label] = {
            "mean_l2_error": mean_error,
            "covariance_frobenius_error": covariance_error,
        }
    return {
        "times": time_metrics,
        "midpoint_occupancy": classify_midpoint_modes(
            rollout[:, DIAGNOSTIC_INDICES["0.5"]]
        ),
    }


def _acceptance_failures(samplers: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for sampler_name, diagnostics in samplers.items():
        for time_label, metrics in diagnostics["times"].items():
            if metrics["mean_l2_error"] > MEAN_ERROR_LIMIT:
                failures.append(
                    f"{sampler_name} mean error at {time_label} exceeds "
                    f"{MEAN_ERROR_LIMIT}"
                )
            if (
                metrics["covariance_frobenius_error"]
                > COVARIANCE_ERROR_LIMIT
            ):
                failures.append(
                    f"{sampler_name} covariance error at {time_label} exceeds "
                    f"{COVARIANCE_ERROR_LIMIT}"
                )
        occupancy = diagnostics["midpoint_occupancy"]
        for mode_name in MODE_NAMES:
            if abs(occupancy[mode_name] - 0.25) > MODE_ERROR_LIMIT:
                failures.append(
                    f"{sampler_name} {mode_name} occupancy differs from 0.25 "
                    f"by more than {MODE_ERROR_LIMIT}"
                )
        if occupancy["other"] > OTHER_LIMIT:
            failures.append(
                f"{sampler_name} other occupancy exceeds {OTHER_LIMIT}"
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
        "ode": _sampler_diagnostics(ode_positions, references),
        "sde": _sampler_diagnostics(sde_positions, references),
    }
    failures = _acceptance_failures(sampler_diagnostics)
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
