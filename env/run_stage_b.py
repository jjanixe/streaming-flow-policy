"""Command-line orchestration for the deterministic toy Stage B1 workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from env.artifacts import (
    load_demonstration_bank,
    save_json,
    train_data_digest,
)
from env.chunk_data import PushTChunkDataset, PushTStats
from env.config import DEFAULT_CONFIG, config_to_dict
from env.demonstrations import DemonstrationBank
from env.evaluate_stage_b import (
    SFPDRollout,
    SFPDRolloutBatch,
    evaluate_sfpd,
    rollout_sfpd,
)
from env.run_stage_a import classify_midpoint_modes
from env.stage_b_artifacts import (
    load_pusht_stats,
    load_sfpd_checkpoint,
    save_sfpd_rollouts,
)
from env.stage_b_config import (
    DEFAULT_STAGE_B1_CONFIG,
    StageB1Config,
    stage_b1_config_to_dict,
)
from env.visualize_stage_b import (
    plot_mode_occupancy,
    plot_trajectory_comparison,
    write_representative_gifs,
)

if TYPE_CHECKING:
    from env.sfp_policies import StreamingFlowPolicyDeterministic
    from env.train_stage_b import TrainResult


RUN_FORMAT_VERSION = 1
ENVIRONMENT_ID = "PointReach2DPreference-v0"
UINT32_SEED_SPACE_SIZE = 2**32


def derive_rollout_seeds(root_seed: int, count: int) -> list[int]:
    """Derive stable, independent uint32 rollout seeds from one stream seed."""
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if count > UINT32_SEED_SPACE_SIZE:
        raise ValueError("count exceeds the uint32 seed space capacity")
    result: list[int] = []
    used: set[int] = set()
    for child in np.random.SeedSequence(root_seed).spawn(count):
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        while seed in used:
            seed = (seed + 1) % UINT32_SEED_SPACE_SIZE
        used.add(seed)
        result.append(seed)
    return result


def _checkpoint_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_device(device: str | torch.device) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return result


def _validate_evaluation_config(config: StageB1Config) -> None:
    for name in ("rollout_count", "integration_steps_per_action"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def load_trained_sfpd(
    checkpoint_path: str | Path,
    stats_path: str | Path,
    *,
    expected_train_data_digest: str,
    device: str | torch.device = "cpu",
) -> tuple[StreamingFlowPolicyDeterministic, PushTStats, dict[str, Any]]:
    """Validate artifacts, reconstruct the policy, and load selected EMA state."""
    from env.models import SFPDVelocityMLP
    from env.sfp_policies import StreamingFlowPolicyDeterministic

    stats, stats_digest = load_pusht_stats(stats_path)
    if stats_digest != expected_train_data_digest:
        raise ValueError("statistics train data digest does not match")
    checkpoint = load_sfpd_checkpoint(
        checkpoint_path,
        expected_train_data_digest=expected_train_data_digest,
    )
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]
    resolved_device = _resolved_device(device)
    with torch.random.fork_rng(devices=[]):
        velocity_net = SFPDVelocityMLP(
            hidden_dim=architecture["hidden_dim"],
            hidden_layers=architecture["hidden_layers"],
        )
    policy = StreamingFlowPolicyDeterministic(
        velocity_net,
        pred_horizon=architecture["pred_horizon"],
        device=resolved_device,
    ).to(resolved_device)
    policy.load_state_dict(checkpoint["ema_state"], strict=True)
    policy.eval()
    for value in policy.state_dict().values():
        if value.is_floating_point():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("loaded EMA state must contain finite float32 tensors")
        if value.device != resolved_device:
            raise ValueError("loaded EMA state is on the wrong device")
    return policy, stats, metadata


@torch.inference_mode()
def _held_out_field_loss(
    policy: StreamingFlowPolicyDeterministic,
    stats: PushTStats,
    bank: DemonstrationBank,
    config: StageB1Config,
    seed: int,
) -> float:
    from env.drake_trajectory import SFPDDrakeTransform

    dataset = PushTChunkDataset(
        bank,
        split="test",
        stats=stats,
        transform=SFPDDrakeTransform(
            config.sigma,
            np.random.default_rng(seed),
        ),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    weighted_loss = 0.0
    sample_count = 0
    for batch in loader:
        loss = policy.loss(batch)
        if loss.dtype != torch.float32 or not torch.isfinite(loss):
            raise FloatingPointError("held-out field loss is non-finite")
        count = int(batch["obs"].shape[0])
        weighted_loss += float(loss.item()) * count
        sample_count += count
    if sample_count == 0:
        raise ValueError("test split produced no held-out samples")
    result = weighted_loss / sample_count
    if not math.isfinite(result):
        raise FloatingPointError("held-out field loss is non-finite")
    return result


def _rollout_equal(first: SFPDRollout, second: SFPDRollout) -> bool:
    return (
        first.seed == second.seed
        and first.success == second.success
        and first.numerical_failure == second.numerical_failure
        and first.action_limit_failure == second.action_limit_failure
        and np.array_equal(first.initial_observation, second.initial_observation)
        and np.array_equal(
            first.executed_positions,
            second.executed_positions,
            equal_nan=True,
        )
        and np.array_equal(
            first.requested_actions,
            second.requested_actions,
            equal_nan=True,
        )
        and np.array_equal(
            first.raw_predicted_chunks,
            second.raw_predicted_chunks,
            equal_nan=True,
        )
    )


def _occupancy_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    occupancy = result["midpoint_occupancy"]
    upper = float(occupancy["upper-narrow"] + occupancy["upper-wide"])
    lower = float(occupancy["lower-narrow"] + occupancy["lower-wide"])
    narrow = float(occupancy["upper-narrow"] + occupancy["lower-narrow"])
    wide = float(occupancy["upper-wide"] + occupancy["lower-wide"])
    result["mode_summary"] = {
        "upper_occupancy": upper,
        "lower_occupancy": lower,
        "upper_lower_abs_difference": abs(upper - lower),
        "narrow_occupancy": narrow,
        "wide_occupancy": wide,
        "narrow_wide_abs_difference": abs(narrow - wide),
    }
    return result


def _unique_executed_trajectory_count(batch: SFPDRolloutBatch) -> int:
    return len({rollout.executed_positions.tobytes() for rollout in batch.rollouts})


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "torch": str(torch.__version__),
        "torchdyn": importlib.metadata.version("torchdyn"),
        "drake": importlib.metadata.version("drake"),
        "gym": importlib.metadata.version("gym"),
    }


def _finite_audit_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    return {
        "count": int(len(array)),
        "finite_count": int(len(finite)),
        "nonfinite_count": int(len(array) - len(finite)),
        "mean": float(finite.mean()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
    }


def _dataset_split_audit(
    bank: DemonstrationBank,
    stats: PushTStats,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        dataset = PushTChunkDataset(bank, split=split, stats=stats)
        corrections: list[float] = []
        endpoint_mismatches: list[float] = []
        for index in range(len(dataset)):
            datum = dataset[index]
            if datum["source_split"] != split:
                raise ValueError("dataset source_split audit metadata is inconsistent")
            corrections.append(float(datum["anchor_correction_raw_l2"]))
            endpoint_mismatches.append(float(datum["anchor_physical_l2"]))
        result[split] = {
            "source_split": split,
            "window_count": len(dataset),
            "boundary_anchor_correction_raw_l2": _finite_audit_summary(
                corrections
            ),
            "normalization_endpoint_mismatch_l2": _finite_audit_summary(
                endpoint_mismatches
            ),
        }
    return result


def _rollout_boundary_jump_summary(
    batch: SFPDRolloutBatch,
) -> dict[str, Any]:
    values: list[float] = []
    for rollout in batch.rollouts:
        chunks = rollout.raw_predicted_chunks
        if len(chunks) > 1:
            jumps = np.linalg.norm(
                chunks[1:, 0] - chunks[:-1, 8],
                axis=1,
            )
            values.extend(float(value) for value in jumps)
    return _finite_audit_summary(values)


def write_stage_b1_outputs(
    destination: str | Path,
    bank: DemonstrationBank,
    config: StageB1Config,
    training: TrainResult,
    checkpoint_metadata: dict[str, Any],
    gaussian_metrics: dict[str, Any],
    gaussian_rollouts: SFPDRolloutBatch,
    centered_metrics: dict[str, Any],
    centered_rollouts: SFPDRolloutBatch,
    *,
    device: str | torch.device,
    held_out_field_loss: float,
    deterministic_replay: bool,
    enforce_acceptance: bool,
) -> dict[str, Any]:
    """Persist all nonconditional Stage B1 diagnostics and visual artifacts."""
    output_dir = Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = train_data_digest(bank)
    if checkpoint_metadata["train_data_digest"] != digest:
        raise ValueError("checkpoint and demonstration digests do not match")
    checkpoint_hash = _checkpoint_digest(training.checkpoint_path)
    stats, stats_digest = load_pusht_stats(training.stats_path)
    if stats_digest != digest:
        raise ValueError("statistics and demonstration digests do not match")
    versions = _runtime_versions()
    gaussian = _occupancy_summary(gaussian_metrics)
    centered = _occupancy_summary(centered_metrics)
    gaussian["acceptance_applicable"] = True
    centered["acceptance_applicable"] = False
    gaussian["unique_executed_trajectory_count"] = (
        _unique_executed_trajectory_count(gaussian_rollouts)
    )
    centered["unique_executed_trajectory_count"] = (
        _unique_executed_trajectory_count(centered_rollouts)
    )
    rollout_seeds = [rollout.seed for rollout in gaussian_rollouts.rollouts]
    centered_seeds = [rollout.seed for rollout in centered_rollouts.rollouts]

    save_sfpd_rollouts(
        output_dir / "rollouts.npz",
        gaussian_rollouts,
        checkpoint_digest=checkpoint_hash,
        train_data_digest=digest,
    )
    save_sfpd_rollouts(
        output_dir / "centered_rollouts.npz",
        centered_rollouts,
        checkpoint_digest=checkpoint_hash,
        train_data_digest=digest,
    )

    test_positions = bank.select("test").positions
    expert_occupancy = classify_midpoint_modes(
        torch.from_numpy(test_positions[:, 32])
    )
    plot_trajectory_comparison(
        output_dir / "trajectory_comparison.png",
        test_positions,
        gaussian_rollouts,
        DEFAULT_CONFIG.environment,
    )
    plot_mode_occupancy(
        output_dir / "mode_occupancy.png",
        expert_occupancy,
        gaussian["midpoint_occupancy"],
        sfpd_classified_count=gaussian["midpoint_classified_count"],
        sfpd_rollout_count=gaussian["rollout_count"],
    )
    gif_paths = write_representative_gifs(
        output_dir,
        gaussian_rollouts,
        DEFAULT_CONFIG.environment,
        center_init=False,
    )

    resolved_config = {
        "format_version": RUN_FORMAT_VERSION,
        "stage": "b1",
        "model_type": "sfpd",
        "environment_id": ENVIRONMENT_ID,
        "environment": config_to_dict(DEFAULT_CONFIG)["environment"],
        "stage_b1": stage_b1_config_to_dict(config),
        "device": str(torch.device(device)),
        "versions": versions,
    }
    seed_payload = {
        "root_seed": checkpoint_metadata["root_seed"],
        "training": dict(training.seed_streams),
        "gaussian_rollouts": rollout_seeds,
        "centered_rollouts": centered_seeds,
    }
    failures = list(gaussian["acceptance_failures"])
    checkpoint_environment = {
        "recorded": {
            "drake": checkpoint_metadata["drake_version"],
            "numpy": checkpoint_metadata["numpy_version"],
            "torch": checkpoint_metadata["torch_version"],
        },
        "matches_runtime": {
            "drake": checkpoint_metadata["drake_version"] == versions["drake"],
            "numpy": checkpoint_metadata["numpy_version"] == versions["numpy"],
            "torch": checkpoint_metadata["torch_version"] == versions["torch"],
        },
    }
    audit = {
        "dataset_splits": _dataset_split_audit(bank, stats),
        "rollout_boundary_jump_l2": {
            "definition": (
                "L2(raw_predicted_chunks[c, 0] - "
                "raw_predicted_chunks[c - 1, 8]) for each replan boundary"
            ),
            "gaussian": _rollout_boundary_jump_summary(gaussian_rollouts),
            "centered": _rollout_boundary_jump_summary(centered_rollouts),
        },
        "anchor_correction_definition": (
            "dataset action[0] replacement needed to equal the final observation "
            "anchor after independent PushT normalization"
        ),
        "normalization_endpoint_mismatch_definition": (
            "physical L2 difference after independently normalized observation "
            "and action endpoints are made equal in normalized coordinates"
        ),
    }
    diagnostics = {
        "format_version": RUN_FORMAT_VERSION,
        "version": "toy-v0.1-stage-b1",
        "model_type": "sfpd",
        "environment_id": ENVIRONMENT_ID,
        "train_data_digest": digest,
        "checkpoint_digest": checkpoint_hash,
        "checkpoint": {
            "selected_update": checkpoint_metadata["selected_update"],
            "validation_loss": checkpoint_metadata["validation_loss"],
            "architecture": checkpoint_metadata["architecture"],
        },
        "checkpoint_environment": checkpoint_environment,
        "audit": audit,
        "teacher_forced_test_field_loss": held_out_field_loss,
        "deterministic_replay": deterministic_replay,
        "expert_midpoint_occupancy": expert_occupancy,
        "gaussian": gaussian,
        "centered": centered,
        "acceptance_enforced": enforce_acceptance,
        "accepted": not failures,
        "failures": failures,
        "dtypes": {
            "drake_interpolation_boundary": "float64",
            "learned_model_and_ode": "float32",
            "rollout_arrays": "float32",
        },
        "versions": versions,
        "artifacts": {
            "rollouts": "rollouts.npz",
            "centered_rollouts": "centered_rollouts.npz",
            "trajectory_comparison": "trajectory_comparison.png",
            "mode_occupancy": "mode_occupancy.png",
            "representative_gifs": {
                label: path.name for label, path in gif_paths.items()
            },
        },
    }
    save_json(output_dir / "resolved_config.json", resolved_config)
    save_json(output_dir / "seed_streams.json", seed_payload)
    save_json(output_dir / "diagnostics.json", diagnostics)
    return {
        **diagnostics,
        "rollout_seeds": rollout_seeds,
        "gaussian_metrics": gaussian,
        "centered_metrics": centered,
        "representative_gifs": {
            label: str(path) for label, path in gif_paths.items()
        },
    }


def run_stage_b1(
    output_dir: str | Path,
    *,
    bank: DemonstrationBank,
    config: StageB1Config = DEFAULT_STAGE_B1_CONFIG,
    seed: int = 0,
    device: str | torch.device = "cpu",
    enforce_acceptance: bool = False,
) -> dict[str, Any]:
    """Train, reload, evaluate, and persist one deterministic B1 experiment."""
    from env.train_stage_b import train_sfpd

    if not isinstance(bank, DemonstrationBank):
        raise ValueError("bank must be a DemonstrationBank")
    _validate_evaluation_config(config)
    if type(enforce_acceptance) is not bool:
        raise ValueError("enforce_acceptance must be a bool")
    resolved_device = _resolved_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training = train_sfpd(
        bank,
        destination,
        config=config,
        root_seed=seed,
        device=resolved_device,
    )
    digest = train_data_digest(bank)
    policy, stats, checkpoint_metadata = load_trained_sfpd(
        training.checkpoint_path,
        training.stats_path,
        expected_train_data_digest=digest,
        device=resolved_device,
    )
    if checkpoint_metadata["stage_b1_config"] != stage_b1_config_to_dict(config):
        raise ValueError("checkpoint Stage B1 configuration does not match")
    rollout_seeds = derive_rollout_seeds(
        training.seed_streams["rollout_initialization"],
        config.rollout_count,
    )
    gaussian_metrics, gaussian_rollouts = evaluate_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        rollout_seeds,
        center_init=False,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    centered_seeds = rollout_seeds[: min(32, len(rollout_seeds))]
    centered_metrics, centered_rollouts = evaluate_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        centered_seeds,
        center_init=True,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    held_out_field_loss = _held_out_field_loss(
        policy,
        stats,
        bank,
        config,
        training.seed_streams["checkpoint_replay"],
    )
    replay = rollout_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        seed=rollout_seeds[0],
        center_init=False,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    deterministic_replay = _rollout_equal(gaussian_rollouts.rollouts[0], replay)
    return write_stage_b1_outputs(
        destination,
        bank,
        config,
        training,
        checkpoint_metadata,
        gaussian_metrics,
        gaussian_rollouts,
        centered_metrics,
        centered_rollouts,
        device=resolved_device,
        held_out_field_loss=held_out_field_loss,
        deterministic_replay=deterministic_replay,
        enforce_acceptance=enforce_acceptance,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the deterministic Stage B1 toy policy."
    )
    parser.add_argument("--stage", choices=("b1",), default="b1")
    parser.add_argument(
        "--demonstrations",
        type=Path,
        default=Path("env/artifacts/demonstrations.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-updates", type=int, default=20_000)
    parser.add_argument("--rollout-count", type=int, default=1024)
    parser.add_argument("--integration-steps-per-action", type=int, default=6)
    parser.add_argument("--enforce-acceptance", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    bank = load_demonstration_bank(args.demonstrations)
    config = replace(
        DEFAULT_STAGE_B1_CONFIG,
        max_updates=args.max_updates,
        rollout_count=args.rollout_count,
        integration_steps_per_action=args.integration_steps_per_action,
    )
    output_dir = args.output_dir or Path(
        f"env/artifacts/stage_b/b1-seed{args.seed}"
    )
    result = run_stage_b1(
        output_dir,
        bank=bank,
        config=config,
        seed=args.seed,
        device=args.device,
        enforce_acceptance=args.enforce_acceptance,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.enforce_acceptance and not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
