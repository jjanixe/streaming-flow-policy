"""Command-line orchestration for deterministic and stochastic Stage B."""

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
    SFPSRollout,
    SFPSRolloutBatch,
    evaluate_sfpd,
    evaluate_sfps,
    rollout_sfpd,
    rollout_sfps,
)
from env.run_stage_a import classify_midpoint_modes
from env.stage_b_artifacts import (
    load_pusht_stats,
    load_sfpd_checkpoint,
    load_sfps_checkpoint,
    save_sfpd_rollouts,
    save_sfps_rollouts,
)
from env.stage_b_config import (
    DEFAULT_STAGE_B1_CONFIG,
    DEFAULT_STAGE_B2_CONFIG,
    StageB1Config,
    StageB2Config,
    stage_b1_config_to_dict,
    stage_b2_config_to_dict,
)
from env.visualize_stage_b import (
    plot_mode_occupancy,
    plot_stage_b_mode_comparison,
    plot_trajectory_comparison,
    write_representative_gifs,
)

if TYPE_CHECKING:
    from env.sfp_policies import (
        StreamingFlowPolicyDeterministic,
        StreamingFlowPolicyStochastic,
    )
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


def derive_stage_seeds(root_seed: int) -> tuple[int, int]:
    """Derive independent uint32 B1 and B2 root seeds."""
    b1_seed, b2_seed = derive_rollout_seeds(root_seed, 2)
    return b1_seed, b2_seed


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


def _validate_evaluation_config(config: StageB1Config | StageB2Config) -> None:
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


def load_trained_sfps(
    checkpoint_path: str | Path,
    stats_path: str | Path,
    *,
    expected_train_data_digest: str,
    device: str | torch.device = "cpu",
) -> tuple[StreamingFlowPolicyStochastic, PushTStats, dict[str, Any]]:
    """Validate artifacts, reconstruct SFPS, and load selected EMA state."""
    from env.models import SFPSVelocityMLP
    from env.sfp_policies import StreamingFlowPolicyStochastic

    stats, stats_digest = load_pusht_stats(stats_path)
    if stats_digest != expected_train_data_digest:
        raise ValueError("statistics train data digest does not match")
    checkpoint = load_sfps_checkpoint(
        checkpoint_path,
        expected_train_data_digest=expected_train_data_digest,
    )
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]
    config = StageB2Config(**metadata["stage_b2_config"])
    resolved_device = _resolved_device(device)
    with torch.random.fork_rng(devices=[]):
        velocity_net = SFPSVelocityMLP(
            hidden_dim=architecture["hidden_dim"],
            hidden_layers=architecture["hidden_layers"],
        )
    policy = StreamingFlowPolicyStochastic(
        velocity_net,
        pred_horizon=architecture["pred_horizon"],
        sigma0=config.sigma0,
        sigma1=config.sigma1,
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


@torch.inference_mode()
def _held_out_sfps_field_loss(
    policy: StreamingFlowPolicyStochastic,
    stats: PushTStats,
    bank: DemonstrationBank,
    config: StageB2Config,
    seed: int,
) -> float:
    from env.drake_trajectory import SFPSDrakeTransform

    dataset = PushTChunkDataset(
        bank,
        split="test",
        stats=stats,
        transform=SFPSDrakeTransform(
            config.sigma0,
            config.sigma1,
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
            raise FloatingPointError("held-out SFPS field loss is non-finite")
        count = int(batch["obs"].shape[0])
        weighted_loss += float(loss.item()) * count
        sample_count += count
    if sample_count == 0:
        raise ValueError("test split produced no held-out samples")
    result = weighted_loss / sample_count
    if not math.isfinite(result):
        raise FloatingPointError("held-out SFPS field loss is non-finite")
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


def _sfps_rollout_equal(first: SFPSRollout, second: SFPSRollout) -> bool:
    return (
        _rollout_equal(first, second)
        and first.environment_seed == second.environment_seed
        and first.latent_seed == second.latent_seed
        and np.array_equal(first.chunk_latents, second.chunk_latents)
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


def _unique_executed_trajectory_count(
    batch: SFPDRolloutBatch | SFPSRolloutBatch,
) -> int:
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
    batch: SFPDRolloutBatch | SFPSRolloutBatch,
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


def _resolve_b1_diagnostics(
    output_dir: Path,
    expected_train_data_digest: str,
    explicit_path: str | Path | None,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    if explicit_path is not None:
        candidates = [Path(explicit_path)]
    else:
        candidates = sorted(output_dir.parent.glob("b1-seed*/diagnostics.json"))
        conventional = output_dir.parent / "b1" / "diagnostics.json"
        if conventional.is_file() and conventional not in candidates:
            candidates.append(conventional)
    if not candidates:
        return None, None, "no compatible B1 diagnostics were supplied or discovered"
    if len(candidates) > 1 and explicit_path is None:
        return None, None, "multiple sibling B1 diagnostics require an explicit path"
    path = candidates[0]
    if not path.is_file():
        raise ValueError(f"B1 diagnostics do not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        record = json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    if not isinstance(record, dict) or record.get("model_type") != "sfpd":
        raise ValueError("B1 diagnostics have an incompatible model type")
    if record.get("train_data_digest") != expected_train_data_digest:
        raise ValueError("B1 and B2 demonstration digests do not match")
    if record.get("environment_id") != ENVIRONMENT_ID:
        raise ValueError("B1 and B2 environment identifiers do not match")
    architecture = record.get("checkpoint", {}).get("architecture", {})
    if architecture.get("pred_horizon") != 16:
        raise ValueError("B1 diagnostics have an incompatible chunk contract")
    field_loss = record.get("teacher_forced_test_field_loss")
    if not isinstance(field_loss, (int, float)) or not math.isfinite(field_loss):
        raise ValueError("B1 diagnostics are not numerically finite")
    if not isinstance(record.get("gaussian"), dict):
        raise ValueError("B1 diagnostics are missing Gaussian metrics")
    return record, path, None


def validate_b1_precondition(
    record_path: str | Path,
    bank: DemonstrationBank,
) -> tuple[dict[str, Any], Path | None]:
    """Validate that B1 is finite and shares B2's data/chunk contract."""
    path = Path(record_path)
    if not path.is_file():
        raise ValueError(f"B1 pipeline record does not exist: {path}")
    digest = train_data_digest(bank)
    if path.suffix == ".pt":
        checkpoint = load_sfpd_checkpoint(
            path,
            expected_train_data_digest=digest,
        )
        metadata = checkpoint["metadata"]
        return {
            "kind": "checkpoint",
            "path": str(path),
            "train_data_digest": digest,
            "model_type": metadata["model_type"],
            "selected_update": metadata["selected_update"],
            "validation_loss": metadata["validation_loss"],
            "distributional_acceptance_required": False,
        }, None
    if path.suffix != ".json":
        raise ValueError("B1 pipeline record must be diagnostics JSON or checkpoint PT")

    record, resolved_path, _ = _resolve_b1_diagnostics(
        path.parent,
        digest,
        path,
    )
    if record is None or resolved_path is None:
        raise ValueError("B1 diagnostics could not be validated")
    validation_loss = record.get("checkpoint", {}).get("validation_loss")
    if (
        not isinstance(validation_loss, (int, float))
        or isinstance(validation_loss, bool)
        or not math.isfinite(validation_loss)
    ):
        raise ValueError("B1 checkpoint validation loss is not finite")
    if record.get("deterministic_replay") is not True:
        raise ValueError("B1 deterministic seed replay did not pass")
    numerical_failures = record["gaussian"].get("numerical_failure_count")
    if type(numerical_failures) is not int or numerical_failures != 0:
        raise ValueError("B1 Gaussian evaluation has numerical failures")
    resolved_config_path = path.with_name("resolved_config.json")
    if resolved_config_path.is_file():
        with resolved_config_path.open(encoding="utf-8") as stream:
            resolved_config = json.load(stream)
        b1_config = resolved_config.get("stage_b1", {})
        horizons = tuple(
            b1_config.get(name)
            for name in ("pred_horizon", "obs_horizon", "action_horizon")
        )
        if horizons != (16, 2, 8):
            raise ValueError("B1 resolved config has an incompatible chunk contract")
    return {
        "kind": "diagnostics",
        "path": str(path),
        "train_data_digest": digest,
        "model_type": "sfpd",
        "selected_update": record["checkpoint"].get("selected_update"),
        "validation_loss": float(validation_loss),
        "distributional_acceptance_required": False,
        "distributional_accepted": record.get("accepted"),
    }, resolved_path


def write_stage_b2_outputs(
    destination: str | Path,
    bank: DemonstrationBank,
    config: StageB2Config,
    training: TrainResult,
    checkpoint_metadata: dict[str, Any],
    gaussian_metrics: dict[str, Any],
    gaussian_rollouts: SFPSRolloutBatch,
    centered_metrics: dict[str, Any],
    centered_rollouts: SFPSRolloutBatch,
    *,
    device: str | torch.device,
    held_out_field_loss: float,
    seeded_replay: bool,
    enforce_acceptance: bool,
    b1_diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist stochastic SFPS diagnostics, rollouts, and visual artifacts."""
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
    centered["acceptance_applicable"] = True
    gaussian["unique_executed_trajectory_count"] = (
        _unique_executed_trajectory_count(gaussian_rollouts)
    )
    centered["unique_executed_trajectory_count"] = (
        _unique_executed_trajectory_count(centered_rollouts)
    )

    save_sfps_rollouts(
        output_dir / "sfps_rollouts.npz",
        gaussian_rollouts,
        checkpoint_digest=checkpoint_hash,
        train_data_digest=digest,
    )
    save_sfps_rollouts(
        output_dir / "sfps_centered_rollouts.npz",
        centered_rollouts,
        checkpoint_digest=checkpoint_hash,
        train_data_digest=digest,
    )

    test_positions = bank.select("test").positions
    expert_occupancy = classify_midpoint_modes(
        torch.from_numpy(test_positions[:, 32])
    )
    plot_trajectory_comparison(
        output_dir / "sfps_trajectory_comparison.png",
        test_positions,
        gaussian_rollouts,
        DEFAULT_CONFIG.environment,
        model_label="SFPS",
    )
    plot_mode_occupancy(
        output_dir / "sfps_mode_occupancy.png",
        expert_occupancy,
        gaussian["midpoint_occupancy"],
        sfpd_classified_count=gaussian["midpoint_classified_count"],
        sfpd_rollout_count=gaussian["rollout_count"],
        model_label="SFPS",
    )
    gif_paths = write_representative_gifs(
        output_dir,
        gaussian_rollouts,
        DEFAULT_CONFIG.environment,
        center_init=False,
    )

    gaussian_environment_seeds = [
        rollout.environment_seed for rollout in gaussian_rollouts.rollouts
    ]
    gaussian_latent_seeds = [
        rollout.latent_seed for rollout in gaussian_rollouts.rollouts
    ]
    centered_environment_seeds = [
        rollout.environment_seed for rollout in centered_rollouts.rollouts
    ]
    centered_latent_seeds = [
        rollout.latent_seed for rollout in centered_rollouts.rollouts
    ]
    resolved_config = {
        "format_version": RUN_FORMAT_VERSION,
        "stage": "b2",
        "model_type": "sfps",
        "environment_id": ENVIRONMENT_ID,
        "environment": config_to_dict(DEFAULT_CONFIG)["environment"],
        "stage_b2": stage_b2_config_to_dict(config),
        "device": str(torch.device(device)),
        "versions": versions,
    }
    seed_payload = {
        "root_seed": checkpoint_metadata["root_seed"],
        "training": dict(training.seed_streams),
        "gaussian_rollouts": {
            "environment_seeds": gaussian_environment_seeds,
            "latent_seeds": gaussian_latent_seeds,
        },
        "centered_rollouts": {
            "environment_seeds": centered_environment_seeds,
            "latent_seeds": centered_latent_seeds,
        },
    }
    failures = [
        *(f"gaussian: {failure}" for failure in gaussian["acceptance_failures"]),
        *(f"centered: {failure}" for failure in centered["acceptance_failures"]),
    ]
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
    b1_record, b1_path, b1_reason = _resolve_b1_diagnostics(
        output_dir,
        digest,
        b1_diagnostics_path,
    )
    if b1_record is None:
        b1_comparison: dict[str, Any] = {
            "available": False,
            "reason": b1_reason,
        }
    else:
        comparison_payload = {
            "format_version": RUN_FORMAT_VERSION,
            "train_data_digest": digest,
            "environment_id": ENVIRONMENT_ID,
            "b1_diagnostics": str(b1_path),
            "denominators": {
                "expert": "all held-out expert trajectories",
                "mode_occupancy": "rollouts reaching state index 32",
                "failed_before_midpoint": "all rollouts",
            },
            "expert_midpoint_occupancy": expert_occupancy,
            "b1_gaussian": b1_record["gaussian"],
            "b2_gaussian": gaussian,
            "b2_centered": centered,
        }
        save_json(
            output_dir / "stage_b1_b2_comparison.json",
            comparison_payload,
        )
        plot_stage_b_mode_comparison(
            output_dir / "stage_b1_b2_mode_comparison.png",
            expert_occupancy,
            b1_record["gaussian"],
            gaussian,
        )
        b1_comparison = {
            "available": True,
            "source": str(b1_path),
            "artifacts": {
                "metrics": "stage_b1_b2_comparison.json",
                "mode_occupancy": "stage_b1_b2_mode_comparison.png",
            },
        }

    diagnostics = {
        "format_version": RUN_FORMAT_VERSION,
        "version": "toy-v0.1-stage-b2",
        "model_type": "sfps",
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
        "seeded_replay": seeded_replay,
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
        "b1_comparison": b1_comparison,
        "versions": versions,
        "artifacts": {
            "rollouts": "sfps_rollouts.npz",
            "centered_rollouts": "sfps_centered_rollouts.npz",
            "trajectory_comparison": "sfps_trajectory_comparison.png",
            "mode_occupancy": "sfps_mode_occupancy.png",
            "representative_gifs": {
                label: path.name for label, path in gif_paths.items()
            },
        },
    }
    save_json(output_dir / "resolved_config.json", resolved_config)
    save_json(output_dir / "sfps_seed_streams.json", seed_payload)
    save_json(output_dir / "sfps_diagnostics.json", diagnostics)
    return {
        **diagnostics,
        "environment_seeds": gaussian_environment_seeds,
        "latent_seeds": gaussian_latent_seeds,
        "gaussian_metrics": gaussian,
        "centered_metrics": centered,
        "representative_gifs": {
            label: str(path) for label, path in gif_paths.items()
        },
    }


def run_stage_b2(
    output_dir: str | Path,
    *,
    bank: DemonstrationBank,
    config: StageB2Config = DEFAULT_STAGE_B2_CONFIG,
    seed: int = 0,
    device: str | torch.device = "cpu",
    enforce_acceptance: bool = False,
    b1_diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train, reload, evaluate, and persist one stochastic B2 experiment."""
    from env.train_stage_b import train_sfps

    if not isinstance(bank, DemonstrationBank):
        raise ValueError("bank must be a DemonstrationBank")
    _validate_evaluation_config(config)
    if type(enforce_acceptance) is not bool:
        raise ValueError("enforce_acceptance must be a bool")
    resolved_device = _resolved_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training = train_sfps(
        bank,
        destination,
        config=config,
        root_seed=seed,
        device=resolved_device,
    )
    digest = train_data_digest(bank)
    policy, stats, checkpoint_metadata = load_trained_sfps(
        training.checkpoint_path,
        training.stats_path,
        expected_train_data_digest=digest,
        device=resolved_device,
    )
    if checkpoint_metadata["stage_b2_config"] != stage_b2_config_to_dict(config):
        raise ValueError("checkpoint Stage B2 configuration does not match")

    environment_seeds = derive_rollout_seeds(
        training.seed_streams["rollout_initialization"],
        config.rollout_count,
    )
    latent_seeds = derive_rollout_seeds(
        training.seed_streams["sfps_latent_rollout"],
        config.rollout_count,
    )
    gaussian_metrics, gaussian_rollouts = evaluate_sfps(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        environment_seeds,
        latent_seeds,
        center_init=False,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    centered_count = min(32, config.rollout_count)
    centered_environment_seeds = [environment_seeds[0]] * centered_count
    centered_latent_seeds = latent_seeds[:centered_count]
    centered_metrics, centered_rollouts = evaluate_sfps(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        centered_environment_seeds,
        centered_latent_seeds,
        center_init=True,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    held_out_field_loss = _held_out_sfps_field_loss(
        policy,
        stats,
        bank,
        config,
        training.seed_streams["checkpoint_replay"],
    )
    replay = rollout_sfps(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        environment_seed=environment_seeds[0],
        latent_seed=latent_seeds[0],
        center_init=False,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    seeded_replay = _sfps_rollout_equal(gaussian_rollouts.rollouts[0], replay)
    return write_stage_b2_outputs(
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
        seeded_replay=seeded_replay,
        enforce_acceptance=enforce_acceptance,
        b1_diagnostics_path=b1_diagnostics_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate deterministic B1 and stochastic B2."
    )
    parser.add_argument("--stage", choices=("b1", "b2", "all"), default="b1")
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
    parser.add_argument(
        "--b1-record",
        type=Path,
        default=None,
        help=(
            "compatible B1 diagnostics JSON or checkpoint PT; required for "
            "standalone b2 unless the default b1-seed path exists"
        ),
    )
    return parser


def _cli_config_overrides(args: argparse.Namespace) -> dict[str, int]:
    if args.max_updates <= 0:
        raise ValueError("max_updates must be positive")
    return {
        "max_updates": args.max_updates,
        "validation_interval": min(1_000, args.max_updates),
        "warmup_updates": min(500, args.max_updates),
        "rollout_count": args.rollout_count,
        "integration_steps_per_action": args.integration_steps_per_action,
    }


def _compact_model_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "model_type",
            "train_data_digest",
            "checkpoint_digest",
            "checkpoint",
            "teacher_forced_test_field_loss",
            "accepted",
            "failures",
            "gaussian_metrics",
            "centered_metrics",
        )
        if key in result
    }


def main() -> None:
    args = _build_parser().parse_args()
    bank = load_demonstration_bank(args.demonstrations)
    overrides = _cli_config_overrides(args)
    b1_config = replace(DEFAULT_STAGE_B1_CONFIG, **overrides)
    b2_config = replace(DEFAULT_STAGE_B2_CONFIG, **overrides)

    if args.stage == "b1":
        output_dir = args.output_dir or Path(
            f"env/artifacts/stage_b/b1-seed{args.seed}"
        )
        result = run_stage_b1(
            output_dir,
            bank=bank,
            config=b1_config,
            seed=args.seed,
            device=args.device,
            enforce_acceptance=args.enforce_acceptance,
        )
    elif args.stage == "b2":
        output_dir = args.output_dir or Path(
            f"env/artifacts/stage_b/b2-seed{args.seed}"
        )
        b1_record = args.b1_record or Path(
            f"env/artifacts/stage_b/b1-seed{args.seed}/diagnostics.json"
        )
        precondition, comparison_path = validate_b1_precondition(b1_record, bank)
        result = run_stage_b2(
            output_dir,
            bank=bank,
            config=b2_config,
            seed=args.seed,
            device=args.device,
            enforce_acceptance=args.enforce_acceptance,
            b1_diagnostics_path=comparison_path,
        )
        result["b1_precondition"] = precondition
    else:
        output_dir = args.output_dir or Path(
            f"env/artifacts/stage_b/all-seed{args.seed}"
        )
        b1_seed, b2_seed = derive_stage_seeds(args.seed)
        b1_output = output_dir / "b1"
        b2_output = output_dir / "b2"
        b1_result = run_stage_b1(
            b1_output,
            bank=bank,
            config=b1_config,
            seed=b1_seed,
            device=args.device,
            enforce_acceptance=args.enforce_acceptance,
        )
        b1_record = b1_output / "diagnostics.json"
        precondition, comparison_path = validate_b1_precondition(b1_record, bank)
        b2_result = run_stage_b2(
            b2_output,
            bank=bank,
            config=b2_config,
            seed=b2_seed,
            device=args.device,
            enforce_acceptance=args.enforce_acceptance,
            b1_diagnostics_path=comparison_path,
        )
        result = {
            "format_version": RUN_FORMAT_VERSION,
            "stage": "all",
            "root_seed": args.seed,
            "b1_seed": b1_seed,
            "b2_seed": b2_seed,
            "train_data_digest": train_data_digest(bank),
            "b1_precondition": precondition,
            "b1": _compact_model_summary(b1_result),
            "b2": _compact_model_summary(b2_result),
            "accepted": b1_result["accepted"] and b2_result["accepted"],
        }
        save_json(output_dir / "stage_b_summary.json", result)

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.enforce_acceptance and not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
