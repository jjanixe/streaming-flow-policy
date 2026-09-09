"""Portable statistics and checkpoint serialization for toy Stage B1."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.chunk_data import PushTStats
from env.stage_b_config import StageB1Config, stage_b1_config_to_dict


CHECKPOINT_FORMAT_VERSION = 1
ROLLOUT_FORMAT_VERSION = 1
REQUIRED_METADATA = {
    "format_version",
    "model_type",
    "architecture",
    "stage_b1_config",
    "selected_update",
    "validation_loss",
    "root_seed",
    "seed_streams",
    "train_data_digest",
    "drake_version",
    "numpy_version",
    "torch_version",
    "optimizer",
    "solver",
}

SEED_STREAM_NAMES = (
    "model_initialization",
    "dataloader_shuffle",
    "train_transform",
    "validation_transform",
    "rollout_initialization",
    "checkpoint_replay",
)
UINT32_MAX = np.iinfo(np.uint32).max


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("train data digest must be a nonempty string")
    return value


def save_pusht_stats(
    path: str | Path,
    stats: PushTStats,
    train_data_digest: str,
) -> None:
    """Save train-only PushT normalization statistics without pickle."""
    digest = _require_digest(train_data_digest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        obs_min=stats.obs_min,
        obs_max=stats.obs_max,
        action_min=stats.action_min,
        action_max=stats.action_max,
        train_data_digest=np.asarray(digest, dtype=np.str_),
    )


def load_pusht_stats(path: str | Path) -> tuple[PushTStats, str]:
    """Load portable PushT statistics and their source data digest."""
    required = {
        "obs_min",
        "obs_max",
        "action_min",
        "action_max",
        "train_data_digest",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"statistics artifact is missing keys: {sorted(missing)}")
        digest_array = payload["train_data_digest"]
        if digest_array.shape != () or digest_array.dtype.kind != "U":
            raise ValueError("train data digest must be a scalar Unicode string")
        stats = PushTStats(
            obs_min=payload["obs_min"].copy(),
            obs_max=payload["obs_max"].copy(),
            action_min=payload["action_min"].copy(),
            action_max=payload["action_max"].copy(),
        )
        digest = _require_digest(str(digest_array.item()))
    return stats, digest


def _padded_float32(
    arrays: list[np.ndarray],
    trailing_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray([len(value) for value in arrays], dtype=np.int64)
    maximum = int(lengths.max())
    padded = np.full(
        (len(arrays), maximum, *trailing_shape),
        np.nan,
        dtype=np.float32,
    )
    mask = np.zeros((len(arrays), maximum), dtype=np.bool_)
    for index, value in enumerate(arrays):
        length = len(value)
        padded[index, :length] = value
        mask[index, :length] = True
    return padded, lengths, mask


def save_sfpd_rollouts(
    path: str | Path,
    batch: object,
    *,
    checkpoint_digest: str,
    train_data_digest: str,
) -> None:
    """Save variable-length SFPD rollouts as numeric, pickle-free arrays."""
    from env.evaluate_stage_b import SFPDRolloutBatch

    if not isinstance(batch, SFPDRolloutBatch):
        raise ValueError("batch must be an SFPDRolloutBatch")
    checkpoint_hash = _require_digest(checkpoint_digest)
    train_hash = _require_digest(train_data_digest)
    rollouts = batch.rollouts
    executed, executed_lengths, executed_mask = _padded_float32(
        [rollout.executed_positions for rollout in rollouts],
        (2,),
    )
    requested, requested_lengths, requested_mask = _padded_float32(
        [rollout.requested_actions for rollout in rollouts],
        (2,),
    )
    chunks, chunk_lengths, chunk_mask = _padded_float32(
        [rollout.raw_predicted_chunks for rollout in rollouts],
        (9, 2),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format_version=np.asarray(ROLLOUT_FORMAT_VERSION, dtype=np.int64),
        model_type=np.asarray("sfpd", dtype=np.str_),
        rollout_seeds=np.asarray(
            [rollout.seed for rollout in rollouts],
            dtype=np.uint64,
        ),
        initial_observations=np.stack(
            [rollout.initial_observation for rollout in rollouts]
        ).astype(np.float32, copy=False),
        executed_positions=executed,
        executed_position_lengths=executed_lengths,
        executed_position_mask=executed_mask,
        requested_actions=requested,
        requested_action_lengths=requested_lengths,
        requested_action_mask=requested_mask,
        raw_predicted_chunks=chunks,
        raw_predicted_chunk_lengths=chunk_lengths,
        raw_predicted_chunk_mask=chunk_mask,
        success_mask=np.asarray(
            [rollout.success for rollout in rollouts],
            dtype=np.bool_,
        ),
        numerical_failure_mask=np.asarray(
            [rollout.numerical_failure for rollout in rollouts],
            dtype=np.bool_,
        ),
        action_limit_failure_mask=np.asarray(
            [rollout.action_limit_failure for rollout in rollouts],
            dtype=np.bool_,
        ),
        failure_mask=np.asarray(
            [not rollout.success for rollout in rollouts],
            dtype=np.bool_,
        ),
        checkpoint_digest=np.asarray(checkpoint_hash, dtype=np.str_),
        train_data_digest=np.asarray(train_hash, dtype=np.str_),
    )


def _validate_portable_metadata(value: object, path: str = "metadata") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_portable_metadata(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_portable_metadata(child, f"{path}[{index}]")
        return
    raise ValueError(f"{path} must contain only primitive containers and strings")


def _copy_checkpoint_state(
    state: Mapping[str, torch.Tensor],
    name: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{name} must be a state mapping")
    copied: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must map string keys to tensors")
        detached = value.detach()
        if detached.is_floating_point():
            if detached.dtype != torch.float32:
                raise ValueError(f"{name} floating tensors must use float32")
            if not torch.isfinite(detached).all():
                raise ValueError(f"{name} must contain finite tensors")
        copied[key] = detached.to(device="cpu", copy=True)
    return copied


def save_sfpd_checkpoint(
    path: str | Path,
    *,
    raw_state: Mapping[str, torch.Tensor],
    ema_state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    """Save a weights-only-compatible deterministic SFPD checkpoint."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    metadata_copy = copy.deepcopy(dict(metadata))
    _validate_portable_metadata(metadata_copy)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "raw_state": _copy_checkpoint_state(raw_state, "raw_state"),
            "ema_state": _copy_checkpoint_state(ema_state, "ema_state"),
            "metadata": metadata_copy,
        },
        destination,
    )


def _validate_architecture(architecture: object) -> dict[str, Any]:
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint architecture must be a dictionary")
    if set(architecture) != {
        "name",
        "hidden_dim",
        "hidden_layers",
        "pred_horizon",
    }:
        raise ValueError("checkpoint architecture has invalid schema")
    if architecture.get("name") != "SFPDVelocityMLP":
        raise ValueError("checkpoint architecture is not SFPDVelocityMLP")
    for key in ("hidden_dim", "hidden_layers", "pred_horizon"):
        value = architecture.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"checkpoint architecture has invalid {key}")
    if architecture["pred_horizon"] != 16:
        raise ValueError("checkpoint architecture has incompatible pred_horizon")
    return architecture


def _canonical_policy_state_contract(
    architecture: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[tuple[int, ...], torch.dtype]],
    dict[str, torch.Tensor],
]:
    """Derive the exact wrapper state schema without changing caller RNG state."""
    from env.models import SFPDVelocityMLP
    from env.sfp_policies import StreamingFlowPolicyDeterministic

    with torch.random.fork_rng(devices=[]):
        policy = StreamingFlowPolicyDeterministic(
            SFPDVelocityMLP(
                hidden_dim=architecture["hidden_dim"],
                hidden_layers=architecture["hidden_layers"],
            ),
            pred_horizon=architecture["pred_horizon"],
            device="cpu",
        )
    schema = {
        name: (tuple(value.shape), value.dtype)
        for name, value in policy.state_dict().items()
    }
    immutable_buffers = {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in policy.named_buffers()
    }
    return schema, immutable_buffers


def _validate_checkpoint_state_schema(
    state: Mapping[str, torch.Tensor],
    schema: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    name: str,
) -> None:
    actual_keys = set(state)
    expected_keys = set(schema)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{name} state schema keys do not match "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for key, value in state.items():
        expected_shape, expected_dtype = schema[key]
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} state schema shape for {key!r} must be "
                f"{expected_shape}, got {tuple(value.shape)}"
            )
        if value.dtype != expected_dtype:
            raise ValueError(
                f"{name} state schema dtype for {key!r} must be "
                f"{expected_dtype}, got {value.dtype}"
            )


def _validate_stage_b1_config(value: object) -> StageB1Config:
    if not isinstance(value, dict):
        raise ValueError("checkpoint stage_b1_config must be a dictionary")
    expected_keys = set(stage_b1_config_to_dict(StageB1Config()))
    if set(value) != expected_keys:
        raise ValueError("checkpoint stage_b1_config has invalid schema")
    try:
        return StageB1Config(**value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint stage_b1_config is invalid: {error}") from error


def _validate_seed_metadata(metadata: Mapping[str, Any]) -> None:
    root_seed = metadata["root_seed"]
    if type(root_seed) is not int or not 0 <= root_seed <= UINT32_MAX:
        raise ValueError("checkpoint root_seed must be a uint32-range integer")
    seed_streams = metadata["seed_streams"]
    if not isinstance(seed_streams, dict) or set(seed_streams) != set(
        SEED_STREAM_NAMES
    ):
        raise ValueError("checkpoint seed_streams has invalid names")
    if any(
        type(seed) is not int or not 0 <= seed <= UINT32_MAX
        for seed in seed_streams.values()
    ):
        raise ValueError("checkpoint seed_streams values must be uint32-range ints")
    children = np.random.SeedSequence(root_seed).spawn(len(SEED_STREAM_NAMES))
    expected_seed_streams = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(SEED_STREAM_NAMES, children)
    }
    if seed_streams != expected_seed_streams:
        raise ValueError(
            "checkpoint seed_streams do not match deterministic root_seed streams"
        )


def _validate_optimizer_metadata(
    value: object,
    config: StageB1Config,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "learning_rate",
        "weight_decay",
        "warmup_updates",
        "schedule",
    }:
        raise ValueError("checkpoint optimizer metadata has invalid schema")
    if (
        type(value["name"]) is not str
        or type(value["learning_rate"]) is not float
        or type(value["weight_decay"]) is not float
        or type(value["warmup_updates"]) is not int
        or type(value["schedule"]) is not str
    ):
        raise ValueError("checkpoint optimizer metadata has invalid field types")
    expected = {
        "name": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_updates": config.warmup_updates,
        "schedule": "linear_warmup_cosine",
    }
    if value != expected:
        raise ValueError("checkpoint optimizer metadata is inconsistent with config")


def _validate_solver_metadata(value: object, config: StageB1Config) -> None:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "sensitivity",
        "atol",
        "rtol",
        "integration_steps_per_action",
    }:
        raise ValueError("checkpoint solver metadata has invalid schema")
    if (
        type(value["name"]) is not str
        or type(value["sensitivity"]) is not str
        or type(value["atol"]) is not float
        or type(value["rtol"]) is not float
        or type(value["integration_steps_per_action"]) is not int
    ):
        raise ValueError("checkpoint solver metadata has invalid field types")
    expected = {
        "name": "dopri5",
        "sensitivity": "adjoint",
        "atol": 1e-4,
        "rtol": 1e-4,
        "integration_steps_per_action": config.integration_steps_per_action,
    }
    if value != expected:
        raise ValueError("checkpoint solver metadata is inconsistent with config")


def _validate_immutable_buffers(
    state: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    name: str,
) -> None:
    for key, expected_value in expected.items():
        if not torch.equal(state[key], expected_value):
            raise ValueError(
                f"{name} immutable buffer {key!r} does not match canonical value"
            )


def load_sfpd_checkpoint(
    path: str | Path,
    *,
    expected_train_data_digest: str,
    expected_architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate an SFPD checkpoint before model construction/loading."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if not {"raw_state", "ema_state", "metadata"}.issubset(payload):
        raise ValueError("checkpoint payload is missing state or metadata")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")

    expected_digest = _require_digest(expected_train_data_digest)
    if metadata.get("train_data_digest") != expected_digest:
        raise ValueError("checkpoint train data digest does not match")

    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"checkpoint metadata is missing keys: {sorted(missing)}")
    _validate_portable_metadata(metadata)
    if (
        type(metadata["format_version"]) is not int
        or metadata["format_version"] != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError("checkpoint format version is incompatible")
    if metadata["model_type"] != "sfpd":
        raise ValueError("checkpoint model type is incompatible")
    architecture = _validate_architecture(metadata["architecture"])
    if expected_architecture is not None and architecture != dict(
        expected_architecture
    ):
        raise ValueError(
            "checkpoint architecture does not match the expected architecture"
        )
    config = _validate_stage_b1_config(metadata["stage_b1_config"])
    if architecture != {
        "name": "SFPDVelocityMLP",
        "hidden_dim": config.hidden_dim,
        "hidden_layers": config.hidden_layers,
        "pred_horizon": config.pred_horizon,
    }:
        raise ValueError("checkpoint architecture is inconsistent with config")
    if (
        type(metadata["selected_update"]) is not int
        or not 1 <= metadata["selected_update"] <= config.max_updates
    ):
        raise ValueError("checkpoint selected_update is outside training bounds")
    if type(metadata["validation_loss"]) is not float or not math.isfinite(
        metadata["validation_loss"]
    ) or metadata["validation_loss"] < 0.0:
        raise ValueError("checkpoint validation_loss must be a finite nonnegative float")
    for version_name in ("drake_version", "numpy_version", "torch_version"):
        version = metadata[version_name]
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"checkpoint {version_name} must be a nonempty version")
    _validate_seed_metadata(metadata)
    _validate_optimizer_metadata(metadata["optimizer"], config)
    _validate_solver_metadata(metadata["solver"], config)

    raw_state = _copy_checkpoint_state(payload["raw_state"], "raw_state")
    ema_state = _copy_checkpoint_state(payload["ema_state"], "ema_state")
    if raw_state.keys() != ema_state.keys():
        raise ValueError("raw and EMA checkpoint states must have identical keys")
    canonical_schema, immutable_buffers = _canonical_policy_state_contract(
        architecture
    )
    _validate_checkpoint_state_schema(raw_state, canonical_schema, "raw_state")
    _validate_checkpoint_state_schema(ema_state, canonical_schema, "ema_state")
    _validate_immutable_buffers(raw_state, immutable_buffers, "raw_state")
    _validate_immutable_buffers(ema_state, immutable_buffers, "ema_state")
    return {
        "raw_state": raw_state,
        "ema_state": ema_state,
        "metadata": copy.deepcopy(metadata),
    }
