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


CHECKPOINT_FORMAT_VERSION = 1
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
}


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
    if architecture.get("name") != "SFPDVelocityMLP":
        raise ValueError("checkpoint architecture is not SFPDVelocityMLP")
    for key in ("hidden_dim", "hidden_layers", "pred_horizon"):
        value = architecture.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"checkpoint architecture has invalid {key}")
    if architecture["pred_horizon"] != 16:
        raise ValueError("checkpoint architecture has incompatible pred_horizon")
    return architecture


def _canonical_policy_state_schema(
    architecture: Mapping[str, Any],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
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
    return {
        name: (tuple(value.shape), value.dtype)
        for name, value in policy.state_dict().items()
    }


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
    if metadata["format_version"] != CHECKPOINT_FORMAT_VERSION:
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
    if not isinstance(metadata["stage_b1_config"], dict):
        raise ValueError("checkpoint stage_b1_config must be a dictionary")
    if (
        not isinstance(metadata["selected_update"], int)
        or isinstance(metadata["selected_update"], bool)
        or metadata["selected_update"] <= 0
    ):
        raise ValueError("checkpoint selected_update must be positive")
    if not isinstance(metadata["validation_loss"], float) or not math.isfinite(
        metadata["validation_loss"]
    ):
        raise ValueError("checkpoint validation_loss must be a finite float")
    if not isinstance(metadata["seed_streams"], dict):
        raise ValueError("checkpoint seed_streams must be a dictionary")

    raw_state = _copy_checkpoint_state(payload["raw_state"], "raw_state")
    ema_state = _copy_checkpoint_state(payload["ema_state"], "ema_state")
    if raw_state.keys() != ema_state.keys():
        raise ValueError("raw and EMA checkpoint states must have identical keys")
    canonical_schema = _canonical_policy_state_schema(architecture)
    _validate_checkpoint_state_schema(raw_state, canonical_schema, "raw_state")
    _validate_checkpoint_state_schema(ema_state, canonical_schema, "ema_state")
    return {
        "raw_state": raw_state,
        "ema_state": ema_state,
        "metadata": copy.deepcopy(metadata),
    }
