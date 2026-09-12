"""Seeded finite-update training for toy deterministic and stochastic policies."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import importlib.metadata
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from env.artifacts import save_json, train_data_digest
from env.chunk_data import (
    MaterializedPushTTrainingDataset,
    PushTChunkDataset,
    fit_pusht_stats,
)
from env.demonstrations import DemonstrationBank
from env.drake_trajectory import (
    SFPDDrakeTransform,
    SFPSDrakeTransform,
    sample_sfpd_training_targets,
    sample_sfps_training_targets,
)
from env.models import SFPDVelocityMLP, SFPSVelocityMLP
from env.sfp_policies import (
    StreamingFlowPolicyDeterministic,
    StreamingFlowPolicyStochastic,
)
from env.stage_b_artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    save_pusht_stats,
    save_sfpd_checkpoint,
    save_sfps_checkpoint,
)
from env.stage_b_config import (
    DEFAULT_STAGE_B1_CONFIG,
    DEFAULT_STAGE_B2_CONFIG,
    StageB1Config,
    StageB2Config,
    stage_b1_config_to_dict,
    stage_b2_config_to_dict,
)


SEED_STREAM_NAMES = (
    "model_initialization",
    "dataloader_shuffle",
    "train_transform",
    "validation_transform",
    "rollout_initialization",
    "checkpoint_replay",
)
SFPS_SEED_STREAM_NAMES = (*SEED_STREAM_NAMES, "sfps_latent_rollout")


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    stats_path: Path
    history_path: Path
    selected_update: int
    validation_loss: float
    seed_streams: dict[str, int]


class ExponentialMovingAverage:
    """Maintain an explicit EMA copy of all module parameters and buffers."""

    def __init__(self, module: nn.Module, decay: float) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }
        _require_finite_float32_state(self.shadow, "EMA shadow")

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        state = module.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA module state keys changed")
        _require_finite_float32_state(state, "EMA source")
        for name, value in state.items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())
        _require_finite_float32_state(self.shadow, "EMA shadow")


def _require_finite_float32_state(
    state: Mapping[str, torch.Tensor],
    name: str,
) -> None:
    for value in state.values():
        if value.is_floating_point():
            if value.dtype != torch.float32:
                raise ValueError(f"{name} floating tensors must use float32")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{name} contains non-finite values")


def _derive_seed_streams(root_seed: int) -> dict[str, int]:
    if type(root_seed) is not int or not 0 <= root_seed <= np.iinfo(np.uint32).max:
        raise ValueError("root_seed must be a uint32-range integer")
    children = np.random.SeedSequence(root_seed).spawn(len(SEED_STREAM_NAMES))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(SEED_STREAM_NAMES, children)
    }


def _derive_sfps_seed_streams(root_seed: int) -> dict[str, int]:
    if type(root_seed) is not int or not 0 <= root_seed <= np.iinfo(np.uint32).max:
        raise ValueError("root_seed must be a uint32-range integer")
    children = np.random.SeedSequence(root_seed).spawn(len(SFPS_SEED_STREAM_NAMES))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(SFPS_SEED_STREAM_NAMES, children)
    }


def _validate_training_config(config: StageB1Config) -> None:
    if config.validation_interval <= 0:
        raise ValueError("validation_interval must be positive")
    if config.warmup_updates < 0:
        raise ValueError("warmup_updates must be nonnegative")
    for name in ("learning_rate", "weight_decay", "ema_decay"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.learning_rate == 0.0:
        raise ValueError("learning_rate must be positive")
    if config.ema_decay >= 1.0:
        raise ValueError("ema_decay must be below one")


def _learning_rate_factor(
    scheduler_step: int,
    *,
    warmup_updates: int,
    max_updates: int,
) -> float:
    if warmup_updates > 0 and scheduler_step < warmup_updates:
        return float(scheduler_step + 1) / float(warmup_updates)
    cosine_updates = max_updates - warmup_updates
    if cosine_updates <= 0:
        return 1.0
    progress = min(
        1.0,
        float(scheduler_step - warmup_updates + 1) / float(cosine_updates),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _training_loader_options(device: torch.device) -> dict[str, int | bool]:
    """Use the reference PushT prefetch pipeline for CUDA training only."""
    use_worker = device.type == "cuda"
    return {
        "num_workers": 1 if use_worker else 0,
        "pin_memory": use_worker,
        "persistent_workers": use_worker,
    }


def _materialize_validation_batches(
    dataset: PushTChunkDataset,
    batch_size: int,
    generator: torch.Generator,
    tensor_names: tuple[str, ...] = ("obs", "x", "v", "t"),
) -> tuple[dict[str, torch.Tensor], ...]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )
    materialized: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        fixed_batch: dict[str, torch.Tensor] = {}
        for name in tensor_names:
            value = batch[name]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise ValueError(f"validation {name} must be a float32 tensor")
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"validation {name} contains non-finite values"
                )
            fixed_batch[name] = value.detach().clone()
        materialized.append(fixed_batch)
    if not materialized:
        raise ValueError("validation split produced no batches")
    return tuple(materialized)


@torch.inference_mode()
def _validation_loss(
    policy: StreamingFlowPolicyDeterministic,
    batches: tuple[dict[str, torch.Tensor], ...],
) -> float:
    policy.eval()
    weighted_loss = 0.0
    sample_count = 0
    for batch in batches:
        loss = policy.loss(batch)
        if loss.dtype != torch.float32 or not torch.isfinite(loss):
            raise FloatingPointError("validation loss is non-finite")
        count = int(batch["obs"].shape[0])
        weighted_loss += float(loss.item()) * count
        sample_count += count
    result = weighted_loss / sample_count
    if not math.isfinite(result):
        raise FloatingPointError("validation loss is non-finite")
    return result


def _require_finite_gradients(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError(f"gradient for {name} is non-finite")


def _cpu_state_copy(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in state.items()
    }


def _architecture_metadata(config: StageB1Config) -> dict[str, Any]:
    return {
        "name": "SFPDVelocityMLP",
        "hidden_dim": config.hidden_dim,
        "hidden_layers": config.hidden_layers,
        "pred_horizon": config.pred_horizon,
    }


def _sfps_architecture_metadata(config: StageB2Config) -> dict[str, Any]:
    return {
        "name": "SFPSVelocityMLP",
        "hidden_dim": config.hidden_dim,
        "hidden_layers": config.hidden_layers,
        "pred_horizon": config.pred_horizon,
        "latent_dim": config.latent_dim,
    }


def train_sfpd(
    bank: DemonstrationBank,
    output_dir: str | Path,
    *,
    config: StageB1Config = DEFAULT_STAGE_B1_CONFIG,
    root_seed: int = 0,
    device: str | torch.device = "cpu",
) -> TrainResult:
    """Train SFPD for exactly ``config.max_updates`` finite optimizer steps."""
    _validate_training_config(config)
    seed_streams = _derive_seed_streams(root_seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stats_path = destination / "pusht_stats.npz"
    checkpoint_path = destination / "sfpd_best.pt"
    history_path = destination / "training_history.json"

    digest = train_data_digest(bank)
    stats = fit_pusht_stats(bank)
    save_pusht_stats(stats_path, stats, digest)

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    train_dataset = MaterializedPushTTrainingDataset(
        PushTChunkDataset(
            bank,
            split="train",
            stats=stats,
            transform=None,
        )
    )
    validation_dataset = PushTChunkDataset(
        bank,
        split="validation",
        stats=stats,
        transform=SFPDDrakeTransform(
            config.sigma,
            np.random.default_rng(seed_streams["validation_transform"]),
        ),
    )
    validation_batches = _materialize_validation_batches(
        validation_dataset,
        config.batch_size,
        torch.Generator().manual_seed(seed_streams["validation_transform"]),
    )
    shuffle_generator = torch.Generator().manual_seed(
        seed_streams["dataloader_shuffle"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        **_training_loader_options(resolved_device),
    )
    target_generator = torch.Generator(device=resolved_device).manual_seed(
        seed_streams["train_transform"]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_streams["model_initialization"])
        velocity_net = SFPDVelocityMLP(
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        )
    policy = StreamingFlowPolicyDeterministic(
        velocity_net,
        pred_horizon=config.pred_horizon,
        device=resolved_device,
    ).to(resolved_device)
    validation_policy = copy.deepcopy(policy)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_factor(
            step,
            warmup_updates=config.warmup_updates,
            max_updates=config.max_updates,
        ),
    )
    ema = ExponentialMovingAverage(policy, config.ema_decay)

    training_history: list[dict[str, float | int]] = []
    validation_history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_update = 0
    best_ema_state: dict[str, torch.Tensor] | None = None
    train_iterator = iter(train_loader)

    for update in range(1, config.max_updates + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        normalized_actions = batch["action"].to(
            device=resolved_device,
            non_blocking=resolved_device.type == "cuda",
        )
        targets = sample_sfpd_training_targets(
            normalized_actions,
            sigma=config.sigma,
            generator=target_generator,
        )
        batch = {"obs": batch["obs"], **targets}

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = policy.loss(batch)
        if loss.dtype != torch.float32 or not torch.isfinite(loss):
            raise FloatingPointError("training loss is non-finite")
        loss.backward()
        _require_finite_gradients(policy)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _require_finite_float32_state(policy.state_dict(), "trained model")
        ema.update(policy)
        scheduler.step()

        training_history.append(
            {
                "update": update,
                "loss": float(loss.detach().item()),
                "learning_rate": learning_rate,
            }
        )

        if update % config.validation_interval == 0 or update == config.max_updates:
            validation_policy.load_state_dict(ema.shadow, strict=True)
            current_validation_loss = _validation_loss(
                validation_policy,
                validation_batches,
            )
            validation_history.append(
                {"update": update, "loss": current_validation_loss}
            )
            if current_validation_loss < best_loss:
                best_loss = current_validation_loss
                best_update = update
                best_ema_state = _cpu_state_copy(ema.shadow)

    if best_ema_state is None or best_update <= 0 or not math.isfinite(best_loss):
        raise RuntimeError("training completed without a finite validation checkpoint")

    history = {
        "train": training_history,
        "validation": validation_history,
    }
    save_json(history_path, history)
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": "sfpd",
        "architecture": _architecture_metadata(config),
        "stage_b1_config": stage_b1_config_to_dict(config),
        "selected_update": best_update,
        "validation_loss": float(best_loss),
        "root_seed": root_seed,
        "seed_streams": seed_streams,
        "train_data_digest": digest,
        "drake_version": importlib.metadata.version("drake"),
        "numpy_version": importlib.metadata.version("numpy"),
        "torch_version": str(torch.__version__),
        "trajectory_targets": {
            "training": "torch_uniform_first_order_hold_float32",
            "validation": (
                "drake_first_order_hold_float64_boundary_float32_output"
            ),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_updates": config.warmup_updates,
            "schedule": "linear_warmup_cosine",
        },
        "solver": {
            "name": "dopri5",
            "sensitivity": "adjoint",
            "atol": 1e-4,
            "rtol": 1e-4,
            "integration_steps_per_action": config.integration_steps_per_action,
        },
    }
    save_sfpd_checkpoint(
        checkpoint_path,
        raw_state=policy.state_dict(),
        ema_state=best_ema_state,
        metadata=metadata,
    )
    return TrainResult(
        checkpoint_path=checkpoint_path,
        stats_path=stats_path,
        history_path=history_path,
        selected_update=best_update,
        validation_loss=float(best_loss),
        seed_streams=dict(seed_streams),
    )


def train_sfps(
    bank: DemonstrationBank,
    output_dir: str | Path,
    *,
    config: StageB2Config = DEFAULT_STAGE_B2_CONFIG,
    root_seed: int = 0,
    device: str | torch.device = "cpu",
) -> TrainResult:
    """Train SFPS for exactly ``config.max_updates`` finite optimizer steps."""
    _validate_training_config(config)
    seed_streams = _derive_sfps_seed_streams(root_seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stats_path = destination / "pusht_stats.npz"
    checkpoint_path = destination / "sfps_best.pt"
    history_path = destination / "sfps_training_history.json"

    digest = train_data_digest(bank)
    stats = fit_pusht_stats(bank)
    save_pusht_stats(stats_path, stats, digest)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    train_dataset = MaterializedPushTTrainingDataset(
        PushTChunkDataset(
            bank,
            split="train",
            stats=stats,
            transform=None,
        )
    )
    validation_dataset = PushTChunkDataset(
        bank,
        split="validation",
        stats=stats,
        transform=SFPSDrakeTransform(
            config.sigma0,
            config.sigma1,
            np.random.default_rng(seed_streams["validation_transform"]),
        ),
    )
    validation_batches = _materialize_validation_batches(
        validation_dataset,
        config.batch_size,
        torch.Generator().manual_seed(seed_streams["validation_transform"]),
        ("obs", "a", "z", "va", "vz", "t"),
    )
    shuffle_generator = torch.Generator().manual_seed(
        seed_streams["dataloader_shuffle"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        **_training_loader_options(resolved_device),
    )
    target_generator = torch.Generator(device=resolved_device).manual_seed(
        seed_streams["train_transform"]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_streams["model_initialization"])
        velocity_net = SFPSVelocityMLP(
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        )
    policy = StreamingFlowPolicyStochastic(
        velocity_net,
        pred_horizon=config.pred_horizon,
        sigma0=config.sigma0,
        sigma1=config.sigma1,
        device=resolved_device,
    ).to(resolved_device)
    validation_policy = copy.deepcopy(policy)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_factor(
            step,
            warmup_updates=config.warmup_updates,
            max_updates=config.max_updates,
        ),
    )
    ema = ExponentialMovingAverage(policy, config.ema_decay)

    training_history: list[dict[str, float | int]] = []
    validation_history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_update = 0
    best_ema_state: dict[str, torch.Tensor] | None = None
    train_iterator = iter(train_loader)
    for update in range(1, config.max_updates + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        normalized_actions = batch["action"].to(
            device=resolved_device,
            non_blocking=resolved_device.type == "cuda",
        )
        targets = sample_sfps_training_targets(
            normalized_actions,
            sigma0=config.sigma0,
            sigma1=config.sigma1,
            generator=target_generator,
        )
        batch = {"obs": batch["obs"], **targets}
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = policy.loss(batch)
        if loss.dtype != torch.float32 or not torch.isfinite(loss):
            raise FloatingPointError("training loss is non-finite")
        loss.backward()
        _require_finite_gradients(policy)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _require_finite_float32_state(policy.state_dict(), "trained model")
        ema.update(policy)
        scheduler.step()
        training_history.append(
            {
                "update": update,
                "loss": float(loss.detach().item()),
                "learning_rate": learning_rate,
            }
        )
        if update % config.validation_interval == 0 or update == config.max_updates:
            validation_policy.load_state_dict(ema.shadow, strict=True)
            current_validation_loss = _validation_loss(
                validation_policy,
                validation_batches,
            )
            validation_history.append(
                {"update": update, "loss": current_validation_loss}
            )
            if current_validation_loss < best_loss:
                best_loss = current_validation_loss
                best_update = update
                best_ema_state = _cpu_state_copy(ema.shadow)

    if best_ema_state is None or best_update <= 0 or not math.isfinite(best_loss):
        raise RuntimeError("training completed without a finite validation checkpoint")
    save_json(
        history_path,
        {"train": training_history, "validation": validation_history},
    )
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": "sfps",
        "architecture": _sfps_architecture_metadata(config),
        "stage_b2_config": stage_b2_config_to_dict(config),
        "selected_update": best_update,
        "validation_loss": float(best_loss),
        "root_seed": root_seed,
        "seed_streams": seed_streams,
        "train_data_digest": digest,
        "drake_version": importlib.metadata.version("drake"),
        "numpy_version": importlib.metadata.version("numpy"),
        "torch_version": str(torch.__version__),
        "trajectory_targets": {
            "training": "torch_uniform_first_order_hold_float32",
            "validation": (
                "drake_first_order_hold_float64_boundary_float32_output"
            ),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_updates": config.warmup_updates,
            "schedule": "linear_warmup_cosine",
        },
        "solver": {
            "name": "dopri5",
            "sensitivity": "adjoint",
            "atol": 1e-4,
            "rtol": 1e-4,
            "integration_steps_per_action": config.integration_steps_per_action,
        },
    }
    save_sfps_checkpoint(
        checkpoint_path,
        raw_state=policy.state_dict(),
        ema_state=best_ema_state,
        metadata=metadata,
    )
    return TrainResult(
        checkpoint_path=checkpoint_path,
        stats_path=stats_path,
        history_path=history_path,
        selected_update=best_update,
        validation_loss=float(best_loss),
        seed_streams=dict(seed_streams),
    )
