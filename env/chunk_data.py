from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from env.demonstrations import DemonstrationBank
from streaming_flow_policy.pusht.dp_state_notebook.dataset import (
    create_sample_indices,
    normalize_data,
    sample_sequence,
    unnormalize_data,
)


@dataclass(frozen=True)
class PushTStats:
    obs_min: np.ndarray
    obs_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "obs_min": (3,),
            "obs_max": (3,),
            "action_min": (2,),
            "action_max": (2,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
        if np.any(self.obs_max <= self.obs_min):
            raise ValueError("observation ranges must be positive")
        if np.any(self.action_max <= self.action_min):
            raise ValueError("action ranges must be positive")


def build_episode_arrays(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build PushT-style observations and next-position actions for one episode."""
    if not isinstance(positions, np.ndarray):
        raise ValueError("positions must be a NumPy array")
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("positions must have shape (T, 2) for positive T")
    if positions.dtype != np.float32 or not np.isfinite(positions).all():
        raise ValueError("positions must contain finite float32 values")

    normalized_time = np.linspace(0.0, 1.0, len(positions), dtype=np.float32)
    observations = np.concatenate((positions, normalized_time[:, None]), axis=1)
    actions = np.concatenate((positions[1:], positions[[-1]]), axis=0)
    return observations.astype(np.float32), actions.astype(np.float32)


def fit_pusht_stats(bank: DemonstrationBank) -> PushTStats:
    """Fit independent observation and action ranges from train episodes only."""
    train = bank.select("train")
    if len(train) == 0:
        raise ValueError("train split contains no trajectories")
    observations, actions = zip(
        *(build_episode_arrays(positions) for positions in train.positions)
    )
    observation_data = np.concatenate(observations, axis=0)
    action_data = np.concatenate(actions, axis=0)
    return PushTStats(
        obs_min=np.min(observation_data, axis=0).astype(np.float32),
        obs_max=np.max(observation_data, axis=0).astype(np.float32),
        action_min=np.min(action_data, axis=0).astype(np.float32),
        action_max=np.max(action_data, axis=0).astype(np.float32),
    )


def normalize_observations(data: np.ndarray, stats: PushTStats) -> np.ndarray:
    return normalize_data(
        data, {"min": stats.obs_min, "max": stats.obs_max}
    ).astype(np.float32)


def unnormalize_observations(data: np.ndarray, stats: PushTStats) -> np.ndarray:
    return unnormalize_data(
        data, {"min": stats.obs_min, "max": stats.obs_max}
    ).astype(np.float32)


def normalize_actions(data: np.ndarray, stats: PushTStats) -> np.ndarray:
    return normalize_data(
        data, {"min": stats.action_min, "max": stats.action_max}
    ).astype(np.float32)


def unnormalize_actions(data: np.ndarray, stats: PushTStats) -> np.ndarray:
    return unnormalize_data(
        data, {"min": stats.action_min, "max": stats.action_max}
    ).astype(np.float32)


class PushTChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bank: DemonstrationBank,
        split: str,
        stats: PushTStats,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        selected = bank.select(split)
        if len(selected) == 0:
            raise ValueError(f"split {split!r} contains no trajectories")
        if selected.positions.shape[1:] != (65, 2):
            raise ValueError("PushTChunkDataset requires 65-state episodes")

        self.stats = stats
        self.transform = transform
        self.source_split = split
        self.trajectory_ids = selected.trajectory_ids
        self.episodes = []
        for positions in selected.positions:
            obs, action = build_episode_arrays(positions)
            self.episodes.append(
                {
                    "obs": normalize_observations(obs, stats),
                    "action": normalize_actions(action, stats),
                }
            )
        local = create_sample_indices(
            np.asarray([65], dtype=np.int64),
            sequence_length=16,
            pad_before=1,
            pad_after=7,
        )
        self.windows = [
            (trajectory_index, row.copy())
            for trajectory_index in range(len(self.episodes))
            for row in local
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_index, row = self.windows[index]
        buffer_start, buffer_end, sample_start, sample_end = map(int, row)
        sample = sample_sequence(
            self.episodes[trajectory_index],
            sequence_length=16,
            buffer_start_idx=buffer_start,
            buffer_end_idx=buffer_end,
            sample_start_idx=sample_start,
            sample_end_idx=sample_end,
        )
        obs_window = sample["obs"][:2].copy()
        action_window = sample["action"].copy()
        before = action_window[[0]].copy()
        if not np.all(action_window[0] == obs_window[-1, :2]):
            action_window[0] = obs_window[-1, :2]
        correction = np.linalg.norm(
            unnormalize_actions(action_window[[0]], self.stats)
            - unnormalize_actions(before, self.stats)
        )
        physical_anchor = unnormalize_actions(action_window[[0]], self.stats)[0]
        physical_observation = unnormalize_observations(
            obs_window[[-1]], self.stats
        )[0, :2]
        anchor_physical_l2 = np.linalg.norm(
            physical_anchor - physical_observation
        )
        sequence_start = buffer_start - sample_start
        datum = {
            "obs": obs_window,
            "action": action_window,
            "trajectory_id": str(self.trajectory_ids[trajectory_index]),
            "source_split": self.source_split,
            "sequence_start": np.int64(sequence_start),
            "anchor_index": np.int64(sequence_start + 1),
            "pad_before": np.int64(sample_start),
            "pad_after": np.int64(16 - sample_end),
            "anchor_correction_raw_l2": np.float32(correction),
            "anchor_physical_l2": np.float32(anchor_physical_l2),
        }
        return self.transform(datum) if self.transform is not None else datum
