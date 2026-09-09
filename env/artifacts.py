import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.demonstrations import DemonstrationBank
from env.features import FeatureNormalizer


def _update_array_digest(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def _update_string_digest(digest: Any, values: np.ndarray) -> None:
    for value in values.tolist():
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)


def train_data_digest(bank: DemonstrationBank) -> str:
    train = bank.select("train")
    digest = hashlib.sha256()
    _update_string_digest(digest, train.trajectory_ids)
    _update_string_digest(digest, train.splits)
    _update_string_digest(digest, train.modes)
    for values in (
        train.signs,
        train.amplitudes,
        train.replay_seeds,
        train.times,
        train.positions,
        train.derivatives,
    ):
        _update_array_digest(digest, values)
    return digest.hexdigest()


def save_demonstration_bank(
    path: str | Path,
    bank: DemonstrationBank,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        trajectory_ids=bank.trajectory_ids,
        splits=bank.splits,
        modes=bank.modes,
        signs=bank.signs,
        amplitudes=bank.amplitudes,
        replay_seeds=bank.replay_seeds,
        times=bank.times,
        positions=bank.positions,
        derivatives=bank.derivatives,
    )


def load_demonstration_bank(path: str | Path) -> DemonstrationBank:
    with np.load(Path(path), allow_pickle=False) as payload:
        return DemonstrationBank(
            trajectory_ids=payload["trajectory_ids"].copy(),
            splits=payload["splits"].copy(),
            modes=payload["modes"].copy(),
            signs=payload["signs"].copy(),
            amplitudes=payload["amplitudes"].copy(),
            replay_seeds=payload["replay_seeds"].copy(),
            times=payload["times"].copy(),
            positions=payload["positions"].copy(),
            derivatives=payload["derivatives"].copy(),
        )


def save_feature_normalizer(
    path: str | Path,
    normalizer: FeatureNormalizer,
    train_hash: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        mu=normalizer.mu,
        scale=normalizer.scale,
        y_scale=np.asarray(normalizer.y_scale, dtype=np.float32),
        train_data_hash=np.asarray(train_hash, dtype=np.str_),
    )


def load_feature_normalizer(
    path: str | Path,
) -> tuple[FeatureNormalizer, str]:
    with np.load(Path(path), allow_pickle=False) as payload:
        normalizer = FeatureNormalizer(
            mu=payload["mu"].copy(),
            scale=payload["scale"].copy(),
            y_scale=float(payload["y_scale"]),
        )
        train_hash = str(payload["train_data_hash"])
    return normalizer, train_hash


def save_rollouts(
    path: str | Path,
    initial_states: np.ndarray,
    ode_positions: np.ndarray,
    sde_positions: np.ndarray,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        initial_states=np.asarray(initial_states, dtype=np.float32),
        ode_positions=np.asarray(ode_positions, dtype=np.float32),
        sde_positions=np.asarray(sde_positions, dtype=np.float32),
    )


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
