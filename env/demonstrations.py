from dataclasses import dataclass

import numpy as np

from env.config import StageAConfig


@dataclass(frozen=True)
class ModeSpec:
    name: str
    sign: float
    width: str


MODES = (
    ModeSpec("upper-narrow", 1.0, "narrow"),
    ModeSpec("upper-wide", 1.0, "wide"),
    ModeSpec("lower-narrow", -1.0, "narrow"),
    ModeSpec("lower-wide", -1.0, "wide"),
)


@dataclass(frozen=True)
class DemonstrationBank:
    trajectory_ids: np.ndarray
    splits: np.ndarray
    modes: np.ndarray
    signs: np.ndarray
    amplitudes: np.ndarray
    replay_seeds: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    derivatives: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.trajectory_ids)
        for name in ("splits", "modes", "signs", "amplitudes", "replay_seeds"):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} must contain {count} entries")
        expected = (count, len(self.times), 2)
        if self.positions.shape != expected:
            raise ValueError(f"positions must have shape {expected}")
        if self.derivatives.shape != expected:
            raise ValueError(f"derivatives must have shape {expected}")
        for name in ("times", "positions", "derivatives", "signs", "amplitudes"):
            if getattr(self, name).dtype != np.float32:
                raise ValueError(f"{name} must use float32")
        if not np.isfinite(self.positions).all() or not np.isfinite(
            self.derivatives
        ).all():
            raise ValueError("demonstration values must be finite")

    def __len__(self) -> int:
        return len(self.trajectory_ids)

    def select(self, split: str) -> "DemonstrationBank":
        return self.subset(np.flatnonzero(self.splits == split))

    def subset(self, indices: np.ndarray) -> "DemonstrationBank":
        selected = np.asarray(indices, dtype=np.int64)
        return DemonstrationBank(
            trajectory_ids=self.trajectory_ids[selected],
            splits=self.splits[selected],
            modes=self.modes[selected],
            signs=self.signs[selected],
            amplitudes=self.amplitudes[selected],
            replay_seeds=self.replay_seeds[selected],
            times=self.times.copy(),
            positions=self.positions[selected],
            derivatives=self.derivatives[selected],
        )


def quintic_progress(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float32)
    return (10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5).astype(np.float32)


def quintic_progress_derivative(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float32)
    return (60.0 * t**2 * (1.0 - t) ** 2).astype(np.float32)


def bump(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float32)
    return (16.0 * t**2 * (1.0 - t) ** 2).astype(np.float32)


def bump_derivative(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float32)
    return (32.0 * t * (1.0 - t) * (1.0 - 2.0 * t)).astype(np.float32)


def evaluate_path(
    sign: float,
    amplitude: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sign_value = np.float32(sign)
    amplitude_value = np.float32(amplitude)
    progress = quintic_progress(times)
    lateral = bump(times)
    progress_derivative = quintic_progress_derivative(times)
    lateral_derivative = bump_derivative(times)

    positions = np.stack(
        (
            -1.0 + 2.0 * progress,
            sign_value * amplitude_value * lateral,
        ),
        axis=-1,
    )
    derivatives = np.stack(
        (
            progress_derivative,
            sign_value * amplitude_value * lateral_derivative,
        ),
        axis=-1,
    )
    return positions.astype(np.float32), derivatives.astype(np.float32)


def _amplitude_bounds(config: StageAConfig, mode: ModeSpec) -> tuple[float, float]:
    if mode.width == "narrow":
        return config.demonstrations.amplitude_narrow
    return config.demonstrations.amplitude_wide


def generate_demonstration_bank(
    config: StageAConfig,
    seed: int | None = None,
) -> DemonstrationBank:
    generation_seed = config.root_seed if seed is None else seed
    times = np.linspace(
        0.0,
        1.0,
        config.environment.horizon_steps + 1,
        dtype=np.float32,
    )
    split_counts = (
        ("train", config.demonstrations.train_per_mode),
        ("validation", config.demonstrations.validation_per_mode),
        ("test", config.demonstrations.test_per_mode),
    )
    split_sequences = np.random.SeedSequence(generation_seed).spawn(
        len(split_counts)
    )

    trajectory_ids: list[str] = []
    splits: list[str] = []
    mode_names: list[str] = []
    signs: list[np.float32] = []
    amplitudes: list[np.float32] = []
    replay_seeds: list[np.uint32] = []
    positions: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []

    for (split, per_mode), split_sequence in zip(
        split_counts,
        split_sequences,
    ):
        trajectory_sequences = split_sequence.spawn(per_mode * len(MODES))
        sequence_index = 0
        for mode in MODES:
            bounds = _amplitude_bounds(config, mode)
            for mode_index in range(per_mode):
                replay_seed = trajectory_sequences[sequence_index].generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
                sequence_index += 1
                rng = np.random.default_rng(int(replay_seed))
                amplitude = np.float32(rng.uniform(*bounds))
                path, derivative = evaluate_path(
                    mode.sign,
                    float(amplitude),
                    times,
                )
                trajectory_ids.append(f"{split}:{mode.name}:{mode_index:03d}")
                splits.append(split)
                mode_names.append(mode.name)
                signs.append(np.float32(mode.sign))
                amplitudes.append(amplitude)
                replay_seeds.append(replay_seed)
                positions.append(path)
                derivatives.append(derivative)

    return DemonstrationBank(
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.str_),
        splits=np.asarray(splits, dtype=np.str_),
        modes=np.asarray(mode_names, dtype=np.str_),
        signs=np.asarray(signs, dtype=np.float32),
        amplitudes=np.asarray(amplitudes, dtype=np.float32),
        replay_seeds=np.asarray(replay_seeds, dtype=np.uint32),
        times=times,
        positions=np.stack(positions).astype(np.float32),
        derivatives=np.stack(derivatives).astype(np.float32),
    )
