"""Static plots and real-environment RGB animations for Stage B1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from env.config import EnvironmentConfig
from env.environment import PointReach2DPreferenceEnv
from env.evaluate_stage_b import SFPDRollout, SFPDRolloutBatch


MODE_NAMES = (
    "upper-narrow",
    "upper-wide",
    "lower-narrow",
    "lower-wide",
)
MODE_COLORS = {
    "upper-narrow": "#689de0",
    "upper-wide": "#3465b8",
    "lower-narrow": "#f4a667",
    "lower-wide": "#cc5b43",
    "other": "#777b85",
}


def _plot_modules() -> tuple[Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch

    return plt, Circle, Patch


def _mode_from_position(position: np.ndarray) -> str:
    y = float(position[1])
    if abs(y) < 0.12:
        return "other"
    width = "wide" if abs(y) >= 0.40 else "narrow"
    vertical = "upper" if y > 0.0 else "lower"
    return f"{vertical}-{width}"


def _require_expert_positions(expert_positions: np.ndarray) -> np.ndarray:
    if (
        not isinstance(expert_positions, np.ndarray)
        or expert_positions.ndim != 3
        or expert_positions.shape[0] == 0
        or expert_positions.shape[1] < 33
        or expert_positions.shape[2] != 2
    ):
        raise ValueError("expert_positions must have shape [N, T, 2] with T >= 33")
    if expert_positions.dtype != np.float32:
        raise ValueError("expert_positions must use float32")
    if not np.isfinite(expert_positions).all():
        raise ValueError("expert_positions must contain only finite values")
    return expert_positions


def _configure_world_axes(ax: Any, config: EnvironmentConfig, circle: Any) -> None:
    ax.set_xlim(*config.visualization_x)
    ax.set_ylim(*config.visualization_y)
    ax.axhline(0.0, color="#dadee5", linewidth=0.8, zorder=0)
    ax.scatter(*config.start, marker="o", color="#2d3139", s=42, zorder=5)
    ax.scatter(*config.goal, marker="*", color="#37914f", s=95, zorder=5)
    ax.add_patch(
        circle(
            config.goal,
            config.goal_tolerance,
            facecolor="#cfebd6",
            edgecolor="#37914f",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def plot_trajectory_comparison(
    path: str | Path,
    expert_positions: np.ndarray,
    rollout_batch: SFPDRolloutBatch,
    environment_config: EnvironmentConfig,
) -> None:
    """Overlay held-out expert and SFPD paths using fixed mode colors."""
    experts = _require_expert_positions(expert_positions)
    if not isinstance(rollout_batch, SFPDRolloutBatch):
        raise ValueError("rollout_batch must be an SFPDRolloutBatch")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt, circle, patch = _plot_modules()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    try:
        for expert in experts:
            mode = _mode_from_position(expert[32])
            axes[0].plot(
                expert[:, 0],
                expert[:, 1],
                color=MODE_COLORS[mode],
                linewidth=1.0,
                alpha=0.45,
            )
        axes[0].set_title("Held-out expert trajectories")

        for rollout in rollout_batch.rollouts:
            positions = rollout.executed_positions
            mode_position = positions[32] if len(positions) > 32 else positions[-1]
            mode = _mode_from_position(mode_position)
            axes[1].plot(
                positions[:, 0],
                positions[:, 1],
                color=MODE_COLORS[mode],
                linewidth=1.3,
                alpha=0.72,
            )
            if not rollout.success:
                axes[1].scatter(
                    positions[-1, 0],
                    positions[-1, 1],
                    marker="x",
                    color="#a40000",
                    s=48,
                    linewidth=1.5,
                    zorder=6,
                )
        axes[1].set_title("SFPD Gaussian-initialized rollouts")

        for axis in axes:
            _configure_world_axes(axis, environment_config, circle)
        figure.legend(
            handles=[
                patch(facecolor=MODE_COLORS[name], label=name)
                for name in (*MODE_NAMES, "other")
            ],
            loc="outside lower center",
            ncol=5,
            frameon=False,
        )
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)


def _validated_occupancy(
    values: Mapping[str, float],
    name: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for mode in (*MODE_NAMES, "other"):
        if mode not in values:
            raise ValueError(f"{name} is missing {mode}")
        value = float(values[mode])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must contain finite nonnegative values")
        result[mode] = value
    return result


def plot_mode_occupancy(
    path: str | Path,
    expert_occupancy: Mapping[str, float],
    sfpd_occupancy: Mapping[str, float],
) -> None:
    """Plot expert and SFPD midpoint occupancies together."""
    expert = _validated_occupancy(expert_occupancy, "expert_occupancy")
    sfpd = _validated_occupancy(sfpd_occupancy, "sfpd_occupancy")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = (*MODE_NAMES, "other")
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.36
    plt, _, patch = _plot_modules()
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    try:
        for index, mode in enumerate(labels):
            color = MODE_COLORS[mode]
            axis.bar(
                x[index] - width / 2,
                expert[mode],
                width,
                color=color,
                alpha=0.38,
                hatch="//",
            )
            axis.bar(
                x[index] + width / 2,
                sfpd[mode],
                width,
                color=color,
                alpha=0.95,
            )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel("midpoint occupancy")
        axis.set_ylim(0.0, max(1.0, *(expert.values()), *(sfpd.values())))
        axis.set_title("Held-out expert vs SFPD mode occupancy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(
            handles=[
                patch(facecolor="#777777", alpha=0.38, hatch="//", label="expert"),
                patch(facecolor="#777777", alpha=0.95, label="SFPD"),
            ],
            frameon=False,
        )
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)


def _render_rollout_frames(
    rollout: SFPDRollout,
    environment_config: EnvironmentConfig,
    *,
    center_init: bool,
) -> list[np.ndarray]:
    environment = PointReach2DPreferenceEnv(
        config=environment_config,
        render_mode="rgb_array",
    )
    try:
        observation, _ = environment.reset(
            seed=rollout.seed,
            options={"center_init": center_init},
        )
        if not np.array_equal(observation, rollout.initial_observation):
            raise ValueError("rollout initial observation cannot be replayed")
        initial_frame = environment.render()
        if not isinstance(initial_frame, np.ndarray):
            raise RuntimeError("rgb_array rendering did not return an array")
        frames = [initial_frame]
        for action in rollout.requested_actions:
            _, _, terminated, truncated, _ = environment.step(action)
            frame = environment.render()
            if not isinstance(frame, np.ndarray):
                raise RuntimeError("rgb_array rendering did not return an array")
            frames.append(frame)
            if terminated or truncated:
                break
        return frames
    finally:
        environment.close()


def write_representative_gifs(
    output_dir: str | Path,
    rollout_batch: SFPDRolloutBatch,
    environment_config: EnvironmentConfig,
    *,
    center_init: bool,
) -> dict[str, Path]:
    """Write one success and one failure replay GIF when each is available."""
    if not isinstance(rollout_batch, SFPDRolloutBatch):
        raise ValueError("rollout_batch must be an SFPDRolloutBatch")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    representatives = {
        "success": next(
            (rollout for rollout in rollout_batch.rollouts if rollout.success),
            None,
        ),
        "failure": next(
            (rollout for rollout in rollout_batch.rollouts if not rollout.success),
            None,
        ),
    }
    written: dict[str, Path] = {}
    for label, rollout in representatives.items():
        if rollout is None:
            continue
        frames = _render_rollout_frames(
            rollout,
            environment_config,
            center_init=center_init,
        )
        path = destination / f"representative_{label}.gif"
        imageio.mimsave(
            path,
            frames,
            format="GIF",
            duration=1.0 / PointReach2DPreferenceEnv.metadata["render_fps"],
            loop=0,
        )
        written[label] = path
    return written
