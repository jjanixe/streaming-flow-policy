"""Static plots and real-environment RGB animations for Stage B1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from env.config import EnvironmentConfig
from env.environment import PointReach2DPreferenceEnv
from env.evaluate_stage_b import (
    SFPDRollout,
    SFPDRolloutBatch,
    SFPSRolloutBatch,
)


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
    "failed-before-midpoint": "#8f949e",
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


def _rollout_display_mode(rollout: SFPDRollout) -> str:
    if rollout.info["step_index"] < 32:
        return "failed-before-midpoint"
    return _mode_from_position(rollout.executed_positions[32])


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
    rollout_batch: SFPDRolloutBatch | SFPSRolloutBatch,
    environment_config: EnvironmentConfig,
    *,
    model_label: str = "SFPD",
) -> None:
    """Overlay held-out expert and SFPD paths using fixed mode colors."""
    experts = _require_expert_positions(expert_positions)
    if not isinstance(rollout_batch, (SFPDRolloutBatch, SFPSRolloutBatch)):
        raise ValueError("rollout_batch must be a streaming rollout batch")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be a nonempty string")
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
            mode = _rollout_display_mode(rollout)
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
        axes[1].set_title(f"{model_label} Gaussian-initialized rollouts")

        for axis in axes:
            _configure_world_axes(axis, environment_config, circle)
        figure.legend(
            handles=[
                patch(facecolor=MODE_COLORS[name], label=name)
                for name in (*MODE_NAMES, "other", "failed-before-midpoint")
            ],
            loc="outside lower center",
            ncol=6,
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


def _occupancy_plot_data(
    expert_occupancy: Mapping[str, float],
    sfpd_occupancy: Mapping[str, float],
    *,
    sfpd_classified_count: int,
    sfpd_rollout_count: int,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...], str]:
    """Build plot values while making the two occupancy denominators explicit."""
    expert = _validated_occupancy(expert_occupancy, "expert_occupancy")
    sfpd = _validated_occupancy(sfpd_occupancy, "sfpd_occupancy")
    if type(sfpd_classified_count) is not int or sfpd_classified_count < 0:
        raise ValueError("sfpd_classified_count must be a nonnegative integer")
    if type(sfpd_rollout_count) is not int or sfpd_rollout_count <= 0:
        raise ValueError("sfpd_rollout_count must be a positive integer")
    if sfpd_classified_count > sfpd_rollout_count:
        raise ValueError("sfpd_classified_count cannot exceed sfpd_rollout_count")
    labels = (*MODE_NAMES, "other", "failed-before-midpoint")
    failed_share = (
        sfpd_rollout_count - sfpd_classified_count
    ) / sfpd_rollout_count
    expert_values = tuple(expert[mode] for mode in (*MODE_NAMES, "other")) + (0.0,)
    sfpd_values = tuple(sfpd[mode] for mode in (*MODE_NAMES, "other")) + (
        failed_share,
    )
    title = (
        "Mode occupancy among rollouts reaching state 32 "
        f"({sfpd_classified_count}/{sfpd_rollout_count}); "
        "failed-before-midpoint uses all rollouts"
    )
    return labels, expert_values, sfpd_values, title


def plot_mode_occupancy(
    path: str | Path,
    expert_occupancy: Mapping[str, float],
    sfpd_occupancy: Mapping[str, float],
    *,
    sfpd_classified_count: int,
    sfpd_rollout_count: int,
    model_label: str = "SFPD",
) -> None:
    """Plot conditional midpoint occupancy and pre-midpoint failure share."""
    labels, expert_values, sfpd_values, title = _occupancy_plot_data(
        expert_occupancy,
        sfpd_occupancy,
        sfpd_classified_count=sfpd_classified_count,
        sfpd_rollout_count=sfpd_rollout_count,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.36
    plt, _, patch = _plot_modules()
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    try:
        for index, mode in enumerate(labels):
            color = MODE_COLORS[mode]
            axis.bar(
                x[index] - width / 2,
                expert_values[index],
                width,
                color=color,
                alpha=0.38,
                hatch="//",
            )
            axis.bar(
                x[index] + width / 2,
                sfpd_values[index],
                width,
                color=color,
                alpha=0.95,
            )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel("conditional occupancy (failure share uses all rollouts)")
        axis.set_ylim(0.0, max(1.0, *expert_values, *sfpd_values))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(
            handles=[
                patch(facecolor="#777777", alpha=0.38, hatch="//", label="expert"),
                patch(facecolor="#777777", alpha=0.95, label=model_label),
            ],
            frameon=False,
        )
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)


def plot_stage_b_mode_comparison(
    path: str | Path,
    expert_occupancy: Mapping[str, float],
    b1_metrics: Mapping[str, Any],
    b2_metrics: Mapping[str, Any],
) -> None:
    """Compare B1 and B2 with identical midpoint/failure denominators."""
    labels, expert_values, b1_values, _ = _occupancy_plot_data(
        expert_occupancy,
        b1_metrics["midpoint_occupancy"],
        sfpd_classified_count=int(b1_metrics["midpoint_classified_count"]),
        sfpd_rollout_count=int(b1_metrics["rollout_count"]),
    )
    b2_labels, _, b2_values, _ = _occupancy_plot_data(
        expert_occupancy,
        b2_metrics["midpoint_occupancy"],
        sfpd_classified_count=int(b2_metrics["midpoint_classified_count"]),
        sfpd_rollout_count=int(b2_metrics["rollout_count"]),
    )
    if labels != b2_labels:
        raise ValueError("B1 and B2 comparison labels must match")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.25
    plt, _, patch = _plot_modules()
    figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    try:
        for index, mode in enumerate(labels):
            color = MODE_COLORS[mode]
            axis.bar(
                x[index] - width,
                expert_values[index],
                width,
                color=color,
                alpha=0.28,
                hatch="//",
            )
            axis.bar(
                x[index],
                b1_values[index],
                width,
                color=color,
                alpha=0.65,
            )
            axis.bar(
                x[index] + width,
                b2_values[index],
                width,
                color=color,
                alpha=1.0,
            )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel("conditional occupancy (failure share uses all rollouts)")
        axis.set_ylim(0.0, max(1.0, *expert_values, *b1_values, *b2_values))
        axis.set_title(
            "B1/B2 midpoint occupancy — "
            f"B1: {int(b1_metrics['midpoint_classified_count'])}/"
            f"{int(b1_metrics['rollout_count'])}, "
            f"B2: {int(b2_metrics['midpoint_classified_count'])}/"
            f"{int(b2_metrics['rollout_count'])}"
        )
        axis.grid(axis="y", alpha=0.2)
        axis.legend(
            handles=[
                patch(facecolor="#777777", alpha=0.28, hatch="//", label="expert"),
                patch(facecolor="#777777", alpha=0.65, label="SFPD (B1)"),
                patch(facecolor="#777777", alpha=1.0, label="SFPS (B2)"),
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
    rollout_batch: SFPDRolloutBatch | SFPSRolloutBatch,
    environment_config: EnvironmentConfig,
    *,
    center_init: bool,
) -> dict[str, Path]:
    """Write one success and one failure replay GIF when each is available."""
    if not isinstance(rollout_batch, (SFPDRolloutBatch, SFPSRolloutBatch)):
        raise ValueError("rollout_batch must be a streaming rollout batch")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for label in ("success", "failure"):
        (destination / f"representative_{label}.gif").unlink(missing_ok=True)
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
    import mediapy as media

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
        media.write_video(
            path,
            frames,
            fps=PointReach2DPreferenceEnv.metadata["render_fps"],
            codec="gif",
        )
        written[label] = path
    return written
