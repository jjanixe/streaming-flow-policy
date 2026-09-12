"""Render saved inference results; does not fit a model or rerun inference."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from env.demonstrations import evaluate_path

OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT.parent
USERS = ("upper_narrow", "upper_wide", "lower_narrow", "lower_wide")
METHODS = (("learned_20", "Learned: 20 comparisons", "#2563eb"),
           ("oracle_best", "Oracle: best of 32", "#087f5b"))


def name(mode):
    return mode.replace("_", " ").capitalize()


def modes_for(data):
    # Same float32 threshold operations as env.run_preference._path_metrics.
    y = data["positions"][:, 32, 1]
    modes = np.full(len(y), "other", dtype="U24")
    modes[(y >= .12) & (y < .4)] = "upper_narrow"
    modes[y >= .4] = "upper_wide"
    modes[(y <= -.12) & (y > -.4)] = "lower_narrow"
    modes[y <= -.4] = "lower_wide"
    modes[data["lengths"] <= 32] = "failed_before_midpoint"
    return modes


def load(key):
    with np.load(SOURCE / (key.replace("/", "--") + ".npz"), allow_pickle=False) as data:
        return {k: data[k].copy() for k in
                ("positions", "lengths", "success", "environment_seeds", "rollout_seeds")}


def render(regime, diagnostics):
    base = load(f"{regime}/base")
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.1), dpi=100)
    fig.patch.set_facecolor("#f8fafc")
    fig.subplots_adjust(left=.045, right=.99, bottom=.12, top=.84, wspace=.17, hspace=.40)
    heading = "Same centered start (-1, 0)" if regime == "centered" else "Gaussian starts: paired seeds across preferences"
    fig.suptitle(f"Frozen SFPS inference | {heading}", x=.5, y=.978,
                 fontsize=18, fontweight="bold", color="#0f172a")
    fig.text(.5, .932,
             "Thin paths: all 16 episodes   |   Bold path + dot: fixed episode 0   |   Gray: base episode 0",
             ha="center", fontsize=11, color="#334155")
    fig.text(.5, .902,
             "Dashed curve: requested reference   |   Target hits: measured mode at action 32   |   Goal: final task success",
             ha="center", fontsize=10.5, color="#334155")
    progress = fig.text(.5, .052, "", ha="center", fontsize=12, color="#0f172a")
    fig.text(.5, .017,
             "Saved executed positions; slowed playback.  Final tallies use all 16 episodes, including task failures.",
             ha="center", fontsize=10, color="#475569")

    panels, summary = [], {}
    for row, (method, label, color) in enumerate(METHODS):
        for col, user in enumerate(USERS):
            key = f"{regime}/{user}/{method}"
            data = load(key)
            assert np.array_equal(data["environment_seeds"], base["environment_seeds"])
            assert np.array_equal(data["rollout_seeds"], base["rollout_seeds"])
            actual = modes_for(data)
            record = diagnostics["conditions"][key]
            hits = int(np.sum(actual == user))
            assert hits == record["midpoint_counts"][user]
            assert int(data["success"].sum()) == record["success_count"]
            summary[key] = {
                "target_hits": hits, "count": len(actual),
                "goal_successes": int(data["success"].sum()),
                "episode_0_mode": str(actual[0]),
                "episode_0_goal_success": bool(data["success"][0]),
                "episode_0_environment_seed": int(data["environment_seeds"][0]),
                "source": key.replace("/", "--") + ".npz",
            }
            axis = axes[row, col]
            axis.set_facecolor("white")
            axis.set_xlim(-1.3, 1.3)
            # Show out-of-reference excursions without clipping positions.
            axis.set_ylim(-1.3, 1.3)
            assert np.nanmax(np.abs(data["positions"])) < 1.3
            axis.set_aspect("equal")
            axis.set_xticks([-1, 0, 1])
            axis.set_yticks([-1, -.5, 0, .5, 1])
            axis.tick_params(labelsize=9, colors="#64748b")
            axis.grid(alpha=.15)
            for spine in axis.spines.values():
                spine.set_color("#cbd5e1")
            axis.set_title(f"{label}\nRequested: {name(user)}", fontsize=11, pad=9,
                           color="#0f172a", fontweight="bold")
            if col == 0:
                axis.set_ylabel("y", fontsize=10)
            if row == 1:
                axis.set_xlabel("x", fontsize=10)
            sign = 1 if user.startswith("upper") else -1
            amplitude = .27 if user.endswith("narrow") else .53
            reference, _ = evaluate_path(sign, amplitude, np.linspace(0, 1, 65, dtype=np.float32))
            axis.plot(reference[:, 0], reference[:, 1], "--", lw=1.2, color="#a1a1aa", zorder=1)
            axis.add_patch(Circle((1, 0), .1, facecolor="#dcfce7", edgecolor="#16a34a", lw=.7))
            axis.scatter([1], [0], marker="*", color="#166534", s=70, zorder=6)
            axis.scatter(*data["positions"][0, 0], color="#0f172a", s=20, zorder=6)
            axis.text(.025, .975, f"Final target hits: {hits}/16   |   Goal: {data['success'].sum()}/16",
                      transform=axis.transAxes, va="top", fontsize=9, color="#334155",
                      bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
            collection = LineCollection([], colors=color, linewidths=.85, alpha=.27, zorder=2)
            axis.add_collection(collection)
            baseline, = axis.plot([], [], color="#9ca3af", lw=2.0, zorder=3)
            bold, = axis.plot([], [], color=color, lw=2.6, zorder=4)
            head, = axis.plot([], [], "o", color=color, markersize=5, zorder=7)
            midpoint, = axis.plot([], [], "s", mfc="white", mec=color, ms=5, zorder=8)
            failed, = axis.plot([], [], "x", color="#b91c1c", ms=5, zorder=9)
            status = axis.text(.025, .025, "", transform=axis.transAxes, fontsize=9,
                               color="#0f172a", va="bottom",
                               bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
            panels.append((data, actual, collection, baseline, bold, head, midpoint, failed, status))

    frames = []
    durations = []
    for step in range(65):
        for data, actual, collection, baseline, bold, head, midpoint, failed, status in panels:
            collection.set_segments([p[:min(step + 1, n)] for p, n in zip(data["positions"], data["lengths"])])
            original = base["positions"][0, :min(step + 1, base["lengths"][0])]
            baseline.set_data(original[:, 0], original[:, 1])
            path = data["positions"][0, :min(step + 1, data["lengths"][0])]
            bold.set_data(path[:, 0], path[:, 1])
            head.set_data(path[-1:, 0], path[-1:, 1])
            if step >= 32 and data["lengths"][0] > 32:
                point = data["positions"][0, 32]
                midpoint.set_data([point[0]], [point[1]])
            else:
                midpoint.set_data([], [])
            mode_text = name(str(actual[0])) if step >= 32 else "pending (action 32)"
            goal_text = ("PASS" if data["success"][0] else "FAIL") if step == 64 else "pending"
            status.set_text(f"Episode 0 mode: {mode_text}\nEpisode 0 goal: {goal_text}")
            if step == 64:
                ends = np.array([p[n - 1] for p, n, ok in zip(data["positions"], data["lengths"], data["success"]) if not ok])
                if len(ends):
                    failed.set_data(ends[:, 0], ends[:, 1])
        progress.set_text(f"Executed action {step:02d}/64   |   Simulated time {step / 32:.2f} s   |   Replan every 8 actions")
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        if step in (0, 32, 64):
            frame.save(OUTPUT / f"{regime}_frame_{step:02d}.png")
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        durations.append(2000 if step == 64 else 650 if step == 32 else 80)
    destination = OUTPUT / f"{regime}_mode_selection.gif"
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    plt.close(fig)
    with Image.open(destination) as saved:
        assert saved.n_frames == 65
        assert saved.size == (1600, 910)
        for i in range(saved.n_frames):
            saved.seek(i)
            saved.load()
    print(f"Verified {destination.name}: 65 frames, {destination.stat().st_size / 1e6:.2f} MB", flush=True)
    return summary


if __name__ == "__main__":
    diagnostics = json.loads((SOURCE / "diagnostics.json").read_text())
    summary = {}
    for regime in ("centered", "gaussian"):
        summary.update(render(regime, diagnostics))
    (OUTPUT / "mode_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
