"""Static figures for the synthetic-user preference pilot."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_preference_results(destination: Path, result: dict) -> None:
    users = list(result["config"]["users"])
    budgets = result["config"]["budgets"]
    conditions = result["conditions"]
    for quantity in ("utility", "success"):
        fig, axes = plt.subplots(2, len(users), figsize=(4 * len(users), 7),
                                  squeeze=False, constrained_layout=True)
        for row, regime in enumerate(("centered", "gaussian")):
            for column, user in enumerate(users):
                axis = axes[row, column]
                def value(key):
                    record = conditions[f"{regime}/{key}"]
                    result_value = record["success_rate"] if quantity == "success" else (
                        record["utility_by_user"][user]["successful_mean"])
                    return np.nan if result_value is None else result_value
                learned = [value(f"{user}/learned_{budget}") for budget in budgets]
                axis.plot(budgets, learned, "o-", color="tab:blue", label="Few-shot soft")
                if quantity == "utility":
                    errors = [conditions[f"{regime}/{user}/learned_{budget}"]["utility_by_user"][user][
                        "successful_standard_error"] for budget in budgets]
                    errors = np.array([np.nan if e is None else e for e in errors])
                    axis.fill_between(budgets, np.array(learned) - errors,
                                      np.array(learned) + errors, color="tab:blue", alpha=.15)
                for name, color, label in (("base", "gray", "Base"),
                        ("task_only", "tab:orange", "Task-only soft"),
                        (f"{user}/oracle_soft", "tab:green", "Oracle soft"),
                        (f"{user}/oracle_best", "tab:red", "Oracle best")):
                    axis.axhline(value(name), color=color, linestyle="--", label=label)
                axis.set_title(f"{regime}: {user}")
                axis.set_xlabel("Pairwise comparisons")
                axis.set_xticks(budgets)
                axis.set_ylabel("True utility (successful episodes)" if quantity == "utility" else "Goal success rate")
                axis.grid(alpha=.2)
                if quantity == "success":
                    axis.set_ylim(-.03, 1.03)
                if row == column == 0:
                    axis.legend(fontsize=8)
        fig.suptitle("Synthetic-user pilot; utility excludes failures, success includes all episodes")
        fig.savefig(destination / f"preference_{quantity}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, len(users), figsize=(4 * len(users), 7),
                              squeeze=False, constrained_layout=True)
    budget = 20 if 20 in budgets else max(budgets)
    for row, regime in enumerate(("centered", "gaussian")):
        for column, user in enumerate(users):
            axis = axes[row, column]
            for method, color, label in (("base", "gray", "Base"),
                    (f"{user}/oracle_soft", "tab:green", "Oracle soft"),
                    (f"{user}/learned_{budget}", "tab:blue", f"Learned K={budget}")):
                with np.load(destination / conditions[f"{regime}/{method}"]["artifact"], allow_pickle=False) as data:
                    for i in range(min(8, len(data["positions"]))):
                        path = data["positions"][i, :data["lengths"][i]]
                        axis.plot(path[:, 0], path[:, 1], color=color, alpha=.45,
                                  label=label if i == 0 else None)
                        if not data["success"][i]:
                            axis.scatter(*path[-1], marker="x", color=color, s=20)
            axis.scatter([-1, 1], [0, 0], c=["black", "red"], s=20)
            axis.set_title(f"{regime}: {user}")
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.grid(alpha=.2)
            if row == column == 0:
                axis.legend(fontsize=8)
    fig.suptitle("Same seeded starts; first 8 episodes per condition; x marks failed endpoints")
    fig.savefig(destination / "preference_trajectories.png", dpi=160)
    plt.close(fig)
