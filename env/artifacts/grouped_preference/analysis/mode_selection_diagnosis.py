"""Read-only diagnostics of the saved grouped pilot; no new policy rollouts.

Run from the B2 worktree: PYTHONPATH=. .venv/bin/python \
    env/artifacts/grouped_preference/analysis/mode_selection_diagnosis.py
The beta sweep reweights the SAME first-decision bank. Its mode probabilities
use four base-policy futures per candidate and are not closed-loop success rates.
"""
import json
from pathlib import Path

import numpy as np

from env.demonstrations import evaluate_path
from env.grouped_preference import grouped_features, grouped_utility, synthetic_grouped_users


ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "pilot-seed0"
MODES = ("upper_narrow", "upper_wide", "lower_narrow", "lower_wide", "other")


def classify(y):
    return np.select(
        [(y >= .12) & (y < .4), y >= .4,
         (y <= -.12) & (y > -.4), y <= -.4],
        np.arange(4), default=4)


def distribution(scores, valid):
    assert np.all(valid.any(axis=-1))
    scores = np.where(valid, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def main():
    users = synthetic_grouped_users()
    report = {
        "source": str(PILOT),
        "scope": "Saved first-decision candidate bank; no new closed-loop inference.",
        "caveats": [
            "Finite-bank absence is not proof of zero probability under every latent.",
            "L=4 futures share each current chunk; 512 is not 512 independent candidates.",
            "Fixed-bank mode probability is not a closed-loop control upper bound.",
            "Synthetic oracle weights are used for the selector diagnostic.",
        ],
        "regimes": {},
    }
    config = json.loads((PILOT / "resolved_config.json").read_text())
    with np.load(config["demonstrations_path"]) as bank:
        train = bank["splits"] == "train"
        modes, counts = np.unique(bank["modes"][train], return_counts=True)
        report["training"] = {
            "mode_counts": dict(zip(modes.tolist(), counts.tolist())),
            "unique_initial_positions": np.unique(bank["positions"][train, 0], axis=0).tolist(),
        }
    for regime in ("centered", "gaussian"):
        with np.load(PILOT / f"{regime}__task_only.npz") as bank:
            y = bank["continuation_positions"][:, 0, :, :, 32, 1]
            valid = bank["current_valid"][:, 0]
            complete = bank["continuation_lengths"][:, 0] == 65
            successful = bank["continuation_success"][:, 0]
            initial = bank["positions"][:, 0]
            features = bank["expected_features"][:, 0]
            costs = bank["goal_costs"][:, 0]
            current_latents = bank["current_latents"][:, 0]
            future_latents = bank["future_latents"][:, 0]
        assert np.all(valid) and np.all(complete)
        # Every user starts with exactly the same proposals for the same episode.
        for user in users:
            with np.load(PILOT / f"{regime}__{user}__oracle_soft.npz") as other:
                np.testing.assert_array_equal(current_latents, other["current_latents"][:, 0])
                np.testing.assert_array_equal(future_latents, other["future_latents"][:, 0])
                np.testing.assert_array_equal(features, other["expected_features"][:, 0])
        modes = classify(y)
        frequencies = np.stack([((modes == i) & successful).mean(axis=-1)
                                for i in range(5)], axis=-1)
        record = {
            "episodes": len(y), "M": y.shape[1], "L": y.shape[2],
            "complete_futures": int(complete.sum()),
            "successful_futures": int(successful.sum()),
            "complete_mode_counts": dict(zip(MODES, np.bincount(modes.ravel(), minlength=5).tolist())),
            "successful_mode_counts": dict(zip(MODES, np.bincount(modes[successful], minlength=5).tolist())),
            "episodes_with_any_successful_target_future": dict(zip(MODES, (frequencies.max(axis=1) > 0).sum(axis=0).tolist())),
            "episodes_detail": [], "fixed_bank_reweighting": {},
        }
        for episode in range(len(y)):
            record["episodes_detail"].append({
                "episode": episode, "initial_position": initial[episode].tolist(),
                "midpoint_y_min": float(y[episode].min()),
                "midpoint_y_max": float(y[episode].max()),
                "complete_mode_counts": dict(zip(MODES, np.bincount(modes[episode].ravel(), minlength=5).tolist())),
            })
        for user, preference in users.items():
            target = MODES.index(user.rsplit("_", 1)[0])
            utility = grouped_utility(features, preference)
            frequency = frequencies[..., target]
            values = {}
            for beta in (0, 16, 64, 256):
                weights = distribution(beta * utility - costs, valid)
                values[f"beta_{beta}"] = {
                    "mean_empirical_success_and_target_probability": float((weights * frequency).sum(axis=-1).mean()),
                    "mean_ESS_at_first_decision": float((1 / (weights ** 2).sum(axis=-1)).mean()),
                }
            best = np.argmax(np.where(valid, 16 * utility - costs, -np.inf), axis=-1)
            values["argmax_beta16_empirical_target_probability"] = float(frequency[np.arange(len(y)), best].mean())
            values["max_target_frequency_in_bank_mean"] = float(np.where(valid, frequency, 0).max(axis=-1).mean())
            record["fixed_bank_reweighting"][user] = values
        report["regimes"][regime] = record

    amplitudes = np.array([.27, .53, -.27, -.53, 0.])
    times = np.linspace(0, 1, 65, dtype=np.float32)
    paths = np.stack([evaluate_path(1. if a >= 0 else -1., abs(a), times)[0]
                      for a in amplitudes])
    features = grouped_features(paths)
    report["prototype_objective_check"] = {
        "modes": MODES, "signed_amplitudes": amplitudes.tolist(),
        "profiles": {},
    }
    for user, preference in users.items():
        utility = grouped_utility(features, preference)
        report["prototype_objective_check"]["profiles"][user] = {
            "utilities": utility.tolist(), "preferred_mode": MODES[int(utility.argmax())]}
    output = ROOT / "mode_selection_diagnosis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    for regime, record in report["regimes"].items():
        print(regime, record["complete_mode_counts"])
        for user in ("upper_narrow_direction", "upper_narrow_width", "upper_wide_direction"):
            values = record["fixed_bank_reweighting"][user]
            print(user, {key: round(values[key]["mean_empirical_success_and_target_probability"], 4)
                         for key in ("beta_0", "beta_16", "beta_64", "beta_256")},
                  "argmax", round(values["argmax_beta16_empirical_target_probability"], 4),
                  "bank maximum", round(values["max_target_frequency_in_bank_mean"], 4))


if __name__ == "__main__":
    main()
