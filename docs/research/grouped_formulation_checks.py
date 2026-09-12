"""Small numerical checks for the grouped formulation audit, not a policy run."""

from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / ".worktrees/stage-b2-sfps/env/artifacts/preference/pilot-seed0"


def kl(q, p):
    return float(np.sum(q * (np.log(q) - np.log(p))))


def run_checks():
    result = {}
    p = np.array([.1, .2, .3, .4])
    utility = np.array([-.5, .2, .7, 1.1])
    beta = 2.3
    log_z = float(np.log(np.sum(p * np.exp(beta * utility))))
    optimal = p * np.exp(beta * utility - log_z)
    gaps = []
    rng = np.random.default_rng(20260911)
    for q in rng.dirichlet(np.ones(4), size=100):
        objective = float(q @ utility) - kl(q, p) / beta
        alternative = log_z / beta - kl(q, optimal) / beta
        assert abs(objective - alternative) < 1e-12
        gaps.append(log_z / beta - objective)
    assert min(gaps) >= 0
    result["gibbs_identity"] = {
        "probability_sum": float(optimal.sum()),
        "checked_distributions": 100,
        "minimum_gap_to_gibbs_objective": min(gaps),
    }

    # Equal mean, different conditional variance; beta=4, M tending to infinity.
    mc = []
    for count in (1, 2, 4, 16, 64, 1024):
        t = 2 / count
        log_ratio = count * (np.logaddexp(t, -t) - np.log(2))
        probability = float(1 / (1 + np.exp(-log_ratio)))
        mc.append({"L": count, "noisy_candidate_probability": probability})
    assert abs(mc[0]["noisy_candidate_probability"] - .7900128291929869) < 1e-12
    assert abs(mc[-1]["noisy_candidate_probability"] - .5) < .001
    result["finite_L_counterexample"] = {"intended_probability": .5, "beta": 4, "large_M_limits": mc}

    # Two different hierarchical parameters with identical overall contrasts.
    d = rng.uniform(-1, 1, size=50)
    excursion = rng.uniform(d * d, 1)
    f_direction = np.stack(((1 + d) / 2, (1 - d) / 2), axis=-1)
    f_width = np.stack((1 - excursion, excursion), axis=-1)
    u1 = .5 * (f_direction @ [.8, .2]) + .5 * (f_width @ [.6, .4])
    u2 = .6 * (f_direction @ [.75, .25]) + .4 * (f_width @ [.625, .375])
    max_difference = float(np.max(np.abs((u1 - u1[0]) - (u2 - u2[0]))))
    assert max_difference < 1e-14
    combined = np.concatenate((f_direction, f_width), axis=-1)
    tangent = np.concatenate((np.eye(3), -np.ones((1, 3))), axis=0)
    contrast_rank = int(np.linalg.matrix_rank((combined - combined[0]) @ tangent))
    assert contrast_rank == 2
    result["overall_only_nonidentifiability"] = {
        "alpha_1": [.5, .5], "alpha_2": [.6, .4],
        "max_overall_contrast_difference": max_difference,
        "simplex_parameter_dof": 3, "overall_contrast_rank": contrast_rank,
    }

    # Verify real saved query features and an exact signed-to-simplex conversion.
    with np.load(PILOT / "feature_normalizer.npz", allow_pickle=False) as data:
        mu, scale = data["mu"].astype(float), data["scale"].astype(float)
        y_scale = float(data["y_scale"])
    with np.load(PILOT / "preferences.npz", allow_pickle=False) as data:
        paths = data["query_trajectories"].astype(float)
        saved_delta = data["query_delta_features"].astype(float)
    rho = np.tanh(paths[:, :, 1:, 1] / y_scale)
    raw = np.stack((rho.mean(axis=-1), (rho * rho).mean(axis=-1)), axis=-1)
    normalized = (raw - mu) / scale
    f_direction = np.stack(((1 + raw[..., 0]) / 2, (1 - raw[..., 0]) / 2), axis=-1)
    f_excursion = np.stack((raw[..., 1], 1 - raw[..., 1]), axis=-1)
    conversion = []
    for signed in ((2, -1), (2, 1), (-2, -1), (-2, 1)):
        a, b = signed
        mass = np.array([2 * abs(a) / scale[0], abs(b) / scale[1]])
        total = float(mass.sum())
        alpha = mass / total
        direction = f_direction[..., 0 if a > 0 else 1]
        excursion = f_excursion[..., 0 if b > 0 else 1]
        grouped = alpha[0] * direction + alpha[1] * excursion
        old = normalized @ np.array(signed)
        old_contrast = old[:, 0] - old[:, 1]
        new_contrast = total * (grouped[:, 0] - grouped[:, 1])
        error = float(np.max(np.abs(old_contrast - new_contrast)))
        saved_error = float(np.max(np.abs(saved_delta @ np.array(signed) - new_contrast)))
        assert error < 1e-12
        assert saved_error < 1e-5
        conversion.append({"old_weights": list(signed), "scale_S": total,
                           "alpha": alpha.tolist(), "algebra_max_error": error,
                           "saved_float32_contrast_max_error": saved_error})
    result["oracle_reparameterization"] = conversion
    endpoints = paths[:, :, -1].reshape(-1, 2)
    lengths = np.linalg.norm(np.diff(paths, axis=2), axis=-1).sum(axis=-1)
    mean_speed = (np.linalg.norm(np.diff(paths, axis=2), axis=-1) / (2 / 64)).mean(axis=-1)
    assert np.ptp(endpoints, axis=0).max() == 0
    assert np.max(np.abs(mean_speed - lengths / 2)) < 1e-12
    result["current_query_support"] = {
        "option_count": int(paths.shape[0] * paths.shape[1]),
        "endpoint_range": np.ptp(endpoints, axis=0).tolist(),
        "fixed_duration_seconds": 2,
        "mean_speed_equals_length_over_duration": True,
        "path_length_range": [float(lengths.min()), float(lengths.max())],
    }
    return result


if __name__ == "__main__":
    result = run_checks()
    destination = Path(__file__).with_name("2026-09-11-grouped-formulation-checks.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
