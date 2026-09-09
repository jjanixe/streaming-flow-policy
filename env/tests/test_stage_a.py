import json
import subprocess
import sys

import pytest
import torch

from env.run_stage_a import (
    acceptance_failures,
    classify_midpoint_modes,
    run_stage_a,
    sample_errors,
)


def test_midpoint_classifier_leaves_center_as_other():
    samples = torch.tensor(
        [
            [0.0, 0.25],
            [0.0, 0.50],
            [0.0, -0.25],
            [0.0, -0.50],
            [0.0, 0.00],
        ],
        dtype=torch.float32,
    )

    occupancy = classify_midpoint_modes(samples)

    assert occupancy == pytest.approx({
        "upper-narrow": 0.2,
        "upper-wide": 0.2,
        "lower-narrow": 0.2,
        "lower-wide": 0.2,
        "other": 0.2,
        "nonfinite": 0.0,
    })


def test_midpoint_classifier_accounts_for_nonfinite_samples():
    samples = torch.tensor(
        [[0.0, 0.25], [0.0, float("nan")]],
        dtype=torch.float32,
    )

    occupancy = classify_midpoint_modes(samples)

    assert occupancy["upper-narrow"] == pytest.approx(0.5)
    assert occupancy["nonfinite"] == pytest.approx(0.5)
    assert sum(occupancy.values()) == pytest.approx(1.0)


def test_identical_samples_have_zero_marginal_error():
    samples = torch.tensor(
        [[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]],
        dtype=torch.float32,
    )

    mean_error, covariance_error = sample_errors(samples, samples.clone())

    assert mean_error == 0.0
    assert covariance_error == 0.0


def test_sample_errors_reject_nonfinite_inputs():
    samples = torch.tensor(
        [[0.0, 0.0], [float("nan"), 1.0]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="finite"):
        sample_errors(samples, torch.zeros_like(samples))


def _passing_sampler_diagnostics():
    return {
        "times": {
            label: {
                "mean_l2_error": 0.0,
                "covariance_frobenius_error": 0.0,
                "nonfinite_count": 0,
            }
            for label in ("0.25", "0.5", "0.75", "1.0")
        },
        "midpoint_occupancy": {
            "upper-narrow": 0.25,
            "upper-wide": 0.25,
            "lower-narrow": 0.25,
            "lower-wide": 0.25,
            "other": 0.0,
            "nonfinite": 0.0,
        },
        "nonfinite_state_count": 0,
    }


def test_acceptance_passes_finite_metrics_and_rejects_threshold_failure():
    diagnostics = {"ode": _passing_sampler_diagnostics()}

    assert acceptance_failures(diagnostics) == []

    diagnostics["ode"]["times"]["0.5"]["mean_l2_error"] = 0.031
    failures = acceptance_failures(diagnostics)

    assert any("mean error" in failure for failure in failures)


def test_acceptance_rejects_nonfinite_metrics_and_occupancy():
    diagnostics = {"ode": _passing_sampler_diagnostics()}
    diagnostics["ode"]["times"]["0.5"]["mean_l2_error"] = float("nan")
    diagnostics["ode"]["midpoint_occupancy"]["upper-narrow"] = float("nan")

    failures = acceptance_failures(diagnostics)

    assert any("non-finite mean error" in failure for failure in failures)
    assert any("non-finite midpoint occupancy" in failure for failure in failures)


def test_small_stage_a_run_writes_complete_artifacts(tmp_path):
    result = run_stage_a(
        tmp_path,
        seed=23,
        num_rollouts=128,
        enforce_acceptance=False,
    )

    assert result["num_rollouts"] == 128
    assert set(result["samplers"]) == {"ode", "sde"}
    assert set(result["samplers"]["ode"]["times"]) == {
        "0.25",
        "0.5",
        "0.75",
        "1.0",
    }
    assert set(result["samplers"]["ode"]["midpoint_occupancy"]) == {
        "upper-narrow",
        "upper-wide",
        "lower-narrow",
        "lower-wide",
        "other",
        "nonfinite",
    }
    for name in (
        "demonstrations.npz",
        "feature_normalizer.npz",
        "rollouts.npz",
        "resolved_config.json",
        "diagnostics.json",
    ):
        assert (tmp_path / name).is_file()
    with (tmp_path / "diagnostics.json").open(encoding="utf-8") as stream:
        assert json.load(stream) == result


def test_stage_a_acceptance_passes_at_practical_sample_count(tmp_path):
    result = run_stage_a(
        tmp_path,
        seed=0,
        num_rollouts=2048,
        enforce_acceptance=True,
    )

    assert result["accepted"] is True
    assert result["failures"] == []


def test_module_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "env.run_stage_a", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--num-rollouts" in completed.stdout
