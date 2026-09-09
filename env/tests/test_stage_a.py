import json
import subprocess
import sys

import pytest
import torch

from env.run_stage_a import (
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
    })


def test_identical_samples_have_zero_marginal_error():
    samples = torch.tensor(
        [[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]],
        dtype=torch.float32,
    )

    mean_error, covariance_error = sample_errors(samples, samples.clone())

    assert mean_error == 0.0
    assert covariance_error == 0.0


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


def test_module_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "env.run_stage_a", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--num-rollouts" in completed.stdout
