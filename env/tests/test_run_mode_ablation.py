import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from env.artifacts import save_demonstration_bank
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.grouped_preference import synthetic_grouped_users
from env.tests.test_grouped_rollout import LinearPolicy, stats


class FrozenLinearPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearPolicy()

    @property
    def device(self):
        return self.linear.device

    def predict_batch(self, *args, **kwargs):
        return self.linear.predict_batch(*args, **kwargs)


class FixedXPolicy:
    device = torch.device("cpu")

    def predict_batch(self, nobs, num_actions, integration_steps_per_action, *, latents):
        physical = nobs[:, -1, :2] * torch.tensor([2, 4])
        delta = torch.stack(
            (torch.full_like(latents[:, 0], 1 / 32), latents[:, 1] / 128), -1)
        points = physical[:, None] + torch.arange(9)[None, :, None] * delta[:, None]
        points[:, 0] = nobs[:, -1, :2] * torch.tensor([3, 2])
        return points / torch.tensor([3, 2])


@pytest.fixture
def ablation_inputs(tmp_path, monkeypatch, stats):
    config = replace(DEFAULT_CONFIG, demonstrations=replace(
        DEFAULT_CONFIG.demonstrations, train_per_mode=1,
        validation_per_mode=1, test_per_mode=1))
    bank = generate_demonstration_bank(config, seed=71)
    bank_path = tmp_path / "demonstrations.npz"
    checkpoint_path = tmp_path / "checkpoint.pt"
    stats_path = tmp_path / "stats.npz"
    save_demonstration_bank(bank_path, bank)
    checkpoint_path.write_bytes(b"fixture-checkpoint")
    stats_path.write_bytes(b"fixture-stats")

    import env.run_mode_ablation as module
    calls = []

    def load(checkpoint, statistics, *, expected_train_data_digest, device):
        calls.append((checkpoint, statistics, expected_train_data_digest, device))
        return FrozenLinearPolicy(), stats, {"architecture": {"fixture": True}}

    monkeypatch.setattr(module, "load_trained_sfps", load)
    return module, calls, dict(
        checkpoint_path=checkpoint_path, stats_path=stats_path,
        demonstrations_path=bank_path, seed=72, device="cpu",
        candidates=(2, 3), continuations=1, rollout_count=1,
        integration_steps_per_action=2, regimes=("centered",),
        users={"upper_narrow_direction": synthetic_grouped_users()[
            "upper_narrow_direction"]}, methods=("soft", "best"), beta=16.,
        proposal_std=1.5,
    )


def test_proposal_std_scales_only_guided_current_latents(stats):
    from env.run_grouped_preference import _closed_loop

    common = dict(
        environment_seeds=[2], rollout_seeds=[4], centered=True,
        candidates=3, continuations=2, beta=16., integration_steps=2,
        feature_kind="mode", proposal_std=2.5,
    )
    preference = synthetic_grouped_users()["upper_narrow_direction"]
    guided = _closed_loop(
        LinearPolicy(), stats, preference, method="best", **common)
    unit = _closed_loop(
        LinearPolicy(), stats, preference, method="best",
        **{**common, "proposal_std": 1.0})
    base = _closed_loop(
        LinearPolicy(), stats, preference, method="base", **common)

    guided_mask = guided["decision_mask"]
    unit_mask = unit["decision_mask"]
    np.testing.assert_array_equal(guided_mask, unit_mask)
    np.testing.assert_allclose(
        guided["current_latents"][guided_mask],
        2.5 * unit["current_latents"][unit_mask], rtol=0, atol=0)
    np.testing.assert_array_equal(
        guided["future_latents"][guided_mask],
        unit["future_latents"][unit_mask])
    np.testing.assert_array_equal(
        base["current_latents"][base["decision_mask"]],
        unit["current_latents"][unit_mask][:, :1])


@pytest.mark.parametrize("proposal_std", [0., -1., np.inf, np.nan, True])
def test_closed_loop_rejects_invalid_proposal_std(stats, proposal_std):
    from env.run_grouped_preference import _closed_loop

    with pytest.raises(ValueError, match="proposal_std"):
        _closed_loop(
            LinearPolicy(), stats, None,
            environment_seeds=[2], rollout_seeds=[4], centered=True,
            candidates=2, continuations=2, method="soft", beta=16.,
            integration_steps=2, proposal_std=proposal_std)


def test_fixture_policy_execution_distinguishes_soft_and_best(stats):
    from env.run_grouped_preference import _closed_loop

    common = dict(
        environment_seeds=[2], rollout_seeds=[2], centered=True,
        candidates=4, continuations=2, beta=16., integration_steps=2,
        feature_kind="mode")
    preference = synthetic_grouped_users()["upper_narrow_direction"]
    soft = _closed_loop(FixedXPolicy(), stats, preference, method="soft", **common)
    best = _closed_loop(FixedXPolicy(), stats, preference, method="best", **common)

    assert soft["selected_indices"][0, 0] == 1
    assert best["selected_indices"][0, 0] == 3
    np.testing.assert_array_equal(
        soft["current_latents"][0, 0], best["current_latents"][0, 0])
    np.testing.assert_array_equal(
        soft["future_latents"][0, 0], best["future_latents"][0, 0])


def test_target_mode_summary_excludes_goal_misses_and_reports_full_distribution():
    from env.run_mode_ablation import _target_mode_summary

    positions = np.zeros((5, 65, 2), np.float32)
    positions[:, 32, 1] = [.2, .2, .5, -.2, -.5]
    output = dict(
        positions=positions, lengths=np.full(5, 65),
        success=np.array([True, False, True, False, True]),
        numerical_failure=np.zeros(5, bool),
        action_limit_failure=np.zeros(5, bool),
    )

    summary = _target_mode_summary(output, "upper_narrow")

    assert summary["target_mode"] == "upper_narrow"
    assert summary["successful_target_count"] == 1
    assert summary["successful_target_rate"] == pytest.approx(.2)
    assert summary["mode_counts"] == {
        "upper_narrow": 2, "upper_wide": 1, "lower_narrow": 1,
        "lower_wide": 1, "other": 0,
    }
    assert summary["successful_mode_counts"] == {
        "upper_narrow": 1, "upper_wide": 1, "lower_narrow": 0,
        "lower_wide": 1, "other": 0,
    }
    assert summary["successful_target_wilson_95"] == pytest.approx(
        [0.03622316096978745, 0.6244717358814613])


def test_runner_saves_complete_conditions_and_pairs_streams_across_m(
        tmp_path, ablation_inputs):
    module, calls, kwargs = ablation_inputs
    output_dir = tmp_path / "ablation"

    result = module.run_mode_ablation(output_dir, **kwargs)

    assert len(calls) == 1
    assert result["checkpoint_digest"] == hashlib.sha256(
        kwargs["checkpoint_path"].read_bytes()).hexdigest()
    assert result["stats_digest"] == hashlib.sha256(
        kwargs["stats_path"].read_bytes()).hexdigest()
    assert result["complete"]
    assert len(result["conditions"]) == 4
    small_key = "centered__upper_narrow_direction__soft__m2"
    large_key = "centered__upper_narrow_direction__soft__m3"
    small = result["conditions"][small_key]
    assert small["regime"] == "centered"
    assert small["profile"] == "upper_narrow_direction"
    assert small["method"] == "soft"
    assert small["M"] == 2
    np.testing.assert_allclose(
        small["preference"]["weights"], [[.9, .1], [.1, .9]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        small["preference"]["alpha"], [.8, .2], rtol=0, atol=0)
    assert "successful_midpoint_counts" in small
    assert "successful_target_count" in small
    assert "task_failure_count" in small
    assert "elapsed_seconds" in small
    with np.load(output_dir / small["artifact"], allow_pickle=False) as a:
        with np.load(
                output_dir / result["conditions"][large_key]["artifact"],
                allow_pickle=False) as b:
            np.testing.assert_array_equal(
                a["current_latents"], b["current_latents"][:, :, :2])
            np.testing.assert_array_equal(
                a["future_latents"], b["future_latents"][:, :, :2])
            assert str(a["condition_key"].item()) == small_key
            assert str(a["config_digest"].item()) == result["config_digest"]
    marker_path = output_dir / small["completion_marker"]
    with marker_path.open(encoding="utf-8") as stream:
        marker = json.load(stream)
    assert marker["artifact_digest"] == hashlib.sha256(
        (output_dir / small["artifact"]).read_bytes()).hexdigest()
    for filename in ("resolved_config.json", "diagnostics.json"):
        with (output_dir / filename).open(encoding="utf-8") as stream:
            json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)))


def test_resume_skips_valid_conditions_and_refuses_mismatch_or_corruption(
        tmp_path, ablation_inputs, monkeypatch):
    module, _, kwargs = ablation_inputs
    output_dir = tmp_path / "resume"
    first = module.run_mode_ablation(output_dir, **kwargs)
    artifact = output_dir / next(iter(first["conditions"].values()))["artifact"]
    original = artifact.read_bytes()

    def should_not_run(*args, **values):
        raise AssertionError("completed condition was rerun")

    monkeypatch.setattr(module, "_closed_loop", should_not_run)
    resumed = module.run_mode_ablation(output_dir, **kwargs)
    assert resumed["conditions"] == first["conditions"]
    assert artifact.read_bytes() == original

    with pytest.raises(ValueError, match="resolved config mismatch"):
        module.run_mode_ablation(output_dir, **{**kwargs, "proposal_std": 2.})

    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact digest"):
        module.run_mode_ablation(output_dir, **kwargs)


def test_partial_artifact_without_completion_marker_is_replaced(
        tmp_path, ablation_inputs):
    module, _, kwargs = ablation_inputs
    output_dir = tmp_path / "partial"
    output_dir.mkdir()
    partial = output_dir / "centered__upper_narrow_direction__soft__m2.npz"
    partial.write_bytes(b"partial")

    result = module.run_mode_ablation(output_dir, **kwargs)

    assert result["complete"]
    with np.load(partial, allow_pickle=False) as payload:
        assert str(payload["condition_key"].item()).endswith("__soft__m2")


def test_runner_accepts_continuous_matched_control(tmp_path, ablation_inputs):
    module, _, kwargs = ablation_inputs

    result = module.run_mode_ablation(
        tmp_path / "continuous", **{
            **kwargs, "candidates": (2,), "methods": ("best",),
            "feature_kind": "continuous"})

    assert result["config"]["feature_kind"] == "continuous"
    condition = next(iter(result["conditions"].values()))
    with np.load(tmp_path / "continuous" / condition["artifact"],
                 allow_pickle=False) as payload:
        decided = payload["decision_mask"]
        values = payload["expected_features"][decided]
        assert np.any((values > 0) & (values < 1))
        from env.run_grouped_preference import _path_metrics
        expected = _path_metrics(
            {name: payload[name] for name in payload.files}, kwargs["users"],
            feature_kind="continuous")
    assert condition["utility_by_user"] == expected["utility_by_user"]
