import hashlib
import json
from dataclasses import replace

import numpy as np
import torch

from env.artifacts import save_demonstration_bank
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.features import fit_feature_normalizer, trajectory_features
from env.run_preference import build_comparison_queries, run_preference
from env.stage_b_config import DEFAULT_STAGE_B2_CONFIG
from env.train_stage_b import train_sfps


def test_queries_span_both_preferences_and_have_independent_heldout_ids():
    training = build_comparison_queries(seed=40, count=40)
    repeated = build_comparison_queries(seed=40, count=40)
    heldout = build_comparison_queries(seed=41, count=80)
    np.testing.assert_array_equal(training["trajectories"], repeated["trajectories"])
    assert not set(training["ids"].flat) & set(heldout["ids"].flat)
    assert training["trajectories"].shape == (40, 2, 65, 2)
    np.testing.assert_array_equal(training["trajectories"][:, :, 0],
                                  np.tile([-1, 0], (40, 2, 1)))
    np.testing.assert_array_equal(training["trajectories"][:, :, -1],
                                  np.tile([1, 0], (40, 2, 1)))
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=0)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    features = trajectory_features(training["trajectories"], normalizer, 1 / 64)
    assert np.linalg.matrix_rank(features[:5, 0] - features[:5, 1]) == 2
    # A pure direction query holds width fixed, and a width query holds sign.
    np.testing.assert_allclose(features[0, 0, 1], features[0, 1, 1], atol=1e-6)
    assert np.sign(training["trajectories"][1, 0, 32, 1]) == np.sign(
        training["trajectories"][1, 1, 32, 1])


def test_small_experiment_records_failures_and_preserves_frozen_checkpoint(tmp_path):
    config = replace(DEFAULT_CONFIG, demonstrations=replace(
        DEFAULT_CONFIG.demonstrations, train_per_mode=2,
        validation_per_mode=1, test_per_mode=1))
    bank = generate_demonstration_bank(config, seed=42)
    bank_path = tmp_path / "demonstrations.npz"
    save_demonstration_bank(bank_path, bank)
    training_config = replace(DEFAULT_STAGE_B2_CONFIG, hidden_dim=8,
        hidden_layers=1, batch_size=16, max_updates=1, validation_interval=1,
        warmup_updates=1, integration_steps_per_action=1)
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = train_sfps(bank, tmp_path / "training",
                              config=training_config, root_seed=43, device="cpu")
        checkpoint = training.checkpoint_path
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        kwargs = dict(checkpoint_path=checkpoint, stats_path=training.stats_path,
            demonstrations_path=bank_path, seed=44, device="cpu", candidates=2,
            rollout_count=2, coverage_count=2, integration_steps_per_action=1,
            budgets=(5,), user_weights={"test_user": (2.0, -1.0)}, make_plots=False)
        result = run_preference(tmp_path / "first", **kwargs)
        repeated = run_preference(tmp_path / "second", **kwargs)
    finally:
        torch.set_num_threads(before_threads)
    assert before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["policy_unchanged"] is True
    assert result["checkpoint_digest"] == before
    assert result["synthetic_users"] is True
    assert result["coverage"].keys() == {"centered", "gaussian_0", "gaussian_1"}
    assert result["conditions"].keys() == repeated["conditions"].keys()
    assert len(result["conditions"]) == 10
    for key, metrics in result["conditions"].items():
        assert metrics["count"] == 2
        assert 0 <= metrics["success_count"] <= 2
        assert metrics["complete_count"] + metrics["incomplete_count"] == 2
        with np.load(tmp_path / "first" / metrics["artifact"], allow_pickle=False) as a:
            with np.load(tmp_path / "second" / repeated["conditions"][key]["artifact"],
                         allow_pickle=False) as b:
                for field in ("positions", "requested_actions", "lengths", "selected_indices"):
                    np.testing.assert_array_equal(a[field], b[field])
                assert a["positions"].dtype == np.float32
                assert a["lengths"].dtype.kind in "iu"
    with (tmp_path / "first" / "diagnostics.json").open() as stream:
        saved = json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert saved["policy_unchanged"] is True
    assert (tmp_path / "first" / "preferences.npz").is_file()
    assert (tmp_path / "first" / "feature_normalizer.npz").is_file()
