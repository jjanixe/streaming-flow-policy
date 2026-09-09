import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from env.artifacts import train_data_digest
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.stage_b_artifacts import load_sfpd_checkpoint
from env.stage_b_config import DEFAULT_STAGE_B1_CONFIG
from env.train_stage_b import ExponentialMovingAverage, train_sfpd


def _small_bank(seed: int):
    demonstration_config = replace(
        DEFAULT_CONFIG.demonstrations,
        train_per_mode=2,
        validation_per_mode=1,
        test_per_mode=1,
    )
    return generate_demonstration_bank(
        replace(DEFAULT_CONFIG, demonstrations=demonstration_config),
        seed=seed,
    )


def _short_config(**overrides):
    values = {
        "hidden_dim": 16,
        "hidden_layers": 1,
        "batch_size": 64,
        "max_updates": 4,
        "validation_interval": 2,
        "warmup_updates": 1,
    }
    values.update(overrides)
    return replace(DEFAULT_STAGE_B1_CONFIG, **values)


def test_exponential_moving_average_updates_float32_and_integer_state():
    module = torch.nn.Module()
    module.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    )
    module.register_buffer("count", torch.tensor(1, dtype=torch.int32))
    ema = ExponentialMovingAverage(module, decay=0.75)
    with torch.no_grad():
        module.weight.fill_(2.0)
        module.count.fill_(3)

    ema.update(module)

    torch.testing.assert_close(
        ema.shadow["weight"], torch.full((2,), 0.5, dtype=torch.float32)
    )
    torch.testing.assert_close(ema.shadow["count"], torch.tensor(3, dtype=torch.int32))


def test_short_cpu_training_writes_finite_best_ema_checkpoint(tmp_path):
    bank = _small_bank(seed=22)
    config = _short_config()

    result = train_sfpd(bank, tmp_path, config=config, root_seed=23, device="cpu")

    assert result.checkpoint_path.is_file()
    assert result.stats_path.is_file()
    assert result.history_path.is_file()
    assert result.selected_update in (2, 4)
    assert np.isfinite(result.validation_loss)
    assert set(result.seed_streams) == {
        "model_initialization",
        "dataloader_shuffle",
        "train_transform",
        "validation_transform",
        "rollout_initialization",
        "checkpoint_replay",
    }
    assert len(set(result.seed_streams.values())) == 6

    loaded = load_sfpd_checkpoint(
        result.checkpoint_path,
        expected_train_data_digest=train_data_digest(bank),
    )
    assert loaded["metadata"]["model_type"] == "sfpd"
    assert loaded["metadata"]["selected_update"] == result.selected_update
    assert loaded["metadata"]["seed_streams"] == result.seed_streams
    assert loaded["metadata"]["stage_b1_config"]["max_updates"] == 4
    assert loaded["metadata"]["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "warmup_updates": 1,
        "schedule": "linear_warmup_cosine",
    }

    with result.history_path.open(encoding="utf-8") as stream:
        history = json.load(stream)
    assert [entry["update"] for entry in history["train"]] == [1, 2, 3, 4]
    assert [entry["update"] for entry in history["validation"]] == [2, 4]
    assert all(np.isfinite(entry["loss"]) for entry in history["train"])
    assert all(np.isfinite(entry["loss"]) for entry in history["validation"])


def test_short_training_is_reproducible_and_restores_global_torch_rng(tmp_path):
    bank = _small_bank(seed=24)
    config = _short_config(max_updates=3, validation_interval=2)
    torch.manual_seed(8675309)
    global_rng_before = torch.random.get_rng_state().clone()

    first = train_sfpd(
        bank,
        tmp_path / "first",
        config=config,
        root_seed=25,
        device="cpu",
    )
    global_rng_after_first = torch.random.get_rng_state().clone()
    second = train_sfpd(
        bank,
        tmp_path / "second",
        config=config,
        root_seed=25,
        device="cpu",
    )
    global_rng_after_second = torch.random.get_rng_state().clone()

    assert torch.equal(global_rng_before, global_rng_after_first)
    assert torch.equal(global_rng_before, global_rng_after_second)
    assert first.seed_streams == second.seed_streams
    assert first.history_path.read_bytes() == second.history_path.read_bytes()
    with first.history_path.open(encoding="utf-8") as stream:
        assert [entry["update"] for entry in json.load(stream)["validation"]] == [2, 3]
    first_checkpoint = load_sfpd_checkpoint(
        first.checkpoint_path,
        expected_train_data_digest=train_data_digest(bank),
    )
    second_checkpoint = load_sfpd_checkpoint(
        second.checkpoint_path,
        expected_train_data_digest=train_data_digest(bank),
    )
    for state_name in ("raw_state", "ema_state"):
        assert (
            first_checkpoint[state_name].keys()
            == second_checkpoint[state_name].keys()
        )
        for name, first_value in first_checkpoint[state_name].items():
            torch.testing.assert_close(
                first_value,
                second_checkpoint[state_name][name],
                rtol=0.0,
                atol=0.0,
            )


def test_training_rejects_nonfinite_model_loss_before_writing_checkpoint(
    tmp_path, monkeypatch
):
    import env.train_stage_b as training_module

    class NonFiniteVelocityMLP(training_module.SFPDVelocityMLP):
        def forward(self, sample, timestep, global_cond):
            finite = super().forward(sample, timestep, global_cond)
            return finite * torch.tensor(float("nan"), dtype=torch.float32)

    monkeypatch.setattr(training_module, "SFPDVelocityMLP", NonFiniteVelocityMLP)
    bank = _small_bank(seed=26)

    with pytest.raises(FloatingPointError, match="loss"):
        train_sfpd(
            bank,
            tmp_path,
            config=_short_config(max_updates=1, validation_interval=1),
            root_seed=27,
            device="cpu",
        )
    assert not (tmp_path / "sfpd_best.pt").exists()
