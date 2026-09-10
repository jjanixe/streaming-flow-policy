from dataclasses import replace

import pytest

from env.stage_b_config import DEFAULT_STAGE_B2_CONFIG, stage_b2_config_to_dict


def test_stage_b2_defaults_match_repository_equal_sigma_contract():
    config = DEFAULT_STAGE_B2_CONFIG

    assert (config.pred_horizon, config.obs_horizon, config.action_horizon) == (
        16,
        2,
        8,
    )
    assert config.latent_dim == 2
    assert config.sigma0 == 0.1
    assert config.sigma1 == 0.1
    assert stage_b2_config_to_dict(config)["sigma0"] == 0.1


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"sigma0": -0.1}, "sigma0"),
        ({"sigma0": 0.2, "sigma1": 0.1}, "sigma0"),
        ({"latent_dim": 3}, "latent_dim"),
        ({"pred_horizon": 15}, "pred_horizon"),
        ({"rollout_count": 0}, "rollout_count"),
    ],
)
def test_stage_b2_rejects_incompatible_parameters(overrides, match):
    with pytest.raises(ValueError, match=match):
        replace(DEFAULT_STAGE_B2_CONFIG, **overrides)


@pytest.mark.parametrize("field", ("sigma0", "sigma1"))
def test_stage_b2_rejects_nonfinite_sigma(field):
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_STAGE_B2_CONFIG, **{field: float("nan")})
