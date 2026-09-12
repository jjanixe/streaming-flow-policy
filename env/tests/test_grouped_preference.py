import numpy as np
import pytest

from env.grouped_preference import (
    GroupedPreference,
    build_grouped_queries,
    classify_modes,
    fit_grouped_preference,
    grouped_features,
    grouped_metrics,
    grouped_probabilities,
    grouped_utility,
    synthetic_grouped_users,
)
from env.demonstrations import evaluate_path


def _preference(direction=0.8, width=0.3, alpha=0.6):
    return GroupedPreference(
        weights=np.array([[direction, 1.0 - direction], [width, 1.0 - width]]),
        alpha=np.array([alpha, 1.0 - alpha]),
    )


def test_grouped_features_use_post_initial_path_and_complementary_bounded_values():
    positions = np.zeros((2, 65, 2), dtype=np.float64)
    positions[0, 0, 1] = -1000.0  # The anchor must not affect the result.
    positions[0, 1:, 1] = 0.35 * np.arctanh(0.5)
    positions[1, 1:, 1] = -0.35 * np.arctanh(0.5)

    features = grouped_features(positions)

    np.testing.assert_allclose(
        features,
        [[[0.75, 0.25], [0.25, 0.75]], [[0.25, 0.75], [0.25, 0.75]]],
        atol=1e-14,
    )
    assert features.dtype == np.float64
    assert np.all((features >= 0.0) & (features <= 1.0))
    np.testing.assert_allclose(features.sum(axis=-1), 1.0)


def test_grouped_features_swap_upper_and_lower_only_swaps_direction_entries():
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(4, 65, 2))
    reflected = positions.copy()
    reflected[..., 1] *= -1.0

    actual = grouped_features(reflected)
    original = grouped_features(positions)

    np.testing.assert_allclose(actual[:, 0], original[:, 0, ::-1], atol=1e-14)
    np.testing.assert_allclose(actual[:, 1], original[:, 1], atol=1e-14)


def test_mode_features_rank_each_requested_mode_first_for_all_eight_profiles():
    times = np.linspace(0.0, 1.0, 65, dtype=np.float32)
    paths = np.stack([
        evaluate_path(sign, amplitude, times)[0]
        for sign, amplitude in ((1, .27), (1, .53), (-1, .27), (-1, .53), (1, 0))
    ])

    features = grouped_features(paths, feature_kind="mode")

    targets = ["upper_narrow", "upper_wide", "lower_narrow", "lower_wide"]
    for name, user in synthetic_grouped_users().items():
        target = targets.index(name.rsplit("_", 1)[0])
        utility = grouped_utility(features, user)
        assert utility.argmax() == target
        assert utility[target] > np.max(np.delete(utility, target))
    np.testing.assert_array_equal(features[-1], np.zeros((2, 2)))
    assert features.dtype == np.float64


def test_mode_classifier_and_features_share_exact_midpoint_boundaries():
    midpoint_y = np.array([
        np.nextafter(.12, -np.inf), .12, np.nextafter(.4, -np.inf), .4,
        np.nextafter(-.12, np.inf), -.12, np.nextafter(-.4, np.inf), -.4, 0.,
    ])
    paths = np.zeros((len(midpoint_y), 65, 2), dtype=np.float64)
    paths[:, 32, 1] = midpoint_y

    np.testing.assert_array_equal(
        classify_modes(midpoint_y),
        np.array([4, 0, 0, 1, 4, 2, 2, 3, 4]),
    )
    np.testing.assert_array_equal(
        grouped_features(paths, feature_kind="mode"),
        np.array([
            [[0, 0], [0, 0]],
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
            [[1, 0], [1, 0]],
            [[0, 0], [0, 0]],
            [[0, 1], [0, 1]],
            [[0, 1], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, 0], [0, 0]],
        ], dtype=np.float64),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mode_classifier_rejects_nonfinite_midpoints(value):
    with pytest.raises(ValueError, match="finite"):
        classify_modes(np.array([value]))


@pytest.mark.parametrize(
    "weights,alpha",
    [
        (np.ones((2, 2), dtype=np.float32) / 2, np.array([0.5, 0.5])),
        (np.array([[0.5, 0.5], [-0.1, 1.1]]), np.array([0.5, 0.5])),
        (np.array([[0.4, 0.4], [0.5, 0.5]]), np.array([0.5, 0.5])),
        (np.ones((2, 2)) / 2, np.array([0.4, 0.4])),
        (np.ones((2, 2)) / 2, np.array([np.nan, np.nan])),
    ],
)
def test_grouped_preference_rejects_non_float64_or_invalid_simplexes(weights, alpha):
    with pytest.raises(ValueError):
        GroupedPreference(weights=weights, alpha=alpha)


def test_grouped_utility_accepts_zero_failure_features():
    features = np.zeros((3, 2, 2), dtype=np.float64)
    assert np.array_equal(grouped_utility(features, _preference()), np.zeros(3))


def test_scoped_probabilities_ignore_alpha_and_match_hand_computed_logits():
    delta = np.array(
        [
            [[0.4, -0.4], [9.0, -9.0]],
            [[7.0, -7.0], [-0.25, 0.25]],
            [[0.4, -0.4], [-0.25, 0.25]],
        ],
        dtype=np.float64,
    )
    scopes = np.array(["direction", "width", "overall"])
    first = _preference(alpha=0.9)
    second = _preference(alpha=0.1)

    p_first = grouped_probabilities(delta, scopes, first, rho=2.0)
    p_second = grouped_probabilities(delta, scopes, second, rho=2.0)

    expected_direction = 1.0 / (1.0 + np.exp(-2.0 * 0.24))
    expected_width = 1.0 / (1.0 + np.exp(-2.0 * 0.1))
    assert p_first[0] == pytest.approx(expected_direction)
    assert p_first[1] == pytest.approx(expected_width)
    np.testing.assert_allclose(p_first[:2], p_second[:2])
    assert p_first[2] != pytest.approx(p_second[2])


def test_overall_probability_logit_matches_gibbs_utility_difference():
    preference = _preference(direction=0.7, width=0.2, alpha=0.65)
    options = np.array(
        [
            [[[0.9, 0.1], [0.2, 0.8]], [[0.3, 0.7], [0.6, 0.4]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.7, 0.3], [0.1, 0.9]]],
        ],
        dtype=np.float64,
    )
    delta = options[:, 0] - options[:, 1]
    beta = 16.0
    probabilities = grouped_probabilities(
        delta, np.array(["overall", "overall"]), preference, rho=beta
    )
    utility_delta = grouped_utility(options[:, 0], preference) - grouped_utility(
        options[:, 1], preference
    )

    np.testing.assert_allclose(
        np.log(probabilities / (1.0 - probabilities)), beta * utility_delta, atol=1e-12
    )


def test_query_bank_has_nested_prefixes_and_fixed_scope_pattern():
    short = build_grouped_queries(123, count=13)
    long = build_grouped_queries(123, count=40)
    repeated = build_grouped_queries(123, count=40)

    assert long["trajectories"].shape == (40, 2, 65, 2)
    assert long["ids"].shape == (40, 2)
    assert long["scopes"].tolist()[:10] == [
        "direction", "width", "overall", "direction", "width",
        "overall", "direction", "width", "overall", "overall",
    ]
    assert {scope: int(np.sum(long["scopes"] == scope)) for scope in np.unique(long["scopes"])} == {
        "direction": 12,
        "width": 12,
        "overall": 16,
    }
    np.testing.assert_array_equal(short["trajectories"], long["trajectories"][:13])
    np.testing.assert_array_equal(short["ids"], long["ids"][:13])
    np.testing.assert_array_equal(long["trajectories"], repeated["trajectories"])


def test_synthetic_users_are_the_eight_requested_simplex_profiles():
    users = synthetic_grouped_users()

    assert list(users) == [
        "upper_narrow_direction", "upper_narrow_width",
        "upper_wide_direction", "upper_wide_width",
        "lower_narrow_direction", "lower_narrow_width",
        "lower_wide_direction", "lower_wide_width",
    ]
    upper_narrow_direction = users["upper_narrow_direction"]
    np.testing.assert_allclose(
        upper_narrow_direction.weights, np.array([[0.9, 0.1], [0.1, 0.9]])
    )
    np.testing.assert_array_equal(upper_narrow_direction.alpha, np.array([0.8, 0.2]))
    np.testing.assert_allclose(users["lower_wide_width"].weights,
                               np.array([[0.1, 0.9], [0.9, 0.1]]))
    np.testing.assert_array_equal(users["lower_wide_width"].alpha, np.array([0.2, 0.8]))


def _fit_data(preference, *, include_scoped=True, seed=8):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.7, 0.7, size=(6000, 2, 2))
    base[..., 1] = -base[..., 0]  # Real feature differences are complementary.
    if include_scoped:
        scopes = np.resize(np.array(["direction", "width", "overall"]), len(base))
    else:
        scopes = np.full(len(base), "overall")
    probabilities = grouped_probabilities(base, scopes, preference)
    labels = np.where(rng.random(len(base)) < probabilities, 1, -1)
    return base, labels, scopes


def test_fit_recovers_synthetic_population_and_reports_local_identifiability():
    truth = synthetic_grouped_users()["upper_narrow_direction"]
    delta, labels, scopes = _fit_data(truth)

    fit = fit_grouped_preference(delta, labels, scopes, l2=0.01, max_steps=400)

    np.testing.assert_allclose(fit.preference.weights[:, 0], truth.weights[:, 0], atol=0.08)
    assert fit.preference.alpha[0] == pytest.approx(truth.alpha[0], abs=0.1)
    assert fit.converged
    assert fit.projected_gradient_norm < 1e-4
    assert fit.jacobian_rank == 3
    assert fit.scope_counts == {"direction": 2000, "width": 2000, "overall": 2000}
    assert fit.iterations > 0
    assert np.isfinite(fit.objective)


def test_overall_only_fit_reports_nonidentifiability():
    truth = synthetic_grouped_users()["upper_wide_width"]
    delta, labels, scopes = _fit_data(truth, include_scoped=False)

    fit = fit_grouped_preference(delta, labels, scopes, l2=0.01)

    assert fit.jacobian_rank <= 2
    assert fit.scope_counts == {"direction": 0, "width": 0, "overall": 6000}


def test_iteration_limit_is_reported_as_nonconvergence():
    truth = synthetic_grouped_users()["upper_wide_direction"]
    delta, labels, scopes = _fit_data(truth, seed=19)

    fit = fit_grouped_preference(delta, labels, scopes, max_steps=1)

    assert not fit.converged
    assert fit.iterations <= 1
    assert fit.projected_gradient_norm > 1e-5


def test_positive_regularization_keeps_unobserved_scope_coordinates_at_uniform():
    delta = np.zeros((20, 2, 2), dtype=np.float64)
    delta[:, 0, 0] = 0.4
    delta[:, 0, 1] = -0.4
    labels = np.ones(20, dtype=int)
    scopes = np.full(20, "direction")

    fit = fit_grouped_preference(delta, labels, scopes, l2=0.2)

    assert fit.preference.weights[0, 0] > 0.5
    assert fit.preference.weights[1, 0] == pytest.approx(0.5, abs=1e-10)
    assert fit.preference.alpha[0] == pytest.approx(0.5, abs=1e-10)
    assert fit.scope_counts == {"direction": 20, "width": 0, "overall": 0}


def test_probabilities_and_metrics_stay_finite_for_large_logits():
    preference = _preference()
    delta = np.array(
        [
            [[1e300, -1e300], [0.0, 0.0]],
            [[-1e300, 1e300], [0.0, 0.0]],
            [[0.0, 0.0], [-1e300, 1e300]],
        ],
        dtype=np.float64,
    )
    scopes = np.array(["direction", "direction", "width"])
    labels = np.array([1, -1, 1])

    probabilities = grouped_probabilities(delta, scopes, preference)
    metrics = grouped_metrics(delta, labels, scopes, preference)

    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(probabilities, np.array([1.0, 0.0, 1.0]))
    assert metrics["aggregate"]["count"] == 3
    assert metrics["aggregate"]["cross_entropy"] == pytest.approx(0.0)
    assert metrics["aggregate"]["accuracy"] == 1.0
    assert metrics["overall"] == {"count": 0, "cross_entropy": None, "accuracy": None}


def test_unrepresentable_finite_rho_is_rejected_instead_of_returning_nan_diagnostics():
    delta = np.array([[[1.0, -1.0], [1.0, -1.0]]], dtype=np.float64)

    with pytest.raises(ValueError, match="representable float64"):
        fit_grouped_preference(
            delta,
            np.array([1]),
            np.array(["overall"]),
            rho=np.finfo(np.float64).max,
            max_steps=2,
        )


def test_extreme_finite_l2_cannot_escape_as_nonfinite_fit_diagnostics():
    delta = np.array(
        [
            [[0.4, -0.4], [0.2, -0.2]],
            [[-0.3, 0.3], [-0.1, 0.1]],
            [[0.2, -0.2], [-0.4, 0.4]],
        ],
        dtype=np.float64,
    )

    try:
        fit = fit_grouped_preference(
            delta,
            np.array([1, -1, 1]),
            np.array(["direction", "width", "overall"]),
            l2=np.finfo(np.float64).max,
            max_steps=3,
        )
    except ValueError as error:
        assert "representable float64" in str(error)
    else:
        assert np.isfinite(fit.objective)
        assert np.isfinite(fit.projected_gradient_norm)
        assert np.isfinite(fit.start_objectives).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda: grouped_features(np.zeros((64, 2))),
        lambda: grouped_features(np.zeros((65, 2)), y_scale=0),
        lambda: grouped_features(np.zeros((65, 2)), feature_kind="other"),
        lambda: grouped_probabilities(np.zeros((1, 2, 2)), ["bad"], _preference()),
        lambda: fit_grouped_preference(np.zeros((0, 2, 2)), [], []),
        lambda: fit_grouped_preference(np.zeros((1, 2, 2)), [0], ["overall"]),
        lambda: fit_grouped_preference(np.zeros((1, 2, 2)), [1], ["overall"], rho=0),
        lambda: fit_grouped_preference(np.zeros((1, 2, 2)), [1], ["overall"], l2=-1),
        lambda: fit_grouped_preference(np.zeros((1, 2, 2)), [1], ["overall"], max_steps=1.5),
        lambda: build_grouped_queries(-1),
        lambda: build_grouped_queries(1, count=0),
    ],
)
def test_invalid_grouped_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()
