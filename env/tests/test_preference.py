import numpy as np
import pytest

from env import preference as preference_module
from env.preference import fit_bradley_terry


def test_fit_recovers_axis_preferences_and_is_invariant_to_pair_swap():
    # Removing the label sign or applying it twice breaks both the learned signs
    # and the A/B representation invariant checked here.
    delta_features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    labels = np.array([1, -1, -1, 1], dtype=np.int8)

    fit = fit_bradley_terry(delta_features, labels)
    swapped = fit_bradley_terry(-delta_features, -labels)

    assert fit.weights[0] > 0.0
    assert fit.weights[1] < 0.0
    np.testing.assert_allclose(fit.weights, swapped.weights, atol=1e-5)


def test_probabilities_are_stable_for_extreme_logits():
    # A direct exp(-logit) implementation overflows at the negative extreme.
    # The last row also catches float32 dot-product overflow before cancellation.
    delta_features = np.array(
        [[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [1.0e30, -1.0e30]],
        dtype=np.float32,
    )
    weights = np.array([1.0e30, 1.0e30], dtype=np.float32)

    probabilities = preference_module.preference_probabilities(
        delta_features,
        weights,
    )

    np.testing.assert_array_equal(
        probabilities,
        np.array([1.0, 0.5, 0.0, 0.5], dtype=np.float32),
    )
    assert probabilities.dtype == np.float32


def test_metrics_report_mean_log_loss_and_tie_positive_accuracy():
    # Summing loss instead of averaging, or treating a 0.5 tie as negative,
    # changes these independently hand-calculated results.
    delta_features = np.array([[2.0], [-2.0], [0.0]], dtype=np.float32)
    labels = np.array([1, -1, 1], dtype=np.int8)

    metrics = preference_module.preference_metrics(
        delta_features,
        labels,
        np.array([1.0], dtype=np.float32),
    )

    assert metrics["log_loss"] == pytest.approx(0.31566775, abs=1e-7)
    assert metrics["accuracy"] == 1.0


def test_metrics_keep_extreme_misclassification_loss_finite():
    # Computing log probabilities after a saturated sigmoid would turn the
    # confidently wrong example into log(0) and an infinite reported loss.
    metrics = preference_module.preference_metrics(
        np.array([[1.0], [-1.0]], dtype=np.float32),
        np.array([-1, -1], dtype=np.int8),
        np.array([1.0e30], dtype=np.float32),
    )

    assert np.isfinite(metrics["log_loss"])
    assert metrics["log_loss"] == pytest.approx(5.0e29, rel=1e-7)
    assert metrics["accuracy"] == 0.5


def test_fit_rejects_non_matrix_features():
    # Without an explicit dimensionality check the solver fails later with an
    # unrelated indexing error, obscuring the public input contract.
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_bradley_terry(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1, -1], dtype=np.int8),
        )


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.array([[1]], dtype=np.int8), "one-dimensional"),
        (np.array([1], dtype=np.int8), "same number"),
        (np.array([0, 1], dtype=np.int8), r"-1 or \+1"),
        (np.array([np.nan, 1.0], dtype=np.float32), "finite"),
    ],
)
def test_fit_rejects_invalid_labels(labels, message):
    # Each malformed label representation would otherwise be broadcast or
    # interpreted as a different statistical model.
    with pytest.raises(ValueError, match=message):
        fit_bradley_terry(
            np.array([[1.0], [-1.0]], dtype=np.float32),
            labels,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_features(bad_value):
    # Nonfinite evidence must not leak into a seemingly valid fit diagnostic.
    with pytest.raises(ValueError, match="delta_features.*finite"):
        fit_bradley_terry(
            np.array([[bad_value], [0.0]], dtype=np.float32),
            np.array([1, -1], dtype=np.int8),
        )


@pytest.mark.parametrize("l2", [0.0, -0.1, np.nan, np.inf, "0.1", None])
def test_fit_requires_finite_positive_regularization(l2):
    # A non-positive penalty removes the promised unique regularized fit and
    # can make Newton's Hessian singular on low-rank evidence.
    with pytest.raises(ValueError, match="l2.*finite and positive"):
        fit_bradley_terry(
            np.array([[1.0], [-1.0]], dtype=np.float32),
            np.array([1, -1], dtype=np.int8),
            l2=l2,
        )


@pytest.mark.parametrize(
    "prior_mean",
    [
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_fit_rejects_invalid_prior(prior_mean):
    # A prior in a different feature space, or with nonfinite entries, cannot
    # define the center of the L2 penalty.
    with pytest.raises(ValueError, match="prior_mean"):
        fit_bradley_terry(
            np.array([[1.0]], dtype=np.float32),
            np.array([1], dtype=np.int8),
            prior_mean=prior_mean,
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_steps", -1),
        ("max_steps", 1.5),
        ("max_steps", True),
        ("tolerance", 0.0),
        ("tolerance", -1.0),
        ("tolerance", np.nan),
        ("tolerance", np.inf),
        ("tolerance", "1e-5"),
        ("tolerance", None),
    ],
)
def test_fit_rejects_invalid_solver_controls(keyword, value):
    # Invalid iteration and stopping controls must fail before they can create
    # misleading convergence diagnostics.
    with pytest.raises(ValueError, match=keyword):
        fit_bradley_terry(
            np.array([[1.0]], dtype=np.float32),
            np.array([1], dtype=np.int8),
            **{keyword: value},
        )


def test_zero_step_budget_reports_nonconvergence_without_hiding_diagnostics():
    # Reporting convergence merely because the loop ended would hide an
    # unoptimized fit from experiment artifacts.
    fit = fit_bradley_terry(
        np.array([[1.0]], dtype=np.float32),
        np.array([1], dtype=np.int8),
        max_steps=0,
    )

    assert not fit.converged
    assert fit.iterations == 0
    assert fit.gradient_norm == pytest.approx(0.5)
    assert fit.objective == pytest.approx(np.log(2.0), abs=1e-7)
    assert fit.feature_rank == 1


def test_balanced_contradictory_labels_return_zero_prior():
    # Letting input order break the exact gradient cancellation would create a
    # preference where the evidence contains no directional signal.
    fit = fit_bradley_terry(
        np.array([[2.0], [2.0]], dtype=np.float32),
        np.array([1, -1], dtype=np.int8),
    )

    np.testing.assert_array_equal(fit.weights, np.array([0.0], dtype=np.float32))
    assert fit.converged
    assert fit.iterations == 0
    assert fit.gradient_norm == 0.0


def test_empty_evidence_returns_nonzero_prior_with_rank_zero():
    # Empty evidence must retain the caller's prior instead of inventing data or
    # silently resetting it to the default zero prior.
    prior = np.array([1.25, -0.75], dtype=np.float32)

    fit = fit_bradley_terry(
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=np.int8),
        prior_mean=prior,
    )

    np.testing.assert_array_equal(fit.weights, prior)
    assert fit.objective == 0.0
    assert fit.gradient_norm == 0.0
    assert fit.converged
    assert fit.iterations == 0
    assert fit.feature_rank == 0


def test_rank_deficiency_is_reported_while_regularization_keeps_fit_finite():
    # Reporting parameter dimension as rank would overstate identifiability;
    # omitting L2 curvature would make this collinear Newton system singular.
    delta_features = np.array(
        [[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0]],
        dtype=np.float32,
    )
    fit = fit_bradley_terry(
        delta_features,
        np.array([1, 1, -1], dtype=np.int8),
    )

    assert fit.feature_rank == 1
    assert fit.converged
    assert fit.gradient_norm <= 1e-5
    assert np.isfinite(fit.weights).all()
    assert fit.weights.dtype == np.float32


@pytest.mark.parametrize(
    ("delta_features", "weights", "message"),
    [
        (np.array([1.0], dtype=np.float32), np.array([1.0]), "two-dimensional"),
        (np.ones((2, 2), dtype=np.float32), np.array([1.0]), "shape"),
        (np.ones((2, 1), dtype=np.float32), np.array([[1.0]]), "one-dimensional"),
        (np.array([[np.nan]], dtype=np.float32), np.array([1.0]), "finite"),
        (np.array([[1.0]], dtype=np.float32), np.array([np.inf]), "finite"),
    ],
)
def test_probabilities_reject_invalid_inputs(delta_features, weights, message):
    # Shape broadcasting and nonfinite values can otherwise produce plausible
    # arrays that are not probabilities for the requested comparisons.
    with pytest.raises(ValueError, match=message):
        preference_module.preference_probabilities(delta_features, weights)


def test_metrics_reject_empty_evaluation_set():
    # NaN metrics cannot be serialized into the strict experiment artifacts and
    # conceal that no held-out comparison was evaluated.
    with pytest.raises(ValueError, match="at least one comparison"):
        preference_module.preference_metrics(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
            np.zeros(2, dtype=np.float32),
        )


def test_well_conditioned_fit_reaches_requested_gradient_tolerance():
    # Float32 objective comparisons can stall backtracking near the optimum even
    # though this regularized two-dimensional problem is well conditioned.
    rng = np.random.default_rng(0)
    delta_features = rng.normal(size=(40, 2)).astype(np.float32)
    generating_weights = np.array([2.0, -1.0], dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-(delta_features @ generating_weights)))
    labels = np.where(rng.random(40) < probabilities, 1, -1).astype(np.int8)

    fit = fit_bradley_terry(delta_features, labels)

    assert fit.converged
    assert fit.iterations < 100
    assert fit.gradient_norm <= 1e-5


def test_collinear_large_features_preserve_ridge_and_report_returned_gradient():
    # At this scale, forming X.T W X loses the 0.1 diagonal ridge to float64
    # rounding. A solver that relies only on that matrix raises Singular matrix.
    delta_features = np.array(
        [[1.0e8, 1.0e8], [-1.0e8, -1.0e8]],
        dtype=np.float32,
    )
    labels = np.array([1, -1], dtype=np.int8)

    fit = fit_bradley_terry(delta_features, labels)

    assert fit.feature_rank == 1
    assert fit.weights.dtype == np.float32
    assert np.isfinite(fit.weights).all()
    assert np.isfinite(fit.objective)
    assert np.isfinite(fit.gradient_norm)

    features64 = delta_features.astype(np.float64)
    labels64 = labels.astype(np.float64)
    weights64 = fit.weights.astype(np.float64)
    margins = labels64 * (features64 @ weights64)
    negative_probabilities = np.exp(-np.logaddexp(0.0, margins))
    gradient = (
        features64.T @ (-labels64 * negative_probabilities)
        + 0.1 * weights64
    )
    returned_gradient_norm = float(np.linalg.norm(gradient))

    assert fit.gradient_norm == pytest.approx(returned_gradient_norm, rel=1e-12)
    assert fit.converged is (returned_gradient_norm <= 1e-5)


def test_backtracking_allows_subepsilon_factor_when_candidate_still_changes():
    # A huge finite Newton direction can need a factor below 2**-52 even while
    # that factor changes the candidate by an order-one amount.
    delta_features = np.array(
        [
            [1.0e16, 1.0e16],
            [-1.0e16, -1.0e16],
            [1.0e16, -1.0e16],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, -1, 1], dtype=np.int8)
    prior = np.array([-1.0, 1.0], dtype=np.float32)

    fit = fit_bradley_terry(delta_features, labels, prior_mean=prior)

    # At zero: three log(2) likelihood terms plus 0.1 prior penalty.
    assert fit.objective < 2.179441541679836
    assert fit.feature_rank == 2
    assert fit.weights.dtype == np.float32
    assert np.isfinite(fit.weights).all()
    assert np.isfinite(fit.gradient_norm)

    features64 = delta_features.astype(np.float64)
    labels64 = labels.astype(np.float64)
    weights64 = fit.weights.astype(np.float64)
    prior64 = prior.astype(np.float64)
    margins = labels64 * (features64 @ weights64)
    losses = np.logaddexp(0.0, -margins)
    returned_objective = float(
        losses.sum() + 0.05 * np.dot(weights64 - prior64, weights64 - prior64)
    )
    negative_probabilities = np.exp(-np.logaddexp(0.0, margins))
    gradient = (
        features64.T @ (-labels64 * negative_probabilities)
        + 0.1 * (weights64 - prior64)
    )
    returned_gradient_norm = float(np.linalg.norm(gradient))

    assert fit.objective == pytest.approx(returned_objective, rel=1e-14)
    assert fit.gradient_norm == pytest.approx(returned_gradient_norm, rel=1e-12)
    assert fit.converged is (returned_gradient_norm <= 1e-5)
