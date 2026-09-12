# Task 1 review: bounded two-group features and fitting

## Verdicts

- **Spec compliance: CHANGES REQUESTED.** The required two-group formulas,
  scope semantics, query pattern, synthetic users, objective weighting,
  deterministic multistart fitting, and diagnostics are present. One numerical
  stability defect violates the finite-input and large-logit requirements.
- **Code quality: CHANGES REQUESTED.** The implementation is otherwise clear,
  well-factored, and backed by meaningful tests, but the optimizer can select a
  nonfinite result and then publish invalid convergence/identifiability
  diagnostics.

## Findings

### [P2] Extreme finite scaling corrupts fitting and its diagnostics

**Locations:** `env/grouped_preference.py:195`,
`env/grouped_preference.py:233`, `env/grouped_preference.py:361`, and
`env/tests/test_grouped_preference.py:226`.

`rho` accepts every finite positive float, and bounded complementary feature
differences are valid fitter input. At sufficiently large finite `rho`, direct
multiplication overflows the logits/Jacobian. The loss-gradient calculation
then forms products such as zero times infinity, producing NaNs. In addition,
`min(results, key=...)` does not reject nonfinite objectives; when the staged
result is first and has `fun=NaN`, Python's ordering can retain it even though
later starts have finite objectives.

A focused check using two valid complementary direction deltas
`[[1,-1],[0,0]]` and `[[-1,1],[0,0]]`, matching labels `[+1,-1]`, and
`rho=np.finfo(np.float64).max` returned:

```text
objective=7.76405003785465e-05
projected_gradient_norm=nan
jacobian_rank=0
start_objectives=(nan, 0.025, 0.075, 0.075, 0.075, 0.075, nan, nan, nan, nan)
```

That result is neither a finite convergence diagnostic nor the true local rank
of the two-row direction design. The existing large-logit test stops at
probabilities and metrics with `rho=16`; it does not exercise fitting or
scaling overflow. Make logit/Jacobian/gradient evaluation stable for every
accepted finite `rho` (or narrow the explicit validation contract if the task
requirements are changed), filter nonfinite optimizer results before selecting
a start, and add a fitting regression test that asserts finite, truthful
diagnostics.

## Confirmed behavior

- Group order is exactly direction `(upper, lower)` and width `(wide, narrow)`;
  alpha is ordered `(direction, width)`.
- `grouped_features` excludes the anchor and averages all 64 future points in
  float64 using the specified bounded complementary formulas.
- Direction/width likelihoods omit alpha; overall likelihoods use the
  alpha-weighted utility.
- The fitted objective is the sum of each nonempty scope's mean logistic loss
  plus the specified three-coordinate L2 penalty.
- The query bank uses the approved ten-query scope pattern, nested seeded
  prefixes, and the same analytic direction/width/tradeoff geometry as
  `env/run_preference.py`.
- A focused check across all eight synthetic profiles recovered all three free
  coordinates within 0.031 absolute error and reported rank 3 and convergence
  for each. The reported pytest suite was not rerun.

## Unresolved cross-task checks

- Later feedback generation must preserve the query scope on every label so
  direction/width judgments continue to exclude alpha.
- Later conditional rollouts must average full-option features over independent
  futures of the same current latent before applying the exponential tilt;
  failure continuations must contribute zero features plus their separate task
  cost.
- Later experiment integration must keep the checkpoint and normalizer frozen,
  execute actual Gym chunks, and preserve the one-anchor/eight-command,
  sixteen-point model-horizon convention.
- Later artifact/report code must retain per-scope metrics, convergence,
  projected-gradient norm, local Jacobian rank, query data, seeds, failures,
  and frozen-policy hashes without interpreting `converged` as a global-optimum
  guarantee.
