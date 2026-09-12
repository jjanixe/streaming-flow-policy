# Preference fitting numerical robustness fix

## Scope and provenance

Only these owned source files were changed:

- `env/preference.py`
- `env/tests/test_preference.py`

Before editing, the original module was copied to
`/tmp/preference-before-robustness.py`. The source and saved copy matched with
SHA-256:

```text
4cdad8bb87e57572abb81d788dd4d3481d3d4a232f5db0f9571473957e4c0306
```

No commit was created.

## Root cause

The reported input reproduces the failure deterministically:

```python
delta_features = np.array(
    [[1e8, 1e8], [-1e8, -1e8]], dtype=np.float32
)
labels = np.array([1, -1], dtype=np.int8)
fit_bradley_terry(delta_features, labels)
```

At the initial iterate, the explicitly formed data Hessian is
`[[5e15, 5e15], [5e15, 5e15]]`. Float64 spacing at each diagonal entry is
`1.0`, so adding the `0.1` ridge is rounded away. The computed eigenvalues are
`[0, 1e16]`, and `np.linalg.solve` raises `LinAlgError: Singular matrix` even
though the regularized objective is mathematically nonsingular.

## Test-first evidence

The new real-code regression test is
`test_collinear_large_features_preserve_ridge_and_report_returned_gradient`.
Before the implementation change it failed at `np.linalg.solve` with
`LinAlgError: Singular matrix` (1 failed). The test checks finite float32
weights, finite objective and gradient diagnostics, feature rank 1, and independently
recomputes the gradient at the returned float32 weights so the convergence flag
must be honest.

## Fix

The normal Newton path remains the existing direct solve. If and only if that
solve raises `LinAlgError`, the fitter now computes the Newton direction from
an SVD of the weighted design. Likelihood-gradient components are evaluated as
`singular_value * U.T @ scaled_residual`, avoiding an unstable projection of a
large row-space gradient into the numerical null space. Ridge denominators are
then applied as `singular_value**2 + l2`, so the `0.1` penalty is retained
without adding it to `5e15`.

For the reproducer the returned fit is:

```text
weights=[1.5600655e-07, 1.5600655e-07] (float32)
objective=5.872924763439068e-14
gradient_norm=7.939315279807463e-06
converged=True
iterations=30
feature_rank=1
```

The regression's independent gradient calculation matches the reported norm;
`7.9393e-6 <= 1e-5`, so convergence is truthful for the returned weights.

## Verification

Focused regression:

```text
UV_CACHE_DIR=/tmp/streaming-policy-uv-cache uv run --frozen pytest \
  env/tests/test_preference.py::test_collinear_large_features_preserve_ridge_and_report_returned_gradient -q

1 passed, 1 warning in 0.20s
```

Full preference-fitting tests after the final cleanup:

```text
UV_CACHE_DIR=/tmp/streaming-policy-uv-cache uv run --frozen pytest \
  env/tests/test_preference.py -q

...........................................                              [100%]
43 passed, 1 warning in 0.30s
```

The warning is the existing `np.bool8` deprecation in `env/__init__.py`.

## Canonical fit stability

Using `env/artifacts/preference/pilot-seed0/preferences.npz`, all four profiles
(`upper_narrow`, `upper_wide`, `lower_narrow`, `lower_wide`) at budgets
5/10/20/40 were refit with both the saved pre-fix module and the current
module. Each of the 16 comparisons satisfied all three checks:

```text
old == stored: True
new == stored: True
old == new: True
max absolute weight difference: 0.0
all_16_bitwise_equal=True
```

Thus the normal-data canonical fitted weights are bitwise unchanged and do not
need regeneration because of this source fix.
