# Preference fitting backtracking step-scale fix

## Scope and provenance

The fix changes only the backtracking termination in `env/preference.py` and
adds one real-code regression in `env/tests/test_preference.py`.

The post-SVD module was preserved before editing at
`/tmp/preference-before-step-fix.py`:

```text
before SHA-256: a5e758104a0f7aa0b177c3cd1a7b381f341a0b1db8f9724fc4f5e418bdd84902
after  SHA-256: 8ac59448c4393ea882ae444ff199695aa1944f9d0c5bfcf5fa6a2423444cf873
```

The scoped source/test diff is saved at
`/tmp/preference-fitting-step-fix.diff`. No commit was created.

## Root cause

For finite float32 input

```python
x = np.array(
    [[1e16, 1e16], [-1e16, -1e16], [1e16, -1e16]],
    dtype=np.float32,
)
y = np.array([1, -1, 1], dtype=np.int8)
prior = np.array([-1, 1], dtype=np.float32)
```

the first regularized SVD Newton direction is approximately
`[6.6784e16, -6.6784e16]`. Factors below float64 machine epsilon can therefore
still move the candidate by order-one amounts. The old absolute
`step_size > 2**-52` condition ended a later line search before it reached a
useful candidate, returning objective `5764528.4173`. The independently
calculated objective at zero is only `3*log(2) + 0.1 = 2.179441541679836`.

## Test-first evidence and fix

The new test
`test_backtracking_allows_subepsilon_factor_when_candidate_still_changes`
failed before the source change:

```text
assert 5764528.417323967 < 2.179441541679836
1 failed, 1 warning in 0.25s
```

The backtracking loop now checks at most 1075 factors, covering every binary64
power from `2**0` through the smallest positive subnormal `2**-1074`. It exits
early when adding the scaled direction no longer changes the candidate's
binary64 representation. This removes the invalid absolute epsilon cutoff
while retaining a finite bound and a concrete progress criterion.

The repaired reproducer returns:

```text
weights=[5.4400928e-15, 1.5392942e-15] (float32)
objective=0.1000000000000004
gradient_norm=0.02059599874020674
converged=False
iterations=100
feature_rank=2
```

The test independently recomputes the objective and gradient at the returned
float32 weights. Both diagnostics match, and `converged=False` honestly
reflects that `0.020596 > 1e-5`.

## Verification

Focused regression after the fix:

```text
UV_CACHE_DIR=/tmp/streaming-policy-uv-cache uv run --frozen pytest \
  env/tests/test_preference.py::test_backtracking_allows_subepsilon_factor_when_candidate_still_changes -q

1 passed, 1 warning in 0.25s
```

Full preference tests:

```text
UV_CACHE_DIR=/tmp/streaming-policy-uv-cache uv run --frozen pytest \
  env/tests/test_preference.py -q

............................................                             [100%]
44 passed, 1 warning in 0.35s
```

The warning is the existing `np.bool8` deprecation in `env/__init__.py`.

All four canonical profiles at budgets 5/10/20/40 were refit from
`env/artifacts/preference/pilot-seed0/preferences.npz` using both the preserved
pre-fix module and the repaired module. All 16 were bitwise identical to each
other and to stored weights:

```text
all_16_bitwise_equal=True
maximum absolute difference=0.0
```

Canonical fitted weights did not change.
