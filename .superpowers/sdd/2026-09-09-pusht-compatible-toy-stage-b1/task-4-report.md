# Task 4 report: toy conditional velocity MLP

## Implementation

- Added `env/models.py` with `LocalTimeFeatures`, using the nine features `[t, sin(2πft), cos(2πft)]` for frequencies `1, 2, 4, 8`.
- Added `SFPDVelocityMLP`, a float32 conditional MLP accepting `sample [B,1,2]`, `timestep [B]`, and `global_cond [B,6]`, and returning velocity `[B,1,2]`.
- Forward validation covers tensor dtype, exact shapes, matching batch sizes, finite values, timestep range `[0,1]`, and common input/model device.
- Added focused tests in `env/tests/test_models.py` covering output/features and all validation paths.

## TDD commands and outputs

RED:

```text
UV_CACHE_DIR=/tmp/sfpd-uv-cache uv run pytest env/tests/test_models.py -v
collected 0 items / 1 error
ModuleNotFoundError: No module named 'env.models'
```

GREEN:

```text
UV_CACHE_DIR=/tmp/sfpd-uv-cache uv run pytest env/tests/test_models.py -v
11 passed, 1 skipped, 1 warning
```

Full environment suite:

```text
UV_CACHE_DIR=/tmp/sfpd-uv-cache uv run pytest env/tests -q
82 passed, 1 skipped, 3 warnings in 15.00s
```

## Files

- `env/models.py` (new)
- `env/tests/test_models.py` (new)

## Self-review

`git diff --check` is clean. Changes are limited to the requested model and tests; dataset and Drake files were not modified. The network has the requested three hidden layers with SiLU activations and a two-dimensional output projection.

## Concerns

- The default uv cache is read-only in this environment, so test commands use `UV_CACHE_DIR=/tmp/sfpd-uv-cache`.
- The device mismatch test is skipped because CUDA is unavailable; CPU validation and all other tests pass.
