# Stage A Toy Environment Design

## Goal

Implement Stage A of the preference-guided streaming-policy toy example described in
2026-09-07-streaming-preference-toy-design.md. The result is a reproducible Gym 0.26
environment, analytic demonstration/data pipeline, analytic probability-flow and score
backend, ODE/SDE samplers, artifacts, and tests. Learned networks, preference fitting,
and STEG are outside this stage.

All floating-point arrays and Torch computations use float32. String identifiers and
integer metadata such as RNG seeds and step indices keep their natural non-floating
dtypes. Tests validate the same floating-point regime that later learned models will use.

## Location and package integration

Create a top-level env/ Python package:

    env/
      __init__.py
      config.py
      environment.py
      demonstrations.py
      features.py
      analytic_fields.py
      samplers.py
      artifacts.py
      run_stage_a.py
      README.md
      tests/

env.__init__ registers PointReach2DPreference-v0. Both of these forms must work:

    import env
    gym.make("PointReach2DPreference-v0")
    gym.make("env:PointReach2DPreference-v0")

Add env to setuptools package discovery. Add pytest as a uv development dependency;
do not alter existing runtime dependency choices.

## Configuration

Use frozen Python dataclasses as the source of truth and avoid adding a YAML parser.
Defaults reproduce the toy document:

- start (-1, 0), goal (1, 0), physical duration 2.0 s
- horizon 64, initial standard deviation 0.04, goal tolerance 0.10
- train/validation/test trajectories per mode 128/32/32
- narrow amplitude [0.22, 0.32], wide amplitude [0.48, 0.58]
- contraction rate k=1.0, diffusion coefficient multiplier kappa=4.0
- feature y-scale 0.35
- root seed 0

The Stage A runner writes the fully resolved dataclass tree to JSON.

## Gym environment

PointReach2DPreferenceEnv follows the Gym 0.26 API:

- observation: float32 [x, y, t]
- action: float32 [next_x, next_y]
- reset(*, seed=None, options=None) -> (observation, info)
- step(action) -> (observation, 0.0, terminated, truncated, info)

Position spaces are unbounded because the visualization bounds are not walls. The time
component is bounded by [0, 1]. The action is followed perfectly without clipping.
Gaussian initialization samples once from N(start, sigma0^2 I) using Gym's seeded RNG.
options={"center_init": True} starts exactly at the nominal start.

The environment never terminates early for reaching the goal. After exactly 64 actions,
terminated=True and truncated=False. info exposes the step index, normalized time,
goal error, success, numerical-failure status, whether the position is outside the
visualization range, and the maximum excursion accumulated so far. Calling step after
termination is an error.

A finite action with any magnitude is accepted. A wrong-shaped action raises ValueError.
A NaN or infinite action preserves the last finite state, marks a numerical failure, and
immediately returns terminated=True, truncated=False, and success=False.

## Demonstrations and artifacts

Generate the four balanced modes upper-narrow, upper-wide, lower-narrow, and
lower-wide. Use the analytic quintic progress and bump functions from the source design
on a shared grid of 65 points. Position and derivative are evaluated directly in
float32; no numerical derivative is stored.

Derive independent split and trajectory RNGs from a NumPy SeedSequence. IDs include
split, mode, and within-mode index, for example train:upper-narrow:000. Store each
trajectory's replay seed. The three splits therefore have disjoint IDs and independently
sampled amplitudes while remaining reproducible from the root seed.

Write compressed NPZ files with allow_pickle=False compatibility:

- demonstration bank: ID, split, mode, sign, amplitude, replay seed, time, position,
  derivative
- feature normalizer: mu, population standard deviation (ddof=0) d, s_y, and a
  SHA-256 digest of canonical train data
- diagnostic rollouts: initial states and ODE/SDE position sequences

The feature normalizer is fit only from train trajectories. Full, prefix, and future
features share a single per-step contribution (f(a) - mu) / d * dt, ensuring that a
full feature equals its prefix plus future parts.

## Analytic field and samplers

The analytic backend owns only the train trajectory bank. Its public methods accept
actions [B, 2] and times [B], preserve torch.float32, and return:

- mixture log density [B]
- responsibilities [B, M]
- full probability-flow velocity [B, 2]
- score [B, 2]
- base SDE drift velocity_pf + epsilon * score

Responsibilities are computed with log-softmax. The implementation validates shapes,
finite values, and times in [0, 1]. It remains differentiable with respect to actions.

samplers.py contains one Euler ODE integrator and one Euler-Maruyama SDE integrator.
Both use the same backend and return the initial state plus all 64 updates. SDE randomness
comes from an explicit seeded torch.Generator; no module-global RNG state is consumed.
Initialization supports Gaussian and centered modes.

## Stage A runner and diagnostics

The command

    uv run python -m env.run_stage_a --output-dir env/artifacts --seed 0

generates data and normalizer artifacts, runs analytic ODE/SDE rollouts, and writes a
diagnostic JSON file. Diagnostics compare empirical marginals against direct samples from
the target Gaussian mixture at normalized times 0.25, 0.5, 0.75, and 1.0.

For each sampler and time, report mean L2 error and covariance Frobenius error. At the
midpoint, report four-mode occupancy and an other fraction. Midpoint classification uses
the sign of y, an other band abs(y) < 0.12, and a narrow/wide boundary abs(y) = 0.40.
With 4,096 default rollouts, Stage A accepts mean and covariance errors no larger than
0.03, each intended mode within 0.08 of probability 0.25, and other <= 0.02.
Diagnostics and tests use deterministic, separately derived random streams.

## Tests and acceptance

Use pytest through the repository's uv environment. Tests cover:

- Gym registration, environment checking, seeded Gaussian and centered resets, direct
  position following, horizon termination, no clipping, and numerical failure
- all path endpoints and expected tensor/array dtypes and shapes
- analytic derivatives against float32 finite differences at interior times
- split balance, ID isolation, amplitude ranges, and deterministic replay
- train-only normalization and full-feature decomposition
- analytic score against Torch autograd of mixture log density
- single-tube contraction against exp(-k t), allowing Euler discretization error
- ODE/SDE shapes, dtype, explicit-RNG reproducibility, marginal diagnostics, and mode
  coverage
- NPZ round trips and train-data digest consistency

Float32-aware tolerances are scale-specific: exact algebraic invariants use approximately
1e-6, derivative and score comparisons use relative tolerances around 1e-3, and the
64-step Euler contraction allows 2% relative error. A tolerance may be tightened after
observing stable results, but it must not be loosened merely to hide a systematic error.

## Non-goals

Stage A does not add neural networks, base training, learned denoisers, preference data,
utility fitting, STEG, rendering, obstacles, action clipping, or early goal termination.
It does not refactor or depend on the existing Drake-based toy implementation.
