# PushT-Compatible Toy Stage B Design

## Goal

Implement and evaluate learned streaming flow policies on the existing
`single_detour` expert demonstrations. The learned policies use the same core
chunk contract as the repository's PushT policies: two observations condition a
local continuous action flow, the current position anchors the prediction, eight
new actions execute, and the policy replans.

Stage B proceeds in two gates:

1. B1 trains a deterministic action-space streaming flow policy (SFPD).
2. B2 trains the repository-style stochastic latent streaming flow policy
   (SFPS) after B1 produces a finite checkpoint and runnable rollouts.

A B1 distributional-gate miss is retained as a scientific result but does not
by itself block B2. B2 is blocked only when B1 is not numerically finite or the
shared chunk/evaluation pipeline cannot run reliably. Preference guidance
remains blocked until the B1 and B2 results have both been interpreted.

The learned implementations are toy-native, CPU-capable, reproducible, and
float32. They preserve the PushT formulations and execution semantics without
reusing the large PushT ConditionalUnet1D, Drake interpolation objects, global
random-number state, or CUDA-only training scripts.

A denoiser and score-corrected Brownian SDE are not part of this stage. They are
a later experimental backend, evaluated against the SFPD/SFPS results produced
here.

## Relationship to the existing PushT code

The PushT SFPD code turns sixteen action waypoints into a continuous local
trajectory, conditions the velocity network on two observations, and integrates
from the current action. At runtime it requests the anchor plus eight future
positions, discards the anchor, executes the eight new actions, and replans.

The PushT SFPS code samples a fresh latent value at each chunk boundary and
integrates a joint action-latent ODE. It does not train a denoiser and does not
use a score-corrected SDE.

This design preserves those contracts. It does not preserve incidental
implementation choices that are inappropriate for the toy, such as the large
UNet or a dependency on `pydrake` merely to interpolate waypoints.

In the existing code, a Drake trajectory is a
`PiecewisePolynomial.FirstOrderHold` object. It linearly interpolates consecutive
waypoints and exposes position and derivative evaluation at arbitrary local
times. It is not used for robot dynamics in this path. Stage B implements the
same first-order-hold calculation directly with Torch operations so it remains
differentiable and float32.

## Fixed chunk and observation contract

- Episode: 64 new position commands and 65 states including the initial state.
- Observation: float32 `[x, y, s]`, where `s` is normalized episode time.
- Observation history: 2 observations.
- Prediction horizon: 16 positions including the current anchor.
- Execution horizon: 8 new actions.
- Chunk-local waypoint time: `lambda_j = j / 15`.
- The chunk-local time resets at every replan.
- Episode time in the observations never resets.
- The environment action remains one absolute next position with shape `[2]`.
- The environment never receives a `[16, 2]` action.
- Every requested environment step remains subject to
  `max_step_distance=0.075`; generated actions are never silently clipped.

At global anchor index `q`, the condition and prediction are

    c_q = (o_{q-1}, o_q)
    Xi_q = (a_q, a_{q+1}, ..., a_{q+15}).

At `q=0`, `o_-1` is a copy of `o_0` with global time zero. The first predicted
position is exactly the current state and is not re-executed. The runner applies
`a_(q+1)` through `a_(q+8)`, or only the remaining valid actions at the episode
end, then refreshes the observation history.

The prediction API accepts a requested number of positions from 1 through 16.
Ordinary execution requests 9 positions (anchor plus eight actions), matching
the current PushT optimization. Diagnostics may request all 16 positions.

## Chunk dataset

Chunks are derived from the saved whole-trajectory demonstration bank. The
trajectory split is authoritative: every chunk from a train, validation, or test
trajectory inherits that split. No chunk-level reshuffling may move windows
across splits.

For each trajectory, sliding-window anchors `q=0,...,63` are available. Sliding
windows improve training coverage even though closed-loop execution replans only
at `q=0,8,...,56`. Define the chunk-local waypoint array as
`Xi_q[j] = state[min(q+j, 64)]` for `j=0,...,15`. A boolean valid mask
distinguishes real episode positions from terminal padding. Padded positions
equal the goal endpoint and their local derivative is zero. The terminal-only
`q=64` window is excluded because it contains no executable action.

The first-order-hold evaluator uses

    segment = min(floor(15 * lambda), 14)
    alpha = 15 * lambda - segment
    xi_q(lambda) = (1-alpha) * Xi_q[segment] + alpha * Xi_q[segment+1]
    d xi_q / d lambda = 15 * (Xi_q[segment+1] - Xi_q[segment]).

All interpolation, derivative, padding, and batch sampling calculations use
Torch float32. Noise and index sampling consume explicit seeded Torch
generators. Dataset artifacts retain trajectory ID, source split, anchor index,
condition, positions, and valid mask for replay and diagnosis.

## Model inputs and architecture

Both models use a small condition-aware MLP rather than ConditionalUnet1D. The
MLP has three hidden layers of width 128 with SiLU activations. Local time uses
the existing toy Fourier features with frequencies 1, 2, 4, and 8.

The condition is the raw six-dimensional concatenation of two observations.
Global episode phase therefore enters through the condition, while integration
uses chunk-local time. Mode IDs, trajectory IDs, signs, and amplitudes are never
model inputs.

Unlike the PushT scripts, Stage B initially applies no learned or dataset-wide
normalization transform. This is deliberate: toy positions and normalized
episode time already occupy fixed, order-one ranges, and retaining physical
coordinates makes the `0.075` action contract directly auditable. If training
diagnostics later justify normalization, its statistics must be fit on the train
split only, saved in the checkpoint, and treated as a documented follow-up
experiment rather than an invisible change to this baseline.

SFPD receives action `[2]`, local-time features `[9]`, and condition `[6]`, and
outputs action velocity `[2]`.

SFPS receives joint action and latent `[4]`, the same local-time features and
condition, and outputs joint velocity `[4]`. The latent dimension equals the
action dimension, matching the PushT implementation.

Model methods validate rank, final dimension, dtype, device consistency, local
time bounds, and finite values. Training and evaluation default to CPU in the
current environment but permit an explicitly selected CUDA device when one is
available.

## Stage B1: deterministic SFPD

For a sampled train chunk and local time, evaluate the first-order-hold position
and derivative, then perturb only the position:

    x = xi(lambda) + sigma * noise
    v_target = d xi / d lambda

The initial toy training noise is `sigma=0.04`. This follows the practical PushT
SFPD target: it does not add the Stage A analytic field's explicit `-k` term.
The learned conditional marginal field may still exhibit data-driven attraction,
but the implementation must not claim that it is the Stage A stabilizing field.

Inference starts at the current observed position and integrates

    da / d lambda = v_theta(a, lambda | c_q)

with a fixed-step differentiable RK4 solver. Solver substeps are configurable and
separate from output action spacing. The condition remains fixed inside a chunk.

## Stage B2: stochastic latent SFPS

SFPS uses the same conditional expert chunk. With latent `z0 ~ N(0, I)`,
independent action noise `epsilon_a0 ~ N(0, sigma0^2 I)`, and
`sigma_r = sqrt(sigma1^2 - sigma0^2)`, training samples are

    a_lambda = xi(lambda) + epsilon_a0 + sigma_r * lambda * z0
    z_lambda = (1 - (1-sigma1) * lambda) * z0 + lambda * xi(lambda)
    v_a = d xi / d lambda + sigma_r * z0
    v_z = xi(lambda) + lambda * d xi / d lambda - (1-sigma1) * z0.

The initial configuration uses `sigma0=sigma1=0.04`, so `sigma_r=0`, matching
the equal-sigma setup in the PushT experiment while using the toy's position
scale. This value is a configurable starting point, not a tuned optimum.

At each chunk boundary, inference sets `a0` to the current observed position,
samples a fresh seeded `z0`, and integrates the learned joint ODE locally. It
executes only the action component. The sampled latent is never exposed to the
Gym environment.

## Training and reproducibility

B1 and B2 train sequentially with independent model-initialization, minibatch,
training-noise, validation, rollout-initialization, and SFPS-latent random
streams derived from a root seed.

Initial training settings follow the practical PushT training loop at toy scale:

- batch size 1024
- maximum 20,000 updates per model
- validation every 1,000 updates
- AdamW learning rate `1e-4`, weight decay `1e-6`
- 500-update linear warmup followed by cosine decay
- exponential moving average with decay 0.999
- best EMA checkpoint selected by the model's validation loss

Validation batches are generated from validation trajectories with a fixed
validation seed and reused at every validation point. The test split is read
only after training and checkpoint selection are complete. Training loss need
not approach zero because a condition may support several expert futures.

Each checkpoint contains raw and EMA state dictionaries, model type and
architecture, chunk and solver configuration, optimizer settings, training
seed streams, selected update and validation loss, code format version, and the
canonical train demonstration digest. Loading rejects an incompatible model
type, architecture, or data digest.

## Streaming evaluation

Evaluation reports teacher-forced held-out field loss and closed-loop streaming
rollouts. Closed-loop evaluation calls the policy at global indices
`q=0,8,...,56`, executes absolute position commands through the Gym contract,
and stops without silently removing numerical or action-limit failures.

B1 uses Gaussian environment initialization to measure diversity arising from
different initial states. A separate centered-initialization diagnostic records
the unique deterministic result and is not required to be multimodal.

B2 evaluates both Gaussian initialization and controlled same-state diversity.
For the controlled diagnostic, every rollout starts from the same centered
state and differs only in explicit SFPS latent streams. This directly tests
whether the stochastic policy can generate more than one mode from an identical
condition.

For each model report:

- goal success rate and final goal-error distribution
- numerical and action-limit failure counts
- action-limit activation count and rate
- maximum requested step distance
- midpoint occupancy for upper/lower narrow/wide and `other`
- upper/lower symmetry and narrow/wide coverage
- per-chunk anchor error and boundary jump
- deterministic replay under identical seeds
- float32 dtype and finite-value checks

Raw generated positions are retained even when a Gym execution would terminate,
and failed trajectory indices are reported separately. This distinguishes model
geometry from the environment safety contract.

The first development gates are recorded rather than assumed. They apply to
each model separately, with the B1 same-state deterministic diagnostic exempt
from multimodal occupancy requirements:

- at least 95% Gym goal success
- no numerical failures
- all four intended modes have at least 5% occupancy
- `other` is at most 5%
- each mode is within 0.10 absolute occupancy of the balanced 0.25 reference
- action-limit activations and affected trajectories are always explicit

A failed gate remains a result and blocks preference guidance. Thresholds are
not loosened merely to make a checkpoint pass. A distributional B1 gate miss
does not prevent the planned B2 comparison, provided B1 and the shared pipeline
are finite and runnable.

## Artifacts and visualization

The Stage B runner writes under an ignored `env/artifacts/stage_b/<run_id>/`
directory:

- resolved configuration and derived seed streams
- SFPD and SFPS checkpoints
- train/validation histories
- teacher-forced and closed-loop diagnostic JSON
- raw B1/B2 rollout NPZ files
- trajectory plots colored by classified mode
- mode-occupancy comparison plots
- representative RGB-array rollout animations

The final report compares expert, B1, and B2 distributions using the same axes
and classification rules. It reports the checkpoint and data hashes used to
generate every visualization.

The canonical full run is exposed as
`uv run python -m env.run_stage_b --stage all --seed 0`. The same runner accepts
`--stage b1` and `--stage b2`, while explicit reduced update and rollout counts
support CPU smoke verification. B2 refuses to start without a compatible B1
pipeline/checkpoint record, but it need not require B1 to pass the distributional
development gates.

## Module boundaries

Stage B adds focused modules under `env/`:

- `chunk_data.py`: split-safe chunk windows, interpolation, target batches
- `models.py`: condition-aware float32 MLPs
- `sfp_policies.py`: SFPD/SFPS targets and differentiable integration
- `train_stage_b.py`: sequential training, EMA, validation, checkpoints
- `evaluate_stage_b.py`: streaming runner, metrics, mode classification
- `run_stage_b.py`: command-line orchestration and artifact writing
- `visualize_stage_b.py`: saved rollout and occupancy figures

Existing `environment.py`, `demonstrations.py`, `artifacts.py`, and Stage A
analytic modules are reused through their public contracts. Existing PushT
modules are read-only references and are not modified.

## Testing

Implementation follows test-driven development. Tests cover:

- exact chunk indices, anchor, history padding, terminal padding, and masks
- split inheritance and absence of trajectory leakage
- float32 first-order-hold values and derivatives at boundaries and interiors
- SFPD and SFPS target formulas against hand-computed examples
- model shape, dtype, finite-value, and input validation
- exact preservation of the current anchor
- execution of exactly eight new actions before replanning
- no reset of environment global time or position at chunk boundaries
- deterministic SFPD replay and seeded SFPS replay/diversity
- checkpoint/data-digest compatibility checks
- action-limit and failed-trajectory accounting
- mode-classification edge cases
- short CPU smoke training for both stages
- end-to-end artifact generation with reduced update and rollout counts

The full existing Stage A and rendering suite remains green. Full 20,000-update
training is an experiment run after unit and smoke tests pass; it is not executed
inside pytest.

## Non-goals

This stage does not implement merge/rebranch demonstrations, a denoiser,
score-corrected Brownian SDE, preference fitting, STEG, obstacles, dynamics, or
PushT code refactoring. It does not claim that SFPD can sample multiple futures
from an identical condition. It does not treat an SFPS mode switch as a task
failure merely because the current obstacle-free environment has no prescribed
mode identity.
