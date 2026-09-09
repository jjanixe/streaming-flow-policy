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

The learned implementations are toy-native, CPU-capable, and reproducible.
They preserve the PushT expert-data transformation, Drake interpolation,
formulations, normalization, and execution semantics without reusing the large
PushT ConditionalUnet1D, global random-number state, or CUDA-only training
scripts. Saved demonstrations, training samples, Torch models, losses, and ODE
states are float32. Only the required Drake interpolation boundary is float64.

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

This design preserves those contracts and uses the same Drake trajectory class.
It does not preserve incidental implementation choices that are inappropriate
for the toy, such as the large UNet, process-global NumPy random state, or a
CUDA-only entry point.

In the existing code, a Drake trajectory is a
`PiecewisePolynomial.FirstOrderHold` object built from a normalized sixteen-step
expert action window at uniformly spaced local times in `[0, 1]`. It linearly
interpolates consecutive waypoints and exposes position and derivative
evaluation at an independently sampled local time. It is not a simulator,
dynamics model, planner, or expert generator. It converts discrete expert
waypoints into the continuous `xi(t)` and `d xi(t)/dt` training targets.

Stage B uses the actual `pydrake.trajectories.PiecewisePolynomial` class rather
than a handwritten Torch substitute. Drake 1.26 exposes the ordinary
`PiecewisePolynomial` specialization over C++ `double`, so construction and
evaluation use float64 even when the source demonstrations are float32. The
evaluated position and derivative are immediately copied to float32 before any
noise, flow-target, Torch, loss, or learned ODE calculation. No gradient is
needed through Drake because it is an off-graph training-datum transform.

The current glibc 2.31 host requires `drake<1.27`; Drake 1.26 cannot import with
NumPy 2 because it uses a removed NumPy API. The environment contract therefore
adds `numpy<2` alongside the existing `drake<1.27` pin and records the resolved
versions in every run artifact.

## Fixed chunk and observation contract

- Episode: 64 new position commands and 65 states including the initial state.
- Observation: float32 `[x, y, s]`, where `s` is normalized episode time.
- Observation history: 2 observations.
- Prediction horizon: 16 normalized position actions including the current
  anchor after the PushT transform.
- Execution horizon: 8 new actions.
- Chunk-local waypoint time: `lambda_j = j / 15`.
- The chunk-local time resets at every replan.
- Episode time in the observations never resets.
- The environment action remains one absolute next position with shape `[2]`.
- The environment never receives a `[16, 2]` action.
- Every requested environment step remains subject to
  `max_step_distance=0.075`; generated actions are never silently clipped.

Conceptually, at global anchor index `q`, the condition and prediction are

    c_q = (o_{q-1}, o_q)
    Xi_q = (p_q, p_{q+1}, ..., p_{q+15}),

with terminal repetition when an index is past `64`. The exact stored-array
construction follows the PushT shift and padding procedure below rather than
constructing this conceptual array directly.

At `q=0`, `o_-1` is a copy of `o_0` with global time zero. Following PushT, the
first predicted position is the current observation in normalized policy
coordinates and is not re-executed. Independent observation/action statistics
can make its action-space inverse transform differ slightly from the physical
current position; that discrepancy is measured rather than hidden. The runner
applies `a_(q+1)` through `a_(q+8)`, or only the remaining valid actions at the
episode end, then refreshes the observation history.

The prediction API accepts a requested number of positions from 1 through 16.
Ordinary execution requests 9 positions (anchor plus eight actions), matching
the current PushT optimization. Diagnostics may request all 16 positions.

## PushT-compatible expert dataset

Chunks are derived from the saved whole-trajectory demonstration bank. The
trajectory split is authoritative: every window from a train, validation, or
test trajectory inherits that split. No chunk-level reshuffling may move
windows across splits, and only train trajectories fit normalization statistics.

For each 65-state toy episode, first construct PushT-style timestep arrays:

    obs[t] = [p_t.x, p_t.y, t / 64],                 t = 0,...,64
    action[t] = p_(min(t+1, 64)),                    t = 0,...,64.

This is the same `next observation as action` shift used by
`PushTStateDatasetWithNextObsAsAction`: the last position is repeated so action
and observation arrays have the same length. The demonstration bank's stored
analytic derivatives are not consumed by Stage B.

Fit independent per-coordinate min/max statistics for `obs` and `action` using
the complete unwindowed train arrays, then normalize both arrays to `[-1, 1]`.
Validation, test, and rollout data use the saved train statistics. Zero-range
coordinates are rejected rather than divided by zero. Statistics and their
source demonstration digest are checkpointed.

Create sequence indices with exactly the PushT parameters:

    sequence_length = pred_horizon = 16
    pad_before = obs_horizon - 1 = 1
    pad_after = action_horizon - 1 = 7.

For an episode length of 65, the sequence start is `r=-1,...,56`, yielding 58
training windows and conceptual current anchors `q=r+1=0,...,57`. Values before
the episode repeat the first timestep; values after it repeat the final
timestep. The model receives only the first two observations from each
sixteen-step sampled sequence and all sixteen actions. This differs deliberately
from the previous draft's `q=0,...,63` windows and matches the repository's
`create_sample_indices` and `sample_sequence` behavior.

After normalization and padding, enforce the repository's anchor correction:

    if action_window[0] != obs_window[-1, :2]:
        action_window[0] = obs_window[-1, :2].

Equality is checked elementwise exactly as in PushT, and the action window is
copied before mutation. Because the toy's observation-position and shifted
action statistics can differ slightly at an endpoint, the runner records the
raw-coordinate size of every anchor correction. The baseline retains the PushT
behavior instead of silently substituting a different shared normalization.

The dataset keeps trajectory ID, source split, sequence start `r`, conceptual
anchor `q`, and padding extents as diagnostic metadata. These fields and any
derived padding mask are never model inputs and do not alter the training loss.
The implementation reuses the repository's generic sample-index, sequence,
normalization, and inverse-normalization helpers where their contracts apply,
while the toy wrapper owns split isolation, metadata, and zero-range checks.

## Drake training-datum transform

For every sampled normalized action window, construct a fresh Drake trajectory:

    tau_j = j / 15,                                  j = 0,...,15
    xi = PiecewisePolynomial.FirstOrderHold(
        linspace(0, 1, 16), action_window.T)

Then sample one independent local time `tau ~ Uniform[0, 1)` and evaluate

    xi_tau = xi.value(tau).T
    xi_dot_tau = xi.EvalDerivative(tau).T.

This occurs in the dataset transform before batching, as in the existing SFPD
and SFPS policies. `tau`, `xi_tau`, and `xi_dot_tau` are converted immediately
to NumPy float32. Model inputs and targets are then assembled in float32. The
transform receives an explicit worker-local NumPy generator derived from the run
seed; it preserves the PushT distribution without relying on process-global
`np.random` state.

For verification only, tests compare Drake output against the closed-form
first-order hold:

    segment = min(floor(15 * tau), 14)
    alpha = 15 * tau - segment
    xi(tau) = (1-alpha) * action[segment] + alpha * action[segment+1]
    xi_dot(tau) = 15 * (action[segment+1] - action[segment]).

The closed form is not the production training path.

## Model inputs and architecture

Both models use a small condition-aware MLP rather than ConditionalUnet1D. The
MLP has three hidden layers of width 128 with SiLU activations. Local time uses
the existing toy Fourier features with frequencies 1, 2, 4, and 8.

The condition is the six-dimensional concatenation of two train-stat-normalized
observations. Global episode phase therefore enters through the condition, while
integration uses chunk-local time. Mode IDs, trajectory IDs, signs, amplitudes,
padding masks, and split labels are never model inputs.

SFPD receives action `[2]`, local-time features `[9]`, and condition `[6]`, and
outputs action velocity `[2]`.

SFPS receives joint action and latent `[4]`, the same local-time features and
condition, and outputs joint velocity `[4]`. The latent dimension equals the
action dimension, matching the PushT implementation.

Model methods validate rank, final dimension, dtype, device consistency, local
time bounds, and finite values. Training and evaluation default to CPU in the
current environment but permit an explicitly selected CUDA device when one is
available.

## Rollout normalization contract

At every replan, the last two raw Gym observations are normalized with saved
train observation statistics and flattened as the fixed condition. As in the
repository policy, the ODE initial action is `normalized_obs[-1, :2]`. Predicted
normalized actions are inverse-transformed with the saved train action
statistics before the anchor is discarded and eight physical absolute-position
commands are sent individually to Gym.

The implementation does not silently replace this with shared position
statistics because that would change the requested PushT baseline. It records
both normalized anchor equality and physical anchor discrepancy. Any later
shared-statistics variant must be a separately named ablation.

## Stage B1: deterministic SFPD

For a sampled train chunk and local time, evaluate the first-order-hold position
and derivative, then perturb only the position:

    x = xi(lambda) + sigma * noise
    v_target = d xi / d lambda

The initial training noise is `sigma=0.1` in normalized action coordinates,
matching the repository's PushT SFPD experiment. It does not add the Stage A
analytic field's explicit `-k` term. The learned conditional marginal field may
still exhibit data-driven attraction, but the implementation must not claim that
it is the Stage A stabilizing field.

Inference starts at the current observed position and integrates

    da / d lambda = v_theta(a, lambda | c_q)

with `torchdyn.core.NeuralODE`, Dopri5, `sensitivity="adjoint"`, and
`atol=rtol=1e-4`, matching the repository policy. The integration grid contains
`1 + num_future_actions * integration_steps_per_action` points over
`[0, num_future_actions / 15]`; every
`integration_steps_per_action`-th state becomes an output action. Grid density is
configurable and separate from output action spacing. The condition remains
fixed inside a chunk, and all learned ODE tensors are float32.

## Stage B2: stochastic latent SFPS

SFPS uses the same conditional expert chunk. With latent `z0 ~ N(0, I)`,
independent action noise `epsilon_a0 ~ N(0, sigma0^2 I)`, and
`sigma_r = sqrt(sigma1^2 - sigma0^2)`, training samples are

    a_lambda = xi(lambda) + epsilon_a0 + sigma_r * lambda * z0
    z_lambda = (1 - (1-sigma1) * lambda) * z0 + lambda * xi(lambda)
    v_a = d xi / d lambda + sigma_r * z0
    v_z = xi(lambda) + lambda * d xi / d lambda - (1-sigma1) * z0.

The initial configuration uses `sigma0=sigma1=0.1` in normalized action
coordinates, so `sigma_r=0`, matching the equal-sigma setup in the PushT
experiment. This value is a configurable starting point, not a tuned optimum.

At each chunk boundary, inference sets `a0` to the current observation's
normalized position coordinates, samples a fresh seeded `z0`, and integrates
the learned joint ODE locally. It executes only the inverse-transformed action
component. The sampled latent is never exposed to the Gym environment.

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

Validation batches are generated from validation trajectories with fixed window,
local-time, and noise seeds and reused at every validation point. The test split
is read only after training and checkpoint selection are complete. Training loss
need not approach zero because a condition may support several expert futures.

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
- normalization endpoint mismatch and anchor-correction magnitude
- deterministic replay under identical seeds
- Drake-boundary dtype plus downstream float32 and finite-value checks

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

- `chunk_data.py`: next-state action shift, train-only statistics, PushT window
  indices, endpoint padding, and diagnostic metadata
- `drake_trajectory.py`: actual Drake FirstOrderHold construction, seeded local
  time evaluation, immediate float32 conversion, and SFPD/SFPS datum transforms
- `models.py`: condition-aware float32 MLPs
- `sfp_policies.py`: SFPD/SFPS targets and differentiable integration
- `train_stage_b.py`: sequential training, EMA, validation, checkpoints
- `evaluate_stage_b.py`: streaming runner, metrics, mode classification
- `run_stage_b.py`: command-line orchestration and artifact writing
- `visualize_stage_b.py`: saved rollout and occupancy figures

Existing `environment.py`, `demonstrations.py`, `artifacts.py`, and Stage A
analytic modules are reused through their public contracts. Existing PushT
modules are not modified; importing their generic dataset helpers is allowed.

## Testing

Implementation follows test-driven development. Tests cover:

- Drake 1.26 / NumPy 1.x import compatibility in the resolved uv environment
- exact next-observation action shift and terminal repetition
- train-only min/max normalization and inverse normalization
- exact PushT sample indices, 58 windows per 65-state episode, history padding,
  terminal padding, and anchor correction
- split inheritance and absence of trajectory leakage
- actual Drake FirstOrderHold values and derivatives at boundaries and interiors
- Drake float64 outputs converted immediately to float32 training data
- Drake results against closed-form FOH values within documented tolerances
- SFPD and SFPS target formulas against hand-computed examples
- model shape, dtype, finite-value, and input validation
- exact normalized-coordinate anchor correction plus measured physical-coordinate
  discrepancy under independent observation/action statistics
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
