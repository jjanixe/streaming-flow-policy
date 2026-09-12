# PointReach2DPreference Stage A

## Few-shot preference steering pilot

`env.run_preference` adds a synthetic-user experiment around a **frozen** B2
SFPS checkpoint. It fits only two utility weights from whole-trajectory A/B
comparisons, then selects full future latent continuations at each replan.
The original B2 training and prediction API are reused.

The completed seed-0 pilot, coverage limits, uncertainty, and verification are
recorded in [the measured results](../docs/research/2026-09-11-preference-steering-results.md).

From this worktree, run:

```bash
uv run --frozen python -m env.run_preference \
  --checkpoint env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt \
  --stats env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz \
  --demonstrations ../../env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/preference/pilot-seed0 \
  --device cuda:1 --seed 0 --candidates 32 --coverage-count 512 \
  --rollout-count 16 --integration-steps-per-action 6 --torch-threads 1
```

Use `--device cpu` for a CPU run and a new output directory for every run.
The demonstrated relative demonstration path assumes this checkout is the
existing project-local `.worktrees/stage-b2-sfps` worktree; otherwise pass the
path to the saved Stage A demonstration bank. Its train digest must match the
checkpoint and normalization statistics.

The experiment first samples 512 continuations independently at the centered
start and at each of two fixed Gaussian starts. It then evaluates centered and
Gaussian closed-loop episodes for base, task-only soft selection, oracle soft
and best selection, and learned soft selection at 5/10/20/40 noisy comparisons
for four synthetic users. Every condition shares evaluation starts and indexed
proposal randomness. Each of 32 candidates rolls out the **remaining episode**,
refreshing history every eight actions. Only the chosen first eight commands
are executed through Gym. The realized prefix stays fixed; whole-trajectory
labels are never copied onto individual chunks.

The score is `w @ normalized_full_path_features - (goal_error / 0.1)**2`.
Candidates with numerical or action-limit failures have zero selection mass.
If no candidate is valid, candidate zero is used as a recorded fallback; actual
Gym execution still rejects invalid commands. Goal tolerance is a soft cost,
not a success guarantee. Base always uses candidate zero with one proposal.

The output contains strict JSON diagnostics/configuration, pickle-free NPZ
comparison data/weights/rollouts, the train-fitted feature normalizer, selected
future trajectories and latent schedules, candidate scores/validity/ESS,
failure counts, checkpoint digest, and utility/success/trajectory PNGs.
Incomplete paths have unavailable full-path utility and remain in failure
denominators. Utility plots condition on successful episodes; their bands are
episode standard errors for **one fitted preference dataset per user**, not
uncertainty over repeated human feedback. Success Wilson intervals are in JSON.

This is finite-candidate resampling, not gradient guidance or exact sampling.
Batch replan timings include a cohort of parallel episodes and are not
single-robot control latency; base uses fewer proposals than steered conditions.
Same-state candidate coverage must be examined before claiming named-mode
control. Arrays and models are float32; the tiny BT Newton solver and softmax
normalization use float64 internal arithmetic for numerical stability.

To fit real A/B feedback after converting completed trajectories with the saved
feature normalizer, call:

```python
from env.preference import fit_bradley_terry

# delta_features[i] = phi(trajectory_A) - phi(trajectory_B)
# labels[i] = +1 for A preferred, -1 for B preferred; no ties in this API.
fit = fit_bradley_terry(delta_features, labels, l2=0.1)
```

The solver reports convergence and feature-difference rank. Few comparisons do
not identify preference directions that the queries never vary. A zero prior
is a regularizer, not a population prior learned from other users.

This package implements Stage A of the preference-guided streaming-policy toy
example. It contains the Gym environment, analytic demonstrations and fields,
feature normalization, ODE/SDE samplers, and reproducible diagnostics.

## Gym environment

Importing the package registers the environment:

~~~python
import gym
import env

instance = gym.make("PointReach2DPreference-v0")
observation, info = instance.reset(seed=0)
~~~

The observation is float32 [x, y, t]. An action is an absolute next float32
[x, y] position. The environment follows it exactly when its L2 distance from
the current position is at most `max_step_distance=0.075`; it never projects or
clips an invalid command. A finite command above that limit preserves the last
valid state and terminates the episode with `action_limit_failure=True`. NaN or
infinite commands similarly terminate with a recorded numerical failure.

The Gym action space remains an unbounded float32 Box because the valid L2 ball
is centered on the current state and therefore cannot be represented by one
static Box. Runtime `info` reports `action_attempt_count`,
`action_limit_activation_count`, `action_limit_activation_rate`, and the last
and maximum requested step distances. Every step returns reward 0.0; task
success and goal error are also reported through `info`. A normal episode ends
after exactly 64 accepted actions.

### Rendering

Both Gym 0.26 render modes are available. The human mode opens a pygame window
and updates it automatically after reset and every step:

~~~python
instance = gym.make("PointReach2DPreference-v0", render_mode="human")
observation, info = instance.reset(seed=0)
observation, reward, terminated, truncated, info = instance.step(action)
instance.close()
~~~

The array mode returns an RGB uint8 frame with shape `[520, 720, 3]`:

~~~python
instance = gym.make("PointReach2DPreference-v0", render_mode="rgb_array")
observation, info = instance.reset(seed=0)
frame = instance.render()
~~~

The four light reference curves represent the upper/lower narrow/wide modes.
The dark line is the current episode trajectory, the purple dot is the current
position, and the green circle is the goal tolerance. Rendering clips only the
display coordinates at the configured visualization boundary. It does not
alter valid environment states or actions.

Gym 0.26 refers to the removed NumPy 2 name np.bool8. Importing env supplies the
equivalent np.bool_ alias so Gym's standard checker and wrappers work in the
repository's current uv environment.

## Run Stage A

~~~bash
uv run python -m env.run_stage_a \
  --output-dir env/artifacts \
  --seed 0 \
  --num-rollouts 4096
~~~

The command writes:

- demonstrations.npz
- feature_normalizer.npz
- rollouts.npz
- resolved_config.json
- diagnostics.json

Generated artifacts are ignored by Git. diagnostics.json compares analytic ODE
and SDE rollout marginals against samples from the target train-bank Gaussian
mixture. ODE and SDE diagnostics intentionally share the same target-reference
sample stream, reducing comparison noise without coupling either rollout
sampler. Its accepted field, not the Gym reward, records whether Stage A meets
the configured marginal-error and midpoint mode-coverage criteria. Any
non-finite state, metric, or occupancy fails acceptance.

Expert generation verifies that every consecutive waypoint is within the same
configured movement limit. Each ODE/SDE sampler diagnostic also records the
number and rate of would-be action-limit activations, plus the indices of
trajectories that activated the limit. These analytic diagnostic rollouts are
left intact rather than clipped, so the affected trajectories can be inspected
separately.

All floating-point NumPy arrays and Torch tensors use float32. String IDs and
integer RNG metadata retain their natural dtypes. For finite actions so extreme
that a true log density falls below the float32 range, the reported log density
saturates at the smallest finite float32 value. Its Torch action gradient still
uses the analytic score, and mixture responsibilities remain finite, normalized,
and correctly ordered.

### Stage B1: PushT-compatible deterministic SFPD

First generate the saved demonstration artifact if it does not already exist:

~~~bash
uv run python -m env.run_stage_a --output-dir env/artifacts --seed 0
~~~

Run the full seeded experiment:

~~~bash
uv run python -m env.run_stage_b \
  --stage b1 \
  --demonstrations env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/stage_b/b1-seed0 \
  --seed 0 \
  --device cpu \
  --max-updates 20000 \
  --rollout-count 1024 \
  --integration-steps-per-action 6 \
  --enforce-acceptance
~~~

The runner always loads the named demonstration artifact; it never silently
regenerates data. The dataset uses next-position actions, train-only `[-1, 1]`
normalization, PushT 16/2/8 windows, and actual Drake FirstOrderHold targets.
Drake evaluates in float64 internally; every learned quantity is float32.

The output directory contains the resolved configuration and seed streams,
train-only normalization statistics, the selected best-EMA checkpoint,
training history, held-out/Gaussian/centered diagnostics, pickle-free padded
Gaussian and centered rollout NPZ files, trajectory and occupancy PNGs, and a
representative success and/or failure GIF whenever that outcome occurs. Each
rollout NPZ stores its checkpoint SHA-256 and train-demonstration digest, raw
nine-position chunks, explicit lengths and validity masks, and failure masks.
The PNGs and GIFs use the same world bounds and actual RGB-array renderer as the
Gym environment.

### Stage B2: stochastic latent SFPS

After a compatible B1 run exists, run B2 on its own with:

~~~bash
uv run python -m env.run_stage_b --stage b2 --seed 0 --device cpu
~~~

By default this validates
`env/artifacts/stage_b/b1-seed0/diagnostics.json` before training. A different
B1 diagnostics JSON or `sfpd_best.pt` checkpoint can be supplied with
`--b1-record`. The precondition checks the demonstration digest, float32/finite
pipeline result, seed replay, and 16/2/8 chunk contract. A B1 mode-coverage gate
miss remains a recorded experiment result and does not by itself block B2.

To train both models sequentially from independently derived seeds, run:

~~~bash
uv run python -m env.run_stage_b --stage all --seed 0 --device cpu
~~~

This writes B1 and B2 beneath `env/artifacts/stage_b/all-seed0/b1/` and
`env/artifacts/stage_b/all-seed0/b2/`, plus a top-level
`stage_b_summary.json`. Reduced `--max-updates`, `--rollout-count`, and
`--integration-steps-per-action` values are supported for CPU smoke runs.

B2 uses the repository-style equal-sigma formulation with
`sigma0=sigma1=0.1`, so `sigma_r=0` initially. At every streaming chunk
boundary it anchors the action state to the current observation, samples a
fresh explicit two-dimensional latent from the recorded latent RNG stream, and
generates a nine-position chunk while executing the next eight positions. The
environment RNG and latent RNG are stored separately, and each chunk latent is
saved in the pickle-free rollout artifact.

The expert action windows are normalized and materialized once before either
Stage B training loop. Training then samples all times, noise, and trajectory
targets for a minibatch directly as float32 Torch tensors on the selected
device. The uniform 16-knot first-order-hold calculation is algebraically the
same trajectory that Drake constructs and is regression-tested against Drake,
but it avoids constructing one CPU Drake object per sample. Held-out validation
still uses `PiecewisePolynomial.FirstOrderHold` through Drake as the independent
reference. Drake receives float64 knot times and values at that API boundary;
its positions and derivatives are immediately converted back to float32. The
learned model, losses, optimizer state, ODE solve, and rollout artifacts remain
float32.

On CUDA, the training loader follows the repository PushT setup with one
persistent worker and pinned memory. The pre-materialized windows remove the
remaining repeated NumPy indexing and normalization work from that worker. B2
evaluation also solves all currently active rollouts together at each chunk
boundary. Every rollout still has its own environment RNG and Torch latent RNG,
so the saved seed and latent provenance is unchanged. Because an adaptive ODE
solver can make slightly different step-size choices for different batch
compositions, exact replay means rerunning the complete recorded seed batch;
the run command performs that batch-level replay check.

Gaussian-initialized rollouts measure ordinary closed-loop behavior. The
centered diagnostic fixes the initial environment state and varies only latent
seeds, directly reporting unique raw and executed trajectories and midpoint
mode occupancy. This B2 implementation has neither a denoiser nor a
controllable preference/mode label: the latent can reveal learned multimodality,
but it does not select a named mode on command.

## Tests

~~~bash
uv run pytest env/tests -q
~~~

## Grouped few-shot preference steering

The grouped pilot uses exactly `direction=(upper, lower)` and
`width=(wide, narrow)`. It fits within-group simplex weights and a separate
between-group importance simplex from scoped Bradley–Terry feedback. Twenty
judgments comprise six direction, six width, and eight overall comparisons.
Group-specific comparisons exclude the between-group importance. P/M/C are not
part of this toy experiment.

The conditional evaluator holds each current latent's eight-command prefix
fixed and averages four independent base-policy continuations before computing
its exponential selection weight. Default `M=8`, `L=4`, and `beta=rho=16` are
pilot settings in bounded feature units. The goal/failure cost makes this a
task-conditioned preference target. It is finite-candidate resampling, not
exact Gibbs sampling or a gradient update to the flow network.

~~~bash
.venv/bin/python -m env.run_grouped_preference \
  --checkpoint env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt \
  --stats env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz \
  --demonstrations ../../env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/grouped_preference/pilot-seed0 \
  --device cuda:1 --candidates 8 --continuations 4 \
  --rollout-count 16 --torch-threads 1
~~~

Run from this B2 worktree; choose a new output directory for each rerun. The
runner fits nested budgets 5/10/20/40 and evaluates learned-20, oracle-soft,
task-only, and raw base through actual Gym. It saves scoped feedback, fitting
diagnostics, current anchors, conditional feature/cost estimates, selected
latents, requested commands, actual paths, failures, seeds, and frozen-policy
hashes.

The checkpoint's sixteen prediction points include the anchor. Nine returned
points give one discarded anchor and eight executed commands, at local times
0 through 8/15, matching upstream SFPS. Observed physical anchors and raw model
anchors are recorded separately because observation/action normalization
statistics can differ. No checkpoint horizon or policy parameters are changed.

Narrow desirability rewards small excursion, including central paths. A higher
utility therefore does not guarantee the corresponding named midpoint mode.
See [the formulation, validation, and results](../docs/research/2026-09-11-grouped-preference-results.md)
for this limitation, numerical checks, inference GIFs, and reproducibility.

### Mode-aligned control and oracle ablation

Use `--feature-kind mode` to align the two grouped desirabilities with the
four midpoint bins; `other` receives four zeros. Synthetic scoped feedback is
regenerated for the selected feature kind. `--selection-method best` selects
the highest-scoring valid current candidate. `--proposal-std` widens guided
current latent draws while base and hypothetical future draws stay unit normal.
It changes the proposal distribution; policy weights and the SFPS anchor stay
fixed. The compatibility defaults remain continuous features, soft selection,
and proposal standard deviation 1.

~~~bash
.venv/bin/python -m env.run_grouped_preference \
  --checkpoint-path env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt \
  --stats-path env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz \
  --demonstrations-path ../../env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/grouped_preference/mode-example \
  --device cuda:0 --feature-kind mode --selection-method best \
  --proposal-std 8 --candidates 128 --continuations 4 --rollout-count 16
~~~

`python -m env.run_mode_ablation` accepts the same checkpoint/stats/data paths,
`--candidates 8 32 128`, and `--methods soft best` for oracle-only comparisons.
Its completed conditions resume only after the explicitly listed source
modules, config, and artifact integrity checks. The source manifest does not
cover every transitive dependency. Use a new output directory after changing
any source or settings, including normalization, loader, or seed helpers.

The checked pilot achieved all four centered modes with learned20; Gaussian
wide modes retain some misses. Counts require both task success and target
mode. See [the paired experiments and actual inference GIFs](../docs/research/2026-09-12-mode-aligned-steering-results.md).
