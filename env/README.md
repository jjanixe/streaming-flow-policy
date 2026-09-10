# PointReach2DPreference Stage A

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
