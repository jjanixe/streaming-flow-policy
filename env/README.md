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

The observation is float32 [x, y, t]. An action is the next float32 [x, y]
position and is followed exactly. Finite actions are never clipped. Every step
returns reward 0.0; task success and goal error are reported through info.
Episodes end after exactly 64 actions unless a NaN or infinite action causes a
recorded numerical failure.

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
display coordinates at the configured visualization boundary; environment
states and actions remain unbounded and unchanged.

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

All floating-point NumPy arrays and Torch tensors use float32. String IDs and
integer RNG metadata retain their natural dtypes. For finite actions so extreme
that a true log density falls below the float32 range, the reported log density
saturates at the smallest finite float32 value. Its Torch action gradient still
uses the analytic score, and mixture responsibilities remain finite, normalized,
and correctly ordered.

## Tests

~~~bash
uv run pytest env/tests -q
~~~
