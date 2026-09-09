# Stage A Toy Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a reproducible float32 Gym 0.26 Stage A environment with analytic demonstrations, features, Gaussian-mixture fields, ODE/SDE samplers, NPZ artifacts, diagnostics, and tests.

**Architecture:** A top-level env package keeps Gym state transitions separate from stateless trajectory mathematics and Torch analytic fields. Frozen dataclass configuration and explicit NumPy/Torch RNGs make every artifact replayable; pytest tests each layer before the Stage A runner combines them.

**Tech Stack:** Python 3.10, uv, Gym 0.26.2, NumPy, PyTorch, pytest, standard-library dataclasses/hashlib/json/pathlib.

**Spec:** docs/superpowers/specs/2026-09-08-stage-a-toy-environment-design.md

## Global Constraints

- Create all Stage A code under the repository-root env/ package.
- Use float32 for every floating-point NumPy array and Torch tensor.
- Keep identifiers as strings and seeds/indices as integers.
- Register PointReach2DPreference-v0 through env.__init__.
- Follow Gym 0.26 reset and five-value step APIs.
- Never clip finite positions and never terminate early on goal success.
- Use only the train split for feature normalization and analytic fields.
- Use explicit, independently derived RNG streams.
- Run Stage A diagnostics on CPU; GPU support begins with later learned-model stages.
- Store arrays in compressed NPZ files loadable with allow_pickle=False.
- Preserve the user's existing Drake pin and unrelated working-tree changes.
- Do not implement learned models, preference fitting, or STEG.

---

## File map

- pyproject.toml: include env in setuptools discovery and declare pytest as a development dependency.
- uv.lock: resolved pytest development dependency while preserving current runtime resolution.
- .gitignore: ignore generated env/artifacts contents.
- env/__init__.py: idempotent Gym registration and public environment export.
- env/config.py: immutable Stage A configuration and JSON-serializable conversion.
- env/environment.py: PointReach2DPreferenceEnv state machine and task metrics.
- env/demonstrations.py: analytic path family, derivatives, split generation, and bank types.
- env/features.py: raw features, train-only normalizer, and prefix/future decomposition.
- env/analytic_fields.py: differentiable train-bank mixture density, PF velocity, score, and SDE drift.
- env/samplers.py: explicit-RNG initialization, Euler ODE, and Euler-Maruyama SDE.
- env/artifacts.py: safe NPZ/JSON serialization and canonical train-data digest.
- env/run_stage_a.py: artifact generation, distribution diagnostics, acceptance result, and CLI.
- env/README.md: environment and Stage A runner usage.
- env/tests/: layer-specific pytest coverage.

### Task 1: Development setup, configuration, and Gym environment

**Files:**
- Modify: pyproject.toml
- Modify: uv.lock
- Create: env/__init__.py
- Create: env/config.py
- Create: env/environment.py
- Create: env/tests/__init__.py
- Create: env/tests/test_environment.py

**Interfaces:**
- Consumes: Gym 0.26.2 and NumPy already present in the uv environment.
- Produces: EnvironmentConfig, DemonstrationConfig, FieldConfig, FeatureConfig, StageAConfig, DEFAULT_CONFIG, config_to_dict, PointReach2DPreferenceEnv, and Gym id PointReach2DPreference-v0.

- [ ] **Step 1: Add pytest and package discovery**

Run:

    uv add --dev pytest

Then ensure setuptools discovery reads:

    [tool.setuptools.packages.find]
    include = ["streaming_flow_policy", "jupyviz", "utils", "env"]

Expected: pyproject.toml contains a dependency-groups.dev entry for pytest and uv.lock resolves it without removing the existing drake<1.27 constraint.

- [ ] **Step 2: Write failing Gym tests**

Create env/tests/test_environment.py with these behaviors:

~~~python
import gym
import numpy as np
import pytest
from gym.utils.env_checker import check_env

import env
from env.config import DEFAULT_CONFIG
from env.environment import PointReach2DPreferenceEnv


def test_gym_registration_and_api():
    wrapped = gym.make("PointReach2DPreference-v0")
    obs, info = wrapped.reset(seed=7, options={"center_init": True})
    np.testing.assert_array_equal(obs, np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    assert info["step_index"] == 0
    assert obs.dtype == np.float32
    check_env(PointReach2DPreferenceEnv(), skip_render_check=True)


def test_seeded_gaussian_reset_is_reproducible():
    instance = PointReach2DPreferenceEnv()
    first, _ = instance.reset(seed=11)
    second, _ = instance.reset(seed=11)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[:2], DEFAULT_CONFIG.environment.start_array())


def test_action_is_followed_without_clipping_and_horizon_terminates():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0, options={"center_init": True})
    far = np.array([9.0, -4.0], dtype=np.float32)
    for step in range(DEFAULT_CONFIG.environment.horizon_steps):
        obs, reward, terminated, truncated, info = instance.step(far)
        assert reward == 0.0
        assert truncated is False
        assert terminated is (step == DEFAULT_CONFIG.environment.horizon_steps - 1)
    np.testing.assert_array_equal(obs[:2], far)
    assert info["outside_visualization"] is True


def test_numerical_failure_preserves_last_state():
    instance = PointReach2DPreferenceEnv()
    initial, _ = instance.reset(seed=0, options={"center_init": True})
    obs, reward, terminated, truncated, info = instance.step(
        np.array([np.nan, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(obs[:2], initial[:2])
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["numerical_failure"] is True
    assert info["success"] is False


def test_bad_action_shape_and_post_terminal_step_raise():
    instance = PointReach2DPreferenceEnv()
    instance.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        instance.step(np.zeros(3, dtype=np.float32))
    instance.step(np.array([np.inf, 0.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="terminated"):
        instance.step(np.zeros(2, dtype=np.float32))
~~~

- [ ] **Step 3: Run the tests and verify red**

Run:

    uv run pytest env/tests/test_environment.py -q

Expected: collection fails because env.config and env.environment do not exist.

- [ ] **Step 4: Implement immutable configuration**

Implement env/config.py with nested frozen dataclasses. EnvironmentConfig exposes start_array() and goal_array() that return new float32 arrays. config_to_dict(config) returns dataclasses.asdict(config), suitable for json.dump.

~~~python
@dataclass(frozen=True)
class EnvironmentConfig:
    start: tuple[float, float] = (-1.0, 0.0)
    goal: tuple[float, float] = (1.0, 0.0)
    physical_duration_s: float = 2.0
    horizon_steps: int = 64
    initial_sigma: float = 0.04
    goal_tolerance: float = 0.10
    visualization_x: tuple[float, float] = (-1.3, 1.3)
    visualization_y: tuple[float, float] = (-0.9, 0.9)

    def start_array(self) -> np.ndarray:
        return np.asarray(self.start, dtype=np.float32)

    def goal_array(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=np.float32)


@dataclass(frozen=True)
class DemonstrationConfig:
    train_per_mode: int = 128
    validation_per_mode: int = 32
    test_per_mode: int = 32
    amplitude_narrow: tuple[float, float] = (0.22, 0.32)
    amplitude_wide: tuple[float, float] = (0.48, 0.58)


@dataclass(frozen=True)
class FieldConfig:
    stabilization_k: float = 1.0
    diffusion_kappa: float = 4.0


@dataclass(frozen=True)
class FeatureConfig:
    y_scale: float = 0.35


@dataclass(frozen=True)
class StageAConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    demonstrations: DemonstrationConfig = field(default_factory=DemonstrationConfig)
    field: FieldConfig = field(default_factory=FieldConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    root_seed: int = 0
    diagnostic_rollouts: int = 4096


DEFAULT_CONFIG = StageAConfig()
~~~

- [ ] **Step 5: Implement Gym registration and environment**

env.__init__ checks Gym's registry before calling register, then exports PointReach2DPreferenceEnv. The environment uses the following state transition:

~~~python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    center = bool((options or {}).get("center_init", False))
    if center:
        self._position = self.config.start_array()
    else:
        noise = self.np_random.normal(
            0.0, self.config.initial_sigma, size=2
        ).astype(np.float32)
        self._position = self.config.start_array() + noise
    self._step_index = 0
    self._done = False
    self._numerical_failure = False
    self._max_abs_x = float(abs(self._position[0]))
    self._max_abs_y = float(abs(self._position[1]))
    return self._observation(), self._info()


def step(self, action):
    if self._done:
        raise RuntimeError("episode has already terminated")
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {value.shape}")
    if not np.isfinite(value).all():
        self._done = True
        self._numerical_failure = True
        return self._observation(), 0.0, True, False, self._info()
    self._position = value.copy()
    self._step_index += 1
    self._max_abs_x = max(self._max_abs_x, float(abs(value[0])))
    self._max_abs_y = max(self._max_abs_y, float(abs(value[1])))
    self._done = self._step_index == self.config.horizon_steps
    return self._observation(), 0.0, self._done, False, self._info()
~~~

Use an unbounded float32 action Box and an observation Box with lower [-inf, -inf, 0] and upper [inf, inf, 1]. success is true only at a normal horizon end whose goal error is at most 0.10.

- [ ] **Step 6: Run Gym tests**

Run:

    uv run pytest env/tests/test_environment.py -q

Expected: 5 passed.

- [ ] **Step 7: Commit the environment slice**

The repository already had uncommitted pyproject.toml and uv.lock changes before this
task. Keep those two files unstaged so the user's changes are not absorbed into this
commit.

    git add env/__init__.py env/config.py env/environment.py env/tests
    git commit -m "feat: add point reach preference Gym environment"

### Task 2: Analytic demonstration bank

**Files:**
- Create: env/demonstrations.py
- Create: env/tests/test_demonstrations.py

**Interfaces:**
- Consumes: StageAConfig and NumPy SeedSequence.
- Produces: ModeSpec, DemonstrationBank with select(split) and subset(indices), quintic_progress(times), quintic_progress_derivative(times), bump(times), bump_derivative(times), evaluate_path(sign, amplitude, times), and generate_demonstration_bank(config, seed).

- [ ] **Step 1: Write failing trajectory tests**

~~~python
import numpy as np

from env.config import DEFAULT_CONFIG
from env.demonstrations import evaluate_path, generate_demonstration_bank


def test_all_modes_have_exact_float32_endpoints():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=13)
    expected_start = DEFAULT_CONFIG.environment.start_array()
    expected_goal = DEFAULT_CONFIG.environment.goal_array()
    np.testing.assert_allclose(bank.positions[:, 0], expected_start, atol=1e-6)
    np.testing.assert_allclose(bank.positions[:, -1], expected_goal, atol=1e-6)
    assert bank.positions.dtype == np.float32
    assert bank.derivatives.dtype == np.float32


def test_analytic_derivative_matches_float32_central_difference():
    times = np.array([0.2, 0.37, 0.63, 0.8], dtype=np.float32)
    step = np.float32(1e-3)
    position, derivative = evaluate_path(1.0, 0.53, times)
    plus, _ = evaluate_path(1.0, 0.53, times + step)
    minus, _ = evaluate_path(1.0, 0.53, times - step)
    finite_difference = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-3, atol=2e-4)


def test_splits_are_balanced_isolated_and_reproducible():
    first = generate_demonstration_bank(DEFAULT_CONFIG, seed=17)
    second = generate_demonstration_bank(DEFAULT_CONFIG, seed=17)
    np.testing.assert_array_equal(first.amplitudes, second.amplitudes)
    np.testing.assert_array_equal(first.replay_seeds, second.replay_seeds)
    split_ids = {
        split: set(first.trajectory_ids[first.splits == split].tolist())
        for split in ("train", "validation", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    for split, per_mode in (("train", 128), ("validation", 32), ("test", 32)):
        for mode in ("upper-narrow", "upper-wide", "lower-narrow", "lower-wide"):
            assert np.sum((first.splits == split) & (first.modes == mode)) == per_mode
~~~

- [ ] **Step 2: Run and verify red**

Run:

    uv run pytest env/tests/test_demonstrations.py -q

Expected: import fails because env.demonstrations does not exist.

- [ ] **Step 3: Implement path functions and bank generation**

Define DemonstrationBank with validated arrays:

~~~python
@dataclass(frozen=True)
class DemonstrationBank:
    trajectory_ids: np.ndarray
    splits: np.ndarray
    modes: np.ndarray
    signs: np.ndarray
    amplitudes: np.ndarray
    replay_seeds: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    derivatives: np.ndarray

    def select(self, split: str) -> "DemonstrationBank":
        mask = self.splits == split
        return self.subset(np.flatnonzero(mask))

    def subset(self, indices: np.ndarray) -> "DemonstrationBank":
        selected = np.asarray(indices, dtype=np.int64)
        return DemonstrationBank(
            self.trajectory_ids[selected],
            self.splits[selected],
            self.modes[selected],
            self.signs[selected],
            self.amplitudes[selected],
            self.replay_seeds[selected],
            self.times.copy(),
            self.positions[selected],
            self.derivatives[selected],
        )
~~~

Use these float32 formulas:

~~~python
def evaluate_path(sign, amplitude, times):
    t = np.asarray(times, dtype=np.float32)
    q = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    b = 16.0 * t**2 * (1.0 - t) ** 2
    dq = 60.0 * t**2 * (1.0 - t) ** 2
    db = 32.0 * sign * amplitude * t * (1.0 - t) * (1.0 - 2.0 * t)
    positions = np.stack((-1.0 + 2.0 * q, sign * amplitude * b), axis=-1)
    derivatives = np.stack((dq, db), axis=-1)
    return positions.astype(np.float32), derivatives.astype(np.float32)
~~~

Spawn one SeedSequence per trajectory, derive a uint32 replay seed, initialize default_rng(replay_seed), and sample exactly one amplitude from the mode range.

- [ ] **Step 4: Run demonstration tests**

Run:

    uv run pytest env/tests/test_demonstrations.py -q

Expected: 3 passed.

- [ ] **Step 5: Commit the demonstration slice**

    git add env/demonstrations.py env/tests/test_demonstrations.py
    git commit -m "feat: generate analytic preference trajectories"

### Task 3: Feature normalization and decomposition

**Files:**
- Create: env/features.py
- Create: env/tests/test_features.py

**Interfaces:**
- Consumes: DemonstrationBank, FeatureConfig, horizon dt.
- Produces: FeatureNormalizer, raw_step_features, raw_trajectory_features, fit_feature_normalizer, step_contributions, trajectory_features, trajectory_feature_parts.

- [ ] **Step 1: Write failing feature tests**

~~~python
import numpy as np

from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.features import (
    fit_feature_normalizer,
    raw_trajectory_features,
    trajectory_feature_parts,
    trajectory_features,
)


def test_normalizer_uses_train_split_and_standardizes_train_features():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=3)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    train = bank.select("train")
    raw = raw_trajectory_features(
        train.positions,
        DEFAULT_CONFIG.features.y_scale,
        1.0 / DEFAULT_CONFIG.environment.horizon_steps,
    )
    np.testing.assert_allclose(normalizer.mu, raw.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(normalizer.scale, raw.std(axis=0, ddof=0), atol=1e-6)


def test_full_feature_equals_prefix_plus_future():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=4)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    trajectories = bank.select("test").positions[:8]
    full = trajectory_features(trajectories, normalizer, dt=1.0 / 64.0)
    for prefix_steps in (0, 1, 17, 63, 64):
        prefix, future = trajectory_feature_parts(
            trajectories, normalizer, prefix_steps=prefix_steps, dt=1.0 / 64.0
        )
        np.testing.assert_allclose(full, prefix + future, rtol=1e-5, atol=1e-6)
    assert full.dtype == np.float32
~~~

- [ ] **Step 2: Run and verify red**

Run:

    uv run pytest env/tests/test_features.py -q

Expected: import fails because env.features does not exist.

- [ ] **Step 3: Implement shared additive features**

~~~python
@dataclass(frozen=True)
class FeatureNormalizer:
    mu: np.ndarray
    scale: np.ndarray
    y_scale: float


def raw_step_features(positions, y_scale):
    values = np.asarray(positions, dtype=np.float32)
    rho = np.tanh(values[..., 1] / np.float32(y_scale))
    return np.stack((rho, rho * rho), axis=-1).astype(np.float32)


def raw_trajectory_features(trajectories, y_scale, dt):
    features = raw_step_features(trajectories[..., 1:, :], y_scale)
    return (np.float32(dt) * features.sum(axis=-2, dtype=np.float32)).astype(np.float32)


def step_contributions(trajectories, normalizer, dt):
    features = raw_step_features(trajectories[..., 1:, :], normalizer.y_scale)
    centered = (features - normalizer.mu) / normalizer.scale
    return (np.float32(dt) * centered).astype(np.float32)


def trajectory_feature_parts(trajectories, normalizer, prefix_steps, dt):
    contributions = step_contributions(trajectories, normalizer, dt)
    if not 0 <= prefix_steps <= contributions.shape[-2]:
        raise ValueError("prefix_steps is outside the trajectory horizon")
    return (
        contributions[..., :prefix_steps, :].sum(axis=-2, dtype=np.float32),
        contributions[..., prefix_steps:, :].sum(axis=-2, dtype=np.float32),
    )
~~~

fit_feature_normalizer filters bank.select("train"), computes raw feature mean and population standard deviation, and raises ValueError if either scale is non-positive.

- [ ] **Step 4: Run feature tests**

Run:

    uv run pytest env/tests/test_features.py -q

Expected: 2 passed.

- [ ] **Step 5: Commit the feature slice**

    git add env/features.py env/tests/test_features.py
    git commit -m "feat: add additive trajectory features"

### Task 4: Safe artifact contracts

**Files:**
- Create: env/artifacts.py
- Create: env/tests/test_artifacts.py
- Modify: .gitignore

**Interfaces:**
- Consumes: DemonstrationBank, FeatureNormalizer, StageAConfig, ODE/SDE arrays.
- Produces: train_data_digest, save/load_demonstration_bank, save/load_feature_normalizer, save_rollouts, save_json.

- [ ] **Step 1: Write failing artifact tests**

~~~python
import json
import numpy as np

from env.artifacts import (
    load_demonstration_bank,
    load_feature_normalizer,
    save_demonstration_bank,
    save_feature_normalizer,
    train_data_digest,
)
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.features import fit_feature_normalizer


def test_bank_and_normalizer_round_trip_without_pickle(tmp_path):
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=5)
    normalizer = fit_feature_normalizer(bank, DEFAULT_CONFIG)
    bank_path = tmp_path / "demonstrations.npz"
    norm_path = tmp_path / "normalizer.npz"
    save_demonstration_bank(bank_path, bank)
    save_feature_normalizer(
        norm_path, normalizer, train_data_digest(bank)
    )
    loaded_bank = load_demonstration_bank(bank_path)
    loaded_norm, digest = load_feature_normalizer(norm_path)
    np.testing.assert_array_equal(loaded_bank.positions, bank.positions)
    np.testing.assert_array_equal(loaded_bank.trajectory_ids, bank.trajectory_ids)
    np.testing.assert_array_equal(loaded_norm.mu, normalizer.mu)
    assert digest == train_data_digest(bank)
    with np.load(bank_path, allow_pickle=False) as payload:
        assert payload["positions"].dtype == np.float32


def test_digest_changes_when_train_data_changes():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=6)
    changed = generate_demonstration_bank(DEFAULT_CONFIG, seed=6)
    changed.positions[0, 1, 1] += np.float32(0.01)
    assert train_data_digest(bank) != train_data_digest(changed)
~~~

- [ ] **Step 2: Run and verify red**

Run:

    uv run pytest env/tests/test_artifacts.py -q

Expected: import fails because env.artifacts does not exist.

- [ ] **Step 3: Implement serialization**

train_data_digest selects train and updates SHA-256 with contiguous bytes from IDs encoded as UTF-8 with null separators, signs, amplitudes, replay seeds, times, positions, and derivatives in fixed order. Save strings as NumPy Unicode arrays and load every NPZ with allow_pickle=False.

~~~python
def save_demonstration_bank(path, bank):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        trajectory_ids=bank.trajectory_ids,
        splits=bank.splits,
        modes=bank.modes,
        signs=bank.signs,
        amplitudes=bank.amplitudes,
        replay_seeds=bank.replay_seeds,
        times=bank.times,
        positions=bank.positions,
        derivatives=bank.derivatives,
    )


def save_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
~~~

save_rollouts stores initial_states, ode_positions, and sde_positions as float32. Add /env/artifacts/ to .gitignore. The runner creates the directory, so do not add a .gitkeep file.

- [ ] **Step 4: Run artifact tests**

Run:

    uv run pytest env/tests/test_artifacts.py -q

Expected: 2 passed.

- [ ] **Step 5: Commit the artifact slice**

    git add .gitignore env/artifacts.py env/tests/test_artifacts.py
    git commit -m "feat: persist Stage A artifacts safely"

### Task 5: Differentiable analytic field

**Files:**
- Create: env/analytic_fields.py
- Create: env/tests/test_analytic_fields.py

**Interfaces:**
- Consumes: train DemonstrationBank and FieldConfig.
- Produces: AnalyticField with sigma, epsilon, centers_and_derivatives, responsibilities, log_density, velocity_pf, score, base_sde_drift, and sample_marginal.

- [ ] **Step 1: Write failing analytic-field tests**

~~~python
import pytest
import torch

from env.analytic_fields import AnalyticField
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank


@pytest.fixture
def field():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=7)
    return AnalyticField(bank.select("train"), DEFAULT_CONFIG)


def test_score_matches_gradient_of_log_density(field):
    actions = torch.tensor(
        [[-0.45, 0.18], [0.0, -0.48], [0.61, 0.21]],
        dtype=torch.float32,
        requires_grad=True,
    )
    times = torch.tensor([0.27, 0.50, 0.79], dtype=torch.float32)
    gradient = torch.autograd.grad(field.log_density(actions, times).sum(), actions)[0]
    torch.testing.assert_close(field.score(actions, times), gradient, rtol=1e-3, atol=2e-4)


def test_field_shapes_dtype_responsibilities_and_drift(field):
    actions = torch.zeros((5, 2), dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
    weights = field.responsibilities(actions, times)
    assert weights.shape == (5, 512)
    assert field.velocity_pf(actions, times).shape == (5, 2)
    assert field.score(actions, times).dtype == torch.float32
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(5), rtol=1e-5, atol=1e-6)
    expected = field.velocity_pf(actions, times) + field.epsilon(times)[:, None] * field.score(actions, times)
    torch.testing.assert_close(field.base_sde_drift(actions, times), expected)


def test_input_contract_is_enforced(field):
    with pytest.raises(ValueError, match="actions"):
        field.score(torch.zeros(2), torch.zeros(1))
    with pytest.raises(ValueError, match="times"):
        field.score(torch.zeros((1, 2)), torch.tensor([1.1]))
    with pytest.raises(ValueError, match="float32"):
        field.score(torch.zeros((1, 2), dtype=torch.float64), torch.zeros(1))
~~~

- [ ] **Step 2: Run and verify red**

Run:

    uv run pytest env/tests/test_analytic_fields.py -q

Expected: import fails because env.analytic_fields does not exist.

- [ ] **Step 3: Implement stable Gaussian-mixture formulas**

Convert train signs and amplitudes once to float32 tensors. centers_and_derivatives vectorizes across batch B and modes M. For delta = actions[:, None, :] - centers:

~~~python
def responsibilities(self, actions, times):
    actions, times = self._validate_inputs(actions, times)
    centers, _ = self.centers_and_derivatives(times)
    sigma = self.sigma(times)[:, None, None]
    logits = -0.5 * (((actions[:, None, :] - centers) / sigma) ** 2).sum(dim=-1)
    return torch.softmax(logits, dim=-1)


def log_density(self, actions, times):
    actions, times = self._validate_inputs(actions, times)
    centers, _ = self.centers_and_derivatives(times)
    sigma = self.sigma(times)
    delta = actions[:, None, :] - centers
    logits = -0.5 * (delta / sigma[:, None, None]).square().sum(dim=-1)
    log_normalizer = torch.log(
        torch.tensor(2.0 * math.pi, dtype=torch.float32)
        * sigma.square()
    )
    return torch.logsumexp(logits, dim=-1) - math.log(self.num_components) - log_normalizer


def velocity_pf(self, actions, times):
    centers, derivatives = self.centers_and_derivatives(times)
    weights = self.responsibilities(actions, times)
    conditional = derivatives - self.k * (actions[:, None, :] - centers)
    return (weights[:, :, None] * conditional).sum(dim=1)


def score(self, actions, times):
    centers, _ = self.centers_and_derivatives(times)
    weights = self.responsibilities(actions, times)
    sigma_sq = self.sigma(times).square()[:, None, None]
    conditional = -(actions[:, None, :] - centers) / sigma_sq
    return (weights[:, :, None] * conditional).sum(dim=1)
~~~

sample_marginal(times, count, generator) samples a bank index and standard normal noise using only the supplied torch.Generator, then returns center + sigma * noise.

- [ ] **Step 4: Run analytic-field tests**

Run:

    uv run pytest env/tests/test_analytic_fields.py -q

Expected: 3 passed.

- [ ] **Step 5: Commit the analytic-field slice**

    git add env/analytic_fields.py env/tests/test_analytic_fields.py
    git commit -m "feat: add analytic Gaussian mixture fields"

### Task 6: ODE and SDE samplers

**Files:**
- Create: env/samplers.py
- Create: env/tests/test_samplers.py

**Interfaces:**
- Consumes: AnalyticField, EnvironmentConfig, initial float32 [B,2], explicit torch.Generator.
- Produces: sample_initial_states, integrate_ode, integrate_sde, all returning float32 tensors with deterministic shapes.

- [ ] **Step 1: Write failing sampler tests**

~~~python
import math
import numpy as np
import torch

from env.analytic_fields import AnalyticField
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank
from env.samplers import integrate_ode, integrate_sde, sample_initial_states


def make_field():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=8)
    return AnalyticField(bank.select("train"), DEFAULT_CONFIG)


def test_initial_and_sampler_shapes_and_reproducibility():
    field = make_field()
    initial_generator = torch.Generator().manual_seed(20)
    initial = sample_initial_states(16, DEFAULT_CONFIG, initial_generator)
    ode = integrate_ode(field, initial, horizon_steps=64)
    first = integrate_sde(
        field, initial, horizon_steps=64, generator=torch.Generator().manual_seed(21)
    )
    second = integrate_sde(
        field, initial, horizon_steps=64, generator=torch.Generator().manual_seed(21)
    )
    assert ode.shape == first.shape == (16, 65, 2)
    assert ode.dtype == first.dtype == torch.float32
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_single_tube_euler_contraction():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=9).select("train")
    one = bank.subset(np.array([0]))
    field = AnalyticField(one, DEFAULT_CONFIG)
    initial = torch.tensor([[-1.0, 0.2]], dtype=torch.float32)
    rollout = integrate_ode(field, initial, horizon_steps=64)
    center, _ = field.centers_and_derivatives(torch.tensor([1.0]))
    final_error = torch.linalg.vector_norm(rollout[:, -1] - center[:, 0], dim=-1)
    expected = torch.tensor([0.2 * math.exp(-1.0)], dtype=torch.float32)
    torch.testing.assert_close(final_error, expected, rtol=2e-2, atol=1e-4)
~~~

- [ ] **Step 2: Run and verify red**

Run:

    uv run pytest env/tests/test_samplers.py -q

Expected: import fails because env.samplers does not exist.

- [ ] **Step 3: Implement explicit-RNG integrators**

~~~python
def integrate_ode(field, initial_states, horizon_steps=64):
    states = _validate_initial(initial_states)
    dt = torch.tensor(1.0 / horizon_steps, dtype=torch.float32, device=states.device)
    trajectory = [states]
    for index in range(horizon_steps):
        times = torch.full(
            (states.shape[0],), index / horizon_steps,
            dtype=torch.float32, device=states.device,
        )
        states = states + dt * field.velocity_pf(states, times)
        trajectory.append(states)
    return torch.stack(trajectory, dim=1)


def integrate_sde(field, initial_states, horizon_steps, generator):
    states = _validate_initial(initial_states)
    dt = torch.tensor(1.0 / horizon_steps, dtype=torch.float32, device=states.device)
    trajectory = [states]
    for index in range(horizon_steps):
        times = torch.full(
            (states.shape[0],), index / horizon_steps,
            dtype=torch.float32, device=states.device,
        )
        epsilon = field.epsilon(times)
        noise = torch.randn(
            states.shape, dtype=torch.float32, device=states.device, generator=generator
        )
        states = (
            states
            + dt * field.base_sde_drift(states, times)
            + torch.sqrt(2.0 * epsilon * dt)[:, None] * noise
        )
        trajectory.append(states)
    return torch.stack(trajectory, dim=1)
~~~

sample_initial_states returns repeated exact starts when center_init=True; otherwise it adds initial_sigma times explicit-generator standard normal noise.

- [ ] **Step 4: Run sampler tests**

Run:

    uv run pytest env/tests/test_samplers.py -q

Expected: 2 passed.

- [ ] **Step 5: Commit the sampler slice**

    git add env/demonstrations.py env/samplers.py env/tests/test_samplers.py
    git commit -m "feat: integrate analytic ODE and SDE samplers"

### Task 7: Stage A diagnostics, documentation, and end-to-end verification

**Files:**
- Create: env/run_stage_a.py
- Create: env/README.md
- Create: env/tests/test_stage_a.py
- Modify: env/artifacts.py

**Interfaces:**
- Consumes: all prior Stage A APIs.
- Produces: compute_marginal_diagnostics, classify_midpoint_modes, run_stage_a, CLI artifacts demonstrations.npz, feature_normalizer.npz, rollouts.npz, resolved_config.json, diagnostics.json.

- [ ] **Step 1: Write failing diagnostic and CLI tests**

~~~python
import json
import subprocess
import sys

from env.run_stage_a import run_stage_a


def test_small_stage_a_run_writes_complete_artifacts(tmp_path):
    result = run_stage_a(tmp_path, seed=23, num_rollouts=128, enforce_acceptance=False)
    assert result["num_rollouts"] == 128
    assert set(result["samplers"]) == {"ode", "sde"}
    assert set(result["samplers"]["ode"]["times"]) == {"0.25", "0.5", "0.75", "1.0"}
    assert set(result["samplers"]["ode"]["midpoint_occupancy"]) == {
        "upper-narrow", "upper-wide", "lower-narrow", "lower-wide", "other"
    }
    for name in (
        "demonstrations.npz",
        "feature_normalizer.npz",
        "rollouts.npz",
        "resolved_config.json",
        "diagnostics.json",
    ):
        assert (tmp_path / name).is_file()
    with (tmp_path / "diagnostics.json").open(encoding="utf-8") as stream:
        assert json.load(stream) == result


def test_module_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "env.run_stage_a", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--num-rollouts" in completed.stdout
~~~

- [ ] **Step 2: Run and verify red**

Run:

    uv run pytest env/tests/test_stage_a.py -q

Expected: import fails because env.run_stage_a does not exist.

- [ ] **Step 3: Implement diagnostics**

At indices 16, 32, 48, and 64, sample an equally sized direct target marginal with its own generator. Compute sample means and unbiased covariance matrices for rollout and target, then report:

~~~python
def sample_errors(samples, reference):
    mean_error = torch.linalg.vector_norm(
        samples.mean(dim=0) - reference.mean(dim=0)
    )
    sample_cov = torch.cov(samples.T)
    reference_cov = torch.cov(reference.T)
    covariance_error = torch.linalg.matrix_norm(sample_cov - reference_cov)
    return float(mean_error.item()), float(covariance_error.item())


def classify_midpoint_modes(samples):
    y = samples[:, 1]
    other = y.abs() < 0.12
    wide = y.abs() >= 0.40
    labels = {
        "upper-narrow": (~other & ~wide & (y > 0)).float().mean(),
        "upper-wide": (~other & wide & (y > 0)).float().mean(),
        "lower-narrow": (~other & ~wide & (y < 0)).float().mean(),
        "lower-wide": (~other & wide & (y < 0)).float().mean(),
        "other": other.float().mean(),
    }
    return {name: float(value.item()) for name, value in labels.items()}
~~~

run_stage_a derives separate NumPy data and Torch initial/ODE-target/SDE-noise/SDE-target streams from the root seed. It saves artifacts before returning the same JSON-safe dictionary written to diagnostics.json. With enforce_acceptance=True, return accepted=False and list every failed criterion; do not conceal or delete outputs. The CLI exits nonzero when full acceptance fails.

- [ ] **Step 4: Run Stage A integration tests**

Run:

    uv run pytest env/tests/test_stage_a.py -q

Expected: 2 passed.

- [ ] **Step 5: Document usage**

env/README.md must show:

    import gym
    import env

    instance = gym.make("PointReach2DPreference-v0")
    observation, info = instance.reset(seed=0)

and:

    uv run python -m env.run_stage_a --output-dir env/artifacts --seed 0
    uv run pytest env/tests -q

Explain that reward is always zero, actions are direct positions, all floating-point computation is float32, generated artifacts are ignored by Git, and diagnostics—not RL reward—define Stage A success.

- [ ] **Step 6: Run the complete unit suite**

Run:

    uv run pytest env/tests -q

Expected: all environment, demonstration, feature, artifact, analytic-field, sampler, and Stage A tests pass.

- [ ] **Step 7: Run full Stage A acceptance**

Run:

    uv run python -m env.run_stage_a       --output-dir env/artifacts       --seed 0       --num-rollouts 4096

Expected: diagnostics.json records accepted=true; ODE and SDE mean/covariance errors are at most 0.03, each midpoint mode is within 0.08 of 0.25, and other is at most 0.02. If a criterion fails, inspect formulas, RNG separation, discretization, and classifier output before considering any tolerance change.

- [ ] **Step 8: Verify repository scope**

Run:

    git diff --check
    git status --short
    git diff -- pyproject.toml uv.lock .gitignore env

Expected: no whitespace errors; only approved Stage A files plus the user's pre-existing pyproject.toml, uv.lock, and source design changes appear.

- [ ] **Step 9: Commit the completed Stage A integration**

Keep pyproject.toml and uv.lock unstaged because they combine approved dependency edits
with the user's pre-existing work. Commit only the independently attributable files.

    git add .gitignore env
    git commit -m "feat: complete Stage A toy environment"
