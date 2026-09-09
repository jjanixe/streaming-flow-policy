# PushT-Compatible Toy Stage B1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a deterministic SFPD policy on the toy expert bank using the repository's PushT windowing, normalization, actual Drake FirstOrderHold transform, and eight-action streaming replanning contract.

**Architecture:** A split-safe toy dataset wraps the repository's generic PushT sampling helpers, shifts the next position into the action array, and normalizes observations/actions with train-only statistics. A Drake-backed datum transform samples one local continuous point and derivative per window, after which a small float32 conditional MLP learns the SFPD velocity and Torchdyn integrates it in normalized action space. A Gym evaluator inverse-normalizes eight-action chunks, records failures without clipping, and writes reproducible metrics and visualizations.

**Tech Stack:** Python 3.10, uv, NumPy 1.x, Drake 1.26, Gym, PyTorch float32, Torchdyn Dopri5, pytest, Matplotlib, mediapy.

**Spec:** `docs/superpowers/specs/2026-09-09-pusht-compatible-toy-stage-b-design.md`

## Global Constraints

- Keep `drake<1.27` because the host uses glibc 2.31; add `numpy<2` because Drake 1.26 does not import with NumPy 2.
- Drake FirstOrderHold construction and evaluation are the only allowed float64 boundary. Convert `tau`, `xi(tau)`, and `xi_dot(tau)` to float32 immediately.
- Saved demonstrations, normalized samples, noise, model parameters, losses, learned ODE states, Gym observations/actions, and rollout artifacts remain float32.
- Preserve `pred_horizon=16`, `obs_horizon=2`, `action_horizon=8`, and chunk-local waypoint times `j/15`.
- Fit normalization statistics only on complete unwindowed train trajectories. Validation, test, and rollout use the saved train statistics.
- Use the PushT next-observation action shift, `pad_before=1`, `pad_after=7`, endpoint repetition, and first-action anchor correction.
- The Gym environment receives one absolute physical `[2]` action per step and enforces `max_step_distance=0.075` without clipping.
- Mode IDs, signs, amplitudes, trajectory IDs, split labels, and padding metadata never enter the learned model.
- Use explicit seeded random streams; do not rely on process-global NumPy or Torch random state.
- Preserve unrelated working-tree changes and stage only the files named by each task.
- Stage B1 artifacts live below ignored `env/artifacts/stage_b/<run_id>/`; do not commit checkpoints or generated media.
- B2/SFPS, denoisers, score-corrected SDEs, preferences, STEG, and merge/rebranch demonstrations are outside this plan.

## File Structure

- Modify `pyproject.toml` and `uv.lock` only for the NumPy/Drake-compatible environment.
- Create `env/stage_b_config.py` for immutable B1 hyperparameters and validation.
- Create `env/chunk_data.py` for next-position shifting, train-only min/max statistics, PushT-compatible windows, normalization, and metadata.
- Create `env/drake_trajectory.py` for actual Drake FirstOrderHold evaluation and SFPD training-datum construction.
- Create `env/models.py` for the small conditional float32 SFPD velocity MLP.
- Create `env/sfp_policies.py` for SFPD loss and Torchdyn streaming integration.
- Create `env/stage_b_artifacts.py` for stats, checkpoint, history, rollout, and compatibility serialization.
- Create `env/train_stage_b.py` for seeded minibatches, validation, AdamW, cosine scheduling, EMA, and best-checkpoint selection.
- Create `env/evaluate_stage_b.py` for closed-loop Gym rollouts, action-limit accounting, occupancy, and development gates.
- Create `env/visualize_stage_b.py` for trajectory, occupancy, and representative rollout media.
- Create `env/run_stage_b.py` for the B1 command-line workflow.
- Modify `env/README.md` with the B1 invocation and artifact contract.
- Add focused tests under `env/tests/` matching each implementation file.

---

### Task 1: Make the uv environment able to import Drake

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `env/tests/test_drake_compatibility.py`

**Interfaces:**
- Consumes: the existing user-selected `drake<1.27` dependency.
- Produces: a locked environment where a fresh Python process imports `pydrake.trajectories.PiecewisePolynomial` with NumPy `<2`.

- [ ] **Step 1: Write the failing subprocess import test**

```python
import subprocess
import sys


def test_drake_first_order_hold_imports_in_fresh_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as m; "
                "from pydrake.trajectories import PiecewisePolynomial; "
                "print(m.version('drake')); "
                "print(m.version('numpy')); "
                "print(PiecewisePolynomial.__name__)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    drake_version, numpy_version, class_name = completed.stdout.splitlines()
    assert drake_version.startswith("1.26.")
    assert int(numpy_version.split(".", maxsplit=1)[0]) < 2
    assert class_name == "PiecewisePolynomial"
```

- [ ] **Step 2: Run the test and verify the current NumPy 2 environment fails**

Run: `uv run pytest env/tests/test_drake_compatibility.py -v`

Expected: FAIL because Drake 1.26 reaches NumPy's removed `ndarray.itemset` API in the subprocess.

- [ ] **Step 3: Add the explicit NumPy compatibility bound and refresh the lock**

Keep the existing Drake comment and dependency, and make the relevant dependency block read:

```toml
    # Drake 1.27+ requires glibc 2.35 or newer on Linux; this project supports
    # manylinux_2_31 hosts as well. Drake 1.26 requires NumPy 1.x.
    "numpy<2",
    "drake<1.27",
```

Run:

```bash
uv lock
uv sync --group dev
```

- [ ] **Step 4: Verify the import and existing suite under the resolved environment**

Run: `uv run pytest env/tests/test_drake_compatibility.py -v`

Expected: PASS with Drake 1.26.x, NumPy 1.26.x, and `PiecewisePolynomial`.

Run: `uv run pytest env/tests -q`

Expected: all existing Stage A/rendering tests plus the compatibility test pass.

- [ ] **Step 5: Commit only the dependency compatibility change**

```bash
git add pyproject.toml uv.lock env/tests/test_drake_compatibility.py
git commit -m "fix: make Drake compatible with the uv environment"
```

---

### Task 2: Add the B1 configuration and PushT-compatible chunk dataset

**Files:**
- Create: `env/stage_b_config.py`
- Create: `env/chunk_data.py`
- Create: `env/tests/test_chunk_data.py`

**Interfaces:**
- Consumes: `env.demonstrations.DemonstrationBank` and the generic PushT helpers `create_sample_indices`, `sample_sequence`, `normalize_data`, and `unnormalize_data`.
- Produces: `StageB1Config`, `PushTStats`, `fit_pusht_stats`, `normalize_observations`, `unnormalize_observations`, `normalize_actions`, `unnormalize_actions`, and `PushTChunkDataset`.

- [ ] **Step 1: Write failing configuration and array-shift tests**

```python
import numpy as np

from env.chunk_data import build_episode_arrays
from env.stage_b_config import DEFAULT_STAGE_B1_CONFIG


def test_stage_b1_defaults_match_pusht_contract():
    config = DEFAULT_STAGE_B1_CONFIG
    assert config.pred_horizon == 16
    assert config.obs_horizon == 2
    assert config.action_horizon == 8
    assert config.sigma == 0.1
    assert config.integration_steps_per_action == 6


def test_episode_actions_are_shifted_next_positions_with_terminal_repeat():
    positions = np.arange(10, dtype=np.float32).reshape(5, 2)
    obs, action = build_episode_arrays(positions)
    np.testing.assert_array_equal(obs[:, :2], positions)
    np.testing.assert_array_equal(action[:-1], positions[1:])
    np.testing.assert_array_equal(action[-1], positions[-1])
    np.testing.assert_array_equal(
        obs[:, 2], np.linspace(0.0, 1.0, 5, dtype=np.float32)
    )
    assert obs.dtype == action.dtype == np.float32
```

- [ ] **Step 2: Run the focused tests and verify imports fail**

Run: `uv run pytest env/tests/test_chunk_data.py -v`

Expected: FAIL because `env.stage_b_config` and `env.chunk_data` do not exist.

- [ ] **Step 3: Implement immutable B1 defaults and validation**

Create `env/stage_b_config.py` with this public shape:

```python
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StageB1Config:
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    sigma: float = 0.1
    hidden_dim: int = 128
    hidden_layers: int = 3
    batch_size: int = 1024
    max_updates: int = 20_000
    validation_interval: int = 1_000
    warmup_updates: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    ema_decay: float = 0.999
    integration_steps_per_action: int = 6
    rollout_count: int = 1024

    def __post_init__(self) -> None:
        if (self.pred_horizon, self.obs_horizon, self.action_horizon) != (16, 2, 8):
            raise ValueError("Stage B1 requires PushT horizons 16/2/8")
        if not 0.0 <= self.sigma:
            raise ValueError("sigma must be nonnegative")
        if self.max_updates <= 0 or self.batch_size <= 0:
            raise ValueError("training counts must be positive")


DEFAULT_STAGE_B1_CONFIG = StageB1Config()


def stage_b1_config_to_dict(config: StageB1Config) -> dict[str, Any]:
    return asdict(config)
```

Implement `build_episode_arrays(positions)` with shape, finite-value, and float32 validation. It returns observation `[position, normalized_episode_time]` and next-position actions exactly as the test specifies.

- [ ] **Step 4: Write failing normalization and split-isolation tests**

```python
import copy

from env.chunk_data import fit_pusht_stats, normalize_actions, unnormalize_actions
from env.config import DEFAULT_CONFIG
from env.demonstrations import generate_demonstration_bank


def test_stats_are_fit_only_from_train_trajectories():
    original = generate_demonstration_bank(DEFAULT_CONFIG, seed=11)
    changed = copy.deepcopy(original)
    changed.positions[changed.splits != "train"] += np.float32(100.0)
    first = fit_pusht_stats(original)
    second = fit_pusht_stats(changed)
    np.testing.assert_array_equal(first.obs_min, second.obs_min)
    np.testing.assert_array_equal(first.obs_max, second.obs_max)
    np.testing.assert_array_equal(first.action_min, second.action_min)
    np.testing.assert_array_equal(first.action_max, second.action_max)


def test_action_normalization_round_trips_in_float32():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=12)
    stats = fit_pusht_stats(bank)
    actions = bank.select("validation").positions[:2, :4]
    restored = unnormalize_actions(normalize_actions(actions, stats), stats)
    np.testing.assert_allclose(restored, actions, atol=2e-7)
    assert restored.dtype == np.float32
```

- [ ] **Step 5: Implement train-only min/max statistics**

Use this exact persisted type:

```python
@dataclass(frozen=True)
class PushTStats:
    obs_min: np.ndarray       # float32 [3]
    obs_max: np.ndarray       # float32 [3]
    action_min: np.ndarray    # float32 [2]
    action_max: np.ndarray    # float32 [2]

    def __post_init__(self) -> None:
        shapes = {
            "obs_min": (3,),
            "obs_max": (3,),
            "action_min": (2,),
            "action_max": (2,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float32 values")
        if np.any(self.obs_max <= self.obs_min):
            raise ValueError("observation ranges must be positive")
        if np.any(self.action_max <= self.action_min):
            raise ValueError("action ranges must be positive")
```

The implementation may replace the comment-only validation body above with explicit checks; no ellipsis remains in production. `fit_pusht_stats(bank)` selects only `train`, builds complete unwindowed arrays, and computes float32 minima/maxima. Wrapper functions call the repository normalization formulas and cast results back to float32.

- [ ] **Step 6: Write failing exact-window tests**

```python
from env.chunk_data import PushTChunkDataset


def test_each_65_state_episode_produces_58_pusht_windows():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=13)
    stats = fit_pusht_stats(bank)
    dataset = PushTChunkDataset(bank, split="validation", stats=stats)
    assert len(dataset) == len(bank.select("validation")) * 58


def test_first_and_last_windows_match_pusht_padding_and_anchor():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=14)
    split = bank.select("validation")
    stats = fit_pusht_stats(bank)
    dataset = PushTChunkDataset(bank, split="validation", stats=stats)
    first = dataset[0]
    last = dataset[57]
    assert first["sequence_start"] == np.int64(-1)
    assert first["anchor_index"] == np.int64(0)
    np.testing.assert_array_equal(first["obs"][0], first["obs"][1])
    np.testing.assert_array_equal(first["action"][0], first["obs"][-1, :2])
    assert first["anchor_physical_l2"] > np.float32(0.0)
    assert first["anchor_physical_l2"] < np.float32(1e-3)
    assert last["sequence_start"] == np.int64(56)
    assert last["anchor_index"] == np.int64(57)
    np.testing.assert_array_equal(last["action"][-1], last["action"][-2])
    assert first["obs"].shape == (2, 3)
    assert first["action"].shape == (16, 2)
    assert first["trajectory_id"] == split.trajectory_ids[0]
```

- [ ] **Step 7: Implement the dataset with repository sampling helpers**

`PushTChunkDataset` has this constructor and datum contract:

```python
class PushTChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bank: DemonstrationBank,
        split: str,
        stats: PushTStats,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        selected = bank.select(split)
        if len(selected) == 0:
            raise ValueError(f"split {split!r} contains no trajectories")
        self.stats = stats
        self.transform = transform
        self.trajectory_ids = selected.trajectory_ids
        self.episodes = []
        for positions in selected.positions:
            obs, action = build_episode_arrays(positions)
            self.episodes.append({
                "obs": normalize_observations(obs, stats),
                "action": normalize_actions(action, stats),
            })
        local = create_sample_indices(
            np.asarray([65], dtype=np.int64),
            sequence_length=16,
            pad_before=1,
            pad_after=7,
        )
        self.windows = [
            (trajectory_index, row.copy())
            for trajectory_index in range(len(self.episodes))
            for row in local
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_index, row = self.windows[index]
        buffer_start, buffer_end, sample_start, sample_end = map(int, row)
        sample = sample_sequence(
            self.episodes[trajectory_index],
            sequence_length=16,
            buffer_start_idx=buffer_start,
            buffer_end_idx=buffer_end,
            sample_start_idx=sample_start,
            sample_end_idx=sample_end,
        )
        obs_window = sample["obs"][:2].copy()
        action_window = sample["action"].copy()
        before = action_window[[0]].copy()
        if not np.all(action_window[0] == obs_window[-1, :2]):
            action_window[0] = obs_window[-1, :2]
        correction = np.linalg.norm(
            unnormalize_actions(action_window[[0]], self.stats)
            - unnormalize_actions(before, self.stats)
        )
        physical_anchor = unnormalize_actions(
            action_window[[0]], self.stats
        )[0]
        physical_observation = unnormalize_observations(
            obs_window[[-1]], self.stats
        )[0, :2]
        anchor_physical_l2 = np.linalg.norm(
            physical_anchor - physical_observation
        )
        sequence_start = buffer_start - sample_start
        datum = {
            "obs": obs_window,                    # float32 [2, 3]
            "action": action_window,              # float32 [16, 2]
            "trajectory_id": str(self.trajectory_ids[trajectory_index]),
            "sequence_start": np.int64(sequence_start),
            "anchor_index": np.int64(sequence_start + 1),
            "pad_before": np.int64(sample_start),
            "pad_after": np.int64(16 - sample_end),
            "anchor_correction_raw_l2": np.float32(correction),
            "anchor_physical_l2": np.float32(anchor_physical_l2),
        }
        return self.transform(datum) if self.transform is not None else datum
```

Use the generic repository functions for index creation and endpoint repetition, then keep only the first two observations, copy before anchor correction, and apply `transform` last when provided. Add `__len__` returning `len(self.windows)` and validate the fixed 65-state episode shape before constructing windows.

- [ ] **Step 8: Run dataset tests and the existing demonstration tests**

Run: `uv run pytest env/tests/test_chunk_data.py env/tests/test_demonstrations.py -v`

Expected: PASS, including exactly 58 windows per trajectory and no validation/test leakage into statistics.

- [ ] **Step 9: Commit the dataset contract**

```bash
git add env/stage_b_config.py env/chunk_data.py env/tests/test_chunk_data.py
git commit -m "feat: add PushT-compatible toy chunk dataset"
```

---

### Task 3: Transform expert windows with actual Drake trajectories

**Files:**
- Create: `env/drake_trajectory.py`
- Create: `env/tests/test_drake_trajectory.py`

**Interfaces:**
- Consumes: a `PushTChunkDataset` datum containing normalized float32 `obs [2,3]` and `action [16,2]`.
- Produces: `evaluate_drake_foh(action, tau)` and `SFPDDrakeTransform(sigma, rng)` returning float32 `obs`, `x [1,2]`, `v [1,2]`, and scalar `t` plus unchanged metadata.

- [ ] **Step 1: Write failing Drake evaluation tests**

```python
import numpy as np

from env.drake_trajectory import evaluate_drake_foh


def test_drake_first_order_hold_matches_linear_segments():
    action = np.stack(
        [np.arange(16, dtype=np.float32), -np.arange(16, dtype=np.float32)],
        axis=-1,
    )
    position, derivative = evaluate_drake_foh(action, np.float32(0.1))
    np.testing.assert_allclose(position, [[1.5, -1.5]], atol=1e-6)
    np.testing.assert_allclose(derivative, [[15.0, -15.0]], atol=1e-6)
    assert position.dtype == derivative.dtype == np.float32


def test_drake_boundary_is_not_used_as_a_float64_training_output():
    action = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2)
    position, derivative = evaluate_drake_foh(action, np.float32(0.37))
    assert position.shape == derivative.shape == (1, 2)
    assert position.dtype == derivative.dtype == np.float32
```

- [ ] **Step 2: Run the tests and verify the transform is missing**

Run: `uv run pytest env/tests/test_drake_trajectory.py -v`

Expected: FAIL because `env.drake_trajectory` does not exist.

- [ ] **Step 3: Implement the actual Drake FirstOrderHold boundary**

```python
from pydrake.trajectories import PiecewisePolynomial


def evaluate_drake_foh(
    action: np.ndarray,
    tau: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    # Validate action is finite float32 [16, 2] and tau is in [0, 1].
    breaks = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    trajectory = PiecewisePolynomial.FirstOrderHold(
        breaks,
        action.astype(np.float64, copy=False).T,
    )
    position = trajectory.value(float(tau)).T.astype(np.float32)
    derivative = trajectory.EvalDerivative(float(tau)).T.astype(np.float32)
    return position, derivative
```

Do not add a production manual interpolation fallback. If Drake import or evaluation fails, raise an error identifying the dependency boundary.

- [ ] **Step 4: Write failing seeded SFPD datum tests**

```python
from env.drake_trajectory import SFPDDrakeTransform


def test_sfpd_transform_is_seeded_and_returns_only_float32_arrays():
    datum = {
        "obs": np.zeros((2, 3), dtype=np.float32),
        "action": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2),
        "trajectory_id": "validation:upper-narrow:000",
    }
    first = SFPDDrakeTransform(0.1, np.random.default_rng(7))(datum)
    second = SFPDDrakeTransform(0.1, np.random.default_rng(7))(datum)
    for key in ("obs", "x", "v", "t"):
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].dtype == np.float32
    assert first["x"].shape == first["v"].shape == (1, 2)
    assert np.asarray(first["t"]).shape == ()
    assert first["trajectory_id"] == datum["trajectory_id"]
```

- [ ] **Step 5: Implement the seeded transform**

```python
class SFPDDrakeTransform:
    def __init__(self, sigma: float, rng: np.random.Generator) -> None:
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and nonnegative")
        self.sigma = np.float32(sigma)
        self.rng = rng

    def __call__(self, datum: dict[str, Any]) -> dict[str, Any]:
        tau = np.float32(self.rng.random())
        xi, xi_dot = evaluate_drake_foh(datum["action"], tau)
        noise = self.rng.standard_normal(xi.shape, dtype=np.float32)
        transformed = {key: value for key, value in datum.items() if key != "action"}
        transformed.update(
            x=(xi + self.sigma * noise).astype(np.float32),
            v=xi_dot.astype(np.float32, copy=False),
            t=np.asarray(tau, dtype=np.float32),
        )
        return transformed
```

- [ ] **Step 6: Run transform and compatibility tests**

Run: `uv run pytest env/tests/test_drake_trajectory.py env/tests/test_drake_compatibility.py -v`

Expected: PASS; no training output is float64.

- [ ] **Step 7: Commit the Drake-backed transform**

```bash
git add env/drake_trajectory.py env/tests/test_drake_trajectory.py
git commit -m "feat: build SFPD targets with Drake trajectories"
```

---

### Task 4: Add the toy conditional velocity MLP

**Files:**
- Create: `env/models.py`
- Create: `env/tests/test_models.py`

**Interfaces:**
- Consumes: `sample` float32 `[B,1,2]`, `timestep` float32 `[B]`, and `global_cond` float32 `[B,6]`.
- Produces: `SFPDVelocityMLP.forward(sample, timestep, global_cond) -> Tensor[B,1,2]` and nine local-time features `[t,sin/cos(2*pi*f*t)]` for frequencies `1,2,4,8`.

- [ ] **Step 1: Write failing shape, dtype, and validation tests**

```python
import pytest
import torch

from env.models import SFPDVelocityMLP


def test_sfpd_velocity_mlp_preserves_sample_shape_and_float32():
    model = SFPDVelocityMLP(hidden_dim=128, hidden_layers=3)
    sample = torch.zeros((5, 1, 2), dtype=torch.float32)
    time = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
    condition = torch.zeros((5, 6), dtype=torch.float32)
    output = model(sample=sample, timestep=time, global_cond=condition)
    assert output.shape == sample.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_sfpd_velocity_mlp_rejects_float64():
    model = SFPDVelocityMLP()
    with pytest.raises(ValueError, match="float32"):
        model(
            sample=torch.zeros((1, 1, 2), dtype=torch.float64),
            timestep=torch.zeros(1, dtype=torch.float32),
            global_cond=torch.zeros((1, 6), dtype=torch.float32),
        )
```

- [ ] **Step 2: Run the tests and verify the model is missing**

Run: `uv run pytest env/tests/test_models.py -v`

Expected: FAIL because `env.models` does not exist.

- [ ] **Step 3: Implement Fourier features and the three-layer MLP**

```python
class LocalTimeFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * torch.pi * timestep[:, None] * self.frequencies[None, :]
        return torch.cat((timestep[:, None], torch.sin(angles), torch.cos(angles)), dim=1)


class SFPDVelocityMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, hidden_layers: int = 3) -> None:
        super().__init__()
        dimensions = [2 + 9 + 6] + [hidden_dim] * hidden_layers + [2]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.time_features = LocalTimeFeatures()
        self.network = nn.Sequential(*layers)
```

In `forward`, validate shapes, float32, common device, finite inputs, and `0 <= timestep <= 1`; concatenate `sample[:,0,:]`, time features, and condition, then restore the `[B,1,2]` axis.

- [ ] **Step 4: Run model tests**

Run: `uv run pytest env/tests/test_models.py -v`

Expected: PASS for shapes and validation.

- [ ] **Step 5: Commit the B1 network**

```bash
git add env/models.py env/tests/test_models.py
git commit -m "feat: add toy SFPD velocity model"
```

---

### Task 5: Implement SFPD loss and PushT-compatible ODE prediction

**Files:**
- Create: `env/sfp_policies.py`
- Create: `env/tests/test_sfp_policies.py`

**Interfaces:**
- Consumes: a velocity network with the Task 4 signature and batches containing `obs`, `x`, `v`, `t`.
- Produces: `StreamingFlowPolicyDeterministic.loss(batch) -> scalar Tensor` and `predict(nobs, num_actions, integration_steps_per_action) -> Tensor[1,N,2]`.

- [ ] **Step 1: Write the failing loss test**

```python
import torch

from env.models import SFPDVelocityMLP
from env.sfp_policies import StreamingFlowPolicyDeterministic


def test_sfpd_loss_is_scalar_finite_float32():
    policy = StreamingFlowPolicyDeterministic(SFPDVelocityMLP(), pred_horizon=16)
    batch = {
        "obs": torch.zeros((4, 2, 3), dtype=torch.float32),
        "x": torch.zeros((4, 1, 2), dtype=torch.float32),
        "v": torch.ones((4, 1, 2), dtype=torch.float32),
        "t": torch.full((4,), 0.5, dtype=torch.float32),
    }
    loss = policy.loss(batch)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run the loss test and verify the policy is missing**

Run: `uv run pytest env/tests/test_sfp_policies.py::test_sfpd_loss_is_scalar_finite_float32 -v`

Expected: FAIL because `env.sfp_policies` does not exist.

- [ ] **Step 3: Implement the policy loss**

```python
class StreamingFlowPolicyDeterministic(nn.Module):
    def __init__(
        self,
        velocity_net: nn.Module,
        pred_horizon: int = 16,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if pred_horizon != 16:
            raise ValueError("pred_horizon must be 16")
        self.velocity_net = velocity_net
        self.device = torch.device(device)
        self.register_buffer("pred_horizon", torch.tensor(16, dtype=torch.int32))

    def loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs"].to(self.device, dtype=torch.float32)
        x = batch["x"].to(self.device, dtype=torch.float32)
        target = batch["v"].to(self.device, dtype=torch.float32)
        timestep = batch["t"].to(self.device, dtype=torch.float32)
        condition = obs.flatten(start_dim=1)
        prediction = self.velocity_net(x, timestep, condition)
        return torch.nn.functional.mse_loss(prediction, target)
```

Reject source batch tensors that are not already float32 instead of silently downcasting; `.to` above handles device transfer only after validation.

- [ ] **Step 4: Write failing constant-field integration tests**

```python
class ConstantVelocity(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        velocity = torch.tensor([0.3, -0.15], dtype=torch.float32, device=sample.device)
        return velocity.expand_as(sample)


def test_prediction_starts_at_anchor_and_samples_eight_future_actions():
    policy = StreamingFlowPolicyDeterministic(ConstantVelocity())
    nobs = torch.tensor(
        [[-0.5, 0.25, -1.0], [-0.4, 0.2, -0.75]],
        dtype=torch.float32,
    )
    actions = policy.predict(nobs, num_actions=9, integration_steps_per_action=2)
    assert actions.shape == (1, 9, 2)
    torch.testing.assert_close(actions[0, 0], nobs[-1, :2])
    torch.testing.assert_close(
        actions[0, -1],
        nobs[-1, :2] + torch.tensor([0.3, -0.15]) * (8.0 / 15.0),
        atol=1e-4,
        rtol=1e-4,
    )
```

- [ ] **Step 5: Implement Torchdyn Dopri5 inference**

Create a `_ConditionedVectorField` wrapper that reshapes Torchdyn's action `[2]` to `[1,1,2]`, expands scalar time to `[1]`, invokes the model with the fixed flattened condition, and returns `[2]`.

`predict` must:

```python
num_future_actions = num_actions - 1
t_max = num_future_actions / (self.pred_horizon.item() - 1)
total_steps = 1 + num_future_actions * integration_steps_per_action
t_span = torch.linspace(
    0.0,
    t_max,
    total_steps,
    dtype=torch.float32,
    device=nobs.device,
)
solver = NeuralODE(
    _ConditionedVectorField(self.velocity_net, condition),
    solver="dopri5",
    sensitivity="adjoint",
    atol=1e-4,
    rtol=1e-4,
)
trajectory = solver.trajectory(x=nobs[-1, :2], t_span=t_span)
indices = torch.arange(0, total_steps, integration_steps_per_action)
return trajectory[indices].unsqueeze(0)
```

Validate `1 <= num_actions <= 16`, positive integration density, input shape `[2,3]`, float32, finite values, and common model/input device.

- [ ] **Step 6: Run policy tests**

Run: `uv run pytest env/tests/test_sfp_policies.py -v`

Expected: PASS with anchor plus eight future samples and no float64 tensors.

- [ ] **Step 7: Commit the deterministic streaming policy**

```bash
git add env/sfp_policies.py env/tests/test_sfp_policies.py
git commit -m "feat: add deterministic streaming flow policy"
```

---

### Task 6: Train with seeded validation, EMA, and compatible checkpoints

**Files:**
- Create: `env/stage_b_artifacts.py`
- Create: `env/train_stage_b.py`
- Create: `env/tests/test_stage_b_artifacts.py`
- Create: `env/tests/test_train_stage_b.py`

**Interfaces:**
- Consumes: `DemonstrationBank`, `StageB1Config`, Task 2 stats/dataset, Task 3 transform, Task 4 model, and Task 5 policy.
- Produces: `save_pusht_stats`, `load_pusht_stats`, `save_sfpd_checkpoint`, `load_sfpd_checkpoint`, `TrainResult`, and `train_sfpd`.

- [ ] **Step 1: Write failing stats and checkpoint round-trip tests**

```python
from env.stage_b_artifacts import (
    load_pusht_stats,
    load_sfpd_checkpoint,
    save_pusht_stats,
    save_sfpd_checkpoint,
)


def test_pusht_stats_round_trip_without_pickle(tmp_path):
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=21)
    stats = fit_pusht_stats(bank)
    path = tmp_path / "pusht_stats.npz"
    save_pusht_stats(path, stats, train_data_digest(bank))
    loaded, digest = load_pusht_stats(path)
    np.testing.assert_array_equal(loaded.obs_min, stats.obs_min)
    np.testing.assert_array_equal(loaded.action_max, stats.action_max)
    assert digest == train_data_digest(bank)


def test_checkpoint_rejects_wrong_demonstration_digest(tmp_path):
    model = SFPDVelocityMLP(hidden_dim=16, hidden_layers=1)
    checkpoint = tmp_path / "model.pt"
    save_sfpd_checkpoint(
        checkpoint,
        raw_state=model.state_dict(),
        ema_state=model.state_dict(),
        metadata={"model_type": "sfpd", "train_data_digest": "correct"},
    )
    with pytest.raises(ValueError, match="digest"):
        load_sfpd_checkpoint(checkpoint, expected_train_data_digest="wrong")
```

- [ ] **Step 2: Run artifact tests and verify the module is missing**

Run: `uv run pytest env/tests/test_stage_b_artifacts.py -v`

Expected: FAIL because `env.stage_b_artifacts` does not exist.

- [ ] **Step 3: Implement portable statistics and weights-only checkpoints**

Save stats with `np.savez_compressed` using only numeric arrays and fixed-width Unicode strings so loading with `allow_pickle=False` is valid. Save checkpoints with `torch.save` containing only tensors, primitive containers, and strings. Load checkpoints with `torch.load(path, map_location="cpu", weights_only=True)`. Require metadata keys:

```python
REQUIRED_METADATA = {
    "format_version",
    "model_type",
    "architecture",
    "stage_b1_config",
    "selected_update",
    "validation_loss",
    "root_seed",
    "seed_streams",
    "train_data_digest",
    "drake_version",
    "numpy_version",
}
```

Reject wrong format version, model type, architecture, missing keys, and digest before loading state into a model.

- [ ] **Step 4: Write failing short-training tests**

```python
from dataclasses import replace

from env.train_stage_b import train_sfpd


def test_short_cpu_training_writes_finite_best_ema_checkpoint(tmp_path):
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=22)
    config = replace(
        DEFAULT_STAGE_B1_CONFIG,
        hidden_dim=16,
        hidden_layers=1,
        batch_size=64,
        max_updates=4,
        validation_interval=2,
        warmup_updates=1,
    )
    result = train_sfpd(bank, tmp_path, config=config, root_seed=23, device="cpu")
    assert result.checkpoint_path.is_file()
    assert result.stats_path.is_file()
    assert result.history_path.is_file()
    assert result.selected_update in (2, 4)
    assert np.isfinite(result.validation_loss)
    loaded = load_sfpd_checkpoint(
        result.checkpoint_path,
        expected_train_data_digest=train_data_digest(bank),
    )
    assert loaded["metadata"]["model_type"] == "sfpd"
```

- [ ] **Step 5: Implement deterministic seed derivation and EMA**

Derive six named seeds from `np.random.SeedSequence(root_seed).spawn(6)` for model initialization, DataLoader shuffle, train transform, validation transform, rollout initialization, and checkpoint replay. Record each generated uint32 seed.

Construct the model inside `torch.random.fork_rng(devices=[])`, call
`torch.manual_seed(model_initialization_seed)` inside that context, and exit the
context immediately after construction. This makes initialization reproducible
while restoring the caller's process-global Torch RNG state. Pass a dedicated
`torch.Generator().manual_seed(dataloader_shuffle_seed)` to the DataLoader.

Implement EMA with float32 shadow tensors:

```python
class ExponentialMovingAverage:
    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())
```

- [ ] **Step 6: Implement the finite-update trainer**

`train_sfpd` must:

1. fit and save train-only stats and digest;
2. build train and validation `PushTChunkDataset` objects;
3. use `num_workers=0`, a seeded shuffle generator, AdamW, 500-step linear warmup followed by cosine decay, and EMA decay 0.999;
4. cycle minibatches until exactly `max_updates` optimizer steps;
5. pre-materialize fixed validation batches from a separate seeded transform;
6. validate at every configured interval and at the final update;
7. reject non-finite losses or gradients immediately;
8. keep the lowest-validation-loss EMA state;
9. write JSON history using `allow_nan=False`; and
10. save raw final weights, best EMA weights, metadata, and seed streams.

Use this result type:

```python
@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    stats_path: Path
    history_path: Path
    selected_update: int
    validation_loss: float
    seed_streams: dict[str, int]
```

- [ ] **Step 7: Run trainer and artifact tests twice for reproducibility**

Run: `uv run pytest env/tests/test_stage_b_artifacts.py env/tests/test_train_stage_b.py -v`

Expected: PASS. Add a second short run with the same root seed and assert identical validation history and EMA state tensors.

- [ ] **Step 8: Commit training and checkpoint support**

```bash
git add env/stage_b_artifacts.py env/train_stage_b.py env/tests/test_stage_b_artifacts.py env/tests/test_train_stage_b.py
git commit -m "feat: train and checkpoint toy SFPD"
```

---

### Task 7: Evaluate eight-action streaming rollouts in Gym

**Files:**
- Create: `env/evaluate_stage_b.py`
- Create: `env/tests/test_evaluate_stage_b.py`

**Interfaces:**
- Consumes: an SFPD policy, `PushTStats`, `EnvironmentConfig`, explicit rollout seeds, and the saved test demonstration distribution.
- Produces: `SFPDRolloutBatch`, `rollout_sfpd`, `evaluate_sfpd`, and `stage_b1_acceptance_failures`.

- [ ] **Step 1: Write a failing streaming-contract test with a scripted policy**

```python
class ScriptedNormalizedPolicy:
    def __init__(self, normalized_positions: np.ndarray) -> None:
        self.normalized_positions = normalized_positions
        self.calls = 0

    def predict(self, nobs, num_actions, integration_steps_per_action):
        start = self.calls * 8
        self.calls += 1
        chunk = self.normalized_positions[start : start + 9]
        return torch.from_numpy(chunk[None]).to(dtype=torch.float32)


def test_rollout_replans_eight_times_and_executes_64_actions():
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=31)
    stats = fit_pusht_stats(bank)
    expert = bank.select("test").positions[0]
    scripted = ScriptedNormalizedPolicy(normalize_actions(expert, stats))
    rollout = rollout_sfpd(
        scripted,
        stats,
        DEFAULT_CONFIG.environment,
        seed=32,
        center_init=True,
        integration_steps_per_action=1,
    )
    assert scripted.calls == 8
    assert rollout.executed_action_count == 64
    assert rollout.success
    np.testing.assert_allclose(rollout.executed_positions, expert, atol=2e-6)
```

- [ ] **Step 2: Run the test and verify evaluation is missing**

Run: `uv run pytest env/tests/test_evaluate_stage_b.py -v`

Expected: FAIL because `env.evaluate_stage_b` does not exist.

- [ ] **Step 3: Implement one closed-loop rollout**

Use this result shape:

```python
@dataclass(frozen=True)
class SFPDRollout:
    seed: int
    initial_observation: np.ndarray        # float32 [3]
    executed_positions: np.ndarray         # float32 [K+1, 2]
    requested_actions: np.ndarray          # float32 [K, 2]
    raw_predicted_chunks: np.ndarray       # float32 [C, 9, 2]
    executed_action_count: int
    success: bool
    numerical_failure: bool
    action_limit_failure: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class SFPDRolloutBatch:
    rollouts: tuple[SFPDRollout, ...]

    @classmethod
    def from_rollouts(
        cls,
        rollouts: Sequence[SFPDRollout],
    ) -> "SFPDRolloutBatch":
        if len(rollouts) == 0:
            raise ValueError("at least one rollout is required")
        return cls(tuple(rollouts))
```

At each replan, normalize the observation deque with train observation stats, call `predict(nobs, num_actions=9, integration_steps_per_action=integration_steps_per_action)`, inverse-normalize all nine positions with action stats, record the full raw chunk, discard index zero, and step Gym through indices one to eight. Stop on termination and never clip or replace a failed action.

- [ ] **Step 4: Write failing failure-accounting and deterministic-replay tests**

Add a scripted policy whose first future action is `0.2` away from the current position. Assert one Gym action-limit activation, unchanged state, preserved raw requested action, and no numerical failure. Run a valid scripted policy twice with the same seed and assert byte-identical arrays.

- [ ] **Step 5: Implement batch evaluation and development gates**

```python
def evaluate_sfpd(
    policy: StreamingFlowPolicyDeterministic,
    stats: PushTStats,
    environment_config: EnvironmentConfig,
    rollout_seeds: Sequence[int],
    *,
    center_init: bool,
    integration_steps_per_action: int,
) -> tuple[dict[str, Any], SFPDRolloutBatch]:
    rollouts = [
        rollout_sfpd(
            policy,
            stats,
            environment_config,
            seed=seed,
            center_init=center_init,
            integration_steps_per_action=integration_steps_per_action,
        )
        for seed in rollout_seeds
    ]
    batch = SFPDRolloutBatch.from_rollouts(rollouts)
    metrics = aggregate_sfpd_metrics(batch, environment_config)
    metrics["acceptance_failures"] = stage_b1_acceptance_failures(metrics)
    return metrics, batch


def stage_b1_acceptance_failures(metrics: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if metrics["goal_success_rate"] < 0.95:
        failures.append("goal success rate is below 0.95")
    if metrics["numerical_failure_count"] != 0:
        failures.append("numerical failures are nonzero")
    for mode in ("upper-narrow", "upper-wide", "lower-narrow", "lower-wide"):
        occupancy = metrics["midpoint_occupancy"][mode]
        if occupancy < 0.05:
            failures.append(f"{mode} occupancy is below 0.05")
        if abs(occupancy - 0.25) > 0.10:
            failures.append(f"{mode} occupancy differs from 0.25 by more than 0.10")
    if metrics["midpoint_occupancy"]["other"] > 0.05:
        failures.append("other occupancy is above 0.05")
    return failures
```

Implement `SFPDRolloutBatch.from_rollouts` and `aggregate_sfpd_metrics` in the same module. Aggregate goal success, final goal errors, numerical failures, action-limit failures/activations, maximum requested step distance, anchor discrepancy, and midpoint occupancy. Full trajectories classify at global state index 32 using the existing Stage A classifier; rollouts that terminate before index 32 receive a separate `failed_before_midpoint` count and no mode. Gates require at least 95% success, zero numerical failures, each intended mode at least 5%, `other` at most 5%, and each mode within 0.10 of 0.25. Always report action-limit counts even when gates pass.

- [ ] **Step 6: Run evaluation tests**

Run: `uv run pytest env/tests/test_evaluate_stage_b.py env/tests/test_environment.py -v`

Expected: PASS for eight replans, 64 physical actions, deterministic replay, and explicit limit failures.

- [ ] **Step 7: Commit closed-loop evaluation**

```bash
git add env/evaluate_stage_b.py env/tests/test_evaluate_stage_b.py
git commit -m "feat: evaluate streaming SFPD rollouts"
```

---

### Task 8: Add artifacts, visualizations, and the B1 command-line runner

**Files:**
- Modify: `env/stage_b_artifacts.py`
- Create: `env/visualize_stage_b.py`
- Create: `env/run_stage_b.py`
- Modify: `env/README.md`
- Create: `env/tests/test_run_stage_b.py`

**Interfaces:**
- Consumes: the Task 6 trainer/checkpoint and Task 7 evaluator.
- Produces: `run_stage_b1`, `python -m env.run_stage_b --stage b1`, complete run artifacts, PNG comparisons, and representative GIFs.

- [ ] **Step 1: Write a failing reduced end-to-end artifact test**

```python
from dataclasses import replace

from env.run_stage_b import run_stage_b1


def test_reduced_stage_b1_run_writes_complete_artifacts(tmp_path):
    bank = generate_demonstration_bank(DEFAULT_CONFIG, seed=41)
    config = replace(
        DEFAULT_STAGE_B1_CONFIG,
        hidden_dim=16,
        hidden_layers=1,
        batch_size=64,
        max_updates=2,
        validation_interval=1,
        warmup_updates=1,
        rollout_count=4,
    )
    result = run_stage_b1(tmp_path, bank=bank, config=config, seed=42)
    assert result["model_type"] == "sfpd"
    for name in (
        "resolved_config.json",
        "seed_streams.json",
        "pusht_stats.npz",
        "sfpd_best.pt",
        "training_history.json",
        "diagnostics.json",
        "rollouts.npz",
        "trajectory_comparison.png",
        "mode_occupancy.png",
    ):
        assert (tmp_path / name).is_file()
```

- [ ] **Step 2: Run the focused test and verify runner imports fail**

Run: `uv run pytest env/tests/test_run_stage_b.py -v`

Expected: FAIL because `env.run_stage_b` does not exist.

- [ ] **Step 3: Add safe rollout serialization**

Extend `env/stage_b_artifacts.py` with `save_sfpd_rollouts(path, batch)`. Store padded float32 arrays plus integer length arrays and boolean failure masks; pad unused requested/executed entries with NaN only in NPZ, never in JSON. Store raw nine-position chunks, rollout seeds, selected checkpoint digest, and train demonstration digest. Load with `allow_pickle=False`.

- [ ] **Step 4: Implement static visualization functions**

Implement `plot_trajectory_comparison(path: str | Path, expert_positions: np.ndarray, rollout_batch: SFPDRolloutBatch, environment_config: EnvironmentConfig) -> None` and `plot_mode_occupancy(path: str | Path, expert_occupancy: Mapping[str, float], sfpd_occupancy: Mapping[str, float]) -> None`. Use the same world bounds as the Gym renderer, draw start/goal markers, use stable colors for four modes, mark failed rollout endpoints, close every Matplotlib figure, and label expert versus SFPD occupancy on the same axes. Both functions create parent directories and reject non-finite expert inputs.

- [ ] **Step 5: Implement the orchestration API and CLI**

```python
def run_stage_b1(
    output_dir: str | Path,
    *,
    bank: DemonstrationBank,
    config: StageB1Config = DEFAULT_STAGE_B1_CONFIG,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training = train_sfpd(
        bank,
        destination,
        config=config,
        root_seed=seed,
        device=device,
    )
    policy, stats, checkpoint_metadata = load_trained_sfpd(
        training.checkpoint_path,
        training.stats_path,
        expected_train_data_digest=train_data_digest(bank),
        device=device,
    )
    rollout_seeds = derive_rollout_seeds(
        training.seed_streams["rollout_initialization"],
        config.rollout_count,
    )
    gaussian_metrics, gaussian_rollouts = evaluate_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        rollout_seeds,
        center_init=False,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    centered_metrics, centered_rollouts = evaluate_sfpd(
        policy,
        stats,
        DEFAULT_CONFIG.environment,
        rollout_seeds[: min(32, len(rollout_seeds))],
        center_init=True,
        integration_steps_per_action=config.integration_steps_per_action,
    )
    return write_stage_b1_outputs(
        destination,
        bank,
        config,
        training,
        checkpoint_metadata,
        gaussian_metrics,
        gaussian_rollouts,
        centered_metrics,
        centered_rollouts,
    )
```

Implement the referenced `load_trained_sfpd`, `derive_rollout_seeds`, and `write_stage_b1_outputs` helpers in `env/run_stage_b.py`. They perform environment-version capture, best-EMA loading, held-out field validation, acceptance evaluation, NPZ/JSON persistence, trajectory and occupancy plots, and representative RGB-array GIFs for at least one successful and one failed rollout when available.

The CLI supports:

```text
--stage b1
--demonstrations env/artifacts/demonstrations.npz
--output-dir env/artifacts/stage_b/b1-seed0
--seed 0
--device cpu
--max-updates 20000
--rollout-count 1024
--integration-steps-per-action 6
--enforce-acceptance
```

Reject any `--stage` other than `b1` in this plan. Load demonstrations through `load_demonstration_bank`; do not regenerate them silently.

- [ ] **Step 6: Document the exact B1 commands and artifacts**

Add to `env/README.md`:

```markdown
### Stage B1: PushT-compatible deterministic SFPD

Run the full seeded experiment:

`uv run python -m env.run_stage_b --stage b1 --seed 0 --device cpu --enforce-acceptance`

The dataset uses next-position actions, train-only `[-1,1]` normalization,
PushT 16/2/8 windows, and actual Drake FirstOrderHold targets. Drake evaluates
in float64 internally; every learned quantity is float32.
```

- [ ] **Step 7: Run the reduced runner test and CLI help test**

Run: `uv run pytest env/tests/test_run_stage_b.py -v`

Expected: PASS with all named artifacts.

Run: `uv run python -m env.run_stage_b --help`

Expected: exit 0 and display every CLI flag above.

- [ ] **Step 8: Commit the runnable B1 workflow**

```bash
git add env/stage_b_artifacts.py env/visualize_stage_b.py env/run_stage_b.py env/README.md env/tests/test_run_stage_b.py
git commit -m "feat: add Stage B1 experiment workflow"
```

---

### Task 9: Verify the full suite and run the canonical B1 experiment

**Files:**
- Modify only if verification exposes a scoped B1 defect: the corresponding implementation and test file from Tasks 1-8.
- Generate, do not commit: `env/artifacts/stage_b/b1-seed0/**`

**Interfaces:**
- Consumes: the complete B1 CLI and the canonical `env/artifacts/demonstrations.npz` bank.
- Produces: a verified checkpoint, metrics, raw rollouts, mode visualization, and an evidence-backed decision about whether B1 is finite, successful, and multimodal across Gaussian initial states.

- [ ] **Step 1: Run focused Stage B tests**

Run: `uv run pytest env/tests/test_drake_compatibility.py env/tests/test_chunk_data.py env/tests/test_drake_trajectory.py env/tests/test_models.py env/tests/test_sfp_policies.py env/tests/test_stage_b_artifacts.py env/tests/test_train_stage_b.py env/tests/test_evaluate_stage_b.py env/tests/test_run_stage_b.py -q`

Expected: all Stage B1 tests pass.

- [ ] **Step 2: Run the complete environment regression suite**

Run: `uv run pytest env/tests -q`

Expected: all Stage A, rendering, environment, and Stage B1 tests pass.

- [ ] **Step 3: Run the canonical full B1 training and evaluation**

Run:

```bash
uv run python -m env.run_stage_b \
  --stage b1 \
  --demonstrations env/artifacts/demonstrations.npz \
  --output-dir env/artifacts/stage_b/b1-seed0 \
  --seed 0 \
  --device cpu \
  --max-updates 20000 \
  --rollout-count 1024 \
  --integration-steps-per-action 6
```

Expected: finite training/validation history, a selected EMA checkpoint, 1024 Gaussian rollouts, centered deterministic diagnostics, and all documented artifacts. A development-gate failure is a valid experimental result and must not trigger threshold changes.

- [ ] **Step 4: Inspect the recorded scientific result**

Read `diagnostics.json` and verify it reports:

- train-data and checkpoint digests;
- resolved Drake/NumPy/Torch versions;
- held-out field loss;
- success and goal-error distribution;
- all action-limit counts and affected trajectory indices;
- maximum requested physical step;
- four-mode, `other`, and failed-before-midpoint occupancy;
- normalized and physical anchor discrepancy;
- centered deterministic replay; and
- the unmodified development-gate failures list.

Open `trajectory_comparison.png`, `mode_occupancy.png`, and representative GIFs. Confirm the plots correspond to the same recorded checkpoint digest.

- [ ] **Step 5: Re-run deterministic replay checks**

Run the evaluator twice from the saved checkpoint with the same rollout seeds and compare saved float32 arrays byte-for-byte. Change only the rollout seed and confirm the Gaussian environment initial states change while a centered SFPD rollout remains deterministic.

- [ ] **Step 6: Commit only scoped fixes and documentation, never artifacts**

If verification required a B1 fix, first add a regression test, rerun the focused and full suites, and commit only the relevant source/test files. Confirm `git status --short` does not stage `env/artifacts/stage_b/`.

- [ ] **Step 7: Report results before planning B2**

Report the exact test counts, selected update and validation loss, success rate, mode occupancy, `other`/failure rates, action-limit statistics, and links to generated visualizations. Do not start B2 until the B1 result and any gate misses have been interpreted.
