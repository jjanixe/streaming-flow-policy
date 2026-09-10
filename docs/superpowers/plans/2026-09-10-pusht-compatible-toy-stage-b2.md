# PushT-Compatible Toy Stage B2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the repository-style stochastic latent streaming flow policy (SFPS) on the same toy expert bank and prove seeded replay plus same-state stochastic mode diversity without adding a denoiser or preference guidance.

**Architecture:** Extend the tested Stage B1 modules instead of creating a parallel pipeline. A seeded Drake transform builds joint action/latent training targets, a float32 conditional MLP predicts their joint velocity, and a Torchdyn ODE integrates from the current normalized action plus an explicitly supplied latent sample. Training, checkpoint validation, Gym rollout accounting, and artifact generation reuse the B1 contracts while recording SFPS latent seeds separately.

**Tech Stack:** Python 3.10, uv, NumPy 1.26, PyTorch 2.7, Torchdyn 1.0, Drake 1.26, Gym 0.26, pytest, Matplotlib, ImageIO

**Spec:** `docs/superpowers/specs/2026-09-09-pusht-compatible-toy-stage-b-design.md`

## Global Constraints

- Preserve the PushT horizons exactly: prediction 16, observation 2, execution 8.
- Use actual `pydrake.trajectories.PiecewisePolynomial.FirstOrderHold`; convert the Drake boundary back to NumPy float32 immediately.
- Keep demonstrations, model inputs, targets, Torch parameters, losses, ODE states, rollouts, and saved numeric arrays float32.
- Use `sigma0=sigma1=0.1` initially, so `sigma_r=0`, and validate `0 <= sigma0 <= sigma1`.
- SFPS latent dimension equals action dimension 2.
- Sample a fresh explicit latent at every chunk boundary; never rely on process-global NumPy or Torch RNG state.
- A fixed environment seed and fixed latent seed stream must replay byte-for-byte. Changing only latent seeds under centered initialization must change at least one generated trajectory in the smoke diagnostic.
- Generated absolute actions remain subject to `max_step_distance=0.075` and are never silently clipped.
- B2 records the same development gates as B1, but a failed distributional gate remains an experimental result.
- Do not implement a denoiser, score-corrected SDE, preference model, STEG, or merge/rebranch data in this plan.
- Generated artifacts remain below ignored `env/artifacts/stage_b/` and are not committed.

---

### Task 1: Add SFPS configuration and seeded Drake targets

**Files:**
- Modify: `env/stage_b_config.py`
- Modify: `env/drake_trajectory.py`
- Modify: `env/tests/test_drake_trajectory.py`
- Create: `env/tests/test_stage_b2_config.py`

**Interfaces:**
- Consumes: `evaluate_drake_foh(action: np.ndarray, tau: np.float32)` and Stage B1 training defaults.
- Produces: `StageB2Config`, `DEFAULT_STAGE_B2_CONFIG`, `stage_b2_config_to_dict`, and `SFPSDrakeTransform(sigma0, sigma1, rng)`.

- [x] **Step 1: Write failing configuration tests**

```python
def test_stage_b2_defaults_match_repository_equal_sigma_contract():
    config = DEFAULT_STAGE_B2_CONFIG
    assert (config.pred_horizon, config.obs_horizon, config.action_horizon) == (16, 2, 8)
    assert config.sigma0 == 0.1
    assert config.sigma1 == 0.1
    assert config.latent_dim == 2


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"sigma0": -0.1}, "sigma0"),
        ({"sigma0": 0.2, "sigma1": 0.1}, "sigma0"),
        ({"latent_dim": 3}, "latent_dim"),
    ],
)
def test_stage_b2_rejects_incompatible_parameters(overrides, match):
    with pytest.raises(ValueError, match=match):
        replace(DEFAULT_STAGE_B2_CONFIG, **overrides)
```

- [x] **Step 2: Run the configuration tests and verify RED**

Run: `uv run pytest env/tests/test_stage_b2_config.py -q`

Expected: collection fails because `StageB2Config` and `DEFAULT_STAGE_B2_CONFIG` do not exist.

- [x] **Step 3: Implement immutable B2 configuration**

```python
@dataclass(frozen=True)
class StageB2Config:
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    latent_dim: int = 2
    sigma0: float = 0.1
    sigma1: float = 0.1
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
        for name in (
            "pred_horizon", "obs_horizon", "action_horizon", "latent_dim",
            "hidden_dim", "hidden_layers", "batch_size", "max_updates",
            "validation_interval", "warmup_updates",
            "integration_steps_per_action", "rollout_count",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an int")
        for name in (
            "sigma0", "sigma1", "learning_rate", "weight_decay", "ema_decay",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if (self.pred_horizon, self.obs_horizon, self.action_horizon) != (16, 2, 8):
            raise ValueError("Stage B2 requires horizons 16/2/8")
        if self.latent_dim != 2:
            raise ValueError("latent_dim must equal action dimension 2")
        if not 0.0 <= self.sigma0 <= self.sigma1:
            raise ValueError("sigma0 and sigma1 must satisfy 0 <= sigma0 <= sigma1")
        for name in (
            "hidden_dim", "hidden_layers", "batch_size", "max_updates",
            "validation_interval", "integration_steps_per_action", "rollout_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_updates <= self.max_updates:
            raise ValueError("warmup_updates must be in [0, max_updates]")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer values are outside their valid ranges")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
```

- [x] **Step 4: Write failing hand-derived SFPS target tests**

```python
def test_sfps_drake_transform_matches_equal_sigma_formula():
    datum = _linear_normalized_datum()
    transform = SFPSDrakeTransform(0.1, 0.1, np.random.default_rng(17))
    actual = transform(datum)
    replay_rng = np.random.default_rng(17)
    tau = np.float32(replay_rng.random())
    xi, xi_dot = evaluate_drake_foh(datum["action"], tau)
    z0 = replay_rng.standard_normal((1, 2), dtype=np.float32)
    epsilon_a0 = np.float32(0.1) * replay_rng.standard_normal((1, 2), dtype=np.float32)
    expected_a = xi + epsilon_a0
    expected_z = (np.float32(1.0) - np.float32(0.9) * tau) * z0 + tau * xi
    expected_va = xi_dot
    expected_vz = xi + tau * xi_dot - np.float32(0.9) * z0
    np.testing.assert_allclose(actual["a"], expected_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["z"], expected_z, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["va"], expected_va, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["vz"], expected_vz, rtol=1e-6, atol=1e-6)
    assert all(actual[name].dtype == np.float32 for name in ("a", "z", "va", "vz", "t"))
```

- [x] **Step 5: Run the transform test and verify RED**

Run: `uv run pytest env/tests/test_drake_trajectory.py -q`

Expected: import or attribute failure for `SFPSDrakeTransform`.

- [x] **Step 6: Implement the seeded SFPS transform**

Compute exactly:

```python
sigma_r = np.float32(np.sqrt(np.float32(sigma1**2 - sigma0**2)))
a = xi + epsilon_a0 + sigma_r * tau * z0
z = (1.0 - (1.0 - sigma1) * tau) * z0 + tau * xi
va = xi_dot + sigma_r * z0
vz = xi + tau * xi_dot - (1.0 - sigma1) * z0
```

Return unchanged diagnostic metadata, drop the original `action`, and emit only finite float32 arrays.

- [x] **Step 7: Run Task 1 tests and commit**

Run: `uv run pytest env/tests/test_stage_b2_config.py env/tests/test_drake_trajectory.py -q`

Commit: `git commit -m "feat: add seeded SFPS Drake targets"`

---

### Task 2: Add the joint velocity model and stochastic policy

**Files:**
- Modify: `env/models.py`
- Modify: `env/sfp_policies.py`
- Modify: `env/tests/test_models.py`
- Modify: `env/tests/test_sfp_policies.py`

**Interfaces:**
- Consumes: normalized observations `[B,2,3]`, joint sample `[B,2,2]`, time `[B]`, explicit `torch.Generator` or latent tensor.
- Produces: `SFPSVelocityMLP.forward(sample, timestep, global_cond) -> Tensor[B,2,2]` and `StreamingFlowPolicyStochastic.loss/predict`.

- [ ] **Step 1: Write failing model shape and validation tests**

```python
def test_sfps_velocity_model_returns_joint_float32_velocity():
    model = SFPSVelocityMLP(hidden_dim=16, hidden_layers=1)
    output = model(
        torch.zeros(3, 2, 2, dtype=torch.float32),
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
        torch.zeros(3, 6, dtype=torch.float32),
    )
    assert output.shape == (3, 2, 2)
    assert output.dtype == torch.float32
```

Add cases rejecting float64, non-finite values, a final joint shape other than `[2,2]`, and time outside `[0,1]`.

- [ ] **Step 2: Run model tests and verify RED**

Run: `uv run pytest env/tests/test_models.py -q`

Expected: import failure for `SFPSVelocityMLP`.

- [ ] **Step 3: Implement `SFPSVelocityMLP`**

Use input width `4 + 9 + 6`, the configured hidden stack, and output width 4 reshaped to `[B,2,2]`. Reuse `LocalTimeFeatures`; keep the B1 model unchanged.

- [ ] **Step 4: Write failing SFPS loss and seeded prediction tests**

```python
def test_sfps_loss_is_finite_float32_scalar():
    policy = StreamingFlowPolicyStochastic(SFPSVelocityMLP(16, 1))
    batch = {
        "obs": torch.zeros(4, 2, 3, dtype=torch.float32),
        "a": torch.zeros(4, 1, 2, dtype=torch.float32),
        "z": torch.zeros(4, 1, 2, dtype=torch.float32),
        "va": torch.ones(4, 1, 2, dtype=torch.float32),
        "vz": torch.ones(4, 1, 2, dtype=torch.float32),
        "t": torch.linspace(0, 1, 4, dtype=torch.float32),
    }
    loss = policy.loss(batch)
    assert loss.shape == () and loss.dtype == torch.float32 and torch.isfinite(loss)


def test_sfps_prediction_replays_with_explicit_latent_generator():
    policy = StreamingFlowPolicyStochastic(_LatentSensitiveVelocity())
    nobs = torch.zeros(2, 3, dtype=torch.float32)
    first = policy.predict(nobs, 9, 1, generator=torch.Generator().manual_seed(7))
    second = policy.predict(nobs, 9, 1, generator=torch.Generator().manual_seed(7))
    third = policy.predict(nobs, 9, 1, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, third)
```

- [ ] **Step 5: Run policy tests and verify RED**

Run: `uv run pytest env/tests/test_sfp_policies.py -q`

Expected: import failure for `StreamingFlowPolicyStochastic`.

- [ ] **Step 6: Implement joint loss and ODE prediction**

Concatenate `a,z` and `va,vz` along the waypoint axis for loss. For prediction, anchor `a0=nobs[-1,:2]`, sample `z0=torch.randn((2,), generator=generator, dtype=float32, device=device)`, integrate joint state `[2,2]`, return only action states with shape `[1,num_actions,2]`, and optionally return the sampled latent for diagnostics through a separate `sample_latent` helper. Match B1's Dopri5, adjoint, tolerances, validation, and numerical-error translation.

- [ ] **Step 7: Run Task 2 tests and commit**

Run: `uv run pytest env/tests/test_models.py env/tests/test_sfp_policies.py -q`

Commit: `git commit -m "feat: add stochastic latent streaming policy"`

---

### Task 3: Add model-typed SFPS checkpoints without weakening B1

**Files:**
- Modify: `env/stage_b_artifacts.py`
- Modify: `env/tests/test_stage_b_artifacts.py`

**Interfaces:**
- Consumes: canonical SFPS wrapper state and B2 metadata.
- Produces: `save_sfps_checkpoint`, `load_sfps_checkpoint`, and model-typed checkpoint schema validation.

- [ ] **Step 1: Write failing SFPS checkpoint tests**

```python
def test_sfps_checkpoint_round_trip_validates_joint_architecture(tmp_path):
    policy = StreamingFlowPolicyStochastic(SFPSVelocityMLP(16, 1))
    metadata = _complete_sfps_metadata()
    path = tmp_path / "sfps.pt"
    save_sfps_checkpoint(path, raw_state=policy.state_dict(), ema_state=policy.state_dict(), metadata=metadata)
    loaded = load_sfps_checkpoint(path, expected_train_data_digest="correct")
    assert loaded["metadata"]["model_type"] == "sfps"
    assert loaded["metadata"]["architecture"]["name"] == "SFPSVelocityMLP"
```

Add mutations for wrong model type, wrong `sigma0/sigma1`, wrong latent dimension, missing state key, wrong state shape/dtype, inconsistent seed streams, and ensure all existing SFPD rejection tests remain unchanged.

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run: `uv run pytest env/tests/test_stage_b_artifacts.py -q`

Expected: import failures for SFPS save/load helpers.

- [ ] **Step 3: Generalize internal validators by explicit model type**

Keep public B1 functions and metadata compatibility. Add B2 metadata key `stage_b2_config`, architecture name `SFPSVelocityMLP`, and wrapper buffers `pred_horizon`, `sigma0`, `sigma1`, `sigma_r`. Construct the canonical policy inside `torch.random.fork_rng` so loading never advances caller RNG state.

- [ ] **Step 4: Run checkpoint tests and commit**

Run: `uv run pytest env/tests/test_stage_b_artifacts.py -q`

Commit: `git commit -m "feat: serialize validated SFPS checkpoints"`

---

### Task 4: Train SFPS with independent deterministic RNG streams

**Files:**
- Modify: `env/train_stage_b.py`
- Modify: `env/tests/test_train_stage_b.py`

**Interfaces:**
- Consumes: `DemonstrationBank`, `StageB2Config`, train-only PushT stats, `SFPSDrakeTransform`, SFPS policy/checkpoint APIs.
- Produces: `train_sfps(bank, output_dir, *, config, root_seed, device) -> TrainResult` with `sfps_best.pt` and `sfps_training_history.json`.

- [ ] **Step 1: Write failing short-training test**

```python
def test_short_sfps_training_is_finite_reproducible_and_separate_from_b1(tmp_path):
    bank = _small_bank(101)
    config = _short_b2_config(max_updates=2)
    first = train_sfps(bank, tmp_path / "first", config=config, root_seed=102)
    second = train_sfps(bank, tmp_path / "second", config=config, root_seed=102)
    one = load_sfps_checkpoint(first.checkpoint_path, expected_train_data_digest=train_data_digest(bank))
    two = load_sfps_checkpoint(second.checkpoint_path, expected_train_data_digest=train_data_digest(bank))
    assert first.validation_loss == second.validation_loss
    for key in one["ema_state"]:
        torch.testing.assert_close(one["ema_state"][key], two["ema_state"][key], rtol=0, atol=0)
    assert one["metadata"]["seed_streams"] == two["metadata"]["seed_streams"]
```

Also assert caller Torch RNG restoration and that every saved floating tensor is finite float32.

- [ ] **Step 2: Run training tests and verify RED**

Run: `uv run pytest env/tests/test_train_stage_b.py -q`

Expected: import failure for `train_sfps`.

- [ ] **Step 3: Implement `train_sfps` using shared training primitives**

Extract only genuinely shared B1/B2 helpers for loader creation, validation, optimizer schedule, EMA, and checkpoint selection. Use independent named streams including `sfps_latent_rollout`; materialize validation batches once; train exactly `max_updates`; select best EMA validation loss; preserve B1 output names and behavior.

- [ ] **Step 4: Run training tests and commit**

Run: `uv run pytest env/tests/test_train_stage_b.py -q`

Commit: `git commit -m "feat: train reproducible toy SFPS"`

---

### Task 5: Evaluate seeded Gaussian and same-state SFPS rollouts

**Files:**
- Modify: `env/evaluate_stage_b.py`
- Modify: `env/tests/test_evaluate_stage_b.py`

**Interfaces:**
- Consumes: an SFPS policy with `predict(nobs, num_actions, integration_steps_per_action, *, generator)`, PushT stats, environment seed, and latent root seed.
- Produces: model-typed `StreamingRollout`, `StreamingRolloutBatch`, `rollout_sfps`, `evaluate_sfps`, latent-seed audit, and the existing acceptance metrics.

- [ ] **Step 1: Write failing seeded replay and fresh-per-chunk tests**

```python
def test_sfps_rollout_uses_fresh_seeded_latent_at_each_chunk():
    policy = RecordingStochasticPolicy(_scripted_normalized_path())
    first = rollout_sfps(policy, stats, env_config, environment_seed=11, latent_seed=12, center_init=True, integration_steps_per_action=1)
    second = rollout_sfps(policy.clone(), stats, env_config, environment_seed=11, latent_seed=12, center_init=True, integration_steps_per_action=1)
    assert len(first.info["chunk_latents"]) == 8
    np.testing.assert_array_equal(first.info["chunk_latents"], second.info["chunk_latents"])
    assert len({row.tobytes() for row in first.info["chunk_latents"]}) > 1
```

Add a centered batch test where the environment seed is fixed and latent seeds vary; assert initial observations are identical, the controlled-diversity metric is applicable, and at least two raw generated trajectories differ for a latent-sensitive fixture.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `uv run pytest env/tests/test_evaluate_stage_b.py -q`

Expected: import failure for `rollout_sfps`/`evaluate_sfps`.

- [ ] **Step 3: Generalize rollout records without losing B1 provenance**

Use a shared immutable rollout record with `model_type`, `environment_seed`, optional `latent_seed`, and float32 `chunk_latents[C,2]` for SFPS. Keep aliases or constructors so existing `SFPDRollout` and `SFPDRolloutBatch` tests continue to pass. The rollout loop must pass one persistent local `torch.Generator` through all eight replans so each boundary consumes a fresh latent deterministically.

- [ ] **Step 4: Implement B2 aggregation and diagnostics**

Reuse goal, failure, anchor, midpoint, and action-limit accounting. Add `same_state_unique_raw_trajectory_count`, `same_state_unique_executed_trajectory_count`, latent seed counts, and `same_state_stochastic_diversity_observed`. Apply distribution gates to both Gaussian and centered B2 batches while reporting failures before midpoint separately.

- [ ] **Step 5: Run evaluation tests and commit**

Run: `uv run pytest env/tests/test_evaluate_stage_b.py -q`

Commit: `git commit -m "feat: evaluate controlled SFPS diversity"`

---

### Task 6: Save SFPS rollouts and B1/B2 comparison artifacts

**Files:**
- Modify: `env/stage_b_artifacts.py`
- Modify: `env/visualize_stage_b.py`
- Modify: `env/run_stage_b.py`
- Modify: `env/tests/test_stage_b_artifacts.py`
- Modify: `env/tests/test_run_stage_b.py`

**Interfaces:**
- Consumes: B2 training result, Gaussian and centered rollout batches, B1 diagnostics when available.
- Produces: `save_sfps_rollouts`, `load_trained_sfps`, `write_stage_b2_outputs`, comparison plots, representative GIFs, and valid JSON diagnostics.

- [ ] **Step 1: Write failing portable rollout tests**

Verify `sfps_rollouts.npz` contains numeric/pickle-free environment seeds, latent seeds, per-chunk latents, masks, requested/executed positions, raw chunks, failure masks, checkpoint digest, and train digest. Mutate missing latent rows and ensure saving rejects inconsistent provenance.

- [ ] **Step 2: Run artifact tests and verify RED**

Run: `uv run pytest env/tests/test_stage_b_artifacts.py -q`

Expected: missing `save_sfps_rollouts`.

- [ ] **Step 3: Implement model-typed rollout serialization**

Preserve the existing B1 NPZ schema and add SFPS-only arrays under a model-typed helper. Padding masks must be explicit and all non-string arrays must remain primitive numeric/bool dtypes.

- [ ] **Step 4: Write failing reduced end-to-end B2 test**

```python
def test_reduced_stage_b2_run_writes_controlled_diversity_artifacts(tmp_path):
    result = run_stage_b2(tmp_path, bank=_small_bank(201), config=_short_b2_config(), seed=202)
    assert result["model_type"] == "sfps"
    assert result["seeded_replay"] is True
    assert result["centered_metrics"]["same_state_diversity_applicable"] is True
    for name in (
        "sfps_best.pt", "sfps_training_history.json", "sfps_diagnostics.json",
        "sfps_rollouts.npz", "sfps_centered_rollouts.npz",
        "sfps_trajectory_comparison.png", "sfps_mode_occupancy.png",
    ):
        assert (tmp_path / name).is_file(), name
```

- [ ] **Step 5: Run workflow test and verify RED**

Run: `uv run pytest env/tests/test_run_stage_b.py -q`

Expected: import failure for `run_stage_b2`.

- [ ] **Step 6: Implement SFPS load/output workflow**

Reconstruct the EMA model after strict digest/config/schema checks. Write Gaussian and centered NPZ files, diagnostics JSON, resolved config, seed streams, mode plots, trajectory plots, and representative GIFs. If a sibling B1 diagnostics file exists, additionally write a comparison JSON/plot using the exact same classifier denominators; absence of B1 diagnostics must be explicit rather than silently treated as zeros.

- [ ] **Step 7: Run Task 6 tests and commit**

Run: `uv run pytest env/tests/test_stage_b_artifacts.py env/tests/test_run_stage_b.py -q`

Commit: `git commit -m "feat: write Stage B2 comparison artifacts"`

---

### Task 7: Extend the CLI for `b2` and sequential `all`

**Files:**
- Modify: `env/run_stage_b.py`
- Modify: `env/README.md`
- Modify: `env/tests/test_run_stage_b.py`

**Interfaces:**
- Consumes: canonical demonstration bank and B1/B2 runners.
- Produces: `--stage b1|b2|all`, compatible output directory layout, and B2 precondition validation.

- [ ] **Step 1: Write failing CLI tests**

Run reduced subprocesses for `--stage b2` and `--stage all`. Assert `b2` refuses a missing/incompatible B1 pipeline record before training, `all` produces `b1/` and `b2/` outputs sequentially, and JSON stdout contains both model summaries without NaN.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `uv run pytest env/tests/test_run_stage_b.py -q`

Expected: parser rejects `b2` and `all`.

- [ ] **Step 3: Implement CLI stage selection**

For `b2`, accept a B1 record/checkpoint path and validate that it is finite and that the shared data digest/chunk contract matches; do not require B1 distributional acceptance. For `all`, run B1 then B2 with independently derived root seeds and write a top-level comparison summary. Keep `--stage b1` backward compatible.

- [ ] **Step 4: Document exact commands**

Document:

```bash
uv run python -m env.run_stage_b --stage b2 --seed 0 --device cpu
uv run python -m env.run_stage_b --stage all --seed 0 --device cpu
```

Explain equal-sigma SFPS, fresh chunk latents, same-state diagnostics, artifact locations, and that B2 does not include a denoiser or controllable mode label.

- [ ] **Step 5: Run CLI tests and commit**

Run: `uv run pytest env/tests/test_run_stage_b.py -q`

Commit: `git commit -m "feat: expose Stage B2 experiment CLI"`

---

### Task 8: Verify and run the canonical B2 experiment

**Files:**
- Modify only when a scoped defect is reproduced by a failing regression test.
- Generate only ignored files under `env/artifacts/stage_b/`.

**Interfaces:**
- Consumes: complete B2 pipeline, canonical demonstrations, verified B1 record.
- Produces: fresh test evidence, canonical B2 checkpoint, Gaussian and centered rollouts, comparison metrics, and visualizations.

- [ ] **Step 1: Run focused Stage B2 tests**

Run: `uv run pytest env/tests/test_stage_b2_config.py env/tests/test_drake_trajectory.py env/tests/test_models.py env/tests/test_sfp_policies.py env/tests/test_stage_b_artifacts.py env/tests/test_train_stage_b.py env/tests/test_evaluate_stage_b.py env/tests/test_run_stage_b.py -q`

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete suite**

Run: `uv run pytest -q`

Expected: all Stage A, rendering, B1, and B2 tests pass.

- [ ] **Step 3: Check the resolved uv environment and CLI**

Run: `uv lock --check && uv run python -m env.run_stage_b --help`

Expected: lock is current and CLI lists `b1`, `b2`, and `all`.

- [ ] **Step 4: Run the canonical B2 experiment**

Run:

```bash
uv run python -m env.run_stage_b \
  --stage b2 \
  --seed 0 \
  --device cpu \
  --max-updates 20000 \
  --rollout-count 1024 \
  --integration-steps-per-action 6
```

Do not pass `--enforce-acceptance` on the first scientific run. Preserve unsuccessful trajectories and unchanged thresholds.

- [ ] **Step 5: Verify replay and controlled diversity from the saved checkpoint**

Repeat evaluation with identical environment/latent seed lists and compare saved float32 arrays byte-for-byte. Then keep the centered environment seed fixed, change only latent seeds, and report unique raw/executed trajectory counts plus occupancy.

- [ ] **Step 6: Record the scientific result**

Report selected update, validation and held-out loss, Gaussian and centered success, numerical and action-limit failures, all occupancy values, `other`, unique same-state trajectories, boundary jumps, checkpoint/data digests, and links to plots/GIFs. A gate miss is retained and blocks preference guidance but does not trigger threshold relaxation.

- [ ] **Step 7: Commit any regression-tested verification fixes**

If the canonical run exposes a defect, first add a failing regression test, implement the minimal fix, rerun focused and full suites, and commit only source/tests. Confirm generated artifacts remain untracked/ignored.
