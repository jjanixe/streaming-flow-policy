# Few-shot Preference Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run and evaluate few-shot preference steering of the frozen B2 SFPS.

**Architecture:** Independent regularized BT fitting and batched full-future
latent simulation feed a reproducible experiment runner. Existing policy,
normalization, and Gym execution semantics remain authoritative.

**Tech Stack:** Existing uv environment, NumPy float32, Torch, pytest, Matplotlib.

**Spec:** docs/superpowers/specs/2026-09-11-few-shot-preference-steering-design.md

## Global Constraints

- Frozen existing B2 checkpoint; preserve prior uncommitted modifications.
- 64 actions, two observations, nine predicted positions, skip anchor, execute eight.
- float32 data/model arrays; positive L2=0.1, beta=1, 32 candidates by default.
- No command projection, whole-label replication, or silent failure removal.
- Generated artifacts under ignored env/artifacts/preference/; source in new modules.

### Task 1: Regularized Bradley–Terry fitting

**Files:** Create env/preference.py and env/tests/test_preference.py.
**Interfaces:** `fit_bradley_terry(delta_features, labels, *, l2=0.1,
prior_mean=None, max_steps=100, tolerance=1e-5) -> PreferenceFit`;
fit has `weights`, `objective`, `gradient_norm`, `converged`, `iterations`,
`feature_rank`. `preference_probabilities(delta_features, weights)` and
`preference_metrics(delta_features, labels, weights) -> dict` are public.

- [x] Write and observe failing meaningful tests, including:
  ```python
  x = np.array([[1,0],[0,1],[-1,0],[0,-1]], dtype=np.float32)
  y = np.array([1,-1,-1,1], dtype=np.int8)
  a = fit_bradley_terry(x, y)
  b = fit_bradley_terry(-x, -y)
  assert a.weights[0] > 0 and a.weights[1] < 0
  np.testing.assert_allclose(a.weights, b.weights, atol=1e-5)
  ```
- [x] Implement stable softplus/sigmoid Newton/backtracking solver, input
  validation, rank diagnostics, and probability/log-loss/accuracy outputs.
- [x] Verify extreme logits, balanced contradictory labels returning prior,
  rank deficiency, shape/label/nonfinite errors and convergence reporting.
- [x] Run `uv run --frozen pytest env/tests/test_preference.py -q` and review.

### Task 2: Batched continuation simulation and latent steering

**Files:** Create env/preference_rollout.py and env/tests/test_preference_rollout.py.
**Interfaces:** `simulate_continuations(policy, stats, prefixes, latents, *,
environment_config, integration_steps_per_action=6) -> ContinuationBatch`.
prefixes [B,q+1,2], latents [B,ceil((64-q)/8),2]. Batch has positions [B,65,2]
(NaN padding), lengths [B] (valid state count), requested_actions [B,64,2],
numerical_failure/action_limit_failure [B], success [B], goal_errors [B].
`select_candidates(scores, valid, *, method, rng) -> (index, ess, fallback)`.

- [x] Write failing tests for prefix preservation, exact 64 valid commands,
  history/time update across chunks and anchor skipping using a simple real
  scripted predictor with independently known positions.
- [x] Implement batched inference using existing normalize/unnormalize helpers
  and `predict_batch` explicit latents. Implement real Gym invalid-command
  semantics and isolate failed rows rather than dropping a whole batch.
- [x] Test finite over-limit and nonfinite requests against Gym, terminal
  handling, all-invalid selection fallback, beta-zero uniform valid weights,
  numerical softmax stability, seeded replay, and future outcome choice.
- [x] Run focused tests and inspect all failure accounting before integration.

### Task 3: Experiment runner, artifacts, and publication audit

**Files:** Create env/run_preference.py, env/tests/test_run_preference.py,
docs/research/2026-09-11-preference-steering-results.md and
the main checkout's docs/research/2026-09-11-preference-publication-audit.md;
append env/README.md. The audit is placed beside the original literature note.
**Interfaces:** `run_preference(output_dir, *, checkpoint_path, stats_path,
demonstrations_path, seed=0, device='cpu', candidates=32, rollout_count=16,
coverage_count=512, integration_steps_per_action=6) -> dict` and CLI.

- [x] Write failing reduced end-to-end artifact test using a small saved SFPS
  checkpoint and real pipeline. Verify no changes to checkpoint state.
- [x] Load checkpoint with existing train digest validation; train-only feature
  normalizer; independent seeded balanced clean query pools and held-out pairs.
- [x] Implement base/soft/best receding candidate selection, full-prefix scoring,
  failure fallback, same-start coverage, four synthetic users and nested budgets.
- [x] Save config, digests, query trajectory IDs/features/labels, model fits,
  execution and candidate diagnostics, per-episode metrics, timings and plots.
- [x] Run reduced runner tests and complete environment regression suite.
- [x] Run canonical pilot, inspect plots and write measured results with limits.
- [x] Verify venues via official sources and author Github source availability;
  write dated cited audit and correct unsupported prior publication assertions.
- [x] Request independent scientific/code review and resolve material issues.

## Progress

- Completed 2026-09-11. The approved design used the existing B2 worktree.
- Task 1: regularized BT fitter and 44 focused tests passed. Independent review
  identified two extreme-scale numerical cases; weighted-design SVD fallback
  and representable-progress backtracking resolved both. Scoped re-review
  approved the fixes with no remaining Important findings.
- Task 2: full-future simulator and selection reviewed and approved. Optimized
  checkpoint batch-versus-individual probe: eight identical latent schedules,
  maximum position difference 5.50e-7, no mode or success disagreement.
- Task 3: experiment runner reviewed and approved. Final scientific report was
  checked against diagnostics and NPZs; equal-rate/different-success-subset
  caveat for Gaussian base versus soft was added as requested by the reviewer.
- Final environment regression: 324 passed, 2 skipped, 13 existing warnings,
  76.84 seconds. Baseline before this work: 258 passed, 2 skipped.
- Canonical GPU-1 pilot completed: 52 conditions / 832 closed-loop episodes and
  1,536 coverage trajectories. Policy state and checkpoint digest unchanged.
  All 16 preference fits are bitwise identical before/after both numerical
  robustness fixes, so saved canonical results remain valid with final code.
- Measured result and limitations:
  [results](../../research/2026-09-11-preference-steering-results.md).
  Utility, success, trajectories, seeds, selections, and verification evidence
  are saved under ignored env/artifacts/preference/pilot-seed0.
- Publication audit verifies 12 papers separately for official publication and
  public source. Unsupported FPL venue attribution and stale UF-OPS paths were
  corrected in the original literature note.
- Existing B2 modifications remain preserved and uncommitted. No merge or push
  was part of this request. Gradient steering remains outside this pilot scope.
