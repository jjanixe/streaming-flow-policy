# Mode-aligned grouped steering implementation and experiment plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Correct the two-group toy objective, measure frozen-policy oracle mode control, and compare few-shot control after the oracle baseline is understood.

**Architecture:** Extend the existing grouped pipeline with an explicit feature kind and guided selector. Keep continuous features as the compatibility default. Use mode indicators for the new experiment, then diagnose missing proposal coverage with wider latent proposals if required. The base checkpoint and the SFPS anchor/execution convention stay fixed.

**Tech stack:** Existing NumPy/SciPy grouped fitter, PyTorch SFPS, Gym simulator, pytest, Matplotlib/Pillow artifact rendering.

**Spec/authorization:** `docs/research/2026-09-12-mode-selection-diagnosis.md`; the user accepted this proposal with “진행해보자”. No new design approval is needed. Research stages are sequential hypotheses, not simultaneous parameter changes. This is an extension of the existing pipeline, with the previously presented design serving as its specification.

**Global constraints:** Work in the existing B2 worktree. Preserve all prior dirty changes and experiment folders. Do not commit, merge, push, delete prior artifacts, retrain the base, or expand P/M/C in the initial feature/oracle experiment. User authorization covers subsequent inference latent exploration if standard candidates remain inadequate. Record seed pairing, feature semantics, selector, M/L/beta, checkpoint/stats hashes, actual successful target modes, and any difference in proposal distribution. No claims that a finite bank proves mathematical unreachability. A new feature requires freshly generated synthetic labels. No extra subagents from implementers/reviewers.

## Task 1: Mode features and pipeline plumbing

**Own files:** `env/grouped_preference.py`, `env/grouped_rollout.py`, `env/run_grouped_preference.py`, `env/tests/test_grouped_preference.py`, `env/tests/test_grouped_rollout.py`, `env/tests/test_run_grouped_preference.py`.

- [x] Add failing tests for desired-mode ordering, exact bin boundaries, other=zero, feature propagation to conditional means/metrics and saved feedback, and actual Gym argmax fallback.
- [x] Implement `grouped_features(positions, *, y_scale=.35, feature_kind='continuous')`. Supported names are exactly `continuous` and `mode`; reject other names. Existing validation and continuous output remain unchanged. For mode, classify `y_32` after float64 conversion with bins UN `[.12,.4)`, UW `[.4,infinity)`, LN `(-.4,-.12]`, LW `(-infinity,-.4]`; other gets zeros. Return float64 `[...,2,2]`, with direction order upper/lower and width order wide/narrow. Use a shared small classifier (public `classify_modes(midpoint_y)` returning integer IDs UN=0,UW=1,LN=2,LW=3,other=4) so new mode metrics agree with features. Reject nonfinite inputs.

```python
# Core desired-mode regression: all eight profiles, including both priorities.
paths = np.stack([evaluate_path(s, a, times)[0]
                  for s, a in [(1,.27),(1,.53),(-1,.27),(-1,.53),(1,0)]])
features = grouped_features(paths, feature_kind='mode')
for name, user in synthetic_grouped_users().items():
    target = ['upper_narrow','upper_wide','lower_narrow','lower_wide'].index(name.rsplit('_',1)[0])
    utility = grouped_utility(features, user)
    assert utility.argmax() == target
    assert utility[target] > np.max(np.delete(utility,target))
assert not features[-1].any()
```

- [x] Thread `feature_kind='continuous'` through `evaluate_conditional_continuations`, `_closed_loop`, `_path_metrics`, and `run_grouped_preference`. All feature calculations in query/heldout labels, conditional continuations, actual utility metrics use the selected kind. Configuration records it explicitly. Add CLI `--feature-kind` choices. Keep incomplete-feature zero behavior, full observed prefix, and frozen anchor intact.
- [x] `_closed_loop` additionally accepts `method='best'`, using existing `select_candidates` semantics. Add `selection_method='soft'` to the public runner and CLI `--selection-method {soft,best}`. Guided task-only/oracle/learned conditions use it. Default condition keys remain unchanged; best oracle is named `oracle_best`. Record method in every condition. All-invalid best must hold just like soft; do not execute invalid candidate 0.
- [x] `_path_metrics` adds `successful_midpoint_counts` computed from complete, successful actual paths; existing `midpoint_counts` stay all reached-midpoint paths. New classifier is the common bin convention. Goal failures must not count as successful target modes.
- [x] Run meaningful focused tests and existing grouped/flat rollout regressions. Use a tiny actual checkpoint for mode+best end-to-end feedback/config validation; preserve current default compatibility test.
- [x] Write task report with files, red/green commands/results, semantic decisions and concerns. No commit (shared dirty worktree policy).

## Task 2: Oracle ablation experiment entry point

**Own files:** `env/run_mode_ablation.py`, `env/tests/test_run_mode_ablation.py`.

- [x] Add a reproducible, resumable experiment CLI around the existing `_closed_loop`. Load/freeze checkpoint and stats once per invocation and validate train digest. Defaults: feature kind mode, candidate counts 8/32/128, selectors soft/best, L=4, beta=16, 16 paired episodes, all eight synthetic users, both centered and gaussian. Accept smaller explicit subsets for smoke experiments. Accept `--feature-kind continuous` for matched old-objective controls. This runner needs only oracle preferences, no feedback fitting.
- [x] Directory records a complete resolved config and hashes BEFORE execution; refuse a config mismatch on resume. Save each completed condition NPZ plus JSON atomically, skipping it only after validating its hashes/metadata. Keep completion marker separate from a partial condition; never silently reuse incompatible arrays. Condition keys include regime, profile, method, and M. Preserve timing fields and all normal `_closed_loop` arrays. Compute actual `success AND target-mode` count/rate, with Wilson interval, and full mode distribution.
- [x] Expose `--proposal-std` positive finite, default 1.0, for diagnostic wider latent proposals. It scales guided CURRENT latent draws only (method=base always retains the original unit-normal current draws), keeps the same underlying current normal draws and independent original N(0,I) futures, and is passed via a new optional `_closed_loop` keyword if Task 1 did not add one. If this touches the core file, controller authorizes the minimal addition once Task 1 is complete. Thread the same optional proposal_std through run_grouped_preference and its CLI/config so the learned comparison uses the identical controller. Save actual scaled current latents and describe this as a changed proposal distribution; no claim to unchanged-prior Gibbs sampling. This is a coverage probe, separate from future-sequence search.
- [x] CLI must flush condition progress and record elapsed time, successful target count, task failures and fallback. Load/save NPZs from one condition at a time to bound RAM. Use Torch threads=1 by default; GPU is selected explicitly. Expose regime/user/M/method lists to run disjoint subsets safely in separate output directories.
- [x] Tests: tiny or fixture policy execution detects soft/best difference, same random streams across M, successful-mode exclusion for goal misses, resume equivalence and mismatch refusal, scaled current latent payload versus untouched future streams. Do not write tests that merely assert CLI implementation text.
- [x] Report focused tests and concerns; no commit.

## Task 3: Scientific evaluation, independent audit and inference GIFs

**Own files:** new experiment artifacts under `env/artifacts/grouped_preference/mode-aligned-*`, analysis scripts under its analysis folder, result document `docs/research/2026-09-12-mode-aligned-steering-results.md`, and `env/README.md` usage addition.

- [x] Measure runtime on a small real-checkpoint oracle run, then execute matched M/selector controls. Retain all eight profile names so alpha priorities are evaluated. If runtime requires an exploratory subset of episode seeds, state it and use the same subset across comparisons; do not select favorable episodes.
- [x] Compare old continuous/soft M8 versus mode/soft M8, then mode/best M8, then M32/M128 with L and beta fixed. Observe coverage and actual successful target modes. Escalate to wider-current-latent proposal exploration only if mode coverage stays inadequate; select scale without using held-out performance as training data. Treat latent sequence optimization or base retraining as a separate subsequent stage if needed, with an explicit design recorded before implementation.
- [x] Regenerate scoped synthetic labels for mode features and fit K5/10/20/40. Compare learned20 and oracle at a specified tested controller setting. Preserve two-group semantics and acknowledge that hard-bin toy fitting does not establish human preference learning.
- [x] Independently replay saved selected commands in Gym, recalculate selected feature means/score/argmax or soft RNG, and verify frozen hashes. Run relevant regression tests and one full `env/tests` suite after changes settle.
- [x] Render GIFs from saved ACTUAL paths: same fixed episode across targets and methods, all episode paths visible, centered and gaussian separately, with success-and-mode counts. Add comparison plots/tables and explicitly state unresolved modes, failures, uncertainty and runtime.
- [x] Obtain task/core review and final independent review, resolve material findings, update this plan and ledger, and report artifacts and evidence to the user.

## Publication follow-up

2026-09-12: the user explicitly requested a method/artifact summary and push. That later authorization supersedes the completed experiment stage's no-commit/no-push constraint. Publish the existing B2 branch after validation; preserve raw local artifacts and do not merge main. See `docs/research/2026-09-12-preference-steering-artifacts.md` for the curated Git payload.
