# Final review: mode-aligned steering

Assessment: **PASS — no blocking findings.** The approved implementation and exploratory evaluation are complete in scope. This verdict does not mean Gaussian wide control is solved; the result document explicitly preserves that limitation.

## Scope and evidence

Reviewed the diagnosis/specification, implementation plan, exact task-1/task-2 scoped patches and task reports/reviews, current core implementation and tests, new README subsection, result document, and seven analysis scripts. Current task-2 source files match their reviewed after-task snapshots; the only later change to task-1 core files is the authorized task-2 proposal plumbing. No attribution was made from the worktree-wide HEAD diff.

Read the existing full-suite log: **389 passed, 2 skipped, 15 warnings in 109.88 s**. A redundant full-suite run was not performed.

Fresh read-only verification in this review passed:

- All **172 formal conditions / 2,752 episodes / 22,016 decisions**: independently recomputed complete-task-success-and-mode counts, checked those against diagnostics and saved independent audits, checked selected current commands against actual requested actions, regenerated every recorded current and future RNG draw, and verified paired environment/rollout seeds across the five formal folders.
- Current checkpoint/stats hashes match every formal run. Both ablation matrices' NPZ hashes and completion records match; their eight recorded source hashes still match current files.
- Primary centered oracle/learned20 counts are 16/16 for all eight profiles. Gaussian narrow is 16/16; upper-wide direction/width are 13/16 and 12/16; lower-wide both priorities are 15/16. Every primary oracle/learned condition has task success 16/16.
- Replayed all nine failure-probe saved command sequences in fresh Gym instances: exact saved positions and **3/9** target repairs. Recomputed all 640 fit-row summary aggregates. Checked coverage diagnostics and the Gaussian episode-6 M512 saved banks: maxima .38334578 (std8) and .38908827 (std16), with no successful upper-wide future.
- Inspected the Gaussian final inference frame and renderer: fixed episode 0, all 16 learned trajectories, and visible lower-wide misses agree with the described presentation. The old/new M8 controls retain matching seed/M/L/beta/integration settings.

## Strengths

- `env/grouped_preference.py:150` and `env/run_grouped_preference.py:178`: the common float64 midpoint classifier, mode feature mapping, and complete-success metric consistently distinguish geometry from task success. Goal misses cannot inflate successful-target counts.
- `env/run_grouped_preference.py:118`: proposal scaling affects guided current draws only. The saved streams confirm unit-normal base/future draws, paired underlying random streams, and correctly executed selected commands. The actual-prefix anchor and nine-points/eight-commands convention remain intact.
- `env/run_mode_ablation.py:139`: atomic artifacts, separate completion markers, resolved configuration, metadata checks, and per-condition processing support reproducible comparisons and detect recorded artifact corruption.
- `docs/research/2026-09-12-mode-aligned-steering-results.md`: claims are appropriately bounded. It identifies changed proposal semantics without importance correction, exploratory reuse of seeds, small-sample uncertainty, synthetic hard-bin fitting, unresolved Gaussian wide cases, and failure-conditioned lookahead. The nine-case probe is kept separate from the main controller/GIFs and uses predetermined future sample zero rather than choosing a successful realized future.

## Issues

- **Low, nonblocking — incomplete runtime source manifest:** `env/run_mode_ablation.py:29` enumerates eight modules, omitting runtime dependencies such as `chunk_data.py` normalization, `run_stage_b.py` loading, and `run_preference.py` seed helpers. A future edit confined to an omitted dependency would not itself trigger the source-hash resume guard. This does not invalidate the present runs: no relevant dependency changed during execution, and the recorded hashes/artifacts pass. The final result document at line 140 and README now explicitly scope validation to the listed eight modules and require a fresh output directory after any code change. Documentation mitigation is accepted for this study.

## Recommendations

In later maintenance, expand the source manifest to the full local runtime dependency set (and test refusal after changing an omitted dependency). Also preserve selected-command linkage and future-RNG checks in the reusable independent audit script; this review independently checked both across every present formal artifact. Neither recommendation requires rerunning this completed study.

Proceed with final plan/ledger completion and report the centered success together with the remaining Gaussian-wide failures. Do not describe the failure probe as a learned-controller improvement or the finite-bank gaps as mathematical unreachability.

Only this report was written by the reviewer. Code, index, HEAD, and experiment artifacts were not modified.
