# Task 1 review: mode features and pipeline plumbing

## Verdicts

- **Spec compliance: PASS.** The pinned Task 1 diff implements every required behavior in the brief, preserves the continuous/soft defaults, and changes only the six owned files.
- **Code quality: PASS.** No concrete material correctness, regression, or maintainability finding was identified.

## Evidence

- `classify_modes` converts inputs to float64, rejects nonfinite values, and implements the specified UN/UW/LN/LW boundary convention with `other=4` (`after-task1/env/grouped_preference.py:149-164`). `grouped_features` preserves the continuous default and maps mode IDs to `[direction: upper/lower, width: wide/narrow]`, leaving `other` as the initialized zero matrix (`after-task1/env/grouped_preference.py:167-188`).
- Conditional rollouts validate and propagate `feature_kind`, while complete continuations receive the selected feature representation and incomplete continuations retain zero features (`after-task1/env/grouped_rollout.py:34-47`, `after-task1/env/grouped_rollout.py:88-105`).
- `_closed_loop` accepts `best`, forwards `feature_kind`, and delegates selection to the existing `select_candidates` contract. Its fallback branch records no selected index and executes repeated current-position commands rather than candidate 0 (`after-task1/env/run_grouped_preference.py:50-58`, `after-task1/env/run_grouped_preference.py:117-148`).
- `_path_metrics` uses the selected feature representation for actual-path utility and the shared classifier for both reached-midpoint and complete-successful midpoint counts (`after-task1/env/run_grouped_preference.py:170-202`).
- The public runner validates and records both options, recomputes query and heldout features and labels under the selected semantics, applies the requested selection method to task-only/oracle/learned conditions, preserves default soft names, names best oracle conditions `oracle_best`, and stores each condition's actual method (`after-task1/env/run_grouped_preference.py:260-280`, `after-task1/env/run_grouped_preference.py:300-310`, `after-task1/env/run_grouped_preference.py:321-354`, `after-task1/env/run_grouped_preference.py:355-374`). The CLI exposes both constrained choices (`after-task1/env/run_grouped_preference.py:388-401`).
- Tests cover all eight desired-mode profiles, exact boundaries, and zero-valued `other` (`after-task1/env/tests/test_grouped_preference.py:57-105`); conditional feature propagation and incomplete-zero behavior (`after-task1/env/tests/test_grouped_rollout.py:100-117`); actual Gym argmax execution, all-invalid fallback for both soft and best, successful-mode filtering, and a tiny actual-checkpoint mode+best artifact/configuration run (`after-task1/env/tests/test_run_grouped_preference.py:32-114`, `after-task1/env/tests/test_run_grouped_preference.py:173-219`).

## Verification assessment

The implementation report records a final grouped-plus-flat regression result of **75 passed, 5 warnings**, plus focused grouped verification and compilation/CLI checks. Per the review brief, I did not rerun those suites. Static inspection raised no concrete doubt requiring an additional focused command. The recorded CPU NVML, Gym/NumPy, and Matplotlib cache warnings are pre-existing limitations and do not affect this verdict.
