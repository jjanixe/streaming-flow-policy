# Task 2 review: conditional continuations and grouped runner

## Verdicts

- **Spec compliance: PASS.** The evaluator and runner preserve the frozen
  16-point SFPS convention, separate the observed and raw model anchors from
  the eight executed commands, share each current candidate across independent
  futures, average full-path features before preference exponentiation, retain
  failed futures, and execute the selected chunk or guided fallback through the
  actual Gym environment.
- **Code quality: PASS.** The implementation keeps the conditional branching in
  the existing rollout executor, validates the public array contract, uses
  explicit deterministic random streams, and records enough masks, seeds,
  failures, traces, and denominators to audit every selection and actual
  outcome. No actionable defects were found in the scoped diff.

## Findings

No findings.

## Evidence reviewed

- `task-2-brief.md`, `task-2-report.md`, and the complete
  `task-2-review.diff`, plus the approved grouped-preference design.
- `env/grouped_rollout.py`: current rows are simulated only through `q+8`, then
  only valid rows are repeated in C-order `[B,M,L]` and continued. Invalid
  current candidates remain recorded and ineligible; incomplete futures remain
  in the `L` denominator with zero features and their terminal-position task
  cost. Completed goal misses retain geometric features and receive the failure
  penalty.
- `env/preference_rollout.py`: `stop_at` preserves the default full-horizon
  behavior, consumes remaining latents relative to the supplied prefix, skips
  point zero, and stores raw anchors at absolute chunk indices. The current and
  future calls therefore use the intended latent rows at every nonzero replan
  boundary.
- `env/run_grouped_preference.py`: current, per-candidate future, and selection
  seeds use separate namespaces; candidate zero is stable when changing `M`,
  and future prefixes are stable when changing `L`. Conditions reuse the same
  rollout/environment seeds. Guided all-invalid batches execute eight hold
  commands with `decision_mask=true`, `selected_indices=-1`, and explicit
  fallback state, while base records and executes candidate zero even when it
  fails.
- Artifact integration records all candidate and continuation arrays, observed
  and model anchors, selection validity, latent seeds, expected features,
  scores, costs, actual failures, and successful/all-outcome utility
  denominators. Scoped labels use the saved scope, held-out data and labels are
  retained, and checkpoint/statistics hashes plus in-memory state are checked
  unchanged.
- The supplied `full-suite.log` reports 366 passed, 2 skipped, and 14 warnings.
  Per instruction, the reported suites were not rerun during this review.
