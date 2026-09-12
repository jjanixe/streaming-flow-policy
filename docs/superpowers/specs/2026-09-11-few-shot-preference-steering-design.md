# Few-shot preference steering of frozen SFPS

The user approved the preceding proposal and requested implementation plus a
publication/code audit. This implements candidate coverage, oracle steering,
and few-shot linear preference fitting. Gradient steering is conditional on
those results and is not required for this first experiment.

## Fixed scope

- Reuse the latest B2 worktree and selected optimized SFPS EMA checkpoint.
- Keep all policy parameters frozen; preserve existing uncommitted B2 changes.
- Use the existing single-detour 64-action environment, position commands,
  max step distance 0.075, two-observation condition, and eight executed actions.
- Each prediction requests nine positions, skips the anchor, refreshes history,
  resets local time, and samples a fresh two-dimensional latent per chunk.
- Use full completed trajectory features from the existing train-normalized
  direction/width representation. Never label each chunk with the full label.
- Fit only user weights with sum Bradley–Terry negative log likelihood plus
  0.1/2 times squared weight norm. Fixed likelihood temperature 1, no intercept.
- Use float32 model/data arrays. Save RNG seeds, labels, fitted weights, raw
  executed/requested trajectories, failures, checkpoint and train-data digests.
- Do not overwrite existing experiment artifacts or commit unrelated changes.

## Preference fitting

Add a standalone arbitrary-low-dimensional regularized BT module. Labels are
strictly -1/+1 and feature differences are finite [K,D]. Use stable softplus
and sigmoid, positive L2, Newton steps with backtracking, explicit convergence
and rank diagnostics. The learner receives feature differences and labels,
never true synthetic weights. Empty/no-information input and rank deficiency
must be reported honestly. A/B swap must leave fitting unchanged.

For the experiment generate an independent clean comparison trajectory pool
using the existing analytic path family, not the base training bank. Balanced
query types cycle direction, width, and joint trade-off; query generation is
independent of user weights. Synthetic profiles are (+/-2, +/-1). Each user gets
one 40-comparison noisy BT stream, with budgets 5/10/20/40 as nested prefixes.
Use a separate held-out trajectory/pair stream and report log loss, noisy label
accuracy and oracle-ordering accuracy. Also evaluate preference prediction on
successful generated trajectory pairs from the same fixed initial condition.
This is a synthetic-user feasibility experiment, not human personalization
evidence. No claim that 5 or 20 labels suffice for arbitrary user preferences.

## Candidate rollout and selection

At every replan sample M=32 remaining latent sequences from the original
product standard Gaussian. Roll out the complete remaining episode for each
candidate, updating predicted two-step observation history at every eight-step
boundary. Preserve the actual executed prefix. Feature utility is evaluated
only for completed paths with finite actions satisfying the environment limit.

Selection scores are beta*w.T*phi - (goal_error/goal_tolerance)^2, beta=1.
All hard-invalid candidates get zero mass. If none are valid, select candidate
zero as an explicitly recorded fallback and execute its first chunk under Gym
semantics (it may fail; never silently project). Goal tolerance itself is a
soft cost, not a hard filter; report actual task success separately.

Support `base` (candidate zero), `best` (argmax) and `soft` (categorical softmax).
Soft selection uses a separate RNG from candidate generation; record ESS,
candidate valid fraction, fallback counts and chosen latent/continuation.
Candidate generation is indexed by rollout seed and global replan, so changing
selection method does not silently consume a different proposal stream.
Replanning evaluates fresh candidates from the realized state. No gradients,
learned critic, policy fine-tuning, latent carryover or exact-sampling claim.

Candidate simulation mirrors rejected-command behavior: preserve last valid
position, terminate immediately, distinguish numerical/action-limit failures,
and do not count repeated terminal padding as executed states or utility.
Validate simulator against real Gym and existing SFPS rollout with the same
latent schedule. Any non-finite batch prediction isolates failures per row.

## Experiment and evidence

First inspect conditional coverage with at least 512 base latent sequences at
the centered start and at two seeded Gaussian starts, separately. Report
coverage denominator, midpoint occupancy, goal success and failures.
Then run base, task-only soft, oracle soft, and learned soft at four budgets
for each profile on the same evaluation starts/seeds. Include an oracle-best
comparison. Start with 16 closed-loop episodes per initial-condition regime;
report this as a small pilot with uncertainty, not a definitive benchmark.
Use centered and Gaussian regimes separately; never reweight across starts.

Persist JSON/NPZ artifacts and static trajectory/utility-vs-budget plots.
Compare successful-trajectory true utility and goal success jointly; failed
episodes remain in success/failure denominators and have unavailable full-path
utility rather than a silently padded score. Report wall time and replan
latency. Confirm policy/checkpoint digest is unchanged after evaluation.

## Literature audit

Use official proceedings/publisher records to distinguish reviewed main-track
conference/journal publication from preprints and unverified venue claims.
Inspect author-linked GitHub code, distinguishing source files from placeholder
repositories or coming-soon links. Record checked date 2026-09-11, direct
evidence URLs, limitations, and any corrections to the prior summary.
