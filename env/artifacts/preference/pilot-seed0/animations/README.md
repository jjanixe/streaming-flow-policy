# Saved inference: requested versus measured mode

These GIFs replay the executed positions saved by the seed-0 preference pilot.
No new inference or model fitting is performed. All 64 executed actions appear,
with a pause at midpoint and at the end; playback is slower than simulated time.

- [Centered start GIF](centered_mode_selection.gif)
- [Gaussian starts GIF](gaussian_mode_selection.gif)

Columns are requested preferences. The top row uses 20-comparison learned
utility and soft selection; the bottom row uses true synthetic utility and
best-of-32 selection. Thin paths show all 16 episodes. The bold path always
shows episode 0, chosen by index before inspecting outcomes. Gray shows the
same base episode 0. The dashed curve is an analytic target reference, not an
inference result. Red crosses mark task-failure endpoints at the final frame.

Every panel reports the final target-mode hit count and goal success count
separately. Mode is measured at state 32 with the existing classifier:
upper narrow `[0.12, 0.4)`, upper wide `[0.4, inf)`, lower narrow `(-0.4, -0.12]`,
lower wide `(-inf, -0.4]`; otherwise Other. The denominator is all 16 episodes,
including task failures. Target hits alone do not imply goal success.

| Requested preference | Centered learned20 | Centered oracle-best | Gaussian learned20 | Gaussian oracle-best |
|---|---:|---:|---:|---:|
| Upper narrow | 2/16 | 6/16 | 3/16 | 2/16 |
| Upper wide | 0/16 | 0/16 | 6/16 | 6/16 |
| Lower narrow | 5/16 | 13/16 | 1/16 | 4/16 |
| Lower wide | 0/16 | 0/16 | 9/16 | 9/16 |

Centered goal success is 16/16 for every panel. Gaussian learned20 is 14/16;
oracle-best is 16/16. Gaussian episode 0 is upper-wide for every requested
preference under both methods. This is evidence of limited preference control
in the saved pilot, not four demonstrations of successful mode selection.

Render from the B2 worktree root:

```bash
MPLCONFIGDIR=/tmp/preference-gif-matplotlib .venv/bin/python \
  env/artifacts/preference/pilot-seed0/animations/render_gifs.py
```

The renderer verifies shared episode seeds, matches all panel counts to
`../diagnostics.json`, verifies bounds, and decodes every GIF frame. Source
NPZ names and episode-0 seeds/modes are saved in [mode_summary.json](mode_summary.json).
