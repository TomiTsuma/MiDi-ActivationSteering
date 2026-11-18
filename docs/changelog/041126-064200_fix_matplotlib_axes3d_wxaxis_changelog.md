# Fix: sampling/visualization crashing with "'Axes3D' object has no attribute 'w_xaxis'"

**Date:** 2026-11-04

## Problem

After the DDP/NCCL fix, `trainer.test(...)` ran successfully through evaluation and
started sampling/visualizing molecules, then crashed in `on_test_epoch_end`:

```
AttributeError: 'Axes3D' object has no attribute 'w_xaxis'
```

Traceback: `midi/diffusion_model.py:sample_batch` -> `visualization.py:visualize`
-> `plot_save_molecule` -> `generatePIL3d` -> `ax.w_xaxis.line.set_color(...)`.

## Root cause

`midi/analysis/visualization.py` used `ax.w_xaxis`, a Matplotlib `Axes3D` alias
for `ax.xaxis` that was deprecated in Matplotlib 3.1 and removed by 3.7. The
installed environment has Matplotlib 3.8.4, where the attribute no longer
exists at all. MiDi was originally written against an older Matplotlib
release where the alias still worked; it is not a logic bug, just a stale
API name — the same function already uses the modern `ax.xaxis`/`ax.yaxis`/
`ax.zaxis` three lines above the failing call.

## Fix

`midi/analysis/visualization.py`, `generatePIL3d()`: replaced both
`ax.w_xaxis.line.set_color(...)` calls with `ax.xaxis.line.set_color(...)`.

## Notes

- No other `w_xaxis`/`w_yaxis`/`w_zaxis` usages exist elsewhere in the repo.
- Related to [[041126-064500_fix_windows_nccl_ddp_strategy_changelog]] — both
  surfaced while getting `python3.11 -m midi.main ... general.test_only=...`
  running end-to-end on Windows with Matplotlib 3.8.4.
