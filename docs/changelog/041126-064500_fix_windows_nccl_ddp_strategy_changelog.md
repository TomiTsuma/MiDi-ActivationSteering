# Fix: single-GPU test/inference failing with "Distributed package doesn't have NCCL built in"

**Date:** 2026-11-04

## Problem

Running `python3.11 -m midi.main ... general.gpus=1 ... general.test_only=...` on Windows
crashed during `trainer.test(...)`:

```
RuntimeError: Distributed package doesn't have NCCL built in
```

## Root cause

`midi/main.py` hardcoded the Lightning `Trainer` strategy to
`"ddp_find_unused_parameters_true"` regardless of device count. For a `'gpu'`
accelerator, Lightning's DDP strategy defaults to the `nccl` process-group
backend. NCCL is a Linux-only NVIDIA library — the official Windows PyTorch
wheels are built without it (Windows distributed only ships `gloo`). So any
run that hit this code path initialized `torch.distributed` with NCCL and
failed immediately, independent of GPU/driver/checkpoint — even for a
single-GPU, single-process test run that never actually needed DDP.

## Fix

`midi/main.py`: only request the DDP strategy when `cfg.general.gpus > 1`;
otherwise use `"auto"`, which resolves to `SingleDeviceStrategy` and never
touches `torch.distributed`.

```python
strategy="ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else "auto",
```

## Notes

- This does not affect multi-GPU training — DDP is still used whenever
  `general.gpus > 1`, preserving the original "needed to load old
  checkpoints" behavior for that case.
- The `[W...] client socket has failed to connect to [ecommerce.local]:...`
  warning seen in the same run is a side effect of Lightning's DDP
  rendezvous trying to resolve the local hostname — it disappears along
  with the NCCL error once DDP is no longer initialized for single-device
  runs.
