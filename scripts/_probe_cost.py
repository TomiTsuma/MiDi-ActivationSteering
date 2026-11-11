"""One cell, timed per feature set, so the grid is a decision and not a guess.

Measures read / standardise / build / fit / null separately, for every feature
set in the manifest, then projects the full grid. Nine of the ten feature sets
are far narrower than `all`, so a projection from `all` alone would be badly
pessimistic; this measures each.

Throwaway. Delete once the number is recorded.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import probe

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "activations_test_full")
LAYER, T_INDEX = 11, 9          # t=50, the cleanest cell
PROP = "polarisability"
N_PERM = 200

man = probe.load_manifest(STORE)
labels = probe.load_labels(STORE)
fsets = probe.feature_sets(man)
n = int(man["n_molecules"])
train, test = probe.make_split(n)
timesteps = list(man["timesteps"])

print(f"store      {STORE}")
print(f"cell       layer {LAYER}, t_index {T_INDEX} -> timestep {timesteps[T_INDEX]}")
print(f"molecules  n={n}  train={len(train)}  test={len(test)}")
print(f"feature sets ({len(fsets)}): "
      + ", ".join(f"{k}:{v[1] - v[0]}" for k, v in fsets.items()))

C = probe.composition_matrix(labels)
y = labels[PROP].astype(np.float64)
r, b = probe.residualize(y, C, train)
print(f"composition-only R2 for {PROP}: {probe.r2(y[test], (C @ b)[test]):.4f}")

t0 = time.time()
P = probe.permute_targets(y, N_PERM, seed=777)
t_perm = time.time() - t0
print(f"\npermute_targets({N_PERM}) once: {t_perm:.2f}s  "
      f"({P.nbytes / 1024**2:.0f} MB per property/variant)")

mm = probe.open_layer(STORE, LAYER)
t0 = time.time()
A = probe.read_cell(mm, T_INDEX)
t_read = time.time() - t0
print(f"read_cell: {t_read:.2f}s for {A.nbytes / 1024**2:.0f} MB\n")

hdr = (f"{'feature_set':<10} {'d':>5} {'std':>7} {'build':>8} {'fit':>7} "
       f"{'null':>8} {'r2':>8} {'null_mu':>9} {'z':>9}")
print(hdr)
print("-" * len(hdr))

tot_build = tot_std = tot_fit = tot_null = 0.0
rows = []
for name, (lo, hi) in fsets.items():
    t0 = time.time()
    Ztr, Zte, d_used, d_drop = probe.standardize(
        A[train, lo:hi].astype(np.float64), A[test, lo:hi].astype(np.float64))
    t_std = time.time() - t0

    t0 = time.time()
    s = probe.CellSolver(Ztr)
    t_build = time.time() - t0

    t0 = time.time()
    fit = probe.fit_cell(s, Zte, y[train], y[test])
    t_fit = time.time() - t0

    t0 = time.time()
    r2_null, _ = probe.fit_many(s, Zte, P[train], P[test])
    t_null = time.time() - t0

    st = probe.floor_stats(fit["r2"], r2_null)
    print(f"{name:<10} {d_used:>5} {t_std:>7.2f} {t_build:>8.2f} {t_fit:>7.2f} "
          f"{t_null:>8.2f} {fit['r2']:>8.4f} {st['r2_null_mean']:>9.4f} "
          f"{st['z_score']:>9.1f}")
    rows.append((name, d_used, t_std, t_build, t_fit, t_null))
    tot_std += t_std
    tot_build += t_build
    tot_fit += t_fit
    tot_null += t_null
    del s

n_layers = int(man["n_layers"])
n_t = len(timesteps)
n_cells = n_layers * n_t
n_targets = 6 * 2          # 6 properties x 2 variants

per_cell = tot_std + tot_build + (tot_fit + tot_null) * n_targets + t_read
total_h = n_cells * per_cell / 3600.0

print("\n--- projection ---")
print(f"per feature set, summed over all {len(fsets)}:")
print(f"  standardise {tot_std:8.2f}s   build {tot_build:8.2f}s")
print(f"  fit (1 target) {tot_fit:5.2f}s   null (1 target, B={N_PERM}) {tot_null:7.2f}s")
print(f"per cell, with {n_targets} targets: {per_cell:.1f}s")
print(f"grid = {n_cells} cells x {len(fsets)} feature sets x {n_targets} targets")
print(f"ESTIMATED TOTAL: {total_h:.2f} hours")
print(f"  of which null: {n_cells * tot_null * n_targets / 3600.0:.2f} h "
      f"({100.0 * tot_null * n_targets / per_cell:.0f}%)")
print(f"  of which build: {n_cells * tot_build / 3600.0:.2f} h")
print(f"\nif the null dominates, B=100 halves it; do that before touching")
print(f"the lambda grid, since fewer permutations costs precision whereas a")
print(f"fixed lambda costs validity.")
