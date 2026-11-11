"""scripts/run_probes.py -- the full probe grid, resumable.

    python3.11 -u scripts/run_probes.py <store> <out.csv>

Grid: 12 layers x 10 timesteps x 10 feature sets x 6 properties x 2 target
variants = 14,400 rows. Measured at about 4.3 hours, of which 94% is the
permutation null.

Loop order is forced by IO: one memmap per layer, one 70 MB read per timestep,
one solver per feature set, then all 12 targets ride that solver.

Rows are appended and fsynced as they are computed, and a restart skips what
is already on disk. That is not caution for its own sake: a 31-hour run on
this machine was already lost to a system shutdown with a buffered log and
nothing written.

No Hydra, so none of the working-directory traps apply. Paths are arguments.
"""
import csv
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import probe

PROPERTIES = ("polarisability", "heat_capacity", "dipole_moment",
              "homo", "lumo", "homo_lumo_gap")
VARIANTS = ("raw", "size_residual")
N_PERM = 200
PERM_SEED = 777

FIELDS = [
    "layer", "t_index", "timestep", "feature_set", "property", "variant",
    "d_used", "d_dropped", "n_train", "n_test", "n_per_dim",
    "lam", "alpha_shrinkage", "df_eff", "df_frac", "lam_at_grid_edge",
    "r2", "r2_size_only",
    "r2_null_mean", "r2_null_sd", "r2_null_p95",
    "n_perm", "perm_exceed", "perm_p", "perm_p_is_bound", "z_score", "z_p",
    "seconds",
]


def row_key(d):
    return (int(d["layer"]), int(d["t_index"]), d["feature_set"],
            d["property"], d["variant"])


def already_done(path):
    """Trust the file, not a state record: a flushed row is a finished row."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        try:
            return {row_key(r) for r in csv.DictReader(f)}
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"{path} is not a probe CSV ({exc}); move it aside")


def write_manifest(path, store, man, n_cells, fsets, elapsed=None):
    with open(path, "w") as f:
        json.dump({
            "store": store,
            "store_manifest": {k: man[k] for k in (
                "mode", "split", "n_molecules", "n_layers", "timesteps",
                "features", "run_seed", "git_commit", "torch", "gpu_name")
                if k in man},
            "split_seed": probe.SPLIT_SEED,
            "test_frac": probe.TEST_FRAC,
            "lam_grid": list(probe.LAM_GRID),
            "n_perm": N_PERM,
            "perm_seed": PERM_SEED,
            "properties": list(PROPERTIES),
            "variants": list(VARIANTS),
            "feature_sets": {k: list(v) for k, v in fsets.items()},
            "composition_elements": list(probe.COMPOSITION_ELEMENTS),
            "n_cells": n_cells,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "elapsed_seconds": elapsed,
        }, f, indent=2)


def acquire_lock(out_csv):
    """Refuse to run if another instance is writing the same CSV.

    Two concurrent runs do not corrupt the file (rows are deterministic, so the
    duplicates are exact copies) but they double the work and both compute the
    whole grid, because each one's resume set is read once at startup and never
    refreshed. This happened: pids 78188 and 68836 on 2026-11-11.

    O_EXCL makes creation atomic, so the race is decided by the filesystem.
    """
    lock = out_csv + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(lock) as f:
                who = f.read().strip()
        except OSError:
            who = "unknown"
        raise SystemExit(
            f"another run_probes instance holds {lock} ({who}).\n"
            f"If that process is definitely gone, delete the lock file and "
            f"re-run; resume will pick up where it stopped.")
    with os.fdopen(fd, "w") as f:
        f.write(f"pid {os.getpid()} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return lock


def main(store, out_csv):
    store = os.path.abspath(store)
    out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    lock = acquire_lock(out_csv)
    try:
        _run(store, out_csv)
    finally:
        try:
            os.unlink(lock)
        except OSError:
            pass


def _run(store, out_csv):

    man = probe.load_manifest(store)
    labels = probe.load_labels(store)
    fsets = probe.feature_sets(man)
    timesteps = list(man["timesteps"])
    n_layers = int(man["n_layers"])
    n = int(man["n_molecules"])
    train, test = probe.make_split(n)

    missing = [p for p in PROPERTIES if p not in labels]
    if missing:
        raise SystemExit(f"labels.npz is missing {missing}")
    if "n_H" not in labels:
        raise SystemExit("labels.npz has no composition columns; run "
                         "scripts/add_composition_labels.py first")

    C = probe.composition_matrix(labels)

    # Targets, their permutations and the composition-only R2 are all
    # independent of the cell, so build them once. r2_size_only depends only
    # on the target and the split, which is why it must come out identical on
    # every one of the 2,400 rows for a property: a cheap, strong check that
    # nothing is being recomputed per cell by accident.
    targets, perms, r2_size_only = {}, {}, {}
    for prop in PROPERTIES:
        y = labels[prop].astype(np.float64)
        r, b = probe.residualize(y, C, train)
        targets[(prop, "raw")] = y
        targets[(prop, "size_residual")] = r
        r2_size_only[prop] = probe.r2(y[test], (C @ b)[test])
        for v in VARIANTS:
            perms[(prop, v)] = probe.permute_targets(
                targets[(prop, v)], N_PERM, seed=PERM_SEED)

    n_cells = n_layers * len(timesteps)
    n_rows = n_cells * len(fsets) * len(PROPERTIES) * len(VARIANTS)
    done = already_done(out_csv)

    print(f"[probes] store      {store}")
    print(f"[probes] molecules  n={n} train={len(train)} test={len(test)}")
    print(f"[probes] grid       {n_layers} layers x {len(timesteps)} timesteps "
          f"x {len(fsets)} feature sets x {len(PROPERTIES)} properties "
          f"x {len(VARIANTS)} variants = {n_rows:,} rows")
    print(f"[probes] null       B={N_PERM} permutations, seed {PERM_SEED}")
    print(f"[probes] composition-only R2: "
          + ", ".join(f"{p}={r2_size_only[p]:.3f}" for p in PROPERTIES))
    if done:
        print(f"[probes] RESUMING: {len(done):,}/{n_rows:,} rows already on disk")
    else:
        print(f"[probes] fresh run")

    write_manifest(os.path.splitext(out_csv)[0] + "_manifest.json",
                   store, man, n_cells, fsets)

    fresh = not os.path.exists(out_csv)
    t_run = time.time()
    written = 0
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
            fh.flush()

        for layer in range(n_layers):
            mm = probe.open_layer(store, layer)
            for t_index, t_value in enumerate(timesteps):
                pending_cell = [
                    (fs, p, v) for fs in fsets for p in PROPERTIES for v in VARIANTS
                    if (layer, t_index, fs, p, v) not in done]
                if not pending_cell:
                    print(f"[probes] layer {layer:2d} t={t_value:3d}  skipped "
                          f"(complete)", flush=True)
                    continue

                t_cell = time.time()
                A = probe.read_cell(mm, t_index)
                for fs_name, (lo, hi) in fsets.items():
                    pending = [(p, v) for p in PROPERTIES for v in VARIANTS
                               if (layer, t_index, fs_name, p, v) not in done]
                    if not pending:
                        continue
                    Ztr, Zte, d_used, d_drop = probe.standardize(
                        A[train, lo:hi].astype(np.float64),
                        A[test, lo:hi].astype(np.float64))
                    s = probe.CellSolver(Ztr)
                    for prop, var in pending:
                        t0 = time.time()
                        y = targets[(prop, var)]
                        fit = probe.fit_cell(s, Zte, y[train], y[test])
                        P = perms[(prop, var)]
                        r2_null, _ = probe.fit_many(s, Zte, P[train], P[test])
                        st = probe.floor_stats(fit["r2"], r2_null)

                        row = {
                            "layer": layer, "t_index": t_index,
                            "timestep": t_value, "feature_set": fs_name,
                            "property": prop, "variant": var,
                            "d_used": d_used, "d_dropped": d_drop,
                            "n_test": len(test),
                            "r2_size_only": r2_size_only[prop],
                            "seconds": round(time.time() - t0, 2),
                        }
                        row.update({k: fit[k] for k in (
                            "lam", "alpha_shrinkage", "df_eff", "df_frac",
                            "n_train", "n_per_dim", "r2", "lam_at_grid_edge")})
                        row.update(st)
                        w.writerow(row)
                        fh.flush()
                        os.fsync(fh.fileno())   # survive a shutdown mid-run
                        written += 1
                    del s

                elapsed = time.time() - t_run
                total_done = len(done) + written
                rate = written / elapsed if elapsed > 0 else 0.0
                eta = (n_rows - total_done) / rate / 3600.0 if rate > 0 else float("nan")
                print(f"[probes] layer {layer:2d} t={t_value:3d}  "
                      f"{time.time() - t_cell:6.1f}s  "
                      f"{total_done:6,}/{n_rows:,} rows "
                      f"({100.0 * total_done / n_rows:5.1f}%)  eta {eta:5.2f}h",
                      flush=True)

    write_manifest(os.path.splitext(out_csv)[0] + "_manifest.json",
                   store, man, n_cells, fsets, elapsed=time.time() - t_run)
    print(f"\n[probes] wrote {written:,} new rows in "
          f"{(time.time() - t_run) / 3600.0:.2f}h -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
