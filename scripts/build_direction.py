"""scripts/build_direction.py -- the steering directions, saved.

    python3.11 -u scripts/build_direction.py <store> <outdir> [property ...]

One .npz per property, each holding a direction at every captured timestep,
computed both ways (raw split and composition-residual split) with the angle
between them, plus the ridge probe's own direction for comparison.

These files are the artefact that reproduces the central claim against a
public checkpoint, so everything needed to regenerate them is recorded inside:
the store's git commit and run_seed, the layer, the block, the split rule and
the molecule counts.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import direction, probe

LAYER = 10          # the probe's best cell, and X is injectable
BLOCK = "X_mean"
REF_T = 50
ALL_PROPS = ("polarisability", "heat_capacity", "dipole_moment",
             "homo", "lumo", "homo_lumo_gap")


def main(store, outdir, props):
    store = os.path.abspath(store)
    os.makedirs(outdir, exist_ok=True)
    man = probe.load_manifest(store)
    print(f"store  {store}")
    print(f"site   layer {LAYER}, block {BLOCK} "
          f"({man['feature_slices'][BLOCK][1] - man['feature_slices'][BLOCK][0]} dims)")
    print(f"       only {direction.INJECTABLE} are injectable; min/max/var have "
          f"no defined 'add v', and 'all' is a concatenation of statistics")
    print()

    for prop in props:
        res = direction.build(store, LAYER, prop, block=BLOCK, ref_t=REF_T)
        path = os.path.join(outdir, f"dir_L{LAYER}_{BLOCK}_{prop}.npz")
        direction.save(res, path)

        ct = res["cos_across_t_resid"]
        off = ct[np.triu_indices_from(ct, k=1)]
        ref_row = next(r for r in res["per_t"] if r["t"] == REF_T)

        print(f"{prop}")
        print(f"  saved            {path}  ({os.path.getsize(path):,} bytes)")
        print(f"  ||d|| at t={REF_T}    raw {ref_row['norm_raw']:.4f}   "
              f"resid {ref_row['norm_resid']:.4f}")
        print(f"  cos(raw, resid)  {ref_row['cos_raw_resid']:+.4f}   "
              f"<- how much of the naive direction is size")
        print(f"  cos(probe w, d)  {res['cos_w_vs_d_resid']:+.4f} vs resid, "
              f"{res['cos_w_vs_d_raw']:+.4f} vs raw")
        print(f"  probe r2 (resid) {res['probe_r2_resid']:.4f} at lam "
              f"{res['probe_lam']:.1e}")
        print(f"  direction rotation across timesteps: cos in "
              f"[{off.min():+.3f}, {off.max():+.3f}]")
        print(f"  split            {ref_row['n_high']} high / "
              f"{ref_row['n_low']} low")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or list(ALL_PROPS))
