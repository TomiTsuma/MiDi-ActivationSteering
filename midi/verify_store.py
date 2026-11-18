"""Verify an activation store before trusting it, and run the sanity probe.

Usage (no GPU, no checkpoint, seconds to minutes):

    python -m midi.verify_store --store <path-to-store>

What it checks, in order of how cheap it is to find a bug this way:

  1. STRUCTURE   one file per layer, the shape the manifest claims, float32,
                 finite everywhere, not all-zero, not constant across molecules.
  2. ALIGNMENT   labels present for every captured molecule, dataset positions
                 contiguous from 0, atom counts in range, and the label
                 distributions matching the published QM9 ranges. A silent
                 mislabelling is the single most likely defect, because the
                 split CSV is shuffled while the dataset is ordered by SDF
                 index, so joining on CSV row order mislabels everything.
  3. PROBE       a diagnostic ladder rather than one number, see below.

The ladder, and why it is a ladder
----------------------------------
Reporting one R-squared cannot distinguish "the property is not linearly
decodable" from "the store is broken". Three rungs can:

  rung 1   atom count from activations.
           The number of atoms is the most trivially present quantity in any
           graph representation. If this is not near 1.0, the capture is wrong:
           either the pooling is reading padding, or the labels are misaligned,
           or the arrays were written at the wrong offsets.

  rung 2   polarisability from atom count alone, by least squares.
           Pure chemistry, no activations involved. Polarisability scales with
           molecular volume, so this should be high. It is the ceiling any
           activation probe is competing with, and it is also the check that
           the LABELS are sane independent of the store.

  rung 3   polarisability from activations.
           Only meaningful once rungs 1 and 2 pass. If rung 1 is high and
           rung 2 is high but rung 3 is flat, that is a scientific result. If
           rung 1 is flat, it is a bug, and rung 3 tells you nothing.

Section 3.6 warns that decodability is not steerability. This script measures
decodability only. The liveness screen in capture.py (mode=liveness) is what
measures whether a site can be steered at all.
"""

import argparse
import json
import os

import numpy as np

# Published QM9 ranges, for the sanity bounds in check 2. Units are those of the
# QM9 release: alpha in Bohr^3, cv in cal/(mol K), mu in Debye, homo/lumo in Hartree.
LABEL_BOUNDS = {
    "polarisability": (6.0, 200.0),
    "heat_capacity": (6.0, 60.0),
    "dipole_moment": (0.0, 30.0),
    "homo": (-0.5, 0.0),
    "lumo": (-0.2, 0.3),
}


def ridge_r2(A, y, lam=1e-2, frac=0.7, seed=0):
    """Held-out coefficient of determination of a ridge probe.

    A : (N, F) activations.  y : (N,) labels.
    Standardises both, fits w minimising ||y - Aw||^2 + lam*||w||^2 on a
    `frac` training split, and reports R-squared on the rest:

        R2 = 1 - SS_res / SS_tot

    Ridge rather than plain least squares because F can exceed N on a pilot run
    and because activations are strongly correlated across coordinates.
    """
    N = A.shape[0]
    rs = np.random.RandomState(seed)
    perm = rs.permutation(N)
    ntr = int(frac * N)
    tr, te = perm[:ntr], perm[ntr:]
    if len(te) < 10:
        return float("nan")

    A = np.asarray(A, dtype=np.float64)
    mu, sd = A[tr].mean(0), A[tr].std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Atr, Ate = (A[tr] - mu) / sd, (A[te] - mu) / sd
    ym = y[tr].mean()
    ytr, yte = y[tr] - ym, y[te] - ym

    F = Atr.shape[1]
    if F <= ntr:
        G = Atr.T @ Atr + lam * ntr * np.eye(F)
        w = np.linalg.solve(G, Atr.T @ ytr)
    else:                                   # dual form, cheaper when F > N
        K = Atr @ Atr.T + lam * ntr * np.eye(ntr)
        w = Atr.T @ np.linalg.solve(K, ytr)
    pred = Ate @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def check_structure(store, man):
    L, T, N, Fd = man["n_layers"], len(man["timesteps"]), man["n_molecules"], man["features"]
    print(f"\n[1] STRUCTURE  expecting {L} files of shape ({T}, {N}, {Fd}) float32")
    problems = []
    for k in range(L):
        p = os.path.join(store, f"pooled_L{k:02d}.npy")
        if not os.path.exists(p):
            problems.append(f"layer {k}: file missing"); continue
        a = np.load(p, mmap_mode="r")
        tag = f"L{k:02d}"
        if a.shape != (T, N, Fd):
            problems.append(f"{tag}: shape {a.shape} != {(T, N, Fd)}")
        if a.dtype != np.float32:
            problems.append(f"{tag}: dtype {a.dtype} != float32")
        sub = np.asarray(a[:, :min(N, 512), :], dtype=np.float32)
        nf = int((~np.isfinite(sub)).sum())
        if nf:
            problems.append(f"{tag}: {nf} non-finite values in the first 512 molecules")
        if np.all(sub == 0):
            problems.append(f"{tag}: all zero, the hook probably never fired")
        # variation ACROSS molecules is the thing that matters for a probe
        var_across = float(sub[0].std(axis=0).mean())
        print(f"    {tag}  shape {a.shape}  finite  mean across-molecule sd {var_across:.4e}"
              + ("   <-- NO VARIATION" if var_across < 1e-9 else ""))
        if var_across < 1e-9:
            problems.append(f"{tag}: activations do not vary across molecules")
    return problems


def check_alignment(store, man):
    print(f"\n[2] ALIGNMENT")
    problems = []
    z = np.load(os.path.join(store, "labels.npz"))
    N = man["n_molecules"]
    for key in ("dataset_pos", "sdf_index", "n_atoms"):
        if key not in z:
            problems.append(f"labels.npz missing {key}")
    if problems:
        return problems, None

    dp, sdf, na = z["dataset_pos"], z["sdf_index"], z["n_atoms"]
    for nm, arr in (("dataset_pos", dp), ("sdf_index", sdf), ("n_atoms", na)):
        if len(arr) != N:
            problems.append(f"{nm} has {len(arr)} entries, store has {N} molecules")
    if not np.array_equal(dp, np.arange(N)):
        problems.append("dataset_pos is not 0..N-1 contiguous")
    if not np.all(np.diff(sdf) > 0):
        problems.append(
            "sdf_index is not strictly increasing. The dataset is ordered by SDF "
            "index, so labels joined on the shuffled CSV row order are misaligned.")
    print(f"    dataset_pos 0..{N-1} contiguous: {np.array_equal(dp, np.arange(N))}")
    print(f"    sdf_index strictly increasing: {bool(np.all(np.diff(sdf) > 0))}"
          f"  range {int(sdf.min())}..{int(sdf.max())}")
    print(f"    n_atoms range {int(na.min())}..{int(na.max())}  mean {na.mean():.1f}")
    if na.max() > man["n_max"] or na.min() < 1:
        problems.append(f"n_atoms outside 1..{man['n_max']}")

    for name, (lo, hi) in LABEL_BOUNDS.items():
        if name not in z:
            problems.append(f"labels.npz missing {name}"); continue
        v = z[name]
        bad = int(((v < lo) | (v > hi)).sum())
        print(f"    {name:16s} min {v.min():10.4f}  max {v.max():10.4f}  "
              f"mean {v.mean():10.4f}  outside [{lo}, {hi}]: {bad}")
        if bad > 0.01 * len(v):
            problems.append(f"{name}: {bad}/{len(v)} values outside the published QM9 range")
        if np.allclose(v.std(), 0):
            problems.append(f"{name}: zero variance, the column is constant")
    return problems, z


def check_probe(store, man, z, layers=None, lam=1e-2):
    print(f"\n[3] PROBE  held-out R-squared, ridge lam={lam}")
    na = z["n_atoms"].astype(np.float64)
    alpha = z["polarisability"].astype(np.float64)
    ts = man["timesteps"]
    L = man["n_layers"]
    layers = layers if layers else list(range(L))

    # rung 2 first: it needs no activations and it bounds the rest.
    r2_size = ridge_r2(na.reshape(-1, 1), alpha, lam=1e-8)
    print(f"\n    rung 2  polarisability from atom count alone: R2 = {r2_size:.4f}")
    print(f"            (pure chemistry; this is the ceiling rung 3 competes with)")

    best = {"atoms": (-9, None), "alpha": (-9, None)}
    print(f"\n    rung 1 and 3, per (layer, timestep):")
    print(f"    {'layer':>5} {'t':>5} {'R2 atom count':>14} {'R2 polarisability':>18}")
    for k in layers:
        a = np.load(os.path.join(store, f"pooled_L{k:02d}.npy"), mmap_mode="r")
        for ti, t in enumerate(ts):
            A = np.asarray(a[ti], dtype=np.float32)
            r2_n = ridge_r2(A, na, lam=lam)
            r2_a = ridge_r2(A, alpha, lam=lam)
            if r2_n > best["atoms"][0]:
                best["atoms"] = (r2_n, (k, t))
            if r2_a > best["alpha"][0]:
                best["alpha"] = (r2_a, (k, t))
            print(f"    {k:>5} {t:>5} {r2_n:>14.4f} {r2_a:>18.4f}")

    print(f"\n    best atom-count R2    {best['atoms'][0]:.4f} at layer/t {best['atoms'][1]}")
    print(f"    best polarisability R2 {best['alpha'][0]:.4f} at layer/t {best['alpha'][1]}")

    verdict, broken = [], False
    if best["atoms"][0] < 0.5:
        broken = True
        verdict.append(
            "CAPTURE LOOKS BROKEN: atom count is not decodable from the activations. "
            "That quantity is trivially present in any graph representation, so a low "
            "value means the store, not the science, is wrong. Check the pooling masks, "
            "the write offsets, and the label alignment before reading anything else.")
    elif r2_size < 0.5:
        broken = True
        verdict.append(
            "LABELS LOOK WRONG: polarisability is not predictable from atom count, "
            "which is basic chemistry. The label join is the first suspect.")
    elif best["alpha"][0] < 0.2:
        verdict.append(
            "Capture looks sound (atom count decodes, labels sane) but polarisability "
            "is not linearly decodable. If this survives the selectivity and "
            "random-projection controls, it is a finding about the model, not a bug.")
    else:
        verdict.append(
            f"Capture looks sound and polarisability is decodable "
            f"(R2 {best['alpha'][0]:.3f} vs the {r2_size:.3f} size ceiling). "
            f"Proceed to the full probe sweep.")
    return verdict, best, r2_size, broken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="restrict the probe to these layers (default: all)")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    store = os.path.abspath(args.store)
    with open(os.path.join(store, "manifest.json")) as f:
        man = json.load(f)
    print(f"store   {store}")
    print(f"mode    {man['mode']}  split={man.get('split')}  "
          f"N={man['n_molecules']}  layers={man['n_layers']}  timesteps={man['timesteps']}")
    print(f"rate    {man.get('molecules_per_gpu_hour', float('nan')):,.0f} mol/GPU-hour")
    print(f"ckpt    {man.get('checkpoint')}")
    print(f"commit  {man.get('git_commit')}  torch {man.get('torch')}")

    problems = check_structure(store, man)
    p2, z = check_alignment(store, man)
    problems += p2

    verdict, broken = [], False
    if z is not None and not args.skip_probe:
        verdict, _, _, broken = check_probe(store, man, z, layers=args.layers, lam=args.lam)

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} STRUCTURAL PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No structural problems found.")
    for v in verdict:
        print(f"\n{v}")
    if broken:
        print("\nVERDICT: do not build on this store. Fix it, re-capture, re-verify.")
    elif not problems:
        print("\nVERDICT: store is usable.")
    print("=" * 70)
    raise SystemExit(1 if (problems or broken) else 0)


if __name__ == "__main__":
    main()
