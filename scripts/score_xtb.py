"""GFN2/PTB-xTB polarisability for molecules already generated, scored post hoc.

Why this exists
---------------
Every property conclusion in this study currently rests on Crippen molar
refractivity (MolMR), an RDKit GRAPH DESCRIPTOR computed from the generated
graph. It is a proxy for polarisability through the Lorentz-Lorenz relation, it
is extensive, and it correlates with molecule size -- the exact confound that
ate the first analysis. A reader will ask whether a physical estimate agrees.
This script produces one, from the saved chunks, without regenerating anything:
each chunk_*.pt already stores atom_types, charges, bond_types and positions.

WHICH xtb NUMBER (this matters, and the obvious choice is wrong)
----------------------------------------------------------------
xtb reports two quantities called polarisability:

  Mol. alpha(0) /au   `--gfn 2 --verbose`, dispersion section. The D4 model's
                      sum of CHARGE-SCALED ATOMIC polarisabilities. Verified on
                      water: O 6.657 + H 1.386 + H 1.386 = 9.429, exactly the
                      printed total. ADDITIVE BY CONSTRUCTION, so it is a
                      composition-and-charge sum. Using it as the independent
                      physical check would REBUILD the size confound the check
                      exists to test. This script does not use it.

  --ptb --alpha       a numerical polarisability TENSOR from the PTB response,
                      with a total isotropic dipole polarisability. A genuine
                      field-response property, and the only one here that reads
                      the 3-D coordinates at all.

Units are a.u. = Bohr^3, which is what QM9 reports alpha in, so no conversion.

Validated against the QM9 DFT labels on 395 labelled test molecules at their
DFT geometries (all 395 converged):

    readout              Pearson   Spearman   R2(affine)   MAE
    xTB PTB alpha         0.9824    0.9783      0.9651     1.25
    descriptor surrogate  0.9740    0.9731      0.9486     1.29
    Crippen MolMR         0.9519    0.9489      0.9061     1.91
    heavy-atom count      0.5585    0.4634      0.3119     5.70

So PTB alpha is the best of the four AND the only physically independent one.
Mean 78.13 against a DFT mean of 75.25, a ratio of 1.038, so it needs no
rescaling to be read as Bohr^3.

GEOMETRY: WHY THIS SCORES BOTH WAYS
-----------------------------------
"Score as-is" is the honest match to what the model produced, and "relax first"
measures the nearest real molecule. The two were treated as a free choice. They
are not, because PTB alpha is acutely and ASYMMETRICALLY sensitive to
coordinate error. Measured by jittering DFT geometries of 120 labelled
molecules and rescoring (paired, same molecules):

    geometry                 Pearson vs DFT   paired shift vs as-is
    as-is (DFT geometry)         0.9791        --
    xtb-relaxed                  0.9831        -1.16 +/- 0.52
    jitter 0.05 A                0.9534        +1.15 +/- 1.98
    jitter 0.15 A                0.7084       +15.95 +/- 9.64 (max 53.4)
    jitter 0.15 A then relax     0.9830        -1.15 +/- 0.53

Coordinate error inflates alpha UPWARD -- 0.15 A of random noise adds ~16
Bohr^3, a 21% rise, and collapses the correlation with truth from 0.98 to 0.71.
Relaxation repairs it completely: jitter-then-relax lands within 0.01 Bohr^3 of
relaxing the clean geometry.

The consequence for this study is not academic. Steering degrades geometry, and
it degrades it most in the arms with the largest |c|. Scoring as-is ALONE would
therefore produce a positive alpha shift concentrated in the most heavily
steered arms, with the same sign as the effect being claimed, and that shift
would be a geometry-quality artefact rather than a property change. So both are
recorded per molecule and the methods section states both:

    alpha_asis   the model's own output, geometry included, warts and all
    alpha_opt    the nearest physically reasonable molecule
    rmsd_opt     how far the relaxation had to move the atoms (Kabsch-aligned)

`rmsd_opt` is the diagnostic that ties them together: it is a direct,
per-molecule measure of how unphysical the generated coordinates were. If
alpha_asis - alpha_opt tracks rmsd_opt within an arm, the as-is number is
reporting geometry quality, not polarisability.

Output and resumability
-----------------------
One CSV per input chunk, mirroring the input tree under --out, written to a
temporary name and renamed, so an interrupted run loses at most one chunk and a
rerun skips every chunk already finished. Same convention as the rest of the
pipeline.

Usage
-----
    python scripts/score_xtb.py --runs results/scan results/pilot \
        --out results/xtb --mode both --workers 4

    python scripts/score_xtb.py --out results/xtb --summary
"""

import argparse
import csv
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATOM_DECODER = ("H", "C", "N", "O", "F")
FIELDS = ("arm", "chunk", "mol_id", "n_atoms", "n_heavy", "net_charge",
          "alpha_asis", "alpha_asis_ang3", "status_asis",
          "alpha_opt", "status_opt", "rmsd_opt", "uhf", "ms")


# ------------------------------------------------------------------ geometry

def kabsch_rmsd(P, Q):
    """RMSD between two coordinate sets after optimal translation+rotation.

    xtb may translate or reorient a structure during optimisation, so a raw
    coordinate difference would report motion that is not a change of shape.
    Kabsch removes exactly that and leaves the deformation.
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if P.shape != Q.shape or len(P) == 0:
        return float("nan")
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    Pr = P @ (V @ D @ Wt)
    return float(np.sqrt(((Pr - Q) ** 2).sum(1).mean()))


# -------------------------------------------------------------------- worker

_XTB = {"dist": None}


def _init(dist):
    _XTB["dist"] = dist


def score_one(job):
    """One molecule -> a dict of CSV fields. Never raises."""
    from midi.xtb_runner import ptb_alpha, write_xyz, xtb_env, uhf_for
    import subprocess, tempfile, shutil, re
    arm, chunk, mol_id, syms, coords, chrg, mode = job
    n_heavy = sum(1 for s in syms if s != "H")
    uhf = uhf_for(syms, chrg)
    row = {"arm": arm, "chunk": chunk, "mol_id": mol_id,
           "n_atoms": len(syms), "n_heavy": n_heavy, "net_charge": chrg,
           "alpha_asis": "", "alpha_asis_ang3": "", "status_asis": "skipped",
           "alpha_opt": "", "status_opt": "skipped", "rmsd_opt": "",
           "uhf": uhf, "ms": 0}
    t0 = time.time()
    dist = _XTB["dist"]

    if mode in ("asis", "both"):
        au, ang, st = ptb_alpha(syms, coords, dist=dist, charge=chrg, uhf=uhf)
        row["status_asis"] = st
        if st == "ok":
            row["alpha_asis"], row["alpha_asis_ang3"] = au, ang

    if mode in ("opt", "both"):
        # Relax with GFN2, keep the relaxed coordinates so the displacement can
        # be measured, then score the relaxed structure with PTB.
        env, exe = xtb_env(dist)
        d = tempfile.mkdtemp(prefix="xtbopt_")
        try:
            write_xyz(os.path.join(d, "m.xyz"), syms, coords)
            r = subprocess.run([exe, "m.xyz", "--gfn", "2", "--opt", "tight",
                                "--norestart", "--chrg", str(chrg),
                                "--uhf", str(uhf)],
                               cwd=d, env=env, capture_output=True, text=True,
                               timeout=600)
            relaxed = os.path.join(d, "xtbopt.xyz")
            if r.returncode != 0 or not os.path.exists(relaxed):
                row["status_opt"] = "opt_failed"
            else:
                xyz = [ln.split() for ln in
                       open(relaxed).read().splitlines()[2:2 + len(syms)]]
                new = [[float(c) for c in ln[1:4]] for ln in xyz if len(ln) >= 4]
                row["rmsd_opt"] = round(kabsch_rmsd(coords, new), 5)
                au, ang, st = ptb_alpha(syms, new, dist=dist, charge=chrg, uhf=uhf)
                row["status_opt"] = st
                if st == "ok":
                    row["alpha_opt"] = au
        except subprocess.TimeoutExpired:
            row["status_opt"] = "timeout"
        except Exception as e:
            row["status_opt"] = f"error:{type(e).__name__}"
        finally:
            shutil.rmtree(d, ignore_errors=True)

    row["ms"] = int(1000 * (time.time() - t0))
    return row


# --------------------------------------------------------------------- chunks

def jobs_from_chunk(path, arm, mode, decoder, limit=None):
    import torch
    o = torch.load(path, weights_only=False)
    chunk = os.path.splitext(os.path.basename(path))[0]
    out = []
    n = len(o["mol_ids"])
    for i in range(n if limit is None else min(n, limit)):
        at = np.asarray(o["atom_types"][i]).reshape(-1)
        ch = np.asarray(o["charges"][i]).reshape(-1)
        po = np.asarray(o["positions"][i], dtype=float)
        keep = at >= 0
        syms = [decoder[int(a)] for a in at[keep]]
        out.append((arm, chunk, int(o["mol_ids"][i]), syms,
                    po[keep].tolist(), int(ch[keep].sum()), mode))
    return out


def find_chunks(runs):
    found = []
    for r in runs:
        for p in sorted(glob.glob(os.path.join(r, "**", "chunk_*.pt"),
                                  recursive=True)):
            arm = os.path.relpath(os.path.dirname(p), os.path.dirname(r.rstrip("/")))
            found.append((p, arm.replace(os.sep, "/")))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("asis", "opt", "both"), default="both")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--xtb", default=os.environ.get("XTB_DIST"))
    ap.add_argument("--limit-per-chunk", type=int, default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    if a.summary:
        return summarise(a.out)

    chunks = find_chunks(a.runs)
    if a.max_chunks:
        chunks = chunks[:a.max_chunks]
    todo = [(p, arm) for p, arm in chunks
            if not os.path.exists(os.path.join(a.out, arm,
                                               os.path.basename(p).replace(".pt", ".csv")))]
    print(f"{len(chunks)} chunks found, {len(todo)} still to do, "
          f"mode={a.mode}, workers={a.workers}", flush=True)

    t_start = time.time()
    done_mols = 0
    with ProcessPoolExecutor(max_workers=a.workers,
                             initializer=_init, initargs=(a.xtb,)) as ex:
        for k, (p, arm) in enumerate(todo, 1):
            jobs = jobs_from_chunk(p, arm, a.mode, ATOM_DECODER,
                                   a.limit_per_chunk)
            rows = list(ex.map(score_one, jobs, chunksize=1))
            dst_dir = os.path.join(a.out, arm)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(p).replace(".pt", ".csv"))
            tmp = dst + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, dst)
            done_mols += len(rows)
            el = time.time() - t_start
            rate = done_mols / max(el, 1e-9)
            left = sum(1 for _ in todo[k:])
            print(f"[{k}/{len(todo)}] {arm}/{os.path.basename(p)} "
                  f"{len(rows)} mols  {el/60:.1f} min elapsed  "
                  f"{rate:.1f} mol/s  ~{left * len(rows) / max(rate,1e-9)/60:.0f} min left",
                  flush=True)
    print(f"done: {done_mols} molecules in {(time.time()-t_start)/60:.1f} min",
          flush=True)


def summarise(out):
    import collections
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "**", "chunk_*.csv"),
                              recursive=True)):
        with open(p) as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        print("no CSVs yet")
        return
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    g = lambda rs, k: np.array([float(r[k]) for r in rs if r[k] not in ("", None)])
    print(f"{'arm':<34}{'n':>6}{'ok_asis':>8}{'ok_opt':>7}"
          f"{'a_asis':>9}{'a_opt':>9}{'a_as-a_op':>10}{'rmsd':>7}")
    for arm in sorted(by):
        rs = by[arm]
        aa, ao = g(rs, "alpha_asis"), g(rs, "alpha_opt")
        rm = g(rs, "rmsd_opt")
        paired = [(float(r["alpha_asis"]), float(r["alpha_opt"])) for r in rs
                  if r["alpha_asis"] and r["alpha_opt"]]
        dif = (np.mean([x - y for x, y in paired]) if paired else float("nan"))
        print(f"{arm:<34}{len(rs):>6}{len(aa):>8}{len(ao):>7}"
              f"{(aa.mean() if len(aa) else float('nan')):>9.2f}"
              f"{(ao.mean() if len(ao) else float('nan')):>9.2f}"
              f"{dif:>10.2f}{(rm.mean() if len(rm) else float('nan')):>7.3f}")
    print()
    print("a_asis   mean PTB alpha on the generated coordinates, Bohr^3")
    print("a_opt    mean PTB alpha after GFN2 relaxation, Bohr^3")
    print("a_as-a_op  paired as-is minus relaxed. Coordinate error biases this")
    print("         UPWARD, so a large value is a geometry-quality signal.")
    print("rmsd     mean Kabsch RMSD the relaxation moved the atoms, Angstrom.")


if __name__ == "__main__":
    main()
