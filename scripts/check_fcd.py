"""scripts/check_fcd.py -- how biased is FCD at the sizes we can afford?

    python3.11 -u scripts/check_fcd.py [device]

THE QUANTITY. ChemNet maps each SMILES to an activation vector in R^d with
d = 512. Fit a Gaussian to each set, (mu_1, Sigma_1) and (mu_2, Sigma_2),
where mu is the mean activation and Sigma the covariance. The Frechet
distance between two Gaussians is

    FCD = ||mu_1 - mu_2||^2 + Tr( Sigma_1 + Sigma_2 - 2 (Sigma_1 Sigma_2)^1/2 )

The first term is how far apart the centres are. The second compares the
shapes: it is zero when Sigma_1 = Sigma_2 and positive otherwise.

WHY SIZE MATTERS SO MUCH HERE. Sigma is estimated from m samples in d = 512
dimensions. A sample covariance from m points has rank at most m - 1, so at
m = 250 the estimate is RANK DEFICIENT: 249 of 512 directions carry
variance and 263 are estimated as exactly zero. The trace term is then
systematically inflated, because each set's covariance looks artificially
flat in directions the other set happens to span. The bias is upward and it
does not vanish with averaging.

THE MEASUREMENT. Split the training partition in two halves. Both halves are
drawn from the SAME distribution, so the true FCD between them is zero and
whatever comes out is pure bias. Sweeping m therefore traces the bias curve
directly, with no modelling assumptions.

WHAT WOULD RESCUE m = 250 ANYWAY. The protocol does not compare an absolute
FCD against zero. It compares a steered-against-unsteered FCD at m = 250
with the distribution of unsteered-against-unsteered FCD at the SAME m. The
bias is common to both and cancels in that comparison. So a large bias is
survivable; what is not survivable is a bias whose run-to-run spread swamps
the effect being looked for. This script reports both, so the power of the
test is known before the baselines are spent.

INPUT FORM. ChemNet was trained on ordinary canonical SMILES, not on the
explicit-hydrogen strings this pipeline produces. Both forms are measured
below, because feeding "[H]C([H])([H])[H]" to a tokeniser built for "C"
would give an embedding of the string rather than of the molecule.
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from rdkit import Chem, RDLogger

from midi import metrics_eval as me

RDLogger.DisableLog("rdApp.*")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PICKLE = os.path.join(REPO, "data", "qm9", "qm9_pyg", "processed",
                            "train_smiles.pickle")
SIZES = (50, 100, 250, 500, 1000, 2500)
SEED = 20261115


def canon_noh(s):
    m = Chem.MolFromSmiles(s, sanitize=True)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(m))
    except Exception:
        return None


def main(device="cpu"):
    from fcd_torch import FCD
    fcd = FCD(device=device, n_jobs=1)

    with open(TRAIN_PICKLE, "rb") as f:
        train_raw = sorted(set(pickle.load(f)))
    train = [c for c in (canon_noh(s) for s in train_raw) if c]
    print(f"training partition: {len(train_raw):,} stored, "
          f"{len(train):,} re-canonicalised without H")

    g = np.random.default_rng(SEED)
    idx = g.permutation(len(train))
    half = len(idx) // 2
    A = [train[i] for i in idx[:half]]
    B = [train[i] for i in idx[half:]]
    print(f"two same-distribution halves of {len(A):,} and {len(B):,}")
    print(f"device {device}, ChemNet d = 512\n")

    print("BIAS CURVE. Both halves come from the same distribution, so the")
    print("true FCD is 0 and every value below is bias.")
    print(f"{'m':>7}{'rank of Sigma':>15}{'FCD mean':>11}{'sd':>9}"
          f"{'reps':>6}{'sec':>7}")
    rows = []
    for m in SIZES:
        reps = 5 if m <= 500 else 3
        vals, t0 = [], time.time()
        for r in range(reps):
            ga = np.random.default_rng(SEED + 1000 * m + r)
            sa = [A[i] for i in ga.choice(len(A), m, replace=False)]
            sb = [B[i] for i in ga.choice(len(B), m, replace=False)]
            vals.append(float(fcd(sa, sb)))
        v = np.array(vals)
        rows.append((m, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0))
        print(f"{m:>7}{min(m - 1, 512):>15}{v.mean():>11.3f}"
              f"{(v.std(ddof=1) if len(v) > 1 else 0.0):>9.3f}"
              f"{reps:>6}{time.time() - t0:>7.1f}")

    print()
    print("POWER AT m = 250. The protocol compares a steered FCD against the")
    print("spread of unsteered-vs-unsteered FCD at the same m, so the bias")
    print("cancels and only the spread limits the test.")
    m250 = [r for r in rows if r[0] == 250][0]
    print(f"  bias at m=250            {m250[1]:.3f}")
    print(f"  run-to-run sd at m=250   {m250[2]:.3f}")
    if m250[1] > 0:
        print(f"  sd as a share of bias    {100 * m250[2] / m250[1]:.1f}%")
    big = rows[-1]
    print(f"  for contrast, m={big[0]}      {big[1]:.3f}")
    print(f"  so shrinking m from {big[0]} to 250 inflates FCD by "
          f"{m250[1] - big[1]:+.3f}")

    print()
    print("INPUT FORM, on a real generated set at m = 250.")
    arm = os.path.join(REPO, "results", "sweep", "control")
    if os.path.isdir(arm):
        from analyse_pilot import load_arm
        out = me.set_metrics(list(load_arm(arm).values()))
        gen_raw = out["unique_smiles"]
        gen_noh = [c for c in (canon_noh(s) for s in gen_raw) if c]
        g2 = np.random.default_rng(SEED + 7)
        tr = [train[i] for i in g2.choice(len(train), 250, replace=False)]
        n = min(250, len(gen_noh), len(gen_raw))
        ir = g2.choice(len(gen_raw), n, replace=False)
        inh = g2.choice(len(gen_noh), n, replace=False)
        v_raw = float(fcd([gen_raw[i] for i in ir], tr))
        v_noh = float(fcd([gen_noh[i] for i in inh], tr))
        print(f"  generated as produced (explicit H) vs train   {v_raw:.3f}")
        print(f"  generated re-canonicalised without H vs train {v_noh:.3f}")
        print(f"  difference                                    "
              f"{v_raw - v_noh:+.3f}")
        print("  The training side is H-free in both rows, so the gap is the")
        print("  cost of feeding explicit-H strings to a tokeniser built for")
        print("  ordinary canonical SMILES. Whichever is smaller is the form")
        print("  that is actually embedding the molecule.")
    else:
        print("  (results/sweep/control not present, skipped)")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
