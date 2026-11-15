"""scripts/build_tolerances.py -- the tolerance table, from the baselines.

    python3.11 -u scripts/build_tolerances.py [--device cuda] [--fcd-reps 200]

Stage 4 of results/tolerance_protocol.md. Reads the ten unsteered baseline
runs and turns them into the table every steered arm is judged against.
Writes results/tolerance_table.md and results/tolerance_table.csv.

THE ARITHMETIC, which is the whole content of this script.

For each metric, let mu be the mean across the S = 10 baseline runs and s the
sample standard deviation across them,

    s = sqrt( (1/(S-1)) * sum_i (x_i - mu)^2 )

with S - 1 = 9 in the denominator because mu was estimated from the same
data. For a proportion there is also an analytic sampling error,

    sigma_bin = sqrt( p (1 - p) / n )

and the scale actually used is

    sigma_eff = max(s, sigma_bin)

so a baseline that happens to scatter less than sampling theory allows cannot
produce an absurdly tight band. Comparing s against sigma_bin is itself a
test of whether the generator is stationary across seeds: s much larger than
sigma_bin means run-to-run variation beyond sampling, and s much smaller
means the ten runs are suspiciously alike.

The band for a single NEW run is

    mu  +-  k * sigma_eff ,   k = t(0.975, S-1) * sqrt(1 + 1/S)
                                = 2.262 * 1.0488 = 2.372

The t quantile is on S - 1 = 9 degrees of freedom because s is estimated.
The sqrt(1 + 1/S) widens the interval from "where the mean lies" to "where
one new observation lies": a new run has its own variance, plus the
uncertainty in mu, and those add as 1 + 1/S.

FCD gets no band of this shape, because it is a distance with no baseline
mean to sit around. Its null is the distribution of FCD between two
INDEPENDENT unsteered runs at matched m, which is 45 pairs from ten runs,
each subsampled R times, and the tolerance is that pooled distribution's
upper 97.5th percentile.

The 0.5 standardised-unit denominators are the per-molecule standard
deviations of each property endpoint, POOLED across the ten baselines, so the
denominator never touches the sweep.
"""
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from midi import fcd_eval as fe, matched_n as mn, metrics_eval as me

SEEDS = tuple(range(90001, 90011))
BASE = os.path.join(REPO, "results", "baseline")
TRAIN_PICKLE = os.path.join(REPO, "data", "qm9", "qm9_pyg", "processed",
                            "train_smiles.pickle")
K = 2.372                      # fixed by the protocol; not to be retuned
M = mn.M_DEFAULT               # 250
R = mn.R_DEFAULT               # 200

# (key in the metrics dict, label, is it a proportion out of n?)
COUNT_METRICS = [
    ("validity_midi", "validity (MiDi, largest fragment)", True),
    ("validity_strict", "validity (strict, whole molecule)", True),
    ("connectivity", "connectivity (single fragment)", True),
    ("mol_stable", "molecule stability (MiDi, integer)", True),
    ("atom_stable", "atom stability (MiDi, integer)", True),
    ("mol_stable_float", "molecule stability (float-corrected)", True),
    ("atom_stable_float", "atom stability (float-corrected)", True),
    ("aromatic_bond_rate", "aromatic bond rate", True),
    ("aromatic_mol_rate", "aromatic molecule rate", True),
]
ENDPOINTS = ("molmr", "mr_per_heavy", "heavy")


def load_train():
    import pickle
    with open(TRAIN_PICKLE, "rb") as f:
        return set(pickle.load(f))


def endpoint_columns(recs):
    """Per-molecule property values, for the 0.5-unit denominators.

    Computed on the MiDi-valid largest-fragment SMILES, which is the same set
    the other metrics use. On this generator that is the same set as the
    strict definition anyway, since connectivity is 98.8% to 100%, so these
    denominators are comparable with the effects measured in
    sweep_findings.md.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen

    out = {k: [] for k in ENDPOINTS}
    for r in recs:
        m = me.to_molecule(r)
        if m is None:
            continue
        smi, _ = me.validity_midi(m)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        heavy = float(mol.GetNumHeavyAtoms())
        if heavy <= 0:
            continue
        molmr = float(Crippen.MolMR(mol))
        out["molmr"].append(molmr)
        out["heavy"].append(heavy)
        out["mr_per_heavy"].append(molmr / heavy)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def band(mu, s, sigma_bin):
    sigma_eff = max(s, sigma_bin) if np.isfinite(sigma_bin) else s
    return mu - K * sigma_eff, mu + K * sigma_eff, sigma_eff


def main(device="cpu", fcd_reps=R):
    from analyse_pilot import load_arm

    present = [s for s in SEEDS
               if os.path.exists(os.path.join(BASE, f"seed{s}", "control",
                                              "chunk_00000.pt"))]
    missing = [s for s in SEEDS if s not in present]
    if missing:
        print(f"WARNING missing baseline seeds {missing}; "
              f"building from {len(present)} runs. The protocol fixes k at "
              f"{K} for S = 10, so a table from fewer runs is PROVISIONAL.")
    if len(present) < 2:
        raise SystemExit("need at least two baseline runs")

    train = load_train()
    print(f"training partition: {len(train):,} SMILES")
    print(f"baselines present: {len(present)} of {len(SEEDS)}\n")

    runs, acts, ep_pool = [], {}, {k: [] for k in ENDPOINTS}
    for s in present:
        t0 = time.time()
        recs = list(load_arm(os.path.join(BASE, f"seed{s}", "control")).values())
        mt = me.set_metrics(recs, train_smiles=train)
        uq = mn.uniqueness_matched(mt["smiles"], m=M, R=R, arm=f"base{s}")
        nv = mn.novelty_matched(mt["smiles"], train, m=M, R=R, arm=f"base{s}")
        # FCD is embedded on the DISTINCT canonical H-free SMILES, per the
        # protocol's input-form decision. Duplicates would weight a repeated
        # molecule twice in the Gaussian fit.
        canon = fe.canon_for_chemnet(mt["unique_smiles"])
        acts[s] = fe.activations(canon, device=device, already_canonical=True)
        cols = endpoint_columns(recs)
        for k in ENDPOINTS:
            ep_pool[k].append(cols[k])
        runs.append({"seed": s, "metrics": mt, "uniq": uq, "nov": nv,
                     "n_act": len(acts[s])})
        print(f"  seed {s}: n={mt['n_generated']}, "
              f"valid {100 * mt['validity_midi']['p']:.1f}%, "
              f"act {len(acts[s])}, {time.time() - t0:.0f}s", flush=True)

    # ---------------- count metrics -------------------------------------
    rows = []
    print()
    print(f"{'metric':<40}{'mu':>9}{'s':>9}{'sig_bin':>9}{'sig_eff':>9}"
          f"{'lo':>9}{'hi':>9}{'s/sbin':>8}")
    for key, label, _ in COUNT_METRICS:
        vals = np.array([r["metrics"][key]["p"] for r in runs], float)
        ks = sum(r["metrics"][key]["k"] for r in runs)
        ns = sum(r["metrics"][key]["n"] for r in runs)
        mu = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        n_per = ns / len(runs)
        sb = float(np.sqrt(mu * (1 - mu) / n_per)) if 0 < mu < 1 else 0.0
        lo, hi, se = band(mu, s, sb)
        ratio = (s / sb) if sb > 0 else float("nan")
        print(f"{label:<40}{mu:>9.5f}{s:>9.5f}{sb:>9.5f}{se:>9.5f}"
              f"{lo:>9.5f}{hi:>9.5f}{ratio:>8.2f}")
        rows.append({"metric": key, "label": label, "mu": mu, "sd_runs": s,
                     "sigma_binom": sb, "sigma_eff": se, "lo": lo, "hi": hi,
                     "sd_over_binom": ratio, "basis": "proportion, full n"})

    for key, label in (("uniq", "uniqueness (matched m=250)"),
                       ("nov", "novelty (MiDi as-is, matched m=250)")):
        vals = np.array([r[key]["mean"] for r in runs], float)
        mu = float(np.nanmean(vals))
        s = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        sb = float(np.sqrt(mu * (1 - mu) / M)) if 0 < mu < 1 else 0.0
        lo, hi, se = band(mu, s, sb)
        ratio = (s / sb) if sb > 0 else float("nan")
        print(f"{label:<40}{mu:>9.5f}{s:>9.5f}{sb:>9.5f}{se:>9.5f}"
              f"{lo:>9.5f}{hi:>9.5f}{ratio:>8.2f}")
        rows.append({"metric": key, "label": label, "mu": mu, "sd_runs": s,
                     "sigma_binom": sb, "sigma_eff": se, "lo": lo, "hi": hi,
                     "sd_over_binom": ratio, "basis": f"matched m={M}"})

    # ---------------- the FCD null --------------------------------------
    print()
    usable = [s for s in present if len(acts[s]) >= M]
    pairs = [(a, b) for i, a in enumerate(usable) for b in usable[i + 1:]]
    print(f"FCD null: {len(usable)} runs with >= {M} canonical SMILES, "
          f"{len(pairs)} pairs x R={fcd_reps} = {len(pairs) * fcd_reps:,} "
          f"evaluations")
    t0, null = time.time(), []
    for a, b in pairs:
        out = fe.fcd_matched(acts[a], acts[b], m=M, R=fcd_reps,
                             seed=(a * 100000 + b))
        if out["evaluable"]:
            null.append(out["mean"])
            print(f"  {a} vs {b}: {out['mean']:.4f} "
                  f"(sd {out['sd_subsample']:.4f})", flush=True)
    null = np.array(null, float)
    if null.size:
        q = float(np.percentile(null, 97.5))
        print(f"  pooled over {null.size} pairs: mean {null.mean():.4f}, "
              f"sd {null.std(ddof=1):.4f}")
        print(f"  TOLERANCE, upper 97.5th percentile: {q:.4f}")
        print(f"  elapsed {time.time() - t0:.0f}s")
        rows.append({"metric": "fcd", "label": f"FCD vs unsteered (m={M})",
                     "mu": float(null.mean()), "sd_runs": float(
                         null.std(ddof=1)) if null.size > 1 else 0.0,
                     "sigma_binom": float("nan"), "sigma_eff": float("nan"),
                     "lo": float("nan"), "hi": q,
                     "sd_over_binom": float("nan"),
                     "basis": f"null over {len(pairs)} independent pairs, "
                              f"upper 97.5 pct"})

    # ---------------- the 0.5-unit denominators -------------------------
    print()
    print("0.5 standardised-unit denominators, pooled per-molecule sd")
    ep_rows = []
    for k in ENDPOINTS:
        allv = np.concatenate(ep_pool[k])
        allv = allv[np.isfinite(allv)]
        sd = float(allv.std(ddof=1))
        print(f"  {k:<14} n={allv.size:<6} mean {allv.mean():>9.4f}  "
              f"sd {sd:>8.4f}  0.5 sd = {0.5 * sd:>8.4f}")
        ep_rows.append({"endpoint": k, "n": int(allv.size),
                        "mean": float(allv.mean()), "sd": sd,
                        "half_sd": 0.5 * sd})

    # ---------------- write ---------------------------------------------
    out_csv = os.path.join(REPO, "results", "tolerance_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out_json = os.path.join(REPO, "results", "tolerance_table.json")
    with open(out_json, "w") as f:
        json.dump({"k": K, "m": M, "R": R, "seeds_present": present,
                   "seeds_missing": missing, "provisional": bool(missing),
                   "bands": rows, "endpoints": ep_rows,
                   "versions": me.versions(os.path.join(
                       REPO, "midi", "checkpoints",
                       "checkpoint_qm9_h_adaptive.ckpt"))}, f, indent=2)
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    dev = "cpu"
    reps = R
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--device" and i + 1 < len(args):
            dev = args[i + 1]
        if a == "--fcd-reps" and i + 1 < len(args):
            reps = int(args[i + 1])
    main(dev, reps)
