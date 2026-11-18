"""scripts/evaluate_arms.py -- the central table: shift beside degradation.

    python3.11 -u scripts/evaluate_arms.py [--device cuda] [--sweep DIR]

Stage 6 of results/tolerance_protocol.md. Applies every metric to every arm,
judges admissibility against results/tolerance_table.json, and joins the
result to the property shift. Writes results/central_table.md and .csv.

THE RULE BEING APPLIED, which is a conjunction and not a score.

An arm is ADMISSIBLE only if every degradation metric lies inside its band

    mu +- k * sigma_eff ,   k = 2.372

and its FCD lies below the null's upper 97.5th percentile. One failure is
enough. There is no weighting and no partial credit, because the question
the gate answers is "did the generator still work", and a generator that
lost a quarter of its validity did not.

An arm that cannot be evaluated on uniqueness, novelty or FCD under the
m = 250 rule is inadmissible by default: the reason it cannot be evaluated
is that fewer than half its molecules are valid.

A property shift is REPORTABLE only from an admissible arm, and only at
0.5 standardised units or more, standardised by the per-molecule sd pooled
across the ten unsteered baselines. Both thresholds were fixed before any
baseline existed, in commit bf3dec5 at 08:36 on 2026-11-15.

So the table has three columns of verdict, in order: did it degrade, was the
shift big enough, and only then is there anything to report. Shift alone is
never reported, which is the whole point of the item this stage comes from.
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from midi import fcd_eval as fe, matched_n as mn, metrics_eval as me

TABLE = os.path.join(REPO, "results", "tolerance_table.json")
TRAIN_PICKLE = os.path.join(REPO, "data", "qm9", "qm9_pyg", "processed",
                            "train_smiles.pickle")
BASE = os.path.join(REPO, "results", "baseline")
REF_SEED = 90001          # the unsteered reference each arm's FCD is against

# The gates, in the order applied. These are the band keys written by
# build_tolerances.py, which calls uniqueness "uniq" and novelty "nov".
GATES = ["validity_midi", "connectivity", "mol_stable", "atom_stable",
         "uniq", "nov", "fcd"]
# Reported beside the gates but NOT gated: diagnostics that explain movement.
# aromatic_smiles_rate is here rather than in GATES deliberately. It is the
# rate that biases novelty, so it has to be visible whenever novelty moves,
# but it is a property of the chemistry rather than a failure of it and
# gating on it would reject an arm for making more rings.
DIAGNOSTICS = ["aromatic_smiles_rate", "validity_strict"]

K = 2.372          # fixed by the protocol in bf3dec5; not to be retuned


def load_bands():
    if not os.path.exists(TABLE):
        raise SystemExit(f"{TABLE} not found; run build_tolerances.py first")
    with open(TABLE) as f:
        t = json.load(f)
    bands = {b["metric"]: b for b in t["bands"]}
    eps = {e["endpoint"]: e for e in t["endpoints"]}
    return t, bands, eps


def inside(band, value):
    """Is value inside this band? FCD is one-sided, the rest two-sided."""
    if value is None or not np.isfinite(value):
        return None
    if band["metric"] == "fcd":
        return bool(value <= band["hi"])
    return bool(band["lo"] <= value <= band["hi"])


def tolerance_distance(band, value):
    """(value - baseline mean) / (k * sigma_eff).

    The arm's distance from the unsteered baseline in units of its OWN
    tolerance, which is what makes seven metrics on different scales
    comparable on one axis: |T| <= 1 is inside tolerance for all of them.
    Sign says which way it moved.

    FCD has no baseline mean to sit around, so its band is one-sided and
    `hi` is a percentile rather than mu + k*sigma. For it, T is expressed
    as a fraction of that percentile: 1.0 means exactly at the limit.
    """
    if value is None or not np.isfinite(value):
        return float("nan")
    if band["metric"] == "fcd":
        return float(value / band["hi"]) if band["hi"] else float("nan")
    half = K * band["sigma_eff"]
    if not np.isfinite(half) or half <= 0:
        return float("nan")
    return float((value - band["mu"]) / half)


def main(device="cpu", sweep=None):
    import pickle
    from analyse_pilot import load_arm
    from rdkit import Chem
    from rdkit.Chem import Crippen

    sweep = sweep or os.path.join(REPO, "results", "sweep")
    t, bands, eps = load_bands()
    if t.get("provisional"):
        print("WARNING the tolerance table is PROVISIONAL "
              f"(seeds present: {t['seeds_present']})")
    with open(TRAIN_PICKLE, "rb") as f:
        train = set(pickle.load(f))

    print(f"tolerance table  k={t['k']}  m={t['m']}  R={t['R']}  "
          f"from {len(t['seeds_present'])} baselines")
    print(f"commit at build  {t['versions'].get('git_commit', '?')[:12]}")
    print()

    # The unsteered reference for FCD: one baseline, held out of nothing,
    # chosen by seed order so the choice is not post hoc.
    ref_recs = list(load_arm(os.path.join(BASE, f"seed{REF_SEED}",
                                          "control")).values())
    ref_m = me.set_metrics(ref_recs)
    ref_act = fe.activations(fe.canon_for_chemnet(ref_m["unique_smiles"]),
                             device=device, already_canonical=True)
    print(f"FCD reference: baseline seed {REF_SEED}, "
          f"{len(ref_act)} canonical SMILES\n")

    # Control arm of the sweep: the paired baseline for property shifts.
    ctrl = {mid: r for mid, r in load_arm(os.path.join(sweep,
                                                       "control")).items()}

    def props(recs_by_id):
        """Per-molecule MolMR, MR/heavy and heavy, keyed by molecule id."""
        out = {}
        for mid, r in recs_by_id.items():
            mol_o = me.to_molecule(r)
            if mol_o is None:
                continue
            smi, _ = me.validity_midi(mol_o)
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            h = float(mol.GetNumHeavyAtoms())
            if h <= 0:
                continue
            mr = float(Crippen.MolMR(mol))
            out[mid] = {"molmr": mr, "heavy": h, "mr_per_heavy": mr / h}
        return out

    ctrl_p = props(ctrl)

    with open(os.path.join(sweep, "summary.csv"), newline="") as f:
        arms = list(csv.DictReader(f))

    rows = []
    for a in arms:
        name = a["arm"]
        recs_by_id = load_arm(os.path.join(sweep, name))
        recs = list(recs_by_id.values())
        m = me.set_metrics(recs, train_smiles=train)
        uq = mn.uniqueness_matched(m["smiles"], arm=name)
        nv = mn.novelty_matched(m["smiles"], train, arm=name)
        act = fe.activations(fe.canon_for_chemnet(m["unique_smiles"]),
                             device=device, already_canonical=True)
        fc = fe.fcd_matched(act, ref_act, arm=name)

        vals = {
            "validity_midi": m["validity_midi"]["p"],
            "validity_strict": m["validity_strict"]["p"],
            "connectivity": m["connectivity"]["p"],
            "mol_stable": m["mol_stable"]["p"],
            "atom_stable": m["atom_stable"]["p"],
            "aromatic_smiles_rate": m["aromatic_smiles_rate"]["p"],
            "uniq": uq["mean"] if uq["evaluable"] else None,
            "nov": nv["mean"] if nv["evaluable"] else None,
            "fcd": fc["mean"] if fc["evaluable"] else None,
        }

        verdicts, failed, tdist = {}, [], {}
        for g in GATES:
            b = bands.get(g)
            if b is None:
                raise SystemExit(
                    f"no band for gate {g!r}; the tolerance table has "
                    f"{sorted(bands)}")
            ok = inside(b, vals[g])
            verdicts[g] = ok
            tdist[g] = tolerance_distance(b, vals[g])
            if ok is not True:
                failed.append(g if ok is False else f"{g}(not evaluable)")
            # Consistency check, and it has to come AFTER the two assignments
            # above: verdicts and tdist are empty on the first pass, so the
            # same test at the top of the loop raises KeyError immediately.
            #
            # ok and T are read off the SAME band, so they cannot disagree.
            # If they do, either the band changed under us or one of the two
            # is computed from the wrong field. FCD gets a one-sided test,
            # because its band has no lower edge: its T is value / limit.
            if ok is True and np.isfinite(tdist[g]):
                over = (tdist[g] > 1.0 + 1e-9 if g == "fcd"
                        else abs(tdist[g]) > 1.0 + 1e-9)
                if over:
                    raise SystemExit(
                        f"{name}: {g} passed the band but T = {tdist[g]:.4f}, "
                        f"which is outside it. ok and T disagree, so one of "
                        f"them is wrong.")
        admissible = not failed

        # Property shift, paired against the sweep's own control, in
        # standardised units with the baseline-pooled denominators.
        arm_p = props(recs_by_id)
        shift = {}
        for k in ("molmr", "mr_per_heavy", "heavy"):
            ids = [i for i in arm_p if i in ctrl_p]
            d = np.array([arm_p[i][k] - ctrl_p[i][k] for i in ids], float)
            d = d[np.isfinite(d)]
            if d.size < 2:
                shift[k] = (np.nan, np.nan, np.nan)
                continue
            sd_pool = eps[k]["sd"]
            shift[k] = (d.mean(), d.mean() / sd_pool if sd_pool else np.nan,
                        d.std(ddof=1) / np.sqrt(d.size))

        PROPERTY_ENDPOINTS = ("molmr", "mr_per_heavy")
        best = max((abs(shift[k][1]) for k in PROPERTY_ENDPOINTS
                    if np.isfinite(shift[k][1])), default=np.nan)
        big_enough = bool(np.isfinite(best) and best >= 0.5)

        rows.append({
            "arm": name, "kind": a["direction"], "coeff": float(a["coeff"]),
            "rho": float(a["rho_mean"]) if a["rho_mean"] != "nan" else np.nan,
            **{f"v_{k}": vals[k] for k in vals},
            **{f"ok_{g}": verdicts[g] for g in GATES},
            **{f"T_{g}": tdist[g] for g in GATES},
            "n_failed": len(failed), "failed": ";".join(failed),
            "admissible": admissible,
            **{f"d_{k}": shift[k][0] for k in shift},
            **{f"std_{k}": shift[k][1] for k in shift},
            "max_abs_std_shift": best,
            "shift_of_interest": big_enough,
            "reportable": bool(admissible and big_enough),
        })
        print(f"  {name:<20} valid {100 * vals['validity_midi']:>5.1f}%  "
              f"fcd {vals['fcd'] if vals['fcd'] else float('nan'):>6.3f}  "
              f"maxstd {best:>6.3f}  "
              f"{'ADMISSIBLE' if admissible else 'fails: ' + failed[0]}",
              flush=True)

    out_csv = os.path.join(REPO, "results", "central_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_adm = sum(r["admissible"] for r in rows)
    n_big = sum(r["shift_of_interest"] for r in rows)
    n_rep = sum(r["reportable"] for r in rows)
    print()
    print(f"admissible                {n_adm} of {len(rows)}")
    print(f"shift >= 0.5 sd           {n_big} of {len(rows)}")
    print(f"REPORTABLE (both)         {n_rep} of {len(rows)}")
    print(f"\nwrote {out_csv}")
    return rows


if __name__ == "__main__":
    dev, sw = "cpu", None
    args = sys.argv[1:]
    for i, x in enumerate(args):
        if x == "--device" and i + 1 < len(args):
            dev = args[i + 1]
        if x == "--sweep" and i + 1 < len(args):
            sw = args[i + 1]
    main(dev, sw)
