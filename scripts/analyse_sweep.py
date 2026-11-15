"""scripts/analyse_sweep.py -- the sweep as paired data, with the controls.

    python3.11 -u scripts/analyse_sweep.py <sweep_dir>

analyse_pilot.py reports raw dMolMR against the control. That is not enough
for the sweep, for two reasons.

ONE: the endpoint has to be size-adjusted. results/coefficient_grid.md showed
that the whole apparent property shift was molecule size, so three readings
of the same paired data are reported here, not one:

    dMolMR              raw paired difference from control
    dMR per heavy atom  MolMR/heavy, paired. Removes size by division.
    dMolMR | dHeavy     the intercept of dMolMR regressed on dHeavy, so the
                        MolMR shift at zero change in heavy-atom count.
                        Removes size by covariate adjustment.

If the raw column is significant and the other two are not, the arm moved
size and MolMR followed.

TWO: the comparison that answers Objective 3 is between ARMS at matched |c|,
not between an arm and the control. Every arm shares the same molecule ids
and run_seed, so molecule i exists in all 23 cells and the control cancels
out of an arm-to-arm difference:

    (MolMR_dom(i) - MolMR_ctrl(i)) - (MolMR_rand(i) - MolMR_ctrl(i))
        = MolMR_dom(i) - MolMR_rand(i)

so the contrast is itself paired and far tighter than comparing two arm
means. Validity is contrasted with McNemar's test on the same pairing, which
uses only the molecules where the two arms disagree.

The sign is meaningful for `dom` only. A random unit vector and its negation
are both random draws and a label permutation has no privileged orientation,
so the control arms are POOLED by |c|, giving them n = 1000 per magnitude.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from analyse_pilot import load_arm, descriptors        # noqa: E402


def paired(a, b):
    """mean, se, t for the paired difference a - b. Arrays, equal length."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, np.nan, np.nan, len(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else np.nan), len(d)


def adjusted_intercept(dy, dx):
    """Intercept of dy = a + b*dx, with its se and t.

    a is the change in dy at zero change in dx, so for dy=dMolMR and
    dx=dHeavy it is the MolMR shift that is NOT explained by the molecule
    gaining or losing heavy atoms.
    """
    dy, dx = np.asarray(dy, float), np.asarray(dx, float)
    k = np.isfinite(dy) & np.isfinite(dx)
    dy, dx = dy[k], dx[k]
    if len(dy) < 3 or dx.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(dx)), dx])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    s2 = float(resid @ resid) / (len(dy) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se_a = float(np.sqrt(cov[0, 0]))
    return float(beta[0]), se_a, (beta[0] / se_a if se_a > 0 else np.nan), \
        float(beta[1])


def mcnemar(valid_a, valid_b):
    """Paired validity contrast. Returns (b, c, chi2) where b is valid in A
    but not B, c the reverse. Only the discordant pairs carry information.
    """
    a = np.asarray(valid_a, bool)
    b = np.asarray(valid_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    tot = n01 + n10
    chi2 = ((abs(n01 - n10) - 1) ** 2) / tot if tot > 0 else np.nan
    return n01, n10, chi2


def main(sweep):
    sweep = os.path.abspath(sweep)
    with open(os.path.join(sweep, "summary.csv"), newline="") as f:
        arms = list(csv.DictReader(f))
    print(f"sweep {sweep}")
    print(f"arms  {len(arms)}\n")

    # ---- load every arm once, keep only the descriptors ----------------
    desc = {}
    for a in arms:
        name = a["arm"]
        recs = load_arm(os.path.join(sweep, name))
        desc[name] = {mid: descriptors(r) for mid, r in recs.items()}
        print(f"  loaded {name:<22} {len(recs):>5} molecules, "
              f"{sum(v['valid'] for v in desc[name].values()):>4} sanitise",
              flush=True)
    ids = sorted(set.intersection(*(set(d) for d in desc.values())))
    print(f"\n{len(ids)} molecule ids present in all {len(arms)} arms\n")

    def vec(name, key, only_valid_with=None):
        """Per-molecule values over `ids`, nan where the molecule is invalid
        in this arm or in the partner arm."""
        out = np.full(len(ids), np.nan)
        for j, i in enumerate(ids):
            d = desc[name][i]
            if not d["valid"]:
                continue
            if only_valid_with is not None and not desc[only_valid_with][i]["valid"]:
                continue
            v = d[key]
            if key == "mr_per_heavy":
                v = d["molmr"] / d["heavy"] if d["heavy"] > 0 else np.nan
            out[j] = v
        return out

    def mrph(name, partner=None):
        out = np.full(len(ids), np.nan)
        for j, i in enumerate(ids):
            d = desc[name][i]
            if not d["valid"] or d["heavy"] <= 0:
                continue
            if partner is not None and not desc[partner][i]["valid"]:
                continue
            out[j] = d["molmr"] / d["heavy"]
        return out

    # ---- TABLE A: every arm against the shared control -----------------
    print("=" * 108)
    print("TABLE A  each arm against the shared unsteered control, paired by "
          "molecule id")
    print("=" * 108)
    hdr = (f"{'arm':<20}{'coeff':>7}{'rho':>7}{'valid%':>8}"
           f"{'dMolMR':>9}{'t':>6}{'dMR/heavy':>11}{'t':>6}"
           f"{'dMolMR|dH':>11}{'t':>6}{'dHeavy':>8}{'t':>6}{'n':>6}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for a in arms:
        name = a["arm"]
        if name == "control":
            continue
        mr_a = vec(name, "molmr", only_valid_with="control")
        mr_c = vec("control", "molmr", only_valid_with=name)
        h_a = vec(name, "heavy", only_valid_with="control")
        h_c = vec("control", "heavy", only_valid_with=name)
        rp_a, rp_c = mrph(name, "control"), mrph("control", name)

        dmr, se_mr, t_mr, n = paired(mr_a, mr_c)
        drp, se_rp, t_rp, _ = paired(rp_a, rp_c)
        dh, se_h, t_h, _ = paired(h_a, h_c)
        a_int, se_int, t_int, slope = adjusted_intercept(mr_a - mr_c, h_a - h_c)

        print(f"{name:<20}{float(a['coeff']):>7.0f}"
              f"{float(a['rho_mean']):>7.2f}{float(a['valid_pct']):>8.1f}"
              f"{dmr:>9.3f}{t_mr:>6.1f}{drp:>11.4f}{t_rp:>6.1f}"
              f"{a_int:>11.3f}{t_int:>6.1f}{dh:>8.3f}{t_h:>6.1f}{n:>6}")
        rows.append({
            "arm": name, "kind": a["direction"], "coeff": float(a["coeff"]),
            "abs_coeff": abs(float(a["coeff"])),
            "rho": float(a["rho_mean"]), "valid_pct": float(a["valid_pct"]),
            "d_molmr": dmr, "se_molmr": se_mr, "t_molmr": t_mr,
            "d_mr_per_heavy": drp, "se_mr_per_heavy": se_rp,
            "t_mr_per_heavy": t_rp,
            "d_molmr_adj": a_int, "se_molmr_adj": se_int,
            "t_molmr_adj": t_int, "slope_mr_on_heavy": slope,
            "d_heavy": dh, "se_heavy": se_h, "t_heavy": t_h, "n_paired": n})

    # ---- TABLE B: arm against arm at matched |c| -----------------------
    print()
    print("=" * 108)
    print("TABLE B  real against each control at MATCHED injected norm, "
           "paired by molecule id")
    print("         the control cancels, so this is a direct dom-minus-control"
          " difference")
    print("=" * 108)
    by = {}
    for a in arms:
        if a["arm"] == "control":
            continue
        by.setdefault((a["direction"], abs(float(a["coeff"]))), []).append(a["arm"])

    hdr2 = (f"{'|c|':>5} {'contrast':<26}{'dMolMR':>9}{'se':>7}{'t':>6}"
            f"{'dHeavy':>9}{'se':>7}{'t':>6}   {'validity (McNemar)':<30}")
    print(hdr2)
    print("-" * len(hdr2))

    # One row per (dom arm, control arm) at matched |c| AND matched sign.
    # Not pooled: a molecule appearing in two control arms would enter the
    # same test twice, which understates the standard error by up to sqrt(2).
    # The spread across draws is reported separately in Table C instead.
    contrasts = []
    for ac in sorted({k[1] for k in by}):
        for sgn in (+1, -1):
            dom_arms = [a["arm"] for a in arms
                        if a["direction"] == "dom"
                        and float(a["coeff"]) == sgn * ac]
            for kind in ("rand", "shuf"):
                ctl_arms = [a["arm"] for a in arms
                            if a["direction"] == kind
                            and float(a["coeff"]) == sgn * ac]
                for da in dom_arms:
                    for ca in ctl_arms:
                        mr_d, mr_k, h_d, h_k, v_d, v_k = [], [], [], [], [], []
                        for i in ids:
                            dd, dk = desc[da][i], desc[ca][i]
                            v_d.append(bool(dd["valid"]))
                            v_k.append(bool(dk["valid"]))
                            if dd["valid"] and dk["valid"]:
                                mr_d.append(dd["molmr"])
                                mr_k.append(dk["molmr"])
                                h_d.append(dd["heavy"])
                                h_k.append(dk["heavy"])
                        dmr, se_mr, t_mr, n = paired(mr_d, mr_k)
                        dh, se_h, t_h, _ = paired(h_d, h_k)
                        n01, n10, chi2 = mcnemar(v_d, v_k)
                        label = f"dom c{sgn * ac:+.0f} - {kind}"
                        print(f"{ac:>5.0f} {label:<26}{dmr:>9.3f}{se_mr:>7.3f}"
                              f"{t_mr:>6.1f}{dh:>9.3f}{se_h:>7.3f}{t_h:>6.1f}"
                              f"   dom-only {n01:>4}, ctl-only {n10:>4}, "
                              f"chi2 {chi2:>6.1f}")
                        contrasts.append({
                            "abs_coeff": ac, "sign": sgn, "dom_arm": da,
                            "control_arm": ca, "control_kind": kind,
                            "d_molmr": dmr, "se_molmr": se_mr,
                            "t_molmr": t_mr, "d_heavy": dh,
                            "se_heavy": se_h, "t_heavy": t_h,
                            "n_paired": n, "mcnemar_dom_only": n01,
                            "mcnemar_ctl_only": n10, "mcnemar_chi2": chi2})

    # ---- TABLE C: is the real arm outside the control DISTRIBUTION? ----
    # Each control arm is ONE draw, so its own shift is a random realisation.
    # The question Objective 3 actually asks is whether the real arm's shift
    # is larger than the spread of shifts that arbitrary directions produce
    # at the same injected norm.
    print()
    print("=" * 108)
    print("TABLE C  the real arm against the SPREAD of control draws at the "
          "same |c|")
    print("=" * 108)
    r_by = {(r["kind"], r["abs_coeff"]): [] for r in rows}
    for r in rows:
        r_by[(r["kind"], r["abs_coeff"])].append(r)
    hdr3 = (f"{'|c|':>5} {'metric':<16}{'dom -c':>9}{'dom +c':>9}"
            f"{'ctl mean':>10}{'ctl sd':>9}{'n draws':>9}  verdict")
    print(hdr3)
    print("-" * len(hdr3))
    for ac in sorted({r["abs_coeff"] for r in rows}):
        ctl = [r for r in rows if r["kind"] in ("rand", "shuf")
               and r["abs_coeff"] == ac]
        if not ctl:
            continue
        for metric in ("d_molmr", "d_heavy"):
            dneg = [r[metric] for r in rows if r["kind"] == "dom"
                    and r["coeff"] == -ac]
            dpos = [r[metric] for r in rows if r["kind"] == "dom"
                    and r["coeff"] == ac]
            cv = np.array([r[metric] for r in ctl], float)
            m, s = float(np.nanmean(cv)), float(np.nanstd(cv, ddof=1))
            best = max(abs(dneg[0]) if dneg else 0,
                       abs(dpos[0]) if dpos else 0)
            inside = s > 0 and best <= abs(m) + 2 * s
            print(f"{ac:>5.0f} {metric:<16}"
                  f"{(dneg[0] if dneg else np.nan):>9.3f}"
                  f"{(dpos[0] if dpos else np.nan):>9.3f}"
                  f"{m:>10.3f}{s:>9.3f}{len(cv):>9}  "
                  f"{'inside the control spread' if inside else 'OUTSIDE'}")

    print()
    print("Reading Table B: dMolMR and dHeavy are the real arm MINUS the")
    print("control arm at the same injected norm. A |t| near zero means the")
    print("real direction did nothing the control did not also do, so the")
    print("effect belongs to perturbing activations at that magnitude rather")
    print("than to this direction's content. McNemar: chi2 above 3.84 is a")
    print("validity difference at the 5% level, and the sign is given by")
    print("which column is larger.")
    print()
    print("Caveat that governs both tables: MolMR is an RDKit graph")
    print("descriptor, not a measurement. It is a proxy for polarisability")
    print("through the Lorentz-Lorenz relation. An independent estimate")
    print("(GFN2-xTB) can be run over these same saved chunks without")
    print("regenerating anything.")

    for nm, data in (("sweep_arms.csv", rows),
                     ("sweep_contrasts.csv", contrasts)):
        p = os.path.join(sweep, nm)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep")
