"""Derive the manuscript tables that were marked "Derive" or did not exist.

    python3 scripts/make_tables.py <repo_root> <outdir>

Writes a .csv (for the record) and a .md (to paste) for each of:

  Table 1  Best probe cell per property, with the size-only baseline beside it.
           Source results/probe_r2.csv.
  Table 2  Direction geometry: the cosine between the probe direction and the
           contrastive direction, against chance, plus the covariance facts
           that explain it. Sources results/sweep/pilot_manifest.json,
           results/directions/, activations_test_full/.
  Table 6  xTB polarisability by arm. New: the template predates the xTB run.
           Source results/xtb_arm_table.csv.

Table 1 uses variant = "size_residual", NOT "raw"
-------------------------------------------------
probe_r2.csv holds both variants. The raw numbers are higher and meaningless
for this claim: polarisability has r2_size_only = 0.871, so a raw probe mostly
reports that the model knows how big the molecule is. The residual variant is
the one the paper's claim rests on and the one Figure 1b plots. Taking the max
over both variants silently reports the raw number, which is the single easiest
way to overstate this result.

`lam_at_grid_edge` is carried through. A cell whose ridge penalty selected the
edge of the search grid has not converged inside the grid, so its R2 is a lower
bound rather than an estimate, and that has to be visible in the table rather
than discovered by a referee. 533 of the 14,400 cells are flagged.
"""

import csv
import json
import os
import sys

import numpy as np


def w(path, rows, cols, title, notes):
    """Write one table as both .csv and .md."""
    with open(path + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})
    with open(path + ".md", "w") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        if notes:
            f.write("\n" + notes.strip() + "\n")
    print(f"  wrote {path}.csv and {path}.md ({len(rows)} rows)")


def fmt(v, n=4):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return ""


# ------------------------------------------------------------------ Table 1

def table1(repo, outdir):
    import pandas as pd
    src = os.path.join(repo, "results", "probe_r2.csv")
    df = pd.read_csv(src, on_bad_lines="skip")
    res = df[df.variant == "size_residual"]
    if res.empty:
        raise SystemExit("no size_residual rows in probe_r2.csv")
    rows = []
    for prop, g in res.groupby("property"):
        b = g.loc[g.r2.idxmax()]
        raw = df[(df.property == prop) & (df.variant == "raw")
                 & (df.layer == b.layer) & (df.timestep == b.timestep)
                 & (df.feature_set == b.feature_set)]
        rows.append({
            "property": prop,
            "features": b.feature_set,
            "layer": int(b.layer),
            "timestep": int(b.timestep),
            "R2 residual": fmt(b.r2),
            "R2 raw": fmt(raw.r2.iloc[0]) if len(raw) else "",
            "R2 size-only": fmt(b.r2_size_only),
            "null p95": fmt(b.r2_null_p95),
            "perm p": fmt(b.perm_p),
            "lam at grid edge": bool(b.lam_at_grid_edge),
            "n test": int(b.n_test),
        })
    order = ["polarisability", "heat_capacity", "homo", "lumo",
             "homo_lumo_gap", "dipole_moment"]
    rows.sort(key=lambda r: order.index(r["property"])
              if r["property"] in order else 99)
    edge = int(df.lam_at_grid_edge.sum())
    w(os.path.join(outdir, "table1_probe"), rows, list(rows[0].keys()),
      "Table 1. Best probe cell per property, composition-controlled, with the "
      "size-only baseline beside it.",
      f"""All cells are held-out R2 on a molecule-level split. "R2 residual" is
the probe after the linear composition component is removed and is the number
the claim rests on; "R2 raw" is the same cell without that control and is shown
only so the size of the correction is visible. "R2 size-only" is a regression
on atom count alone, so the gap between it and the residual column is what the
activations add beyond knowing how big the molecule is. "null p95" is the 95th
percentile of a 200-shuffle permutation floor for that cell. {edge} of the
{len(df)} cells in the full grid selected a ridge penalty at the edge of the
search grid; where that flag is True the R2 is a lower bound.""")
    return rows


# ------------------------------------------------------------------ Table 2

def table2(repo, outdir):
    sys.path.insert(0, repo)
    man_path = os.path.join(repo, "results", "sweep", "pilot_manifest.json")
    man = json.load(open(man_path))
    layer, block = int(man["layer"]), man["block"]
    prop = man["property"]
    cos_obs = float(man["cos_probe_vs_dom"])

    dpath = os.path.join(repo, "results", "directions",
                         f"dir_L{layer}_{block}_{prop}.npz")
    z = np.load(dpath)
    ts = np.asarray(z["timesteps"])
    i50 = int(np.where(ts == 50)[0][0]) if 50 in ts else len(ts) - 1
    d = np.asarray(z["d_resid"])[i50]
    wv = np.asarray(z["w_probe_raw"])
    D = len(wv)
    unit = lambda v: v / np.linalg.norm(v)

    rows = [{"quantity": "dimension of the steered block", "value": str(D),
             "note": f"{block} at layer {layer}"}]
    rows.append({"quantity": "cos(d, w) observed", "value": fmt(cos_obs),
                 "note": "contrastive direction vs probe direction"})
    chance = 1.0 / np.sqrt(D)
    rows.append({"quantity": "cos at chance (RMS, 1/sqrt(D))",
                 "value": fmt(chance),
                 "note": "two unrelated directions in this many dimensions"})
    rows.append({"quantity": "ratio observed / chance",
                 "value": fmt(cos_obs / chance, 2),
                 "note": "1.0 would be exactly orthogonal on average"})

    try:
        from midi import direction
        A = direction.load_block(
            os.path.join(repo, "activations_test_full"), layer, i50, block)
        s_d = float((A @ unit(d)).std())
        s_w = float((A @ unit(wv)).std())
        a_norm = float(np.linalg.norm(A, axis=1).mean())
        ev = np.linalg.eigvalsh(np.cov(A - A.mean(0), rowvar=False))[::-1]
        ev = np.clip(ev, 0, None)
        share10 = float(ev[:10].sum() / ev.sum())
        rows += [
            {"quantity": "data spread along d", "value": fmt(s_d),
             "note": "sd of the activations projected on the direction"},
            {"quantity": "data spread along w", "value": fmt(s_w),
             "note": "the probe direction is a LOW-variance direction"},
            {"quantity": "spread ratio d / w", "value": fmt(s_d / s_w, 1),
             "note": "why the same injected length means different things"},
            {"quantity": "variance in the top 10 of "
                         f"{D} directions", "value": fmt(share10, 3),
             "note": "covariance anisotropy, which is what rotates w off d"},
            {"quantity": "mean activation norm", "value": fmt(a_norm, 2),
             "note": "scale the injection is measured against"},
        ]
        for c in (20.0,):
            L = c * float(np.linalg.norm(np.asarray(z["d_resid"])[i50]))
            L = c  # injected length is coeff * ||d|| * unit, reported in units of ||d||
        inj = 20.0 * float(np.linalg.norm(d)) / float(np.linalg.norm(d))
        rows.append({"quantity": "off-manifold distance at matched "
                                 "injected norm (c = 20)",
                     "value": f"{20.0 / s_d:.1f} sd along d, "
                              f"{20.0 / s_w:.1f} sd along w",
                     "note": "matched length cannot also match plausibility"})
    except Exception as e:
        rows.append({"quantity": "covariance statistics", "value": "unavailable",
                     "note": f"{type(e).__name__}: {e}"})

    w(os.path.join(outdir, "table2_geometry"), rows,
      ["quantity", "value", "note"],
      "Table 2. Direction geometry: the probe direction and the contrastive "
      "direction are near-orthogonal, and why.",
      """d is the composition-residualised difference of means between high and
low deciles of the property. w is the ridge probe weight mapped back to raw
activation space, w proportional to Sigma^-1 d. The whitening is what separates
them: because the covariance is strongly anisotropic, Sigma^-1 rotates d almost
out of the subspace the data occupies. The consequence is the paper's
mechanism. The direction that is most informative for READING the property is a
low-variance direction, which is the worst direction for WRITING it: matching
the off-manifold distance makes the perturbation too small to change any
discrete output, and matching the injected norm puts the activation
implausibly far from anything the model has seen.""")
    return rows


# ------------------------------------------------------------------ Table 6

def table6(repo, outdir):
    src = os.path.join(repo, "results", "xtb_arm_table.csv")
    with open(src, newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit("empty " + src)

    def key(r):
        fam = r["family"]
        return ({"control": 0, "dom": 1, "rand": 2, "shuf": 3}.get(fam, 9),
                float(r["coeff"]))

    rows = []
    for r in sorted(raw, key=key):
        rows.append({
            "arm": r["arm"],
            "c": int(float(r["coeff"])),
            "n ok": f"{r['ok_opt']}/{r['n']}",
            "alpha relaxed": fmt(r["alpha_opt_mean"], 2),
            "d_alpha": fmt(r["d_opt"], 3),
            "t": fmt(r["d_opt_t"], 2),
            "median": fmt(r["d_opt_median"], 3),
            "d_alpha/heavy": fmt(r["d_opt_per_heavy"], 4),
            "t ": fmt(r["d_opt_per_heavy_t"], 2),
            "d/heavy as-is": fmt(r["d_asis_per_heavy"], 4),
            "RMSD": fmt(r["rmsd_mean"], 3),
        })
    w(os.path.join(outdir, "table6_xtb"), rows, list(rows[0].keys()),
      "Table 6. GFN2/PTB-xTB polarisability by arm, paired against the "
      "unsteered control (n = 500 per arm).",
      """alpha is the PTB isotropic dipole polarisability in Bohr^3, a field
response property, not the additive D4 sum that xtb also prints. All
differences are paired on shared molecule ids. Read three things. The sign test
fails: the + and - arms of the same magnitude move the same way at four of five
magnitudes. The controls match the treatment: rand and shuf reach the same
"highly significant" d_alpha as dom, so the large t values are perturbation
damage rather than property control. And the medians sit near zero while the
means do not, which is the signature of a minority of molecules blowing up
rather than a distribution shifting. The as-is column is shown against the
relaxed one because scoring the generated coordinates alone produces a positive
plateau across c = +20 to +160 that relaxation erases; coordinate error inflates
this estimator upward and steering degrades coordinates most at large |c|.""")
    return rows


def main(repo, outdir):
    os.makedirs(outdir, exist_ok=True)
    table1(repo, outdir)
    table2(repo, outdir)
    table6(repo, outdir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_tables.py <repo_root> <outdir>")
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
