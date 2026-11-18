"""Figure 2. Dose-response with controls, from the xTB polarisability scoring.

    python3 scripts/figure2.py <xtb_sweep_rows.csv> <central_table.csv> <outdir>

Produces
    figure2_dose_response.png   the 2x2 main figure
    figure2_data.csv            every number plotted, one row per arm

What the four panels argue, in order
------------------------------------
(a) The result. Paired change in polarisability per heavy atom against the
    steering coefficient, for the real direction (dom) and for the two
    information-free controls (rand, shuf). The grey band is the envelope the
    CONTROLS span at each coefficient. A direction that encodes the property
    leaves that band. dom does not, except at c=+20, which panel (a) also shows
    is not sustained at larger coefficients.

(b) Why the geometry decision mattered. The same dom arms scored two ways. On
    the generated coordinates as-is there is a clean positive plateau across
    four consecutive coefficients, which is what working steering looks like.
    After GFN2 relaxation it is gone. The as-is curve is a geometry-quality
    artefact: coordinate error inflates the PTB polarisability upward, and
    steering degrades coordinates most at large |c|.

(c) The damage, measured without reference to the property at all. Kabsch RMSD
    between the generated coordinates and their relaxed structure. This rises
    with |c| for the real direction while the controls stay near baseline, so
    the injection is doing something -- it is just not moving the property.

(d) The cost. MiDi validity against coefficient. Read with (a): the arms that
    survive do not move the property, and the arms that move anything do not
    survive.

Style is inherited from figure1.py deliberately, so the two figures read as one
pair in the manuscript: same rcParams, same ink and muted greys, and the two
series colours are figure 1's diverging endpoints (#b2182b, #2166ac) reused as
categorical hues. The third hue #b35806 was chosen by validation, not by eye:
the set passes the lightness band, chroma floor, CVD separation (worst adjacent
pair dE 21.1 protan, 25.9 tritan), the normal-vision floor (dE 27.4) and 3:1
contrast against white. Every series also carries a distinct marker, so colour
is never the only thing distinguishing them.

Panel (b) deliberately does NOT reuse the categorical palette. Its two series
are geometry modes, not arm families, and giving them hues that mean "rand" and
"shuf" two panels earlier would be a lie about what colour encodes. It uses one
hue with solid/dashed and filled/open markers instead.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#1a1a1a", "#666666"
BAND = "#e8e8e8"

# Categorical hues. Fixed order, never cycled. dom is figure 1's warm pole and
# rand its cool pole, so the two figures agree about what red and blue mean.
FAM = ("dom", "rand", "shuf")
COLOR = {"dom": "#b2182b", "rand": "#2166ac", "shuf": "#b35806"}
MARKER = {"dom": "o", "rand": "s", "shuf": "^"}
LABEL = {"dom": "dom (real direction)", "rand": "rand (control)",
         "shuf": "shuf (control)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False,
    "savefig.facecolor": "white",
})


# ------------------------------------------------------------------- loading

def coeff_of(arm):
    """Signed steering coefficient from an arm name, 0 for the control."""
    base = arm.split("/")[-1]
    if base == "control":
        return 0.0
    tail = base.rsplit("_c", 1)[-1]
    sign = -1.0 if tail.startswith("m") else 1.0
    return sign * float(tail[1:])


def family_of(arm):
    base = arm.split("/")[-1]
    return "control" if base == "control" else base.split("_")[0]


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_rows(path):
    by_arm = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_arm[r["arm"].split("/")[-1]][(r["chunk"], int(r["mol_id"]))] = r
    return by_arm


def paired(arm_rows, ctrl_rows, key, per_heavy):
    """Paired mean difference against the control, with its standard error.

    Molecules are matched on (chunk, mol_id): every arm shares the ids and the
    run seed, so molecule i in an arm and molecule i in the control start from
    the same size and the same noise and differ only by the injection. That is
    what makes a 500-molecule arm informative.
    """
    d = []
    for k, r in arm_rows.items():
        c = ctrl_rows.get(k)
        if c is None:
            continue
        x, y = fnum(r[key]), fnum(c[key])
        if per_heavy:
            a, b = fnum(r["n_heavy"]), fnum(c["n_heavy"])
            if not (a > 0 and b > 0):
                continue
            x, y = x / a, y / b
        if np.isfinite(x) and np.isfinite(y):
            d.append(x - y)
    d = np.asarray(d)
    if len(d) < 2:
        return np.nan, np.nan, len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def mean_se(arm_rows, key):
    v = np.array([fnum(r[key]) for r in arm_rows.values()])
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def load_validity(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["arm"].split("/")[-1]] = fnum(r.get("v_validity_midi", ""))
    return out


# -------------------------------------------------------------------- panels

def series(ax, pts, fam, key_y, key_e, **kw):
    """One family's line, sorted by coefficient, with +/-1 se bars."""
    xs = sorted(pts[fam])
    y = [pts[fam][c][key_y] for c in xs]
    e = [pts[fam][c][key_e] for c in xs] if key_e else None
    ax.errorbar(xs, y, yerr=e, color=COLOR[fam], marker=MARKER[fam],
                markersize=3.4, linewidth=1.2, elinewidth=0.7, capsize=1.8,
                markeredgewidth=0, zorder=3, **kw)


def control_band(ax, pts):
    """Grey envelope spanning what the information-free directions do.

    This is the panel's actual argument made visual: anything inside this band
    is doing no more than a direction carrying no property information.
    """
    # Drawn as two polygons, one each side of zero. Every family passes
    # through exactly 0 at c=0 by construction, so a single polygon would
    # pinch to a point there and leave two spurious wedges that look like a
    # finding. The span includes +/-1 se, so the band is "where the
    # information-free directions actually sit", uncertainty included.
    drawn = []
    for side in (-1, +1):
        xs = sorted(c for c in set(pts["rand"]) | set(pts["shuf"])
                    if np.sign(c) == side)
        lo, hi, keep = [], [], []
        for c in xs:
            vals = []
            for f in ("rand", "shuf"):
                p = pts[f].get(c)
                if p and np.isfinite(p["y"]):
                    e = p["se"] if np.isfinite(p["se"]) else 0.0
                    vals += [p["y"] - e, p["y"] + e]
            if len(vals) >= 4:
                keep.append(c); lo.append(min(vals)); hi.append(max(vals))
        if len(keep) >= 2:
            ax.fill_between(keep, lo, hi, color=BAND, zorder=1)
            drawn += keep
    return drawn


def zero_line(ax):
    ax.axhline(0.0, color=MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=2)


def main(rows_csv, central_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    by_arm = load_rows(rows_csv)
    if "control" not in by_arm:
        raise SystemExit("no control arm in " + rows_csv)
    ctrl = by_arm["control"]
    validity = load_validity(central_csv)

    # Assemble every plotted number once, so the figure and the CSV cannot drift.
    pts = {f: {} for f in FAM}
    export = []
    for arm, rows in by_arm.items():
        fam = family_of(arm)
        if fam not in pts:
            continue
        c = coeff_of(arm)
        m_rel, se_rel, n = paired(rows, ctrl, "alpha_opt", True)
        m_as, se_as, _ = paired(rows, ctrl, "alpha_asis", True)
        m_raw, se_raw, _ = paired(rows, ctrl, "alpha_opt", False)
        rm, rm_se = mean_se(rows, "rmsd_opt")
        val = validity.get(arm, np.nan)
        pts[fam][c] = {"y": m_rel, "se": se_rel, "asis": m_as, "asis_se": se_as,
                       "rmsd": rm, "rmsd_se": rm_se, "val": val,
                       "val_se": (np.sqrt(val * (1 - val) / n)
                                  if np.isfinite(val) and n else np.nan)}
        export.append({"arm": arm, "family": fam, "coeff": c, "n_paired": n,
                       "d_alpha_per_heavy_relaxed": m_rel, "se_relaxed": se_rel,
                       "d_alpha_per_heavy_asis": m_as, "se_asis": se_as,
                       "d_alpha_relaxed": m_raw, "se_alpha_relaxed": se_raw,
                       "rmsd_mean": rm, "rmsd_se": rm_se, "validity": val})

    # The unsteered control sits at c = 0 in every family: it is the same run,
    # and its paired difference against itself is exactly zero by construction.
    base = {"y": 0.0, "se": 0.0, "asis": 0.0, "asis_se": 0.0,
            "rmsd": mean_se(ctrl, "rmsd_opt")[0],
            "rmsd_se": mean_se(ctrl, "rmsd_opt")[1],
            "val": validity.get("control", np.nan), "val_se": np.nan}
    for f in FAM:
        pts[f][0.0] = dict(base)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6), constrained_layout=True)
    (a, b), (c, d) = axes

    # (a) the result
    control_band(a, pts)
    zero_line(a)
    for f in FAM:
        series(a, pts, f, "y", "se", label=LABEL[f])
    a.set_title(r"a. $\Delta\alpha$ per heavy atom, relaxed", loc="left")
    a.set_ylabel(r"$\Delta\alpha$ / heavy atom  (Bohr$^3$)")

    # (b) the artefact
    zero_line(b)
    xs = sorted(pts["dom"])
    b.errorbar(xs, [pts["dom"][x]["asis"] for x in xs],
               yerr=[pts["dom"][x]["asis_se"] for x in xs],
               color=COLOR["dom"], marker="o", markersize=3.4, linewidth=1.2,
               linestyle=(0, (4, 2)), markerfacecolor="white",
               markeredgecolor=COLOR["dom"], markeredgewidth=1.0,
               elinewidth=0.7, capsize=1.8, zorder=3,
               label="as generated")
    b.errorbar(xs, [pts["dom"][x]["y"] for x in xs],
               yerr=[pts["dom"][x]["se"] for x in xs],
               color=COLOR["dom"], marker="o", markersize=3.4, linewidth=1.2,
               elinewidth=0.7, capsize=1.8, markeredgewidth=0, zorder=4,
               label="after GFN2 relaxation")
    b.set_title("b. dom only: as-generated vs relaxed", loc="left")
    b.set_ylabel(r"$\Delta\alpha$ / heavy atom  (Bohr$^3$)")
    b.legend(loc="upper left", fontsize=7)

    # (c) damage, property-free
    for f in FAM:
        series(c, pts, f, "rmsd", "rmsd_se", label=LABEL[f])
    c.set_title("c. Geometry damage", loc="left")
    c.set_ylabel(u"Kabsch RMSD (Å)")

    # (d) the cost
    for f in FAM:
        series(d, pts, f, "val", "val_se", label=LABEL[f])
    d.set_title("d. MiDi validity", loc="left")
    d.set_ylabel("valid fraction")

    for ax in (a, b, c, d):
        ax.set_xlabel("steering coefficient $c$")
        ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], color=COLOR[f], marker=MARKER[f],
                          markersize=3.4, linewidth=1.2, markeredgewidth=0,
                          label=LABEL[f]) for f in FAM]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND,
                                 label="control envelope (mean $\\pm$ 1 se)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=7.5)

    path = os.path.join(outdir, "figure2_dose_response.png")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote " + path)

    cols = list(export[0].keys())
    cpath = os.path.join(outdir, "figure2_data.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(export, key=lambda r: (r["family"], r["coeff"])):
            w.writerow(r)
    print(f"  wrote {cpath} ({len(export)} arms)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    main(sys.argv[1], sys.argv[2], sys.argv[3])
