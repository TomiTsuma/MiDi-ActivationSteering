"""scripts/figure1.py -- figure 1 and its data.

    python3.11 -u scripts/figure1.py <store> <probe_r2.csv> <outdir>

Produces, for the concatenated feature set:
  figure1_raw.png        layer x timestep heatmaps, one panel per property
  figure1_residual.png   the same after the composition control (the result)
  figure2_channels.png   feature set x layer at each property's best timestep
  figure1_data.csv       exactly the rows that were plotted

Design choices, and why:

  * Magnitude on two categorical axes is a heatmap. Raw R2 lies in [0, 1] and
    is pure magnitude, so it gets a SINGLE-HUE sequential ramp, light to dark.
    No rainbow: a multi-hue ramp invents structure that is not in the data.
  * The residual R2 straddles zero, which is polarity, not magnitude, so it
    gets a DIVERGING ramp with a NEUTRAL GREY midpoint pinned at exactly zero.
    Grey at zero matters: it is the "nothing here" colour, and a hue there
    would read as a finding.
  * A cell that fails to beat the 95th percentile of its own permutation floor
    is HATCHED. That is a second, non-colour encoding of the thing a reader
    most needs, so it survives greyscale printing and colour-vision
    deficiency.
  * Only the best cell per panel is labelled. A number in all 120 cells is
    noise, not information.
  * Ten feature sets are NOT ten colours. Cycled hues past about eight series
    are unreadable, so the channel comparison is a second heatmap.
  * No interaction layer and no dark mode: this is a print figure for a
    dissertation, so it is light-surface only.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import probe

MAIN_FS = "all"
INK, MUTED = "#1a1a1a", "#666666"
SEQ = "Blues"                                    # one hue, light -> dark
DIV = LinearSegmentedColormap.from_list(          # two hues + neutral grey
    "coolgreywarm", ["#2166ac", "#dddddd", "#b2182b"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.facecolor": "white",
})
# NB: no savefig.bbox="tight" -- it fights constrained layout, which is what
# keeps the long panel titles from colliding across columns.


def baselines(store, properties):
    labels = probe.load_labels(store)
    n = len(labels["n_atoms"])
    train, test = probe.make_split(n)
    C = probe.composition_matrix(labels)
    S = np.column_stack([np.ones(n), labels["n_atoms"].astype(np.float64)])
    out = {}
    for p in properties:
        y = labels[p].astype(np.float64)
        bc, *_ = np.linalg.lstsq(C[train], y[train], rcond=None)
        bs, *_ = np.linalg.lstsq(S[train], y[train], rcond=None)
        out[p] = (probe.r2(y[test], C[test] @ bc),
                  probe.r2(y[test], S[test] @ bs))
    return out


def grid(df, prop, variant, fs, layers, timesteps, col="r2"):
    """(n_layers, n_timesteps) with timesteps ordered 500 -> 50, NaN if absent."""
    sub = df[(df["property"] == prop) & (df["variant"] == variant)
             & (df["feature_set"] == fs)]
    M = np.full((len(layers), len(timesteps)), np.nan)
    for _, r in sub.iterrows():
        M[layers.index(int(r["layer"])), timesteps.index(int(r["timestep"]))] = r[col]
    return M


def draw(ax, M, below, timesteps, layers, cmap, norm):
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", origin="lower")
    # a thin surface gap between cells, per the mark spec
    ax.set_xticks(np.arange(-0.5, len(timesteps), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    # second, non-colour encoding: hatched = did not beat its own floor
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if below[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                    edgecolor="#999999", linewidth=0.0))
    ax.set_xticks(range(len(timesteps)))
    ax.set_xticklabels(timesteps, rotation=90)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def label_best(ax, M):
    if np.all(np.isnan(M)):
        return
    i, j = np.unravel_index(np.nanargmax(M), M.shape)
    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
            color="white" if M[i, j] > np.nanmax(M) * 0.6 else INK,
            fontweight="bold")


def panels(df, properties, variant, fs, layers, timesteps, base, cmap, norm,
           title, cbar_label, path):
    ncol = 3
    nrow = int(np.ceil(len(properties) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow),
                             squeeze=False, layout="constrained")
    im = None
    for k, prop in enumerate(properties):
        ax = axes[k // ncol][k % ncol]
        M = grid(df, prop, variant, fs, layers, timesteps)
        P95 = grid(df, prop, variant, fs, layers, timesteps, col="r2_null_p95")
        below = ~(M > P95)
        below[np.isnan(M)] = False
        im = draw(ax, M, below, timesteps, layers, cmap, norm)
        label_best(ax, M)
        comp, atoms = base[prop]
        # Short enough not to collide with the next column's title. The full
        # words are in the caption; the panel only needs the two numbers a
        # reader compares the cells against.
        ax.set_title(f"{prop}\ncomp {comp:.2f} · size {atoms:.2f}",
                     color=INK, fontsize=8.5)
        if k % ncol == 0:
            ax.set_ylabel("layer")
        if k // ncol == nrow - 1:
            ax.set_xlabel("timestep (noisy -> clean)")
    for k in range(len(properties), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    # The hatch note rides in the suptitle: constrained layout reserves no
    # room for a free-floating figtext, so one placed below the axes gets
    # clipped or lands on the tick labels.
    fig.suptitle(title + "\nhatched = does not exceed the 95th percentile of "
                 "its own permutation floor", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=axes, shrink=0.72, pad=0.02)
    cb.set_label(cbar_label, color=MUTED)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def channels(df, properties, layers, base, outdir):
    """Feature set x layer at each property's best timestep.

    A heatmap rather than ten coloured series: past about eight categories,
    cycled hues stop being distinguishable and the chart lies.
    """
    fsets = [f for f in df["feature_set"].unique() if f != MAIN_FS] + [MAIN_FS]
    ncol, nrow = 3, int(np.ceil(len(properties) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow),
                             squeeze=False, layout="constrained")
    # Build every panel's matrix FIRST, so one norm can cover all of them.
    # A per-panel norm with a single shared colourbar would mean the bar
    # describes only the last panel while each one is on its own scale, which
    # is worse than having no colourbar at all.
    built = {}
    for prop in properties:
        sub = df[(df["property"] == prop) & (df["variant"] == "size_residual")]
        if not len(sub):
            continue
        best_t = int(sub.loc[sub["r2"].idxmax(), "timestep"])
        s = sub[sub["timestep"] == best_t]
        M = np.full((len(fsets), len(layers)), np.nan)
        B = np.zeros_like(M, dtype=bool)
        for _, r in s.iterrows():
            i, j = fsets.index(r["feature_set"]), layers.index(int(r["layer"]))
            M[i, j] = r["r2"]
            B[i, j] = not (r["r2"] > r["r2_null_p95"])
        built[prop] = (best_t, M, B)

    if built:
        allv = np.concatenate([m[np.isfinite(m)].ravel() for _, m, _ in built.values()])
        norm = TwoSlopeNorm(vmin=float(min(-1e-6, allv.min())), vcenter=0.0,
                            vmax=float(max(1e-6, allv.max())))
    else:
        norm = TwoSlopeNorm(vmin=-1e-6, vcenter=0.0, vmax=1e-6)

    im = None
    for k, prop in enumerate(properties):
        ax = axes[k // ncol][k % ncol]
        if prop not in built:
            ax.axis("off")
            continue
        best_t, M, B = built[prop]
        im = draw(ax, M, B, layers, fsets, DIV, norm)
        ax.set_yticks(range(len(fsets)))
        ax.set_yticklabels(fsets, fontsize=6.5)
        ax.set_xlabel("layer")
        ax.set_title(f"{prop} at t={best_t}\ncomposition {base[prop][0]:.2f}",
                     color=INK, fontsize=8.5)
    for k in range(len(properties), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    # No y=1.00: constrained layout places the suptitle itself, and forcing a
    # position drops it on top of the middle panel's title.
    fig.suptitle("Size-residual R2 by channel, at each property's best "
                 "timestep\nhatched = does not exceed its own permutation "
                 "floor", fontsize=10, color=INK)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.72, pad=0.02)
        cb.set_label("size-residual R2", color=MUTED)
        cb.outline.set_visible(False)
    path = os.path.join(outdir, "figure2_channels.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main(store, csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    man = probe.load_manifest(store)
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    layers = sorted(df["layer"].unique().tolist())
    timesteps = sorted(df["timestep"].unique().tolist(), reverse=True)
    properties = [p for p in ("polarisability", "heat_capacity", "dipole_moment",
                              "homo", "lumo", "homo_lumo_gap")
                  if p in set(df["property"])]
    base = baselines(store, properties)

    full = int(man["n_layers"]) * len(man["timesteps"])
    have = df.groupby(["layer", "t_index"]).ngroups
    if have < full:
        print(f"PARTIAL: {have}/{full} cells present; missing cells are blank")

    panels(df, properties, "raw", MAIN_FS, layers, timesteps, base,
           plt.get_cmap(SEQ), plt.Normalize(0.0, 1.0),
           "Figure 1a. Held-out R2 from activations, raw property",
           "held-out R2", os.path.join(outdir, "figure1_raw.png"))

    res = df[(df["variant"] == "size_residual") & (df["feature_set"] == MAIN_FS)]
    lo = float(min(-1e-6, res["r2"].min())) if len(res) else -0.1
    hi = float(max(1e-6, res["r2"].max())) if len(res) else 0.1
    panels(df, properties, "size_residual", MAIN_FS, layers, timesteps, base,
           DIV, TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
           "Figure 1b. Held-out R2 after removing the linear composition "
           "component", "size-residual R2",
           os.path.join(outdir, "figure1_residual.png"))

    channels(df, properties, layers, base, outdir)

    keep = df[df["feature_set"].notna()]
    out_csv = os.path.join(outdir, "figure1_data.csv")
    keep.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({len(keep):,} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
