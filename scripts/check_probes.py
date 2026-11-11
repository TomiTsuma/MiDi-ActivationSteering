"""scripts/check_probes.py -- does the probe grid hold up?

    python3.11 -u scripts/check_probes.py <store> <probe_r2.csv>

Exits non-zero if a HARD check fails. Soft findings are printed and counted
but do not fail the run, because some of them are judgement calls.

The point of this script is that wrong probe code does not crash, it reports.
probe.py once returned R2 = -79.8 with a confidently chosen lambda and nothing
in the output looked like an error, so the results get audited the same way
the store did.

Works on a partial CSV, so it can be run while the grid is still going.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import probe

T_MAX = 500                 # the noisiest captured step
MONO_TOL = 0.02             # an R2 drop larger than this counts as a violation
RESID_AT_TMAX_TOL = 0.05    # size-residual R2 allowed at t=500
BH_Q = 0.05

hard_failures = []
soft_findings = []


def hard(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        hard_failures.append(msg)


def soft(ok, msg):
    print(f"  [{'ok  ' if ok else 'WARN'}] {msg}")
    if not ok:
        soft_findings.append(msg)


def baselines(store, properties):
    """Composition-only and ATOM-COUNT-only R2 per property.

    Atom count matters on its own here because the node count is the one thing
    the model never loses: the mask comes from the real molecule, so even at
    t=500, with types, charges and positions destroyed, the activations still
    know exactly how many atoms there are. That makes the atom-count R2 the
    predicted value of the raw column at t=500, and the figure against which
    the size confound is read.
    """
    labels = probe.load_labels(store)
    n = len(labels["n_atoms"])
    train, test = probe.make_split(n)
    ones = np.ones(n)
    C_comp = probe.composition_matrix(labels)
    C_size = np.column_stack([ones, labels["n_atoms"].astype(np.float64)])
    out = {}
    for prop in properties:
        y = labels[prop].astype(np.float64)
        bc, *_ = np.linalg.lstsq(C_comp[train], y[train], rcond=None)
        bs, *_ = np.linalg.lstsq(C_size[train], y[train], rcond=None)
        out[prop] = {
            "comp": probe.r2(y[test], C_comp[test] @ bc),
            "atoms": probe.r2(y[test], C_size[test] @ bs),
        }
    return out


def main(store, csv_path):
    store, csv_path = os.path.abspath(store), os.path.abspath(csv_path)
    man = probe.load_manifest(store)
    fsets = probe.feature_sets(man)
    timesteps = list(man["timesteps"])
    n_layers = int(man["n_layers"])

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    properties = sorted(df["property"].unique()) if len(df) else []
    variants = sorted(df["variant"].unique()) if len(df) else []
    expected = n_layers * len(timesteps) * len(fsets) * len(properties) * len(variants)

    print(f"csv       {csv_path}")
    print(f"rows      {len(df):,}")
    print(f"grid      {n_layers} layers x {len(timesteps)} timesteps x "
          f"{len(fsets)} feature sets x {len(properties)} properties x "
          f"{len(variants)} variants = {expected:,}")
    if not len(df):
        raise SystemExit("no rows yet")

    partial = len(df) < expected
    if partial:
        print(f"\nPARTIAL: {len(df):,}/{expected:,} rows "
              f"({100.0 * len(df) / expected:.1f}%). Checks run on what is here; "
              f"completeness is reported as a soft finding.")

    base = baselines(store, properties)
    print("\nbaselines (independent of any activation cell)")
    print(f"  {'property':<16} {'composition':>12} {'atom count':>12}")
    for p in properties:
        print(f"  {p:<16} {base[p]['comp']:>12.4f} {base[p]['atoms']:>12.4f}")

    key = ["layer", "t_index", "feature_set", "property", "variant"]

    # ---- 1. integrity ---------------------------------------------------
    print("\n1. integrity")
    dupes = df.duplicated(subset=key).sum()
    hard(dupes == 0, f"duplicate row keys: {dupes}")
    hard(df["r2"].notna().all(), f"NaN in r2: {int(df['r2'].isna().sum())}")
    hard(df["r2_null_sd"].gt(0).all(),
         f"non-positive null sd: {int(df['r2_null_sd'].le(0).sum())}")
    soft(not partial, f"completeness: {len(df):,}/{expected:,}")

    # ---- 2. r2_size_only constant per property --------------------------
    print("\n2. r2_size_only depends only on the target and the split")
    for p in properties:
        v = df.loc[df["property"] == p, "r2_size_only"]
        spread = float(v.max() - v.min())
        hard(spread < 1e-9,
             f"{p}: spread across {len(v):,} rows = {spread:.2e} "
             f"(value {v.iloc[0]:.4f}, composition baseline {base[p]['comp']:.4f})")
        soft(abs(float(v.iloc[0]) - base[p]["comp"]) < 1e-6,
             f"{p}: matches the independently computed composition baseline")

    # ---- 3. the t=500 predictions ---------------------------------------
    print(f"\n3. behaviour at t={T_MAX}, where the only surviving information "
          f"is the atom count")
    at_max = df[df["timestep"] == T_MAX]
    if not len(at_max):
        soft(False, f"no rows at t={T_MAX} yet")
    else:
        res = at_max[at_max["variant"] == "size_residual"]
        if len(res):
            worst = float(res["r2"].max())
            bad = res[res["r2"] > RESID_AT_TMAX_TOL]
            hard(len(bad) == 0,
                 f"size_residual at t={T_MAX}: max r2 = {worst:.4f}, "
                 f"{len(bad)} cells above {RESID_AT_TMAX_TOL} "
                 f"(size is removed, so nothing should be predictable)")
        raw = at_max[at_max["variant"] == "raw"]
        for p in properties:
            r = raw[(raw["property"] == p) & (raw["feature_set"] == "all")]
            if not len(r):
                continue
            med = float(r["r2"].median())
            exp = base[p]["atoms"]
            soft(abs(med - exp) < 0.15,
                 f"{p}: raw median r2 at t={T_MAX} = {med:.4f} vs atom-count "
                 f"baseline {exp:.4f} (delta {med - exp:+.4f})")

    # ---- 4. monotonicity in t -------------------------------------------
    print(f"\n4. r2 rises as the timestep falls (500 -> 50)")
    viol = tot = 0
    worst_drop = (0.0, None)
    for grp, g in df.groupby(["layer", "feature_set", "property", "variant"]):
        g = g.sort_values("timestep", ascending=False)
        d = np.diff(g["r2"].to_numpy())
        tot += len(d)
        v = d < -MONO_TOL
        viol += int(v.sum())
        if len(d) and d.min() < -worst_drop[0]:
            worst_drop = (-float(d.min()), grp)
    pct = 100.0 * viol / tot if tot else 0.0
    soft(pct < 5.0,
         f"violations (drop > {MONO_TOL}): {viol}/{tot} steps ({pct:.1f}%), "
         f"worst {worst_drop[0]:.3f} at {worst_drop[1]}")

    # ---- 5. lambda not at a grid edge -----------------------------------
    print("\n5. the lambda grid is wide enough")
    at_edge = df["lam_at_grid_edge"].astype(str).str.lower().eq("true")
    # An edge hit only indicates a too-narrow grid when there is signal to
    # fit. On a target with none -- every size_residual cell at t=500, for
    # instance -- running to maximum shrinkage IS the right answer, and
    # flagging it would bury the real cases under thousands of false alarms.
    has_signal = df["r2"] > df["r2_null_p95"]
    bad = int((at_edge & has_signal).sum())
    benign = int((at_edge & ~has_signal).sum())
    soft(bad == 0, f"lambda at a grid edge WITH signal: {bad} "
                   f"(plus {benign} benign, on targets that beat no floor)")

    # ---- 6. floor calibration -------------------------------------------
    print("\n6. the floor sits just below zero")
    mu = df["r2_null_mean"]
    hard(mu.abs().max() < 0.02,
         f"|r2_null_mean| max = {mu.abs().max():.4f} "
         f"(mean {mu.mean():+.5f}, sd across cells {mu.std():.5f})")
    soft(mu.mean() < 0, f"mean floor is negative as expected: {mu.mean():+.5f}")

    # ---- 7. multiplicity -------------------------------------------------
    print(f"\n7. Benjamini-Hochberg across the whole family at q={BH_Q}")
    p = df["z_p"].to_numpy(dtype=float)
    mask = probe.benjamini_hochberg(p, q=BH_Q)
    print(f"  {int(mask.sum()):,}/{len(df):,} rows survive "
          f"({100.0 * mask.sum() / len(df):.1f}%)")
    print(f"  NOTE z-based p-values assume the null R2 is approximately "
          f"normal. With z around 1000 the p-value is not the interesting "
          f"quantity; effect size against r2_size_only is.")

    # ---- the actual result ----------------------------------------------
    print("\n" + "=" * 78)
    print("TOP CELLS BY SIZE-RESIDUAL R2 (this is the result)")
    print("=" * 78)
    res = df[df["variant"] == "size_residual"]
    for p_ in properties:
        sub = res[res["property"] == p_].nlargest(5, "r2")
        if not len(sub):
            continue
        print(f"\n{p_}  (composition explains {base[p_]['comp']:.3f}, "
              f"atom count {base[p_]['atoms']:.3f})")
        print(f"  {'layer':>5} {'t':>4} {'feature_set':<10} {'r2_resid':>9} "
              f"{'r2_raw':>8} {'floor_p95':>10} {'df':>8} {'alpha':>9}")
        for _, r in sub.iterrows():
            raw_r = df[(df["layer"] == r["layer"]) & (df["t_index"] == r["t_index"])
                       & (df["feature_set"] == r["feature_set"])
                       & (df["property"] == p_) & (df["variant"] == "raw")]
            raw_v = float(raw_r["r2"].iloc[0]) if len(raw_r) else float("nan")
            print(f"  {int(r['layer']):>5} {int(r['timestep']):>4} "
                  f"{r['feature_set']:<10} {r['r2']:>9.4f} {raw_v:>8.4f} "
                  f"{r['r2_null_p95']:>10.4f} {r['df_eff']:>8.1f} "
                  f"{r['alpha_shrinkage']:>9.2e}")

    print("\n" + "=" * 78)
    if hard_failures:
        print(f"HARD FAILURES: {len(hard_failures)}")
        for m in hard_failures:
            print(f"  - {m}")
        raise SystemExit(1)
    print(f"all hard checks passed; {len(soft_findings)} soft finding(s)")
    for m in soft_findings:
        print(f"  - {m}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
