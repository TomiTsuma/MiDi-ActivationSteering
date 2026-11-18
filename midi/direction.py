"""midi/direction.py -- contrastive steering directions from the store.

The direction is the difference of mean activations between the high and low
halves of a property, computed in the UNSTANDARDISED X_mean block so it can be
added straight into the model's forward pass.

Three facts drive the design.

1. Only three of the ten feature sets are injectable. `X_mean`, `E_mean` and
   `y` have a well-defined "add v" upstream of the pooling; min, max and var
   do not, and `all` is a concatenation of statistics, not an activation.
   pool_nodes uses a MASKED mean (denom = node_mask.sum), so adding v to every
   real atom shifts X_mean by exactly v, shifts X_min and X_max by exactly v
   too, and leaves X_var exactly unchanged.

2. The coefficient is expressed in units of ||d||, the low-to-high gap, not in
   raw activation units. c = 1 then means "move this molecule by the full
   average gap between the low and high halves", which transfers across sites,
   layers and properties. The identity that makes this exact is v . d = ||d||:
   the gap between the mean projections of the two halves onto the unit
   direction is exactly ||d||.

3. Splitting on the RAW property would bake in the size confound. Composition
   alone explains 0.871 of polarisability, so a naive high/low split separates
   big molecules from small ones and the direction is substantially a size
   direction. This module computes both and reports the angle between them, so
   the choice is measured rather than assumed.
"""
import json
import os

import numpy as np

from midi import probe

INJECTABLE = ("X_mean", "E_mean", "y")


def load_block(store, layer, t_index, block="X_mean"):
    """(N, d) raw activations for one block of one cell. No standardisation."""
    man = probe.load_manifest(store)
    if block not in man["feature_slices"]:
        raise SystemExit(f"{block} is not in the store; have "
                         f"{sorted(man['feature_slices'])}")
    if block not in INJECTABLE:
        raise SystemExit(
            f"{block} is not injectable. Only {INJECTABLE} have a defined "
            f"'add v' upstream of the pooling; you cannot add to a min, a max "
            f"or a variance.")
    lo, hi = man["feature_slices"][block]
    mm = probe.open_layer(store, layer)
    return np.asarray(mm[t_index][:, lo:hi], dtype=np.float64)


def diff_of_means(A, y, q=0.5, min_per_side=10):
    """mean(high) - mean(low). q=0.5 is the median split the item specifies.

    Returns (d, n_high, n_low). Smaller q (0.25, say) contrasts the tails and
    gives a cleaner direction at the cost of using fewer molecules.

    min_per_side guards against a split that is degenerate on real data, where
    a handful of molecules either side would give a direction made of noise.
    Lower it only for toy inputs in tests.
    """
    lo_thr, hi_thr = np.quantile(y, q), np.quantile(y, 1.0 - q)
    L, H = y <= lo_thr, y >= hi_thr
    if H.sum() < min_per_side or L.sum() < min_per_side:
        raise ValueError(f"degenerate split: {int(H.sum())} high, "
                         f"{int(L.sum())} low, need {min_per_side} each")
    return A[H].mean(axis=0) - A[L].mean(axis=0), int(H.sum()), int(L.sum())


def random_unit(dim, seed):
    """A seeded draw from the unit sphere in R^dim.

    The first control arm of Section 3.8. Gaussian coordinates normalised to
    length one is the only construction that is uniform on the sphere; drawing
    each coordinate uniformly from [-1, 1] and normalising concentrates on the
    diagonals and would make the control easier to beat than chance.

    Seeded, and the seed goes in the manifest, so the control direction
    regenerates from the record alone rather than having to be shipped.
    """
    g = np.random.default_rng(seed)
    u = g.standard_normal(int(dim))
    return u / np.linalg.norm(u)


def shuffled_direction(A, y, seed=None, q=0.5, min_per_side=10, perm=None):
    """diff_of_means on PERMUTED labels: the second control arm.

    The geometry of the construction is identical to the real direction. Same
    activations, same median split, same quantile, same minimum per side. Only
    the pairing between a molecule and its property is destroyed, which is the
    one thing the real direction is supposed to be using.

    `perm` exists so a test can pass the identity permutation and require the
    result to equal diff_of_means exactly. If that gate fails, the two arms
    differ in more than their labelling and the control proves nothing.
    """
    y = np.asarray(y, dtype=np.float64)
    if perm is None:
        if seed is None:
            raise ValueError("shuffled_direction needs a seed or an explicit perm")
        perm = np.random.default_rng(seed).permutation(len(y))
    perm = np.asarray(perm, dtype=int)
    if perm.shape != y.shape:
        raise ValueError(f"perm has shape {perm.shape}, labels have {y.shape}")
    return diff_of_means(A, y[perm], q, min_per_side)


def probe_weight_raw(A, y, train, test):
    """The ridge probe's own direction, mapped back to RAW activation space.

    The probe fits on standardised features, z_i = (a_i - mu_i)/sd_i, so the
    gradient of its prediction with respect to the RAW activation is w_i/sd_i,
    not w_i. Comparing w to a difference-of-means without that division
    compares two different spaces and the cosine is meaningless.
    """
    Ztr, Zte, d_used, _ = probe.standardize(A[train], A[test])
    s = probe.CellSolver(Ztr)
    g, yc, ym = s.prepare(y[train])
    lam, _, _ = s.choose_lam(g, yc)
    w = s.coef_from_g(g, lam)
    r2 = probe.r2(y[test], Zte @ w + ym)

    sd = A[train].std(axis=0)
    keep = sd > 1e-12
    w_raw = np.zeros(A.shape[1], dtype=np.float64)
    w_raw[keep] = w / sd[keep]
    return w_raw, float(lam), float(r2)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def build(store, layer, prop, block="X_mean", q=0.5, ref_t=50):
    """Directions at EVERY captured timestep, plus the diagnostics.

    One direction per captured timestep rather than a single averaged one: the
    representation can rotate between timesteps, and averaging would quietly
    blunt the direction without saying so. The cross-timestep cosine matrix
    reports how much it rotates, so holding one direction across a window is a
    decision you can defend with a number.
    """
    man = probe.load_manifest(store)
    labels = probe.load_labels(store)
    timesteps = list(man["timesteps"])
    n = int(man["n_molecules"])
    train, test = probe.make_split(n)

    y_raw = labels[prop].astype(np.float64)
    C = probe.composition_matrix(labels)
    y_res, _ = probe.residualize(y_raw, C, train)

    D_raw = np.zeros((len(timesteps), 0))
    rows = []
    d_raw_all, d_res_all = [], []
    for ti, t in enumerate(timesteps):
        A = load_block(store, layer, ti, block)
        d_raw, nh, nl = diff_of_means(A, y_raw, q)
        d_res, _, _ = diff_of_means(A, y_res, q)
        d_raw_all.append(d_raw)
        d_res_all.append(d_res)
        rows.append({
            "t": t, "t_index": ti,
            "norm_raw": float(np.linalg.norm(d_raw)),
            "norm_resid": float(np.linalg.norm(d_res)),
            "cos_raw_resid": cosine(d_raw, d_res),
            "n_high": nh, "n_low": nl,
        })

    d_raw_all = np.asarray(d_raw_all)
    d_res_all = np.asarray(d_res_all)

    # How much does the direction rotate between captured timesteps?
    nt = len(timesteps)
    cos_t = np.eye(nt)
    for i in range(nt):
        for j in range(i + 1, nt):
            cos_t[i, j] = cos_t[j, i] = cosine(d_res_all[i], d_res_all[j])

    ref_i = timesteps.index(ref_t)
    A_ref = load_block(store, layer, ref_i, block)
    w_raw, lam, r2 = probe_weight_raw(A_ref, y_res, train, test)

    return {
        "block": block, "layer": layer, "property": prop, "split_q": q,
        "timesteps": np.asarray(timesteps),
        "d_raw": d_raw_all, "d_resid": d_res_all,
        "cos_across_t_resid": cos_t,
        "w_probe_raw": w_raw,
        "cos_w_vs_d_resid": cosine(w_raw, d_res_all[ref_i]),
        "cos_w_vs_d_raw": cosine(w_raw, d_raw_all[ref_i]),
        "ref_t": ref_t, "probe_lam": lam, "probe_r2_resid": r2,
        "per_t": rows,
        "store_commit": man.get("git_commit", ""),
        "store_run_seed": man.get("run_seed", -1),
        "n_molecules": n,
    }


def load_all_blocks(store, layer, block="X_mean"):
    """Every captured timestep's activations for one block, loaded once.

    13,055 x 256 float64 is 26.7 MB per timestep, and the shuffled control
    needs all ten of them once per permutation. Loading inside build_control
    would re-read 1.6 GB for six control arms, and pay it again on every
    resume after a crash. The caller loads once and passes the list in.
    """
    man = probe.load_manifest(store)
    return [load_block(store, layer, ti, block)
            for ti in range(len(man["timesteps"]))]


def build_control(store, layer, prop, kind, seed, d_real, block="X_mean",
                  q=0.5, min_per_side=10, blocks=None):
    """Unit directions for one control arm, matched to the real arm's shape.

    Section 3.8 names two controls and they destroy different things.

      kind "rand"  a single seeded draw from the unit sphere, held across the
                   window. Destroys everything: the direction has no relation
                   to the activations or to the labels. It answers "is the
                   effect simply what perturbing activations at this magnitude
                   does?"
      kind "shuf"  the difference of means recomputed from permuted labels.
                   Destroys only the labelling, keeping the activations, the
                   split rule, the quantile and the molecule grouping. It
                   answers the sharper question: "is the effect in this
                   direction's CONTENT?"

    ONE permutation, and one random draw, shared across every captured
    timestep. The real direction at adjacent timesteps has cosine about 0.99
    because the same molecules sit on the same side of the split. A fresh
    permutation per timestep would give mutually near-orthogonal controls, so
    the arm would differ from the real one in its cross-timestep coherence as
    well as in its content, and a difference in outcome could be either.
    Sharing the permutation preserves that structure.

    Returns unit directions only. The caller must inject at the REAL arm's
    norms, not at `norm_own`: a shuffled split gives a difference of means
    made of noise, so ||d_shuf|| is well below ||d_real||, and injecting at
    its own norm would confound different content with smaller magnitude.
    `norm_ratio` is reported instead as a diagnostic, and for kind "shuf" it
    is a direct measure of how much of the real direction is label-driven.
    For kind "rand" it carries no meaning, since norm_own is 1 by definition.
    """
    d_real = np.asarray(d_real, dtype=np.float64)
    man = probe.load_manifest(store)
    timesteps = list(man["timesteps"])
    if d_real.ndim != 2 or d_real.shape[0] != len(timesteps):
        raise ValueError(f"d_real must be ({len(timesteps)}, d); "
                         f"got {d_real.shape}")
    lo, hi = man["feature_slices"][block]
    dim = hi - lo
    if d_real.shape[1] != dim:
        raise ValueError(f"d_real has {d_real.shape[1]} dims, block {block} "
                         f"has {dim}")

    if kind == "rand":
        D = np.repeat(random_unit(dim, seed)[None, :], len(timesteps), axis=0)
    elif kind == "shuf":
        labels = probe.load_labels(store)
        n = int(man["n_molecules"])
        train, _ = probe.make_split(n)
        y_raw = labels[prop].astype(np.float64)
        C = probe.composition_matrix(labels)
        y_res, _ = probe.residualize(y_raw, C, train)
        perm = np.random.default_rng(seed).permutation(n)
        if blocks is None:
            blocks = load_all_blocks(store, layer, block)
        if len(blocks) != len(timesteps):
            raise ValueError(f"got {len(blocks)} blocks for "
                             f"{len(timesteps)} timesteps")
        D = np.stack([
            shuffled_direction(A_t, y_res, q=q, min_per_side=min_per_side,
                               perm=perm)[0]
            for A_t in blocks])
    else:
        raise SystemExit(f"unknown control kind {kind!r}; use 'rand' or 'shuf'")

    norm_own = np.linalg.norm(D, axis=1)
    if not np.all(norm_own > 0):
        raise ValueError(f"control direction {kind} is zero at some timestep")
    nr = np.linalg.norm(d_real, axis=1)

    # What |cos| to the real direction SHOULD this control show? The two arms
    # have different answers and the difference is the point.
    #
    # "rand" is isotropic, so E[cos^2] = 1/d and the rms is 1/sqrt(d).
    # "shuf" is not. A permuted split is still a difference between the means
    # of two halves of the same activations, so d_shuf is a random vector
    # drawn with the DATA's covariance, giving
    #     rms cos(d_shuf, u) = sqrt( u' Sigma u / trace(Sigma) )
    # which is far above 1/sqrt(d) whenever the real direction points
    # somewhere the activation cloud is wide. Measured at layer 10, t=50 on
    # QM9: predicted 0.4069, observed 0.3969 over 300 permutations, against
    # 0.0622 for the unit sphere. See scripts/check_controls.py.
    #
    # So the shuffled arm is a CONSERVATIVE control, sharing the manifold
    # with the real direction by construction, and the random arm is the
    # clean null. Section 3.8 needs both because they bracket the question.
    if kind == "rand":
        cos_pred = np.full(len(timesteps), 1.0 / np.sqrt(dim))
    else:
        cos_pred = np.empty(len(timesteps))
        for i, A_t in enumerate(blocks):
            Ac = A_t - A_t.mean(axis=0)
            u = d_real[i] / np.linalg.norm(d_real[i])
            cos_pred[i] = np.sqrt(np.var(Ac @ u)
                                  / ((Ac ** 2).sum() / len(Ac)))

    return {
        "kind": kind, "seed": int(seed), "layer": int(layer), "block": block,
        "property": prop, "split_q": q,
        "cos_expected_rms": cos_pred,
        "timesteps": np.asarray(timesteps),
        "V": D / norm_own[:, None],
        "norm_own": norm_own,
        "norm_real": nr,
        "norm_ratio": norm_own / np.maximum(nr, 1e-300),
        "cos_to_real": np.array([cosine(D[i], d_real[i])
                                 for i in range(len(timesteps))]),
    }


def save(res, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    arrays["_meta_json"] = np.array(json.dumps(meta))
    with open(path, "wb") as f:          # a handle, not a name: np.savez
        np.savez(f, **arrays)            # appends .npz to a bare string path
    return path


def load(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "_meta_json"}
    out.update(json.loads(str(z["_meta_json"])))
    return out


def unit_for_t(res, t_int, use="resid"):
    """Unit direction and ||d|| at the CAPTURED timestep nearest t_int."""
    ts = np.asarray(res["timesteps"])
    i = int(np.argmin(np.abs(ts - t_int)))
    d = np.asarray(res["d_resid" if use == "resid" else "d_raw"])[i]
    nrm = float(np.linalg.norm(d))
    return d / nrm, nrm, int(ts[i])
