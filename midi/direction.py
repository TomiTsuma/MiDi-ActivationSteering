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
