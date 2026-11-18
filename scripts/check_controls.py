"""scripts/check_controls.py -- are the two control arms actually controls?

    python3.11 -u scripts/check_controls.py <store> <dirs> [property]

The dry run showed the shuffled-label direction sitting at |cos| about 0.45
to the real direction, against 0.06 for a draw from the unit sphere. Thirty
times chance needs an explanation before a 15-hour sweep is launched on it.

The explanation to test: a shuffled split is still a difference between the
means of two halves of the SAME activations, so

    d_shuf  ~  (2/n) sum_i eps_i a_i ,   eps_i = +-1 balanced

which is a random vector drawn with the data's covariance, not an isotropic
one. For a fixed unit vector u,

    E[(u . d_shuf)^2]  propto  u' Sigma u
    E[||d_shuf||^2]    propto  trace(Sigma)
    so   rms cos(d_shuf, u)  =  sqrt( u' Sigma u / trace(Sigma) )

where Sigma is the covariance of the pooled activations about their mean and
u' Sigma u is the variance along u. If the real direction points somewhere
the activation cloud is wide, that ratio is far above 1/d and the shuffled
arm is PARTLY the real direction by construction.

That is not a bug and not a reason to drop the arm. It decides how the arm is
described: a conservative control, which the real arm must beat despite
sharing the manifold, rather than a clean null.

CAVEAT on the formula. It is a ratio of EXPECTATIONS, and it equals the
expectation of the ratio only while the share is small, because (u.d)^2 and
||d||^2 become strongly correlated once u carries most of the variance.
Measured: at a share of 0.67 the prediction overshoots the observed rms by 19
percent, at this store's 0.17 it is within 2.5 percent. So if the share
printed below is above about 0.4, treat the agreement check as uninformative
rather than as a failure, and compare the empirical columns directly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import direction, probe

LAYER = 10
BLOCK = "X_mean"
REF_T = 50
N_DRAWS = 300


def main(store, dirs, prop="polarisability"):
    store, dirs = os.path.abspath(store), os.path.abspath(dirs)
    man = probe.load_manifest(store)
    labels = probe.load_labels(store)
    ts = list(man["timesteps"])
    ti = ts.index(REF_T)
    n = int(man["n_molecules"])

    A = direction.load_block(store, LAYER, ti, BLOCK)
    res = direction.load(os.path.join(dirs, f"dir_L{LAYER}_{BLOCK}_{prop}.npz"))
    d_real = np.asarray(res["d_resid"])[ti]
    v_real = d_real / np.linalg.norm(d_real)

    train, _ = probe.make_split(n)
    y_raw = labels[prop].astype(np.float64)
    y_res, _ = probe.residualize(y_raw, probe.composition_matrix(labels), train)

    print(f"store {store}")
    print(f"site  layer {LAYER}, {BLOCK}, t={REF_T}, n={n}, d={A.shape[1]}")
    print(f"      ||d_real|| = {np.linalg.norm(d_real):.4f}")
    print()

    # ---- the covariance prediction -------------------------------------
    Ac = A - A.mean(axis=0)
    var_along_v = float(np.var(Ac @ v_real))
    trace_sigma = float((Ac ** 2).sum() / len(Ac))
    pred = np.sqrt(var_along_v / trace_sigma)
    iso = 1.0 / np.sqrt(A.shape[1])
    print("PREDICTION from the covariance")
    print(f"  variance along v_real       {var_along_v:.6g}")
    print(f"  trace(Sigma)                {trace_sigma:.6g}")
    print(f"  share of total variance     {var_along_v / trace_sigma:.4f}"
          f"   (isotropic would be {1 / A.shape[1]:.5f})")
    print(f"  predicted rms |cos|         {pred:.4f}")
    print(f"  isotropic rms |cos| 1/sqrt(d) {iso:.4f}")
    print(f"  ratio                       {pred / iso:.1f}x chance")
    print()

    # ---- empirical: permuted labels ------------------------------------
    cs_shuf, ratios = [], []
    for s in range(N_DRAWS):
        d_s, _, _ = direction.shuffled_direction(A, y_res, seed=10_000 + s)
        cs_shuf.append(direction.cosine(d_s, d_real))
        ratios.append(np.linalg.norm(d_s) / np.linalg.norm(d_real))
    cs_shuf, ratios = np.array(cs_shuf), np.array(ratios)

    # ---- empirical: isotropic draws ------------------------------------
    cs_rand = np.array([direction.cosine(direction.random_unit(A.shape[1],
                                                               20_000 + s),
                                         d_real) for s in range(N_DRAWS)])

    def line(name, c):
        print(f"  {name:<22} rms {np.sqrt((c ** 2).mean()):.4f}   "
              f"mean|cos| {np.abs(c).mean():.4f}   "
              f"max|cos| {np.abs(c).max():.4f}   "
              f"P(|cos|>0.25) {np.mean(np.abs(c) > 0.25):.3f}")

    print(f"EMPIRICAL over {N_DRAWS} draws")
    line("shuffled labels", cs_shuf)
    line("unit sphere", cs_rand)
    print(f"  shuffled/real norm     mean {ratios.mean():.4f}   "
          f"sd {ratios.std():.4f}")
    print()

    obs = float(np.sqrt((cs_shuf ** 2).mean()))
    rel = abs(obs - pred) / pred
    print(f"VERDICT  predicted rms {pred:.4f} vs observed {obs:.4f} "
          f"({100 * rel:.1f}% apart)")
    if rel < 0.25:
        print("  The covariance prediction holds. The shuffled arm shares the")
        print("  activation manifold with the real direction by construction,")
        print("  so it is a CONSERVATIVE control, not a clean null. Report it")
        print("  that way: the real arm has to beat a direction that already")
        print(f"  carries about {obs:.2f} of its orientation.")
    else:
        print("  The covariance prediction does NOT hold. Something other")
        print("  than the data covariance is aligning the shuffled direction")
        print("  with the real one; do not launch until it is understood.")

    # ---- where does v_real sit in the spectrum? ------------------------
    # Cheap: the top few eigenvalues via the Gram trick is overkill at d=256,
    # so do the full eigendecomposition of the 256x256 covariance.
    S = (Ac.T @ Ac) / len(Ac)
    w, Q = np.linalg.eigh(S)
    w, Q = w[::-1], Q[:, ::-1]
    load = (Q.T @ v_real) ** 2
    print()
    print("WHERE v_real SITS in the activation spectrum")
    print(f"  top-1 PC holds {w[0] / w.sum():.3f} of the variance, "
          f"top-10 {w[:10].sum() / w.sum():.3f}")
    print(f"  v_real puts {load[:1].sum():.3f} of its length^2 on PC1, "
          f"{load[:10].sum():.3f} on the top 10")
    k = int(np.searchsorted(np.cumsum(load), 0.9) + 1)
    print(f"  {k} PCs are needed to hold 90% of v_real")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or []))
