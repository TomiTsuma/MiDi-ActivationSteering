"""midi/probe.py -- ridge probes from activations to properties.

One Gram matrix and one eigendecomposition per (cell, feature set). After that
every lambda, every property and every label permutation is arithmetic on
vectors already in hand, and leave-one-out cross-validation needs no refits.

The model fitted is ridge with an UNPENALISED intercept:

    minimise  sum_j (y_j - b0 - z_j . w)^2  +  lambda * ||w||^2

Centring y on the training mean and fitting with no intercept column is
equivalent to that, and is what this code does. Three consequences, all of
which were wrong in the first draft and produced R2 = -79.8 on a cell whose
true value is 0.989:

  * the intercept must be ADDED BACK before prediction. standardize() makes
    every feature mean-zero, so Z^T y cannot see y's mean: the coefficients
    come out right and the predictions come out centred on zero, while e.g.
    polarisability lives around 75.3.
  * it also corrupts lambda selection. With the target unreachable, the LOO
    criterion runs to maximum shrinkage, so the reported lambda is meaningless
    too, not merely the R2.
  * the intercept contributes 1/n to every hat-matrix diagonal and 1 to the
    effective degrees of freedom. Negligible at n = 10,444, but free to be
    exact about.
"""
import numpy as np

import json
import os

SPLIT_SEED = 20261111
TEST_FRAC = 0.20
LAM_GRID = np.logspace(-6, 3, 19)     # lam is lambda / mean eigenvalue

COMPOSITION_ELEMENTS = ("H", "C", "N", "O", "F")


def composition_matrix(labels, elements=COMPOSITION_ELEMENTS):
    """[1, n_H, n_C, n_N, n_O, n_F] -- the size control's basis.

    These six columns already span n_atoms and n_heavy as linear combinations,
    so adding either would add nothing and make C^T C singular. lstsq is used
    downstream rather than an explicit inverse so that a degenerate split (no
    fluorine anywhere, say) degrades instead of exploding.
    """
    n = len(labels["n_atoms"])
    cols = [np.ones(n)] + [labels[f"n_{e}"].astype(np.float64) for e in elements]
    return np.column_stack(cols)


def residualize(y, C, train):
    """y minus its linear composition component, with the fit on TRAIN ONLY.

    Fitting on all molecules would let the test labels define the target,
    which is a real leak. The price is that the residual is exactly orthogonal
    to C on the training set and only approximately so on the held-out set.
    """
    b, *_ = np.linalg.lstsq(C[train], y[train], rcond=None)
    return y - C @ b, b


def permute_targets(y, n_perm, seed):
    """(n, B) matrix of permuted copies of the FULL target vector.

    The whole vector is permuted, not the training part: the null is "this
    target carries no information about these activations", so the association
    must be absent everywhere, not just where the probe is fitted.
    """
    rs = np.random.RandomState(seed)
    n = len(y)
    Y = np.empty((n, n_perm), dtype=np.float64)
    for b in range(n_perm):
        Y[:, b] = y[rs.permutation(n)]
    return Y


def make_split(n, test_frac=TEST_FRAC, seed=SPLIT_SEED):
    """One split on MOLECULE index, reused for every cell and property.

    Fixed so that differences between cells are differences between cells and
    not differences between splits. Splitting on molecule index also means no
    molecule can straddle train and test across the 10 timesteps at which it
    appears. Sorted on return, which keeps memmap reads sequential.
    """
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    n_test = int(round(test_frac * n))
    test, train = perm[:n_test], perm[n_test:]
    assert not set(train.tolist()) & set(test.tolist())
    return np.sort(train), np.sort(test)


def standardize(A_train, A_test, tol=1e-12):
    """Centre and scale on TRAINING statistics; drop constant columns.

    Population sd (ddof=0) on purpose: it makes sum_j z_ji^2 == n exactly for
    every kept column, so trace(Z^T Z) == n*d and the mean eigenvalue is
    exactly n. That is what makes `lam` = lambda/n a dimensionless shrinkage
    strength, comparable across cells of different widths. Verified:
    mean_eig/n = 1.000000000000 at layer 11, t=50.

    Constant columns are dropped rather than given sd=1, since they carry no
    information and would overstate d in the molecules-per-dimension
    bookkeeping Section 3.10 asks for. On the full training set they are rare
    (none at layer 11, t=50); a 64-molecule spot check suggested otherwise,
    which is why the count is measured per cell and reported rather than
    assumed.
    """
    mu = A_train.mean(axis=0)
    sd = A_train.std(axis=0)
    keep = sd > tol
    Ztr = (A_train[:, keep] - mu[keep]) / sd[keep]
    Zte = (A_test[:, keep] - mu[keep]) / sd[keep]
    return Ztr, Zte, int(keep.sum()), int((~keep).sum())


class CellSolver:
    """Everything for one (cell, feature set) that does not depend on the target.

    Build once, then every target and every lambda is cheap. Peak memory is
    about 2 * n * d * 8 bytes, roughly 235 MB at n = 10,444 and d = 1408, so
    build one at a time and let it go.
    """

    def __init__(self, Z, check=True):
        self.Z = np.ascontiguousarray(Z, dtype=np.float64)
        self.n, self.d = self.Z.shape
        G = self.Z.T @ self.Z                       # (d, d) Gram matrix
        s2, V = np.linalg.eigh(G)                   # symmetric: real, ascending
        self.s2 = np.clip(s2, 0.0, None)            # tiny negatives are rounding
        self.V = V
        # Only the SQUARES of P = Z V are ever needed, for the hat diagonal.
        # Keeping P2 makes hat_diag one matrix-vector product instead of an
        # (n, d) temporary allocated on every call: at these sizes that is a
        # 118 MB allocation avoided 19 times per lambda search. Squared in
        # place so the peak stays at two arrays, not three.
        P = self.Z @ V
        np.square(P, out=P)
        self.P2 = P
        self.mean_eig = float(self.s2.sum() / self.d)
        if check:
            assert abs(self.mean_eig / self.n - 1.0) < 1e-8, (
                f"mean eigenvalue {self.mean_eig:.6g} != n {self.n}; Z was not "
                "standardised with population sd, so `lam` is not dimensionless")

    def _lam(self, lam_rel):
        return float(lam_rel) * self.n              # convention of ridge_r2

    # ---- target-specific precompute -------------------------------------
    def prepare(self, y):
        """Project the target once: g = V^T Z^T y_centred.

        Returns (g, y_centred, y_mean). Only training targets are passed here,
        so the mean is the training mean. Doing this once rather than inside
        the lambda loop removes 18 of the 19 Z^T y products.
        """
        y = np.asarray(y, dtype=np.float64)
        ym = float(y.mean())
        yc = y - ym
        return self.V.T @ (self.Z.T @ yc), yc, ym

    def coef_from_g(self, g, lam_rel):
        """Ridge coefficients. `g` must come from prepare(), not a raw y."""
        return self.V @ (g / (self.s2 + self._lam(lam_rel)))

    def hat_diag(self, lam_rel):
        """Diagonal of H(lam), including the unpenalised intercept's 1/n."""
        return self.P2 @ (1.0 / (self.s2 + self._lam(lam_rel))) + 1.0 / self.n

    def df(self, lam_rel):
        """trace H(lam): penalised directions plus 1 for the intercept."""
        lam = self._lam(lam_rel)
        return float((self.s2 / (self.s2 + lam)).sum()) + 1.0

    def loo_sse(self, g, yc, lam_rel):
        """Exact leave-one-out sum of squares. No refits.

        e_j = (y_j - yhat_j) / (1 - h_jj) is exact for any linear smoother,
        and y_j - yhat_j equals yc_j - z_j.w because the intercept cancels.
        """
        resid = yc - self.Z @ self.coef_from_g(g, lam_rel)
        denom = np.clip(1.0 - self.hat_diag(lam_rel), 1e-12, None)
        return float(((resid / denom) ** 2).sum())

    def choose_lam(self, g, yc, grid=LAM_GRID):
        sse = np.array([self.loo_sse(g, yc, L) for L in grid])
        k = int(np.argmin(sse))
        return float(grid[k]), sse, k in (0, len(grid) - 1)

    def prepare_many(self, Y):
        """Column-wise prepare(). Y is (n, B). Returns (G (d,B), Yc, ym (B,))."""
        Y = np.asarray(Y, dtype=np.float64)
        ym = Y.mean(axis=0)
        Yc = Y - ym
        return self.V.T @ (self.Z.T @ Yc), Yc, ym

    def coefs_from_G(self, G, lam_rel):
        return self.V @ (G / (self.s2 + self._lam(lam_rel))[:, None])

    def loo_sse_many(self, G, Yc, lam_rel):
        """(B,) of exact LOO sums of squares. The hat diagonal is shared."""
        R = Yc - self.Z @ self.coefs_from_G(G, lam_rel)
        denom = np.clip(1.0 - self.hat_diag(lam_rel), 1e-12, None)[:, None]
        return ((R / denom) ** 2).sum(axis=0)

    def choose_lam_many(self, G, Yc, grid=LAM_GRID):
        sse = np.stack([self.loo_sse_many(G, Yc, L) for L in grid])  # (n_lam, B)
        k = sse.argmin(axis=0)
        return np.asarray(grid)[k], sse

def r2(y_true, y_pred):
    """1 - SS_res/SS_tot, with SS_tot taken about the TEST mean.

    The test mean is the conventional denominator and is what
    verify_store.ridge_r2 uses, so these numbers stay comparable with the
    figures already on record. It is marginally more generous than the
    training mean, which is the strictly out-of-sample baseline.
    """
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def alpha_from_lam(lam_rel):
    """Section 3.10's shrinkage coefficient.

    Ridge adds lambda*I to Z^T Z, which is the covariance estimate shrunk
    toward a scaled identity. Matching (1-a)*Sigma + a*mu*I with mu = 1 for
    standardised features and lambda = lam*n gives a = lam/(1+lam).
    """
    return float(lam_rel / (1.0 + lam_rel))


def fit_cell(solver, Zte, y_train, y_test, grid=LAM_GRID):
    """Choose lambda by exact LOO on the training set, then evaluate once."""
    g, yc, ym = solver.prepare(y_train)
    lam, sse, at_edge = solver.choose_lam(g, yc, grid)
    w = solver.coef_from_g(g, lam)
    pred = Zte @ w + ym                      # intercept added back
    return {
        "lam": lam,
        "alpha_shrinkage": alpha_from_lam(lam),
        "df_eff": solver.df(lam),
        "df_frac": solver.df(lam) / (solver.d + 1.0),
        "d_used": solver.d,
        "n_train": solver.n,
        "n_per_dim": solver.n / solver.d,
        "r2": r2(np.asarray(y_test, dtype=np.float64), pred),
        "loo_sse_at_lam": float(sse.min()),
        "lam_at_grid_edge": bool(at_edge),
    }


def fit_many(solver, Zte, Y_train, Y_test, grid=LAM_GRID):
    """Held-out R2 for B targets at once, each with its own chosen lambda.

    Columns are grouped by their selected lambda. Null targets cluster on the
    same heavily-shrunk value, so there are usually only a few groups and the
    per-column selection costs almost nothing.
    """
    G, Yc, ym = solver.prepare_many(Y_train)
    lam, _ = solver.choose_lam_many(G, Yc, grid)
    out = np.empty(Y_train.shape[1])
    for L in np.unique(lam):
        m = np.flatnonzero(lam == L)
        W = solver.coefs_from_G(G[:, m], L)
        pred = Zte @ W + ym[m]
        for i, j in enumerate(m):
            out[j] = r2(Y_test[:, j], pred[:, i])
    return out, lam


def floor_stats(r2_obs, r2_null):
    """Empirical and parametric summaries of the null.

    perm_p is floored at 1/(B+1): with B permutations nothing smaller is
    resolvable, so report it as that bound rather than as zero. z_score and
    z_p use the null's mean and sd, which 200 draws estimate well even though
    they cannot resolve the tail, and that is what feeds Benjamini-Hochberg.
    z_p assumes the null R2 is approximately normal, which it is as a sum over
    2,611 test molecules, but it IS an assumption and should be stated.
    """
    from scipy.stats import norm
    B = len(r2_null)
    mu, sd = float(r2_null.mean()), float(r2_null.std(ddof=1))
    exceed = int((r2_null >= r2_obs).sum())
    z = (r2_obs - mu) / sd if sd > 0 else np.inf
    return {
        "r2_null_mean": mu,
        "r2_null_sd": sd,
        "r2_null_p95": float(np.percentile(r2_null, 95)),
        "n_perm": B,
        "perm_exceed": exceed,
        "perm_p": (1 + exceed) / (B + 1),
        "perm_p_is_bound": exceed == 0,
        "z_score": float(z),
        "z_p": float(norm.sf(z)),
    }


def benjamini_hochberg(p, q=0.05):
    """Returns a boolean mask of rejections controlling FDR at q."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) * q) / m
    passing = np.flatnonzero(p[order] <= thresh)
    mask = np.zeros(m, dtype=bool)
    if len(passing):
        mask[order[: passing[-1] + 1]] = True
    return mask

def load_manifest(store):
    with open(os.path.join(store, "manifest.json")) as f:
        return json.load(f)


def feature_sets(manifest, include_all=True):
    """{name: (lo, hi)} from the manifest, plus the concatenation.

    Read from the manifest rather than hardcoded, so a store captured with a
    different layout cannot be silently misread.
    """
    fs = {k: tuple(v) for k, v in manifest["feature_slices"].items()}
    if include_all:
        fs["all"] = (0, int(manifest["features"]))
    return fs


def open_layer(store, layer):
    return np.load(os.path.join(store, f"pooled_L{layer:02d}.npy"), mmap_mode="r")


def read_cell(mm, t_index):
    """One (N, F) float32 slice. t_index indexes the manifest's timestep LIST.

    mm[0] is t=500 and mm[9] is t=50, so passing a bare integer you worked out
    by eye is how the figure gets inverted. Callers pass an index obtained from
    enumerate(manifest["timesteps"]) and carry the t VALUE alongside it.
    """
    return np.asarray(mm[t_index])

# ---------------------------------------------------------------------------
# Store reader
# ---------------------------------------------------------------------------

def load_manifest(store):
    import json
    with open(os.path.join(store, "manifest.json")) as f:
        return json.load(f)


def load_labels(store):
    return dict(np.load(os.path.join(store, "labels.npz"), allow_pickle=False))


def feature_sets(manifest, include_all=True):
    """{name: (lo, hi)} from the manifest, plus the concatenation of all of it.

    Read from the manifest rather than hardcoded, so a store captured with a
    different channel layout cannot be silently misread. For the current store
    this gives y (128), X_{mean,min,max,var} (256 each), E_{mean,min,max,var}
    (64 each), and all (1408).
    """
    fs = {k: tuple(v) for k, v in manifest["feature_slices"].items()}
    if include_all:
        fs["all"] = (0, int(manifest["features"]))
    return fs


def open_layer(store, layer):
    return np.load(os.path.join(store, f"pooled_L{layer:02d}.npy"), mmap_mode="r")


def read_cell(mm, t_index):
    """One (N, F) float32 slice. t_index indexes the manifest's timestep LIST.

    mm[0] is t=500, the noisiest, and mm[9] is t=50, the cleanest. Indexing
    with a bare integer worked out by eye is how the whole figure gets
    inverted, silently and without an error. Callers obtain t_index from
    enumerate(manifest["timesteps"]) and carry the timestep VALUE alongside it
    into every output row.
    """
    return np.asarray(mm[t_index])
