"""tests/test_probe.py -- the probe harness has to be right, because when it
is wrong it does not crash, it reports.

The first draft of probe.py returned R2 = -79.8 with a confidently chosen
lam = 100 on a cell whose true value is 0.989, and nothing in the output
looked like an error. These tests exist for that failure mode.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import probe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(REPO, "activations_test_full")
HAVE_STORE = os.path.exists(os.path.join(STORE, "labels.npz"))
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="activation store absent")


# ---------------------------------------------------------------- helpers ---

def synthetic(n=4000, d=20, snr=4.0, seed=0):
    """y = Zw + noise with a known signal-to-noise ratio and a large offset.

    The offset is deliberate: it is what catches a missing intercept.
    """
    rs = np.random.RandomState(seed)
    Z = rs.randn(n, d)
    Z = (Z - Z.mean(0)) / Z.std(0)
    sig = Z @ rs.randn(d)
    sig = sig / sig.std() * np.sqrt(snr)      # var = snr
    noise = rs.randn(n)                        # var = 1
    return Z, 40.0 + sig + noise


# ------------------------------------------------------------------ tests ---

def test_split_integrity():
    train, test = probe.make_split(13055)
    assert len(train) + len(test) == 13055
    assert not set(train.tolist()) & set(test.tolist())
    assert len(test) == round(0.20 * 13055)
    t2, s2 = probe.make_split(13055)           # same seed, same split
    assert np.array_equal(train, t2) and np.array_equal(test, s2)


def test_mean_eigenvalue_is_n():
    """`lam` is only dimensionless if the mean eigenvalue is exactly n."""
    Z, _ = synthetic()
    s = probe.CellSolver(Z)
    assert abs(s.mean_eig / s.n - 1.0) < 1e-10


def test_df_behaves():
    Z, _ = synthetic()
    s = probe.CellSolver(Z)
    assert abs(s.df(0.0) - (s.d + 1)) < 1e-8          # +1 for the intercept
    vals = [s.df(L) for L in probe.LAM_GRID]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))   # monotone
    assert s.df(1e6) < 1.01                           # only the intercept left


def test_loo_shortcut_is_exact():
    """THE load-bearing test. Every selected lambda depends on this formula.

    The brute-force comparison must re-centre BOTH Z and y on the reduced set.
    With an unpenalised intercept b0 = ybar - zbar.w, and removing point j
    leaves zbar_-j = -z_j/(n-1), which is NOT zero even though the full set is
    centred. Omitting that term makes the brute force wrong by order 1/(n-1):
    at n=150 that is 6.7e-3, and it is exactly the 1e-3 to 6e-3 disagreement
    that first looked like a bug in the solver. Do not "simplify" this.
    """
    n, d = 150, 12
    Z, y = synthetic(n=n, d=d, snr=4.0, seed=1)
    s = probe.CellSolver(Z)
    for lam_rel in (1e-4, 1e-2, 1.0, 10.0):
        lam = lam_rel * n
        g, yc, ym = s.prepare(y)
        fast = s.loo_sse(g, yc, lam_rel)
        brute = 0.0
        for j in range(n):
            keep = np.ones(n, dtype=bool); keep[j] = False
            Zt, yt = Z[keep], y[keep]
            mz, my = Zt.mean(0), yt.mean()
            Zc = Zt - mz
            w = np.linalg.solve(Zc.T @ Zc + lam * np.eye(d), Zc.T @ (yt - my))
            brute += (y[j] - (my + (Z[j] - mz) @ w)) ** 2
        assert abs(fast - brute) / brute < 1e-10, f"lam_rel={lam_rel}"


def test_recovers_known_snr():
    """Held-out R2 must land near SNR/(1+SNR), here 4/5 = 0.80."""
    Z, y = synthetic(n=4000, d=20, snr=4.0, seed=2)
    train, test = probe.make_split(len(y))
    Ztr, Zte, d_used, d_drop = probe.standardize(Z[train], Z[test])
    s = probe.CellSolver(Ztr)
    row = probe.fit_cell(s, Zte, y[train], y[test])
    assert d_drop == 0
    assert abs(row["r2"] - 0.80) < 0.03, row["r2"]


def test_intercept_is_restored():
    """A large target offset must not cost any accuracy.

    This is the regression test for the -79.8 bug. Shifting y by +10,000
    changes nothing a correct probe reports.
    """
    Z, y = synthetic(n=2000, d=20, snr=9.0, seed=3)
    train, test = probe.make_split(len(y))
    Ztr, Zte, _, _ = probe.standardize(Z[train], Z[test])
    s = probe.CellSolver(Ztr)
    a = probe.fit_cell(s, Zte, y[train], y[test])["r2"]
    b = probe.fit_cell(s, Zte, y[train] + 1e4, y[test] + 1e4)["r2"]
    assert a > 0.85, a
    assert abs(a - b) < 1e-8, (a, b)


def test_leakage_is_visible():
    """Paste the target into a feature: R2 must be ~1.

    Confirms the pipeline can see signal at all, which a floor test cannot.
    """
    Z, y = synthetic(n=2000, d=20, seed=4)
    Z = np.column_stack([Z, (y - y.mean()) / y.std()])
    train, test = probe.make_split(len(y))
    Ztr, Zte, _, _ = probe.standardize(Z[train], Z[test])
    s = probe.CellSolver(Ztr)
    assert probe.fit_cell(s, Zte, y[train], y[test])["r2"] > 0.999


def test_pure_noise_floor():
    """Random features against a real-shaped target must score about zero.

    Calibrates the floor machinery Stage 3 depends on. Not exactly zero:
    held-out R2 on noise is slightly negative in expectation.
    """
    rs = np.random.RandomState(5)
    n, d = 4000, 50
    Z = rs.randn(n, d)
    y = 40.0 + rs.randn(n)
    train, test = probe.make_split(n)
    Ztr, Zte, _, _ = probe.standardize(Z[train], Z[test])
    s = probe.CellSolver(Ztr)
    r2v = probe.fit_cell(s, Zte, y[train], y[test])["r2"]
    assert -0.05 < r2v < 0.02, r2v


@needs_store
def test_composition_gate():
    """THE other load-bearing test, and the Stage 0 gate.

    The composition basis contains atom count as a linear combination, so it
    MUST explain at least as much of polarisability as atom count alone. If it
    does not, the element counts are misaligned with the labels and every
    size-controlled number downstream is meaningless. Measured: 0.871 against
    0.584.
    """
    z = np.load(os.path.join(STORE, "labels.npz"))
    y = z["polarisability"].astype(np.float64)
    n = len(y)
    train, test = probe.make_split(n)
    ones = np.ones(n)
    C = np.column_stack([ones, z["n_H"], z["n_C"], z["n_N"],
                         z["n_O"], z["n_F"]]).astype(np.float64)
    C1 = np.column_stack([ones, z["n_atoms"]]).astype(np.float64)
    b, *_ = np.linalg.lstsq(C[train], y[train], rcond=None)
    b1, *_ = np.linalg.lstsq(C1[train], y[train], rcond=None)
    r2_comp = probe.r2(y[test], C[test] @ b)
    r2_atoms = probe.r2(y[test], C1[test] @ b1)
    assert r2_atoms > 0.55, r2_atoms
    assert r2_comp >= r2_atoms - 1e-9, (r2_comp, r2_atoms)
    assert r2_comp > 0.582, r2_comp
    # counts must reconcile with the independently recorded atom count
    assert np.array_equal(z["n_atoms_from_counts"], z["n_atoms"])


@needs_store
def test_agrees_with_verify_store():
    """Cross-check against the code path that produced the 0.990 on record.

    probe.py and verify_store.ridge_r2 share nothing but the mathematics.
    probe.py should be slightly better, because its lambda is tuned while
    ridge_r2 fixes lam=1e-2. Measured 0.9887 against 0.9859.
    """
    from midi.verify_store import ridge_r2
    z = np.load(os.path.join(STORE, "labels.npz"))
    y = z["polarisability"].astype(np.float64)
    mm = np.load(os.path.join(STORE, "pooled_L11.npy"), mmap_mode="r")
    A = np.asarray(mm[9])
    train, test = probe.make_split(len(y))
    Ztr, Zte, _, _ = probe.standardize(
        A[train].astype(np.float64), A[test].astype(np.float64))
    s = probe.CellSolver(Ztr)
    mine = probe.fit_cell(s, Zte, y[train], y[test])["r2"]
    ref = ridge_r2(A, y, lam=1e-2, frac=0.8, seed=0)
    assert mine > 0.95, mine
    assert mine >= ref - 0.01, (mine, ref)

# =========================================================================
# Stage 3: the floor and the size controls
#
# Two of these are load-bearing. test_size_control_kills_size_only_probe
# proves the control actually controls something, and
# test_size_control_passes_real_signal proves it is not simply returning zero
# for everything. Either one alone can pass while the machinery is broken.
# =========================================================================

def composition_world(n=4000, seed=10, extra_signal=0.0, n_act=30):
    """Synthetic activations that encode ONLY composition, plus optionally one
    independent signal that composition cannot explain.

    counts  (n, 5) integer element counts, standing in for n_H..n_F
    y       a linear function of counts plus noise, so composition explains
            most of it, mirroring the real 0.871 for polarisability
    A       a random linear mixture of counts plus small noise: it can
            reconstruct composition and knows nothing else
    extra   when extra_signal > 0, a variable injected into BOTH y and A, so
            the composition residual becomes genuinely predictable
    """
    rs = np.random.RandomState(seed)
    counts = rs.randint(0, 10, size=(n, 5)).astype(np.float64)
    C = np.column_stack([np.ones(n), counts])
    w_true = np.array([2.0, 1.5, 0.5, 3.0, 1.0])
    y = 50.0 + counts @ w_true + 2.0 * rs.randn(n)
    A = counts @ rs.randn(5, n_act) + 0.1 * rs.randn(n, n_act)
    if extra_signal > 0:
        extra = rs.randn(n)
        y = y + extra_signal * extra
        A = np.column_stack([A, extra + 0.1 * rs.randn(n)])
    return C, A, y


def _raw_and_residual_r2(C, A, y):
    train, test = probe.make_split(len(y))
    Ztr, Zte, _, _ = probe.standardize(A[train], A[test])
    s = probe.CellSolver(Ztr)
    raw = probe.fit_cell(s, Zte, y[train], y[test])["r2"]
    r, _ = probe.residualize(y, C, train)
    res = probe.fit_cell(s, Zte, r[train], r[test])["r2"]
    return raw, res


def test_size_control_kills_size_only_probe():
    """LOAD-BEARING. Activations that know only composition must score ~0 on
    the composition residual, however well they score on the raw property.

    This is Case A of the Stage 0 derivation as executable code. If this test
    fails, every size-controlled number in the results table is meaningless
    and objective 1 cannot be defended against the size confound.
    """
    C, A, y = composition_world(extra_signal=0.0)
    raw, res = _raw_and_residual_r2(C, A, y)
    assert raw > 0.90, f"raw R2 should be high, got {raw}"
    assert abs(res) < 0.05, f"residual R2 should be ~0, got {res}"


def test_size_control_passes_real_signal():
    """LOAD-BEARING companion. Without this, the test above could pass simply
    because the residual path always returns zero."""
    C, A, y = composition_world(extra_signal=6.0)
    raw, res = _raw_and_residual_r2(C, A, y)
    assert raw > 0.90, raw
    assert res > 0.70, f"residual R2 should recover the injected signal, got {res}"


def test_batched_equals_unbatched():
    """fit_many on repeated columns must equal fit_cell on that target.

    Protects the whole batching optimisation the same way the brute-force
    comparison protects the LOO shortcut.
    """
    Z, y = synthetic(n=1500, d=25, snr=4.0, seed=11)
    train, test = probe.make_split(len(y))
    Ztr, Zte, _, _ = probe.standardize(Z[train], Z[test])
    s = probe.CellSolver(Ztr)
    one = probe.fit_cell(s, Zte, y[train], y[test])
    Ytr = np.column_stack([y[train]] * 5)
    Yte = np.column_stack([y[test]] * 5)
    many, lam = probe.fit_many(s, Zte, Ytr, Yte)
    assert np.allclose(many, many[0], atol=1e-12), many
    assert np.allclose(lam, one["lam"]), (lam, one["lam"])
    assert abs(many[0] - one["r2"]) < 1e-10, (many[0], one["r2"])


def test_permutation_is_deterministic():
    y = np.arange(100.0)
    a = probe.permute_targets(y, 7, seed=3)
    b = probe.permute_targets(y, 7, seed=3)
    c = probe.permute_targets(y, 7, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    for k in range(7):                       # a permutation, not a resample
        assert np.array_equal(np.sort(a[:, k]), y)


def test_floor_is_calibrated():
    """A permuted target, treated as if observed, must get a roughly uniform
    p-value. About 5% should fall below 0.05.

    If far more do, the null is too tight and every cell in the grid will look
    significant. This is the test that makes the floor mean what it claims.
    """
    rs = np.random.RandomState(12)
    n, d, n_obs, B = 800, 30, 40, 200
    A = rs.randn(n, d)
    y = 40.0 + rs.randn(n)
    train, test = probe.make_split(n)
    Ztr, Zte, _, _ = probe.standardize(A[train], A[test])
    s = probe.CellSolver(Ztr)

    null = probe.permute_targets(y, B, seed=99)
    r2_null, _ = probe.fit_many(s, Zte, null[train], null[test])
    obs = probe.permute_targets(y, n_obs, seed=1234)
    r2_obs, _ = probe.fit_many(s, Zte, obs[train], obs[test])

    ps = np.array([probe.floor_stats(v, r2_null)["perm_p"] for v in r2_obs])
    n_small = int((ps < 0.05).sum())
    assert 0 <= n_small <= 8, f"{n_small}/{n_obs} p-values below 0.05: {ps}"
    assert ps.min() >= 1.0 / (B + 1) - 1e-12


def test_floor_stats_fields():
    r2_null = np.random.RandomState(0).randn(200) * 0.01
    st = probe.floor_stats(0.5, r2_null)
    assert st["perm_exceed"] == 0 and st["perm_p_is_bound"] is True
    assert abs(st["perm_p"] - 1.0 / 201) < 1e-12
    assert st["z_score"] > 10 and st["z_p"] < 1e-10
    st2 = probe.floor_stats(float(np.median(r2_null)), r2_null)
    assert 0.3 < st2["perm_p"] < 0.7, st2["perm_p"]


def test_benjamini_hochberg():
    """The worked example from the plan: m=5, q=0.05, first four reject."""
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.3])
    mask = probe.benjamini_hochberg(p, q=0.05)
    assert mask.tolist() == [True, True, True, True, False]
    assert probe.benjamini_hochberg(np.full(100, 0.9)).sum() == 0
    assert probe.benjamini_hochberg(np.full(100, 1e-9)).sum() == 100


def test_residualize_fits_on_train_only():
    """The residualiser must never see the test labels.

    Changing a held-out label must not change the fitted coefficients, and the
    training residuals must be orthogonal to every column of C.
    """
    C, A, y = composition_world(n=1200, seed=13)
    train, test = probe.make_split(len(y))
    r1, b1 = probe.residualize(y, C, train)
    y2 = y.copy()
    y2[test] += 1000.0
    r2v, b2 = probe.residualize(y2, C, train)
    assert np.allclose(b1, b2), "test labels leaked into the residualiser"
    assert np.abs(C[train].T @ r1[train]).max() < 1e-6 * np.abs(r1[train]).max() * len(train)
