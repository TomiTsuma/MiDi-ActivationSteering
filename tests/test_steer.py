"""tests/test_steer.py -- the injection side.

Fast tests need nothing but numpy and torch. The three marked `slow` load the
checkpoint and run real sampling; run them with -m slow before any pilot.

    python3.11 -m pytest tests/test_steer.py -v -m "not slow"
    python3.11 -m pytest tests/test_steer.py -v -m slow      # ~6 min, needs GPU
"""
import os
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import direction, steer
from midi.capture import pool_nodes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "midi", "checkpoints",
                    "checkpoint_qm9_h_adaptive.ckpt")
slow = pytest.mark.slow


# ----------------------------------------------------------------- direction

def test_diff_of_means_toy():
    """The 6-molecule, 2-dimensional worked example, checked end to end."""
    A = np.array([[1., 0.], [2., 1.], [3., 1.], [4., 3.], [5., 3.], [6., 5.]])
    y = np.array([1., 2., 3., 6., 7., 8.])
    d, nh, nl = direction.diff_of_means(A, y, q=0.5, min_per_side=3)
    assert nh == 3 and nl == 3
    np.testing.assert_allclose(d, [3.0, 3.0])
    assert abs(np.linalg.norm(d) - np.sqrt(18)) < 1e-12


def test_projection_gap_equals_norm():
    """v . d == ||d||, which is what makes c=1 mean 'the full low-high gap'.

    If this fails the coefficient has no interpretable unit and the numbers
    stop transferring between sites.
    """
    rs = np.random.RandomState(0)
    A = rs.randn(400, 32)
    y = A @ rs.randn(32) + 0.5 * rs.randn(400)
    d, _, _ = direction.diff_of_means(A, y)
    v = d / np.linalg.norm(d)
    lo, hi = np.quantile(y, 0.5), np.quantile(y, 0.5)
    gap = (A[y >= hi] @ v).mean() - (A[y <= lo] @ v).mean()
    assert abs(gap - np.linalg.norm(d)) < 1e-9


def test_cosine_edges():
    a = np.array([1.0, 0.0])
    assert abs(direction.cosine(a, a) - 1.0) < 1e-12
    assert abs(direction.cosine(a, -a) + 1.0) < 1e-12
    assert abs(direction.cosine(a, np.array([0.0, 1.0]))) < 1e-12
    assert np.isnan(direction.cosine(a, np.zeros(2)))


# -------------------------------------------------------- the pooling identity

def test_adding_v_shifts_mean_exactly():
    """Adding v to every REAL atom must shift X_mean by exactly v, leave
    X_var exactly unchanged, and shift X_min and X_max by exactly v.

    This is the whole justification for computing the direction in the
    X_mean block and injecting it into X. pool_nodes uses a masked mean
    (denom = node_mask.sum), so the identity is exact rather than approximate
    -- but only if the injection respects the mask. An injection that also
    touched padding would break var and min.
    """
    torch.manual_seed(0)
    bs, n, dx = 4, 9, 16
    X = torch.randn(bs, n, dx, dtype=torch.float64)
    nm = torch.zeros(bs, n, dtype=torch.bool)
    for b in range(bs):
        nm[b, : 3 + b] = True              # 3, 4, 5, 6 real atoms
    X = X * nm.unsqueeze(-1)

    v = torch.randn(dx, dtype=torch.float64)
    before = pool_nodes(X, nm)
    Xs = X + v.view(1, 1, -1) * nm.unsqueeze(-1)
    after = pool_nodes(Xs, nm)

    mean_b, min_b, max_b, var_b = before.split(dx, dim=-1)
    mean_a, min_a, max_a, var_a = after.split(dx, dim=-1)

    assert torch.allclose(mean_a, mean_b + v, atol=1e-12)
    assert torch.allclose(min_a, min_b + v, atol=1e-12)
    assert torch.allclose(max_a, max_b + v, atol=1e-12)
    assert torch.allclose(var_a, var_b, atol=1e-12)


# --------------------------------------------------------------- diagnostics

def test_rho_and_cos_match_the_worked_table():
    """The hand-computed table from the plan, reproduced exactly.

    a = (2,1), v = (0.7071, 0.7071), ||d|| = sqrt(18).
    """
    a = np.array([2.0, 1.0])
    v = np.array([1.0, 1.0]) / np.sqrt(2.0)
    dn = np.sqrt(18.0)
    want = {0.0: (1.0000, 0.9487), 0.5: (1.9235, 0.9864),
            1.0: (2.8634, 0.9939), 2.0: (4.7538, 0.9978)}
    for c, (rho_w, cos_w) in want.items():
        a2 = a + c * dn * v
        rho = np.linalg.norm(a2) / np.linalg.norm(a)
        cos = float(a2 @ v) / np.linalg.norm(a2)
        assert abs(rho - rho_w) < 1e-3, (c, rho, rho_w)
        assert abs(cos - cos_w) < 1e-3, (c, cos, cos_w)
    # the point of recording both: the angle saturates while the norm does not
    assert want[2.0][1] - want[1.0][1] < 0.01
    assert want[2.0][0] / want[1.0][0] > 1.5


# -------------------------------------------------------------- window logic

def _state(coeff=1.0, window=(50, 100), force=False):
    return steer.SteerState(layer=10, channel="X", v=np.ones(8) / np.sqrt(8),
                            d_norm=2.0, coeff=coeff, window=window, T=500,
                            device="cpu", force_apply=force)


def test_window_logic():
    s = _state()
    assert s.window_size() == 51
    for t, want in ((500, False), (101, False), (100, True), (75, True),
                    (50, True), (49, False), (1, False)):
        s.t_now = t
        assert s.active() is want, t
    s.t_now = None
    assert s.active() is False


def test_coeff_zero_is_inert_but_force_apply_still_runs():
    s0, sf = _state(coeff=0.0), _state(coeff=0.0, force=True)
    s0.t_now = sf.t_now = 75
    assert s0.active() is False
    assert sf.active() is True
    assert sf.scale == 0.0


def test_scale_is_in_units_of_d_norm():
    s = _state(coeff=0.5)
    assert s.scale == 1.0          # 0.5 * d_norm 2.0


# ------------------------------------------------------------------ the gates

@lru_cache(maxsize=1)
def _load_model():
    # Cached: loading the checkpoint takes about 90 s and all three gates
    # share one model.
    import omegaconf, hydra
    from hydra import initialize_config_dir, compose
    from hydra.core.hydra_config import HydraConfig
    import midi._compat  # noqa: F401
    from midi.generate import _setup
    cfgdir = os.path.join(REPO, "configs")

    # Two things compose() alone does not give you:
    #   1. HydraConfig must be set, or get_original_cwd() raises, and
    #      QM9DataModule calls it.
    #   2. get_original_cwd() records the cwd at compose time, and the dataset
    #      root is resolved as parents[0] of it. From the repo root that is
    #      the repo's PARENT, where a stray duplicate QM9 tree already exists.
    #      Composing from <repo>/midi is what every other entry point does.
    prev = os.getcwd()
    os.chdir(os.path.join(REPO, "midi"))
    try:
        with initialize_config_dir(version_base="1.3", config_dir=cfgdir):
            cfg = compose(config_name="config", return_hydra_config=True,
                          overrides=[
                              "dataset=qm9", "dataset.remove_h=False",
                              "+experiment=qm9_with_h_adaptive",
                              "general.wandb=disabled", "general.gpus=1",
                              f"general.test_only={CKPT.replace(os.sep, '/')}"])
            HydraConfig.instance().set_config(cfg)
    finally:
        os.chdir(prev)
    with omegaconf.open_dict(cfg):
        cfg.gen = omegaconf.OmegaConf.create(
            {"mode": "generate", "target": 4, "chunk": 4, "out": "_steer_test",
             "run_seed": 1234, "metrics_at_end": False, "metrics_device": "cpu"})
    return _setup(cfg)


def _sample(model, ids, seed=1234):
    from midi import rng
    from midi.generate import N_MAX
    n_nodes = rng.sample_n_nodes(seed, ids, model.node_dist.prob)
    with torch.no_grad():
        return model.sample_batch(n_nodes=n_nodes.tolist(), mol_ids=ids,
                                  run_seed=seed, n_max_fixed=N_MAX, test=True,
                                  save_final=0, keep_chain=0, visualize=False,
                                  batch_id=0)


def _fingerprint(mols):
    return [(m.atom_types.cpu().numpy().copy(), m.charges.cpu().numpy().copy(),
             m.bond_types.cpu().numpy().copy(), m.positions.cpu().numpy().copy())
            for m in mols]


@slow
@pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint absent")
def test_zero_coefficient_is_bit_identical():
    """THE gate. Running the add path with scale 0 must change nothing.

    force_apply=True so the arithmetic really executes. If this fails, the
    hook is perturbing the forward pass and every steered number is suspect.
    """
    device, dinfos, model, _ = _load_model()
    ids = [0, 1, 2, 3]
    base = _fingerprint(_sample(model, ids))

    v = np.zeros(256); v[0] = 1.0
    st = steer.SteerState(layer=10, channel="X", v=v, d_norm=1.0, coeff=0.0,
                          window=(1, 500), T=model.T, device=device,
                          force_apply=True)
    h = steer.attach(model, st)
    try:
        got = _fingerprint(_sample(model, ids))
    finally:
        steer.detach(h)

    assert st.n_calls == model.T, (st.n_calls, model.T)
    assert st.n_injected == model.T
    for i, (b, g) in enumerate(zip(base, got)):
        for name, x, y in zip(("atom_types", "charges", "bond_types", "positions"),
                              b, g):
            assert np.array_equal(x, y), f"molecule {i} {name} changed at coeff=0"


@slow
@pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint absent")
def test_window_fires_only_inside_itself():
    device, dinfos, model, _ = _load_model()
    v = np.zeros(256); v[0] = 1.0
    st = steer.SteerState(layer=10, channel="X", v=v, d_norm=1.0, coeff=0.0,
                          window=(50, 100), T=model.T, device=device,
                          force_apply=True)
    h = steer.attach(model, st)
    try:
        _sample(model, [0, 1])
    finally:
        steer.detach(h)
    assert st.n_calls == model.T
    assert st.n_injected == st.window_size() == 51
    ts = sorted(r["t"] for r in st.steps)
    assert ts[0] == 50 and ts[-1] == 100


@slow
@pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint absent")
def test_nonzero_coefficient_actually_changes_molecules():
    """The converse of the gate: a real coefficient must do something.

    Without this, test_zero_coefficient_is_bit_identical would also pass if
    the hook were never firing at all.
    """
    device, dinfos, model, _ = _load_model()
    ids = [0, 1, 2, 3]
    base = _fingerprint(_sample(model, ids))
    rs = np.random.RandomState(0)
    v = rs.randn(256); v /= np.linalg.norm(v)
    st = steer.SteerState(layer=10, channel="X", v=v, d_norm=1.0, coeff=5.0,
                          window=(50, 200), T=model.T, device=device)
    h = steer.attach(model, st)
    try:
        got = _fingerprint(_sample(model, ids))
    finally:
        steer.detach(h)
    changed = sum(not np.array_equal(b[3], g[3]) for b, g in zip(base, got))
    assert changed > 0, "a coefficient of 5 changed nothing; the hook is inert"
    rho, cos = st.per_molecule()
    assert (rho > 1.0).all(), rho
    assert np.isfinite(cos).all()


# ------------------------------------------------------- the control arms

def test_shuffled_identity_permutation_reproduces_real():
    """THE gate for the shuffled-label arm.

    The arm's whole claim is that the construction is identical to the real
    direction and only the labelling differs. Feed it the identity permutation
    and it must return the real direction bit for bit. If this fails, the two
    arms differ in something else as well, and a difference in outcome cannot
    be attributed to the labelling.
    """
    rs = np.random.RandomState(0)
    A = rs.randn(200, 16)
    y = A @ rs.randn(16) + 0.3 * rs.randn(200)
    d_real, nh, nl = direction.diff_of_means(A, y)
    d_id, nh2, nl2 = direction.shuffled_direction(A, y,
                                                  perm=np.arange(len(y)))
    np.testing.assert_array_equal(d_real, d_id)
    assert (nh, nl) == (nh2, nl2)


def test_shuffled_direction_destroys_the_signal():
    """Permuting the labels must cost both alignment and length.

    Two numbers, and they mean different things. The cosine going to chance
    says the direction no longer points anywhere near the real one. The norm
    ratio falling well below 1 is why the arm must be injected at the REAL
    arm's norm: left at its own, it would differ in magnitude as well as in
    content and a null result would be uninterpretable.
    """
    rs = np.random.RandomState(1)
    A = rs.randn(600, 64)
    y = A @ rs.randn(64) + 0.1 * rs.randn(600)
    d_real, _, _ = direction.diff_of_means(A, y)

    cosines, ratios = [], []
    for s in range(20):
        d_s, _, _ = direction.shuffled_direction(A, y, seed=s)
        cosines.append(abs(direction.cosine(d_s, d_real)))
        ratios.append(np.linalg.norm(d_s) / np.linalg.norm(d_real))
    assert np.mean(cosines) < 0.25, f"mean |cos| {np.mean(cosines):.3f}"
    assert np.mean(ratios) < 0.60, f"mean norm ratio {np.mean(ratios):.3f}"


def test_shuffled_direction_is_seed_reproducible():
    rs = np.random.RandomState(2)
    A, y = rs.randn(150, 8), rs.randn(150)
    a, _, _ = direction.shuffled_direction(A, y, seed=5)
    b, _, _ = direction.shuffled_direction(A, y, seed=5)
    c, _, _ = direction.shuffled_direction(A, y, seed=6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shuffled_direction_requires_a_seed_or_perm():
    """A silent default seed would make every control arm the same draw."""
    rs = np.random.RandomState(3)
    A, y = rs.randn(40, 4), rs.randn(40)
    with pytest.raises(ValueError):
        direction.shuffled_direction(A, y)
    with pytest.raises(ValueError):
        direction.shuffled_direction(A, y, perm=np.arange(5))


def test_random_unit_is_unit_and_seeded():
    u = direction.random_unit(256, 42)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-12
    np.testing.assert_array_equal(u, direction.random_unit(256, 42))
    assert not np.array_equal(u, direction.random_unit(256, 43))


def test_random_unit_is_near_orthogonal_in_high_dimension():
    """E[cos^2] = 1/d for two independent unit vectors, so rms cos = 1/sqrt(d).

    At d = 256 that is 0.0625. This is what makes the random arm a control
    rather than a weak version of the real one, and it is the number the
    sweep's |cos| gate is set against.
    """
    d = 256
    ref = direction.random_unit(d, 0)
    cs = np.array([direction.cosine(direction.random_unit(d, s), ref)
                   for s in range(1, 201)])
    rms = float(np.sqrt(np.mean(cs ** 2)))
    expected = 1.0 / np.sqrt(d)
    assert abs(rms - expected) < 0.2 * expected, f"rms cos {rms:.4f}"
    assert np.abs(cs).max() < 0.25, f"max |cos| {np.abs(cs).max():.4f}"


def test_control_injects_at_matched_norm():
    """The matched-norm requirement of Section 3.8, as a number.

    The injected vector is coeff * d_norm[t] * v[t] with v a unit vector, so
    its length is |coeff * d_norm[t]| whatever v happens to be. Replacing the
    unit direction and keeping d_norm therefore matches the injected norm at
    every timestep exactly. Checked here rather than trusted, because getting
    it wrong makes the control answer a different question.
    """
    nt, dim = 10, 64
    rs = np.random.RandomState(7)
    D_real = rs.randn(nt, dim)
    nrm = np.linalg.norm(D_real, axis=1)
    V_real = D_real / nrm[:, None]
    V_rand = np.repeat(direction.random_unit(dim, 99)[None, :], nt, axis=0)
    ts = np.arange(10, 10 * nt + 1, 10)

    kw = dict(layer=0, channel="X", coeff=20.0, window=(10, 100), T=500,
              device="cpu", v_timesteps=ts)
    s_real = steer.SteerState(v=V_real, d_norm=nrm, **kw)
    s_rand = steer.SteerState(v=V_rand, d_norm=nrm, **kw)

    for t in ts:
        s_real.t_now = s_rand.t_now = int(t)
        v_r, sc_r = s_real.current()
        v_c, sc_c = s_rand.current()
        assert sc_r == sc_c, f"t={t}: scales differ, {sc_r} vs {sc_c}"
        n_r = float(torch.linalg.vector_norm(v_r * sc_r))
        n_c = float(torch.linalg.vector_norm(v_c * sc_c))
        assert abs(n_r - n_c) <= 1e-5 * max(n_r, 1.0), \
            f"t={t}: injected norms differ, {n_r} vs {n_c}"
    # And the norms genuinely vary across t, so the test is not vacuous.
    assert nrm.max() / nrm.min() > 1.01


def test_shuffled_cos_matches_the_covariance_prediction():
    """Why the two control arms show different alignment, as a formula.

    A permuted split is still a difference between the means of two halves of
    the SAME activations, so d_shuf is a random vector drawn with the data's
    covariance rather than an isotropic one. For any fixed unit u,

        E[(u . d_shuf)^2] / E[||d_shuf||^2] = u' Sigma u / trace(Sigma)

    so the rms cosine is the square root of the share of total variance lying
    along u, not 1/sqrt(d). That is the whole reason the shuffled arm sits
    near 0.40 against the real direction at layer 10 while a unit-sphere draw
    sits at 0.06, and why the shuffled arm is a conservative control rather
    than a clean null. Verified on the real store to 2.5% by
    scripts/check_controls.py.

    The formula is a ratio of EXPECTATIONS and only equals the expectation of
    the ratio while the share is small, because (u.d)^2 and ||d||^2 become
    strongly correlated once u carries most of the variance. Measured: at a
    share of 0.67 the prediction overshoots by 19 percent (0.82 against 0.66),
    at the store's 0.17 it is within 2.5 percent. So the synthetic case here
    is set in the regime the real site occupies, not at an extreme that would
    only test the approximation's failure mode.
    """
    rs = np.random.RandomState(11)
    n, dim = 2000, 64
    scale = np.ones(dim)
    scale[0] = 4.0                       # share = 16/79 = 0.20, like the store
    A = rs.randn(n, dim) * scale
    u = np.zeros(dim)
    u[0] = 1.0                           # and we look along it
    y = rs.randn(n)                      # labels are irrelevant: permuted

    Ac = A - A.mean(axis=0)
    share = float(np.var(Ac @ u) / ((Ac ** 2).sum() / n))
    pred = np.sqrt(share)
    assert 0.1 < share < 0.3, f"share {share:.3f} is outside the tested regime"

    cs = np.array([direction.cosine(
        direction.shuffled_direction(A, y, seed=s)[0], u) for s in range(200)])
    obs = float(np.sqrt(np.mean(cs ** 2)))
    assert abs(obs - pred) < 0.15 * pred, f"pred {pred:.4f}, obs {obs:.4f}"
    # And it is far above the isotropic value, which is the point of the test.
    assert pred > 3.0 / np.sqrt(dim), f"{pred:.4f} vs {3 / np.sqrt(dim):.4f}"
