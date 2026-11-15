"""tests/test_fcd_eval.py -- FCD at matched size.

The load-bearing test is test_activation_route_equals_the_direct_call: the
module embeds each set ONCE and then subsamples rows, instead of calling
fcd_torch per subsample. That is a 200-fold saving on the ten-baseline null,
and it is only legitimate if it gives the same number as the direct call.
Marked slow because it loads ChemNet.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import fcd_eval as fe

slow = pytest.mark.slow

# Small, real, parseable molecules. Explicit-H forms on the left of each pair
# are what the pipeline actually produces.
EXPLICIT = ["[H]C([H])([H])[H]", "[H]O[H]", "[H]C([H])([H])O[H]",
            "[H]C#C[H]", "[H]N([H])[H]"]
PLAIN = ["C", "O", "CO", "C#C", "N"]


def test_canon_for_chemnet_strips_hydrogens():
    got = fe.canon_for_chemnet(EXPLICIT)
    assert got == PLAIN, got


def test_canon_for_chemnet_drops_unparseable_without_raising():
    got = fe.canon_for_chemnet(["C", "not_a_smiles", "O", "C(((", "N"])
    assert got == ["C", "O", "N"], got


def test_bias_floor_follows_the_measured_fit():
    """FCD ~ 725/m, from the measured curve in the module docstring."""
    assert abs(fe.bias_floor(250) - 2.9) < 0.1
    assert abs(fe.bias_floor(2500) - 0.29) < 0.01
    assert fe.bias_floor(500) < fe.bias_floor(250)
    assert np.isnan(fe.bias_floor(0))


def test_matched_is_not_evaluable_below_m():
    Xa = np.random.RandomState(0).randn(100, 8)
    Xb = np.random.RandomState(1).randn(400, 8)
    out = fe.fcd_matched(Xa, Xb, m=250, R=3)
    assert out["evaluable"] is False
    assert "100" in out["note"]


def test_matched_on_synthetic_activations_is_reproducible():
    """fcd_from_activations is pure linear algebra, so it can be exercised
    without ChemNet by handing it activation blocks directly."""
    rs = np.random.RandomState(4)
    Xa = rs.randn(400, 16)
    Xb = rs.randn(400, 16) + 0.5          # shifted mean
    a = fe.fcd_matched(Xa, Xb, m=200, R=8, arm="same")
    b = fe.fcd_matched(Xa, Xb, m=200, R=8, arm="same")
    c = fe.fcd_matched(Xa, Xb, m=200, R=8, arm="other")
    assert a["evaluable"] and a["mean"] > 0
    assert a["mean"] == b["mean"]
    assert a["mean"] != c["mean"]
    assert a["n_a"] == 400 and a["n_b"] == 400


def test_identical_blocks_give_zero_and_a_mean_shift_shows_up():
    rs = np.random.RandomState(5)
    X = rs.randn(300, 12)
    assert abs(fe.fcd_from_activations(X, X)) < 1e-6
    # ||mu_1 - mu_2||^2 is the first term, so a shift of 1.0 in every one of
    # 12 dimensions must contribute at least 12 to the distance.
    shifted = X + 1.0
    assert fe.fcd_from_activations(X, shifted) > 11.0


def test_fcd_full_is_flagged_as_not_comparable():
    rs = np.random.RandomState(6)
    out = fe.fcd_full(rs.randn(300, 8), rs.randn(120, 8))
    assert out["comparable"] is False
    assert "sample size" in out["note"]
    assert out["n_a"] == 300 and out["n_b"] == 120


@slow
def test_activation_route_equals_the_direct_call():
    """THE gate. Embedding once and subsampling rows must agree with
    fcd_torch's own end-to-end call, or the 200-fold saving is unsound."""
    from fcd_torch import FCD
    a = PLAIN * 12                      # 60 molecules
    b = ["CC", "CCO", "CCC", "CN", "CCN"] * 12
    eng = FCD(device="cpu", n_jobs=1)
    direct = float(eng(a, b))

    Xa = fe.activations(a, device="cpu", already_canonical=True)
    Xb = fe.activations(b, device="cpu", already_canonical=True)
    assert Xa.shape == (60, fe.CHEMNET_DIM)
    via = fe.fcd_from_activations(Xa, Xb)
    assert abs(via - direct) < 1e-4, (via, direct)


@slow
def test_explicit_hydrogens_are_never_embedded_raw():
    """activations() canonicalises by default, so the two forms of the same
    molecules must give identical activations."""
    Xe = fe.activations(EXPLICIT * 10, device="cpu")
    Xp = fe.activations(PLAIN * 10, device="cpu", already_canonical=True)
    assert Xe.shape == Xp.shape
    assert np.allclose(Xe, Xp, atol=1e-5)
