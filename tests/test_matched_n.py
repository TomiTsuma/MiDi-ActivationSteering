"""tests/test_matched_n.py -- the matched-size harness.

One continuous example again, six valid molecules with three distinct SMILES:

    A A A B B C        so n = 6, multiplicities c = (3, 2, 1), K = 3

Everything below is that set. The closed form says the expected number of
distinct SMILES in a subsample of m drawn without replacement is

    E[K_m] = sum_j [ 1 - C(n - c_j, m) / C(n, m) ]

and at m = 3, with C(6,3) = 20,

    A: 1 - C(3,3)/20 = 1 - 1/20  = 0.95
    B: 1 - C(4,3)/20 = 1 - 4/20  = 0.80
    C: 1 - C(5,3)/20 = 1 - 10/20 = 0.50
                                   ------
                        E[K_3]   = 2.25

so uniqueness at m = 3 is 2.25/3 = 0.75, against 3/6 = 0.5 at the full set.
A smaller sample reports HIGHER uniqueness, which is the artefact the whole
module exists to remove. test_closed_form_matches_brute_force checks the 2.25
by enumerating all twenty 3-subsets by hand.
"""
import itertools
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import matched_n as mn

TOY = ["A", "A", "A", "B", "B", "C"]        # n = 6, c = (3, 2, 1)


# ------------------------------------------------- the closed form itself

def test_closed_form_matches_brute_force():
    """E[K_3] = 2.25, checked against all C(6,3) = 20 subsets.

    The enumeration by hand, for the record:
        AAA           1 subset  x 1 distinct =  1
        AA + B        6         x 2          = 12
        AA + C        3         x 2          =  6
        A  + BB       3         x 2          =  6
        A  + B  + C   6         x 3          = 18
        BB + C        1         x 2          =  2
                     --                       --
                     20                       45     ->  45/20 = 2.25
    """
    exact = mn.expected_distinct(Counter(TOY).values(), 3)
    assert abs(exact - 2.25) < 1e-12, exact

    total, count = 0, 0
    for idx in itertools.combinations(range(6), 3):
        total += len({TOY[i] for i in idx})
        count += 1
    assert count == 20
    assert total == 45
    assert abs(total / count - exact) < 1e-12


def test_uniqueness_falls_as_the_sample_grows():
    """m = 1 gives exactly 1, m = n gives the honest K/n, monotone between.

    This is the bias in one line: an arm with fewer valid molecules reports
    higher uniqueness for no chemical reason.
    """
    vals = [mn.expected_uniqueness(TOY, m) for m in range(1, 7)]
    assert abs(vals[0] - 1.0) < 1e-12          # a single molecule is "unique"
    assert abs(vals[-1] - 3 / 6) < 1e-12       # the full set: K/n
    assert abs(vals[2] - 0.75) < 1e-12         # m = 3, from the docstring
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(5)), vals
    assert vals[0] > vals[-1] + 0.4             # and the gap is large


def test_expected_distinct_edges():
    assert mn.expected_distinct([3, 2, 1], 6) == 3.0      # m >= n
    assert mn.expected_distinct([3, 2, 1], 99) == 3.0
    assert mn.expected_distinct([], 5) == 0.0
    assert mn.expected_distinct([3, 2, 1], 0) == 0.0


def test_a_group_too_large_to_miss_contributes_exactly_one():
    """With n = 6, c = (5, 1) and m = 5: only 1 molecule is outside group A,
    so no 5-subset can avoid A and its term is exactly 1. Group B has
    C(5,5)/C(6,5) = 1/6 chance of being missed, so it contributes 5/6.

        E[K_5] = 1 + 5/6 = 1.8333...
    """
    got = mn.expected_distinct([5, 1], 5)
    assert abs(got - (1.0 + 5.0 / 6.0)) < 1e-12, got

    total, count = 0, 0
    items = ["A"] * 5 + ["B"]
    for idx in itertools.combinations(range(6), 5):
        total += len({items[i] for i in idx})
        count += 1
    assert count == 6
    assert abs(total / count - got) < 1e-12


def test_log_choose_agrees_with_exact_integers():
    """lgamma rather than big integers, so the arithmetic must still be right.

    C(500, 250) is about 1.17e149, which fits a float, so even the large case
    can be checked against math.comb directly.
    """
    from math import comb
    for a, b in ((6, 3), (10, 4), (20, 7)):
        assert abs(np.exp(mn._log_choose(a, b)) - comb(a, b)) < 1e-6, (a, b)
    for a, b in ((52, 5), (500, 250)):
        want = float(np.log(float(comb(a, b))))
        assert abs(mn._log_choose(a, b) - want) < 1e-9, (a, b)
    assert mn._log_choose(5, 9) == float("-inf")
    assert mn._log_choose(5, -1) == float("-inf")


# --------------------------------------------- the Monte Carlo estimator

def test_monte_carlo_matches_the_closed_form():
    """THE gate on the harness.

    Uniqueness has an exact answer, so the subsampling estimator is checked
    against arithmetic rather than trusted. Agreement to within a few Monte
    Carlo standard errors; a real divergence means the subsampling is wrong.
    """
    rs = np.random.RandomState(0)
    # 600 molecules over 300 distinct SMILES with uneven multiplicities
    smi = []
    for j in range(300):
        smi += [f"S{j}"] * (1 + int(rs.poisson(1.0)))
    smi = smi[:600]
    assert len(smi) == 600

    exact = mn.expected_uniqueness(smi, 250)
    got = mn.uniqueness_matched(smi, m=250, R=400, arm="gate")
    assert got["evaluable"]
    assert abs(got["mean"] - exact) < 4 * got["se_mean"] + 1e-9, \
        (got["mean"], exact, got["se_mean"])
    assert abs(got["expected_closed_form"] - exact) < 1e-12
    # and the matched figure differs from the own-n figure
    assert abs(got["own_n"] - got["mean"]) > 1e-4


def test_not_evaluable_below_m_rather_than_evaluated_smaller():
    """The protocol's rule: an arm with fewer than m valid molecules is not
    quietly reported at its own n."""
    out = mn.uniqueness_matched(["A"] * 100, m=250, R=10, arm="short")
    assert out["evaluable"] is False
    assert out["R"] == 0
    assert np.isnan(out["mean"])
    assert "100" in out["note"] and "250" in out["note"]


def varied(n_distinct=400, seed=7):
    """A set with enough distinct SMILES that the subsample actually varies.

    TOY * 100 will NOT do: with only three distinct SMILES every 250-subsample
    contains all three, so uniqueness is 3/250 whatever the seed, and a
    reproducibility test on it cannot fail.
    """
    rs = np.random.RandomState(seed)
    out = []
    for j in range(n_distinct):
        out += [f"V{j}"] * (1 + int(rs.rand() < 0.5))
    return out


def test_subsample_is_reproducible_from_the_arm_name():
    pool = varied()
    a = mn.uniqueness_matched(pool, m=250, R=20, arm="dom_late_cp20")
    b = mn.uniqueness_matched(pool, m=250, R=20, arm="dom_late_cp20")
    c = mn.uniqueness_matched(pool, m=250, R=20, arm="dom_late_cp40")
    assert a["mean"] == b["mean"], "same arm name must redraw identically"
    assert a["mean"] != c["mean"], "different arm names must differ"
    assert mn.arm_seed("x") != mn.arm_seed("y")


def test_subsamples_are_without_replacement_and_exactly_m():
    seen = {}

    def fn(sub):
        seen["n"] = len(sub)
        seen["distinct_idx"] = len(set(sub))
        return 0.0

    items = [f"u{i}" for i in range(400)]     # all distinct
    mn.subsample_metric(items, fn, m=250, R=1, seed=3)
    assert seen["n"] == 250
    assert seen["distinct_idx"] == 250, "drew the same item twice"


def test_reported_spread_names_say_which_is_which():
    """sd_subsample is how much the subsample choice matters. se_mean is that
    divided by sqrt(R). Neither is the tolerance scale, which comes from the
    ten baselines; the names exist so they cannot be confused."""
    out = mn.uniqueness_matched(varied(seed=11), m=250, R=64, arm="spread")
    assert out["R"] == 64
    assert out["sd_subsample"] > 0
    assert abs(out["se_mean"] - out["sd_subsample"] / 8.0) < 1e-12   # sqrt(64)
    assert out["se_mean"] < out["sd_subsample"]


# ------------------------------------------------------------- novelty

def test_novelty_matched_uses_the_distinct_denominator():
    """MiDi computes len(novel)/len(unique), so the denominator is the
    distinct count inside the subsample, not the subsample size."""
    smi = [f"S{j}" for j in range(300)] * 2          # 600 items, 300 distinct
    train = {f"S{j}" for j in range(150)}            # half are known
    out = mn.novelty_matched(smi, train, m=250, R=200, arm="nov")
    assert out["evaluable"]
    assert abs(out["own_n"] - 0.5) < 1e-12
    # Inside a subsample of 250 from a set where every SMILES appears twice,
    # the distinct count is below 250, and roughly half of those are novel.
    assert 0.35 < out["mean"] < 0.65, out["mean"]
    assert out["n_unique_all"] == 300


def test_novelty_all_novel_and_none_novel():
    smi = [f"S{j}" for j in range(400)]
    assert abs(mn.novelty_matched(smi, set(), m=250, R=20,
                                  arm="a")["mean"] - 1.0) < 1e-12
    assert abs(mn.novelty_matched(smi, set(smi), m=250, R=20,
                                  arm="b")["mean"] - 0.0) < 1e-12


# ------------------------------------------------------------ paired side

def test_paired_subsample_draws_m_from_both_sides():
    sizes = []

    def fn2(sa, sb):
        sizes.append((len(sa), len(sb)))
        return 0.0

    a = [f"a{i}" for i in range(300)]
    b = [f"b{i}" for i in range(500)]
    out = mn.paired_subsample_metric(a, b, fn2, m=250, R=3, seed=1)
    assert out["evaluable"]
    assert sizes == [(250, 250)] * 3


def test_paired_not_evaluable_when_either_side_is_short():
    a = [f"a{i}" for i in range(100)]
    b = [f"b{i}" for i in range(500)]
    out = mn.paired_subsample_metric(a, b, lambda x, y: 0.0, m=250, R=3)
    assert out["evaluable"] is False
    assert "100" in out["note"]
