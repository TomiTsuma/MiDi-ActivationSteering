"""midi/matched_n.py -- evaluating n-dependent metrics at a matched size.

Stage 2 of results/tolerance_protocol.md. Uniqueness, novelty and FCD all
move with how many molecules go into them, and the sweep's valid counts run
from 261 to 493, a factor of 1.9. Comparing arms at their own n would
manufacture differences out of validity alone. Every such metric therefore
goes through a subsample of exactly m = 250 valid molecules, R = 200 times.

WHY UNIQUENESS MOVES WITH n, exactly.

Let the valid set hold n molecules made of K distinct SMILES with
multiplicities c_1 ... c_K, so sum_j c_j = n. Draw m of the n without
replacement. Group j is MISSED entirely when all m draws come from the other
n - c_j molecules, which happens with probability

    P(miss j) = C(n - c_j, m) / C(n, m)

where C(a, b) is "a choose b". Distinct groups are either seen or not, so the
expected number seen is the sum of their individual probabilities of being
seen, whatever their overlap:

    E[K_m] = sum_j [ 1 - C(n - c_j, m) / C(n, m) ]

and uniqueness at size m is E[K_m] / m.

Two consequences. At m = 1 it is exactly 1, because a single molecule is
always "all distinct". At m = n it is K/n, the honest figure. In between it
falls monotonically, so **a set with fewer valid molecules reports HIGHER
uniqueness for no chemical reason at all**. That is the artefact the matched
size exists to remove, and `expected_distinct` below computes the closed form
so the Monte Carlo estimator can be checked against arithmetic rather than
trusted.

WHAT THE REPORTED SPREAD IS, AND IS NOT.

Each call returns the mean over R subsamples and the standard deviation
across them. That standard deviation measures only how much the choice of
subsample matters; it shrinks as R grows and it is NOT the uncertainty in the
metric. The uncertainty that the tolerance band rests on is the between-run
standard deviation across the ten unsteered baselines, each of which has
itself been through this harness. Using the subsample spread as a tolerance
scale would be a mistake, so both numbers are returned under names that say
which is which: `sd_subsample` and `se_mean = sd_subsample / sqrt(R)`.
"""
import hashlib
from collections import Counter
from math import lgamma, sqrt

import numpy as np

M_DEFAULT = 250           # fixed a priori: half the generated set size
R_DEFAULT = 200


def arm_seed(name):
    """A reproducible seed from an arm name, so every figure redraws the same
    subsamples. Not time or order dependent."""
    h = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _log_choose(a, b):
    """log C(a, b), as lgamma rather than big integers.

    C(500, 250) is a 150-digit integer; exact arithmetic is correct but slow
    and the ratio is all that is wanted, so the logs cancel cleanly.
    """
    if b < 0 or b > a:
        return float("-inf")
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def expected_distinct(multiplicities, m):
    """E[K_m] in closed form: the expected number of distinct SMILES in a
    subsample of m drawn without replacement.

        E[K_m] = sum_j [ 1 - C(n - c_j, m) / C(n, m) ]

    multiplicities: the counts c_j, in any order.
    """
    c = [int(x) for x in multiplicities if int(x) > 0]
    n = sum(c)
    if m <= 0 or n <= 0:
        return 0.0
    if m >= n:
        return float(len(c))
    log_den = _log_choose(n, m)
    total = 0.0
    for cj in c:
        if n - cj < m:
            total += 1.0            # group j cannot be missed
        else:
            total += 1.0 - np.exp(_log_choose(n - cj, m) - log_den)
    return float(total)


def expected_uniqueness(smiles, m):
    """The closed-form uniqueness at size m for a list of SMILES."""
    if not smiles:
        return float("nan")
    return expected_distinct(Counter(smiles).values(), m) / min(m, len(smiles))


# ------------------------------------------------------------- the harness

def _pack(vals, m, R, note=""):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "sd_subsample": float("nan"),
                "se_mean": float("nan"), "m": m, "R": 0,
                "evaluable": False, "note": note or "no finite values"}
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {"mean": float(v.mean()), "sd_subsample": sd,
            "se_mean": sd / sqrt(v.size), "m": m, "R": int(v.size),
            "evaluable": True, "note": note}


def subsample_metric(items, fn, m=M_DEFAULT, R=R_DEFAULT, seed=0):
    """Apply fn to R random subsamples of exactly m items, without replacement.

    Returns the pack described in the module docstring. A set with fewer than
    m items is NOT evaluated at a smaller size: it comes back
    evaluable=False, because reporting it at its own n is the artefact this
    module exists to prevent, and under the protocol it is inadmissible by
    default.
    """
    n = len(items)
    if n < m:
        return {"mean": float("nan"), "sd_subsample": float("nan"),
                "se_mean": float("nan"), "m": m, "R": 0, "evaluable": False,
                "note": f"only {n} items, need {m}"}
    arr = np.asarray(items, dtype=object)
    g = np.random.default_rng(seed)
    out = []
    for _ in range(R):
        idx = g.choice(n, size=m, replace=False)
        out.append(fn([arr[i] for i in idx]))
    return _pack(out, m, R)


def paired_subsample_metric(a_items, b_items, fn2, m=M_DEFAULT, R=R_DEFAULT,
                            seed=0):
    """fn2(sub_a, sub_b) over R independent subsamples of BOTH sides at m.

    This is what FCD needs: its estimate is biased upward at small m, so a
    comparison is only meaningful when both sides carry the same m. Stage 3
    passes the FCD function in here.
    """
    na, nb = len(a_items), len(b_items)
    if na < m or nb < m:
        return {"mean": float("nan"), "sd_subsample": float("nan"),
                "se_mean": float("nan"), "m": m, "R": 0, "evaluable": False,
                "note": f"sizes {na} and {nb}, need {m} each"}
    A = np.asarray(a_items, dtype=object)
    B = np.asarray(b_items, dtype=object)
    g = np.random.default_rng(seed)
    out = []
    for _ in range(R):
        ia = g.choice(na, size=m, replace=False)
        ib = g.choice(nb, size=m, replace=False)
        out.append(fn2([A[i] for i in ia], [B[i] for i in ib]))
    return _pack(out, m, R)


# --------------------------------------------------------- the two metrics

def _uniqueness(sub):
    return len(set(sub)) / len(sub) if sub else float("nan")


def uniqueness_matched(smiles, m=M_DEFAULT, R=R_DEFAULT, arm=""):
    """Uniqueness at matched m, with the closed-form value beside it.

    `expected` is E[K_m]/m from the module docstring. The Monte Carlo mean
    must agree with it; a divergence means the subsampling is wrong, and a
    test enforces the agreement.
    """
    out = subsample_metric(smiles, _uniqueness, m, R, arm_seed(arm))
    out["expected_closed_form"] = (expected_uniqueness(smiles, m)
                                   if len(smiles) >= m else float("nan"))
    out["own_n"] = (len(set(smiles)) / len(smiles) if smiles else float("nan"))
    out["n_valid"] = len(smiles)
    return out


def novelty_matched(smiles, train_smiles, m=M_DEFAULT, R=R_DEFAULT, arm=""):
    """Novelty at matched m: novel distinct over distinct, inside each subsample.

    The denominator is the DISTINCT count within the subsample, matching
    MiDi's len(novel)/len(unique), so this metric has a random denominator
    nested inside a random denominator. That is the reason it gets no
    analytic standard error and has to be subsampled.
    """
    ts = set(train_smiles)

    def fn(sub):
        u = set(sub)
        if not u:
            return float("nan")
        return sum(1 for s in u if s not in ts) / len(u)

    out = subsample_metric(smiles, fn, m, R, arm_seed(arm) ^ 0x5EED)
    u_all = set(smiles)
    out["own_n"] = (sum(1 for s in u_all if s not in ts) / len(u_all)
                    if u_all else float("nan"))
    out["n_valid"] = len(smiles)
    out["n_unique_all"] = len(u_all)
    return out
