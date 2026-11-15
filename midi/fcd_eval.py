"""midi/fcd_eval.py -- Frechet ChemNet Distance at a matched sample size.

Stage 3 of results/tolerance_protocol.md.

THE QUANTITY. ChemNet maps a SMILES to an activation in R^d, d = 512. Fit a
Gaussian to each set, (mu_1, Sigma_1) and (mu_2, Sigma_2), where mu is the
mean activation and Sigma its covariance. The Frechet distance between two
Gaussians is

    FCD = ||mu_1 - mu_2||^2 + Tr( Sigma_1 + Sigma_2 - 2 (Sigma_1 Sigma_2)^1/2 )

The first term is the squared distance between the centres. The second
compares shapes and is zero exactly when Sigma_1 = Sigma_2.

THE BIAS, MEASURED rather than assumed. Sigma is estimated from m points in
512 dimensions, and a sample covariance from m points has rank at most
m - 1, so below m = 513 the estimate is rank deficient and the trace term is
inflated. Measured by scripts/check_fcd.py on two halves of the training
partition, which come from the same distribution so the true FCD is 0 and
every value is pure bias:

        m      rank     FCD    sd
       50        49   7.518  0.503
      100        99   4.881  0.361
      250       249   2.362  0.097
      500       499   1.328  0.070
     1000       512   0.714  0.017
     2500       512   0.289  0.013

The product m * FCD runs 376, 488, 590, 664, 714, 723, so the bias settles
into roughly 725/m, about 1.4d/m. At the protocol's m = 250 it is 2.362,
which is larger than the FCD a good generator would report at full size, so
**an absolute FCD at m = 250 must never be read as a distribution mismatch.**

WHY m = 250 IS STILL USABLE. The protocol never compares an absolute FCD to
zero. It compares a steered-against-unsteered FCD at m = 250 with the
distribution of unsteered-against-unsteered FCD at the same m, so the bias is
common to both sides and cancels. What limits the test is the run-to-run
spread, and that is 0.097 against a bias of 2.362, which is 4.1%. The
comparison therefore has power even though the offset is large.

One measured consequence worth keeping: the sweep's unsteered control arm
scores 2.358 against the training partition at m = 250, against a pure-bias
floor of 2.362. The unsteered generator is indistinguishable from the
training distribution at this sample size, so FCD here has essentially no
headroom to detect improvement and plenty to detect degradation. That is the
right way round for a degradation metric.

INPUT FORM. ChemNet was trained on ordinary canonical SMILES, not the
explicit-hydrogen strings this pipeline produces. Measured on the control arm
against the training partition at m = 250: 2.450 as produced with explicit H
against 2.358 re-canonicalised without H, a difference of 0.093, which is
about one standard deviation of the null. The H-free form is both lower and
matches what ChemNet was trained on, so `canon_for_chemnet` is applied to
both sides and the raw strings are never embedded.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem, RDLogger

from midi.matched_n import M_DEFAULT, R_DEFAULT, _pack, arm_seed

RDLogger.DisableLog("rdApp.*")
CHEMNET_DIM = 512
_ENGINE = {}


def _engine(device="cpu"):
    """One FCD object per device. Loading ChemNet is the slow part, so it is
    cached rather than rebuilt per call."""
    if device not in _ENGINE:
        from fcd_torch import FCD
        _ENGINE[device] = FCD(device=device, n_jobs=1)
    return _ENGINE[device]


def canon_for_chemnet(smiles):
    """Re-canonicalise and strip explicit hydrogens, dropping what will not
    parse. Returns a list, possibly shorter than the input."""
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s, sanitize=True)
        if m is None:
            continue
        try:
            c = Chem.MolToSmiles(Chem.RemoveHs(m))
        except Exception:
            continue
        if c:
            out.append(c)
    return out


def activations(smiles, device="cpu", already_canonical=False):
    """The (n, 512) per-molecule ChemNet activations.

    Embedding is done ONCE per set. Subsampling then works on rows of this
    array, which turns R = 200 repeats from 200 ChemNet passes into 200
    covariance-and-sqrtm computations. Without this the ten-baseline null,
    45 pairs at 200 repeats each, would take hours instead of minutes.
    """
    smi = smiles if already_canonical else canon_for_chemnet(smiles)
    if len(smi) < 2:
        return np.zeros((0, CHEMNET_DIM), dtype=np.float64)
    X = _engine(device).get_predictions(list(smi))
    return np.asarray(X, dtype=np.float64)


def _trace_sqrt_product(s1, s2):
    """Tr( (s1 s2)^1/2 ) for symmetric positive-semidefinite s1, s2.

    The trace term of the Frechet distance needs the matrix square root of a
    PRODUCT of two covariances. scipy.linalg.sqrtm does that with a general
    Schur decomposition, which is correct for any matrix and costs 1.42 s at
    d = 512. Measured: the 45-pair null at R = 200 would take 3.6 hours.

    A product of two symmetric PSD matrices is similar to a symmetric PSD
    matrix, so the same trace is available from eigenvalues alone. Write
    s1 = V diag(w) V^T, which exists because s1 is symmetric, and set
    s1^1/2 = V diag(sqrt(w)) V^T. Then

        s1^1/2 s2 s1^1/2     is symmetric PSD, and
        Tr( (s1 s2)^1/2 )  = sum_i sqrt( eig_i( s1^1/2 s2 s1^1/2 ) )

    because s1 s2 and s1^1/2 s2 s1^1/2 are similar, via the map
    X -> s1^-1/2 X s1^1/2, so they share a spectrum, and the square root of
    a PSD matrix has the square roots of those eigenvalues.

    That replaces one Schur decomposition with two symmetric eigenproblems,
    which `eigh` solves far faster. Eigenvalues are clipped at zero before
    the square root: the covariances are rank deficient at m < d, so the
    exact zeros come back as tiny negatives from rounding.

    Gated by test_fast_route_matches_fcd_torch against the library's own
    value, since this is an optimisation of a published quantity.
    """
    w, V = np.linalg.eigh(s1)
    w = np.clip(w, 0.0, None)
    s1_half = (V * np.sqrt(w)) @ V.T
    ev = np.linalg.eigvalsh(s1_half @ s2 @ s1_half)
    return float(np.sqrt(np.clip(ev, 0.0, None)).sum())


def frechet(mu_a, s_a, mu_b, s_b):
    """||mu_a - mu_b||^2 + Tr(s_a + s_b - 2 (s_a s_b)^1/2)."""
    d = np.asarray(mu_a) - np.asarray(mu_b)
    return float(d @ d + np.trace(s_a) + np.trace(s_b)
                 - 2.0 * _trace_sqrt_product(s_a, s_b))


def fcd_from_activations(Xa, Xb):
    """FCD between two activation blocks, via the formula in the docstring."""
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    return frechet(Xa.mean(0), np.cov(Xa.T), Xb.mean(0), np.cov(Xb.T))


def fcd_from_activations_reference(Xa, Xb):
    """The same number through fcd_torch's own scipy.linalg.sqrtm route.

    Kept only so a test can hold the fast path to it. Too slow to use.
    """
    from fcd_torch.utils import calculate_frechet_distance
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    return float(calculate_frechet_distance(Xa.mean(0), np.cov(Xa.T),
                                            Xb.mean(0), np.cov(Xb.T)))


def fcd_matched(Xa, Xb, m=M_DEFAULT, R=R_DEFAULT, arm="", seed=None):
    """FCD over R subsamples of exactly m rows from each side.

    Both sides are subsampled, because the bias depends on m and a comparison
    is only meaningful when both carry the same m. A side with fewer than m
    rows comes back evaluable=False rather than being evaluated smaller.
    """
    na, nb = len(Xa), len(Xb)
    if na < m or nb < m:
        return {"mean": float("nan"), "sd_subsample": float("nan"),
                "se_mean": float("nan"), "m": m, "R": 0, "evaluable": False,
                "note": f"sizes {na} and {nb}, need {m} each"}
    g = np.random.default_rng(arm_seed(arm) if seed is None else seed)
    vals = []
    for _ in range(R):
        ia = g.choice(na, size=m, replace=False)
        ib = g.choice(nb, size=m, replace=False)
        vals.append(fcd_from_activations(Xa[ia], Xb[ib]))
    out = _pack(vals, m, R)
    out["n_a"], out["n_b"] = na, nb
    return out


def fcd_full(Xa, Xb):
    """FCD at whatever size the two sets happen to be.

    Reported for reference only. It is NOT comparable between arms, because
    the bias above depends on m and the arms differ in how many molecules
    survive. Admissibility uses fcd_matched.
    """
    return {"fcd": fcd_from_activations(Xa, Xb),
            "n_a": len(Xa), "n_b": len(Xb),
            "comparable": False,
            "note": "biased by sample size; use fcd_matched to compare arms"}


def bias_floor(m, c=725.0):
    """A rough expectation of the pure-bias FCD at size m, from the measured
    fit FCD ~ 725/m. For sanity-checking a number, not for correcting one:
    the protocol cancels the bias by comparison, it does not subtract it."""
    return float(c / m) if m > 0 else float("nan")
