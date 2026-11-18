"""midi/direction_calib.py -- choose coefficients that actually move things.

Measured at layer 10, t=50: the X_mean activation has norm about 7.76, while
||d|| is 0.30 to 1.00 depending on the property. So a coefficient of 1 adds a
vector an eighth as long as the activation, which moves the alignment from
0.00 to 0.07 and the norm by 0.1%. A pilot run at c = 0.5, 1, 2 would report
"the coefficient does nothing" for the trivial reason that nothing was added.

Worse, ||d|| differs threefold between properties, so one fixed grid is not
comparable across them: c=5 is a moderate push for polarisability and a feeble
one for dipole moment.

The fix is to choose coefficients by the quantity Section 3.9 cares about.
Given a target alignment cos(a', v), solve for the c that achieves it at the
median molecule. The three arms then sit at matched ANGLE, so differences in
the norm ratio between properties mean something.

    a' = a + c||d||v,   s = c||d||
    cos = (a.v + s) / sqrt(||a||^2 + 2 s (a.v) + s^2)
"""
import numpy as np


def rho_and_cos(A, v, s):
    """Exact per-molecule (rho, cos) for an added vector s*v."""
    proj = A @ v
    n2 = (A ** 2).sum(axis=1)
    n2s = np.maximum(n2 + 2.0 * s * proj + s * s, 1e-12)
    return np.sqrt(n2s / np.maximum(n2, 1e-12)), (proj + s) / np.sqrt(n2s)


def coeff_for_target_cos(A, v, d_norm, target, hi=200.0, tol=1e-4):
    """Smallest c whose MEDIAN alignment reaches `target`.

    Bisection rather than the closed form: cos is monotone in s for s >= 0,
    and the median over molecules has no tidy inverse.
    """
    def med(c):
        return float(np.median(rho_and_cos(A, v, c * d_norm)[1]))

    if med(hi) < target:
        return float("nan")
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if med(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def pick_coeffs(A, v, d_norm, targets=(0.2, 0.5, 0.8)):
    """One coefficient per target alignment, with its rho and cos."""
    out = []
    for t in targets:
        c = coeff_for_target_cos(A, v, d_norm, t)
        if not np.isfinite(c):
            out.append({"target_cos": t, "coeff": float("nan"),
                        "rho_median": float("nan"), "cos_median": float("nan")})
            continue
        rho, cos = rho_and_cos(A, v, c * d_norm)
        out.append({"target_cos": t, "coeff": float(c),
                    "rho_median": float(np.median(rho)),
                    "cos_median": float(np.median(cos))})
    return out
