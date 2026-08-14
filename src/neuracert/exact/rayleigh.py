"""Float-side generalised Rayleigh machinery: Ritz, pruning, rational polish.

Problem-agnostic.  Any bound of the form

    lambda = max_c  c^T B c / c^T A c,   A PSD,

can use all of this.  That covers Maynard, the zeta-zero-gap and pair
correlation extremal problems, and any other kernel-Gram formulation.  It
does NOT cover the LP-type bounds (Delsarte, Cohn-Elkies), which need a
simplex/interior-point step instead -- those plug into the same exact CRT
backend but bypass this module.

The float layer NEVER certifies anything.  It steers pruning and produces
the trial vector c; validity comes entirely from the exact backend applied
to whatever c it happens to output.
"""

from __future__ import annotations

import numpy as np


def ritz(A: np.ndarray, B: np.ndarray, rank_tol: float = 1e-12
         ) -> tuple[float, np.ndarray]:
    """Max generalised eigenpair of (B, A) for numerically SEMIdefinite A.

    A is PSD in exact arithmetic but at large problem size its entries span
    enough orders of magnitude that Cholesky-based generalised solvers fail
    on indefinite rounding.  Instead: equilibrate by the diagonal,
    eigendecompose A, restrict to its numerical range, and solve the
    standard symmetric problem in whitened coordinates.

    Restricting to range(A) is the CORRECT regularisation rather than a
    convenience: null-space components contribute ~0 to c^T A c and would
    only inflate the quotient spuriously.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m = A.shape[0]
    d = np.diag(A).copy()
    alive = np.isfinite(d) & (d > 0)
    if not np.any(alive):
        return 0.0, np.zeros(m)
    ia = np.where(alive)[0]
    s = 1.0 / np.sqrt(d[ia])
    As = s[:, None] * A[np.ix_(ia, ia)] * s[None, :]
    Bs = s[:, None] * B[np.ix_(ia, ia)] * s[None, :]
    As = 0.5 * (As + As.T)
    Bs = 0.5 * (Bs + Bs.T)
    w, V = np.linalg.eigh(As)
    keep = w > rank_tol * float(w[-1])
    if not np.any(keep):
        return 0.0, np.zeros(m)
    W = V[:, keep] / np.sqrt(w[keep])[None, :]
    lams, U = np.linalg.eigh(W.T @ Bs @ W)
    c = np.zeros(m)
    c[ia] = s * (W @ U[:, -1])
    return float(lams[-1]), c


def greedy_prune(A: np.ndarray, B: np.ndarray, tol: float,
                 min_channels: int = 1, verbose: bool = True,
                 rank_tol: float = 1e-10) -> tuple[list[int], np.ndarray]:
    """Backward elimination under a cumulative relative budget on lambda.

    Validity is untouched -- any explicit c on any channel subset certifies
    -- so only bound quality can move, and the float pass measures that
    before any exact work is done.  Exact cost falls quadratically in the
    number of dropped channels.
    """
    m = A.shape[0]
    active = list(range(m))
    lam0, c0 = ritz(A, B, rank_tol)
    if tol <= 0.0 or lam0 <= 0.0:
        return active, c0
    while len(active) > max(1, min_channels):
        best_lam, best_i = -np.inf, None
        for i in range(len(active)):
            sub = active[:i] + active[i + 1:]
            lam_i, _ = ritz(A[np.ix_(sub, sub)], B[np.ix_(sub, sub)], rank_tol)
            if lam_i > best_lam:
                best_lam, best_i = lam_i, i
        if (lam0 - best_lam) / lam0 <= tol:
            dropped = active.pop(best_i)
            if verbose:
                print(f"    prune: drop channel {dropped:3d} (rel. loss "
                      f"{max(0.0, (lam0 - best_lam) / lam0):.3e}, "
                      f"{len(active)} remain)")
        else:
            break
    lam_f, c = ritz(A[np.ix_(active, active)], B[np.ix_(active, active)],
                    rank_tol)
    if verbose:
        print(f"    prune: kept {len(active)}/{m} channels (pairs "
              f"{m * (m + 1) // 2} -> {len(active) * (len(active) + 1) // 2}),"
              f" float rel. loss {max(0.0, (lam0 - lam_f) / lam0):.3e}")
    return active, c


def rationalise_vector(c: np.ndarray, bits: int = 64) -> list[int]:
    """Dyadic integer numerators for a trial vector, on a common scale.

    The vector is normalised to max|c| = 1 first, which is free: a Rayleigh
    quotient is invariant under c -> alpha c.
    """
    c = np.asarray(c, dtype=np.float64)
    peak = float(np.max(np.abs(c)))
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("trial vector is zero or non-finite")
    unit = 1 << bits
    return [int(round(float(v) / peak * unit)) for v in c]
