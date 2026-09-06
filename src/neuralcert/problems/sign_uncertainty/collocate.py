#!/usr/bin/env python3
"""
Collocation solver for the Gaussian-mixture sign-uncertainty ansatz.

This is the Cohn-Elkies / Cohn-Kumar / Cohn-Goncalves method transplanted from
the Laguerre basis into the Gaussian-mixture basis, and it replaces the LP's
cutting-plane loop by a single square linear solve.

    f(u) = sum_{j=1}^{k} c_j phi_{a_j,s}(u),
    phi_{a,s}(u) = e^{-a u} + s a^{-d/2} e^{-u/a},   u = pi r^2.

DEGREE-OF-FREEDOM COUNT (differs from the polynomial version, so worth stating)

    k coefficients
      - 1  normalisation (pin the slowest-decaying term's coefficient to +1)
      - 1  the constraint f(0) = 0
      = k - 2 left, consumed by m double roots at 2 conditions each:

            m = k/2 - 1,      k even.

    So k = 4 -> 1 double root, k = 8 -> 3, k = 16 -> 7.  The LP confirms this
    independently: its k = 4 optimum at d = 4 comes out with exactly 1 double
    root and 1 simple root.

WHY THIS IS THE RIGHT INSTRUMENT NOW

  * One k x k solve instead of thousands of grid constraints, so extended
    precision is affordable.  The LP's cost blew up precisely because
    longdouble evaluation on a 12000-node verify grid ran inside a cut loop
    inside a leading-width sweep inside a bisection.

  * The tail normalisation is built into the ansatz, so the "leading
    coefficient is 1e-9 with an arbitrary sign" degeneracy cannot occur.

  * It fixes the output-precision problem.  The LP returns coefficients of
    size ~1e5 against an O(1) function; rounding those to float64 perturbs f
    by ~1e-16 * 1e5 = 1e-11, which is the same size as the dips being
    adjudicated -- so a float64 coefficient vector cannot even be checked,
    let alone certified.  Here the structure (which widths, where the double
    roots are) is found cheaply, and then the SAME system is re-solved in
    mpmath at 50 digits, giving coefficients whose rounding error is
    irrelevant.  Discovery and precision are decoupled.

  * The known price, stated by Cohn-Goncalves themselves: the last sign change
    is not a continuous function of the root locations, and their Newton
    iteration fails on exactly that in d <= 4.  The LP has no such failure
    mode.  That is the reason to keep both: the LP is the falsifier, the
    collocation is the microscope.

Usage
    python collocate.py --d 1 --k 8 --tune-widths
    python collocate.py --d 12 --k 12 --seed lp --dps 60
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .gaussian_mixture import (
    XP, Family, analyse_roots, best_lower_bound, cdg_asymptote, make_scales,
    min_rho, published_upper,
)

try:
    import mpmath as mp
except ImportError:                                          # pragma: no cover
    mp = None


# ---------------------------------------------------------------------------
# Extended-precision linear algebra (numpy demotes longdouble to float64)
# ---------------------------------------------------------------------------

def solve_xp(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting, kept in longdouble."""
    A = np.array(A, dtype=XP, copy=True)
    b = np.array(b, dtype=XP, copy=True)
    n = len(b)
    for i in range(n):
        p = i + int(np.argmax(np.abs(A[i:, i])))
        if A[p, i] == 0:
            raise np.linalg.LinAlgError("singular collocation system")
        if p != i:
            A[[i, p]] = A[[p, i]]
            b[i], b[p] = b[p], b[i]
        piv = A[i, i]
        for r in range(i + 1, n):
            f = A[r, i] / piv
            if f != 0:
                A[r, i:] -= f * A[i, i:]
                b[r] -= f * b[i]
    x = np.zeros(n, dtype=XP)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def basis_xp(fam: Family, u) -> np.ndarray:
    """Rescaled basis g = e^{nu u} f; every exponent >= 0, so no underflow."""
    u = np.atleast_1d(np.asarray(u, dtype=XP))
    a = np.asarray(fam.scales, dtype=XP)
    nu = XP(1.0) / a[-1]
    logw = -XP(0.5) * XP(fam.d) * np.log(a)
    return (np.exp(-np.outer(u, a - nu))
            + XP(fam.s) * np.exp(logw[None, :] - np.outer(u, XP(1.0) / a - nu)))


def dbasis_xp(fam: Family, u) -> np.ndarray:
    u = np.atleast_1d(np.asarray(u, dtype=XP))
    a = np.asarray(fam.scales, dtype=XP)
    nu = XP(1.0) / a[-1]
    logw = -XP(0.5) * XP(fam.d) * np.log(a)
    hr, tr = a - nu, XP(1.0) / a - nu
    return (-hr[None, :] * np.exp(-np.outer(u, hr))
            - (XP(fam.s) * tr)[None, :] * np.exp(logw[None, :] - np.outer(u, tr)))


# ---------------------------------------------------------------------------
# The collocation system
# ---------------------------------------------------------------------------

def n_double_roots(k: int) -> int:
    """The SQUARE choice, m = k/2 - 1.  A default, not a law -- see free_indices."""
    if k % 2:
        raise ValueError("square collocation needs k even; pass m explicitly instead")
    return k // 2 - 1


def free_indices(k: int, m: int) -> list[int]:
    """
    Which coefficients to pin when m is smaller than the square value.

    With k widths and m double roots the constraints number 2m + 2, leaving
    nu = k - 2m - 2 degrees of freedom.  Rather than build a nullspace basis
    (whose derivative is ill-conditioned exactly where the widths crowd
    together, which is where the optimiser wants to go), pin nu individual
    coefficients to values eta and keep the system SQUARE -- every existing
    exact solver, in longdouble and in mpmath, then applies verbatim.

    The payoff is continuity in k, which the square rule destroys.  Under the
    square rule k = 12 is not "the good k = 8 function plus four spare basis
    functions"; it is a different problem forced to carry two more double
    roots, in a different basin.  Here eta = 0 embeds the smaller solution
    EXACTLY -- the pinned widths carry zero coefficient -- so a k = 8 optimum
    is a legal starting point for k = 12 and Adam explores away from it.

    Indices are spread over 1..k-2, avoiding the first width and the leading
    one, which carries the tail normalisation.
    """
    nu = k - 2 * m - 2
    if nu < 0:
        raise ValueError(f"k={k} cannot support m={m} double roots")
    if nu == 0:
        return []
    pool = list(range(1, k - 1))
    if nu > len(pool):
        raise ValueError("too many free coefficients to pin")
    step = len(pool) / nu
    return [pool[int((i + 0.5) * step)] for i in range(nu)]


def collocate_xp(fam: Family, taus, eta=None, pinned=None) -> np.ndarray:
    """
    Coefficients of f with f(0) = 0, double roots at taus, the slowest term
    normalised to +1, and (when m < k/2 - 1) the pinned coefficients set to
    eta.  Rows are written for the rescaled g = e^{nu u} f, whose roots and
    signs are those of f.
    """
    taus = np.asarray(taus, dtype=XP)
    m = len(taus)
    if pinned is None:
        pinned = free_indices(fam.k, m)
    if eta is None:
        eta = np.zeros(len(pinned), dtype=XP)
    eta = np.asarray(eta, dtype=XP).ravel()
    if 2 * m + 2 + len(pinned) != fam.k:
        raise ValueError(f"k={fam.k}, m={m}: need {fam.k - 2*m - 2} pinned "
                         f"coefficients, got {len(pinned)}")
    rows = [basis_xp(fam, XP(0.0))[0]]
    rows.extend(basis_xp(fam, taus))
    rows.extend(dbasis_xp(fam, taus))
    lead = np.zeros(fam.k, dtype=XP)
    lead[-1] = XP(fam.s) * np.asarray(fam.scales, dtype=XP)[-1] ** (-XP(fam.d) / 2)
    rows.append(lead)
    rhs = [XP(0.0)] * (1 + 2 * m) + [XP(1.0)]
    for idx, val in zip(pinned, eta):
        e = np.zeros(fam.k, dtype=XP)
        e[idx] = XP(1.0)
        rows.append(e)
        rhs.append(XP(val))
    return solve_xp(np.array(rows, dtype=XP), np.array(rhs, dtype=XP))


def tail_dominance_xp(fam: Family, c: np.ndarray, safety=XP(1e3)) -> float:
    """u beyond which the slowest surviving term alone fixes the sign."""
    a = np.asarray(fam.scales, dtype=XP)
    nu = XP(1.0) / a[-1]
    beta = np.concatenate([a, XP(1.0) / a]) - nu
    gamma = np.concatenate([c, XP(fam.s) * a ** (-XP(fam.d) / 2) * c])
    o = np.argsort(beta)
    beta, gamma = beta[o], gamma[o]
    gmax = np.max(np.abs(gamma))
    keep = np.where(np.abs(gamma) > XP(1e-25) * gmax)[0]
    if len(keep) == 0 or gamma[keep[0]] <= 0:
        return math.inf
    b0, g0 = beta[keep[0]], gamma[keep[0]]
    us = [0.0]
    for b, g in zip(beta, np.abs(gamma)):
        if g <= 0 or b <= b0:
            continue
        us.append(float(np.log(safety * g / g0) / (b - b0)))
    return max(us)


def scan_grid(u_hi: float, n: int = 4000) -> np.ndarray:
    return np.unique(np.concatenate([
        np.linspace(1e-9, min(u_hi, 40.0), n),
        np.geomspace(1e-6, u_hi, n),
        np.array([u_hi])]))


def last_sign_change_xp(fam: Family, c: np.ndarray, n: int = 4000):
    """
    (u0, ok): the last sign change of f, and whether f >= 0 beyond it.

    'ok' is False when the tail is eventually negative or when the scan finds
    a dip below zero past u0 that the scan itself resolved -- the ansatz has
    then split a double root and this configuration is not admissible.
    """
    u_dom = tail_dominance_xp(fam, c)
    if not math.isfinite(u_dom):
        return math.inf, False
    u_hi = max(1.5 * u_dom, 10.0)
    u = scan_grid(u_hi, n)
    v = basis_xp(fam, u) @ c
    sg = np.sign(v)
    idx = np.where(sg[1:] * sg[:-1] < 0)[0]
    if len(idx) == 0:
        return 0.0, bool(v[-1] > 0)
    i = int(idx[-1])
    lo, hi = u[i], u[i + 1]
    for _ in range(200):
        mid = (lo + hi) / 2
        if float(basis_xp(fam, mid)[0] @ c) * float(v[i]) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-17 * max(hi, 1.0):
            break
    u0 = float((lo + hi) / 2)
    return u0, True


COND_MAX = 1e12


def collocation_cond(fam: Family, taus, pinned=None) -> float:
    """cond(M) in float64.  Above ~1e12 the longdouble solve (eps 1.1e-19) has
    too few digits left to trust, and rho_of will silently return a number for
    a function it never computed -- e.g. longdouble 2.0423 where the mpmath
    replay of the same geometry says 19.0675."""
    taus = np.asarray(taus, dtype=float)
    if pinned is None:
        pinned = free_indices(fam.k, len(taus))
    rows = [np.asarray(basis_xp(fam, 0.0)[0], dtype=float)]
    rows += list(np.asarray(basis_xp(fam, taus), dtype=float))
    rows += list(np.asarray(dbasis_xp(fam, taus), dtype=float))
    lead = np.zeros(fam.k)
    lead[-1] = fam.s * float(fam.scales[-1]) ** (-fam.d / 2.0)
    rows.append(lead)
    for idx in pinned:
        e = np.zeros(fam.k); e[idx] = 1.0
        rows.append(e)
    try:
        return float(np.linalg.cond(np.array(rows)))
    except Exception:
        return math.inf


def rho_of(fam: Family, taus, n: int = 4000, eta=None, pinned=None,
           cond_max: float = COND_MAX) -> float:
    if collocation_cond(fam, taus, pinned) > cond_max:
        return math.inf            # refuse, do not guess
    try:
        c = collocate_xp(fam, taus, eta=eta, pinned=pinned)
    except (np.linalg.LinAlgError, ValueError):
        return math.inf
    u0, ok = last_sign_change_xp(fam, c, n)
    if not ok or not math.isfinite(u0):
        return math.inf
    return math.sqrt(u0 / math.pi)


# ---------------------------------------------------------------------------
# Optimisation over root locations and widths
# ---------------------------------------------------------------------------

def _taus_from_theta(theta: np.ndarray) -> np.ndarray:
    """theta -> strictly increasing positive root locations."""
    return np.cumsum(np.exp(np.clip(np.asarray(theta, dtype=float), -12.0, 8.0)))


def _theta_from_taus(taus) -> np.ndarray:
    t = np.asarray(taus, dtype=float)
    return np.log(np.maximum(np.diff(np.concatenate([[0.0], t])), 1e-12))


def _widths_from_phi(phi: np.ndarray, s: int) -> np.ndarray:
    steps = np.exp(np.clip(np.asarray(phi, dtype=float), -8.0, 6.0))
    a = 1.0 + np.cumsum(steps)
    return np.concatenate([[1.0], a]) if s == 1 else a


def _phi_from_widths(a, s: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if s == 1:
        a = a[1:]
    return np.log(np.maximum(np.diff(np.concatenate([[1.0], a])), 1e-9))


@dataclass
class CollocationResult:
    fam: Family
    taus: np.ndarray
    coeffs: np.ndarray
    rho: float
    n_evals: int


def optimise(d: int, s: int, scales, taus0=None, tune_widths: bool = False,
             maxiter: int = 800, restarts: int = 3, n_scan: int = 3000,
             m: int | None = None, pinned=None, cond_max: float = COND_MAX,
             verbose: bool = True) -> CollocationResult | None:
    fam0 = Family(d, s, scales)
    if m is None:
        m = n_double_roots(fam0.k)
    if pinned is None:
        pinned = free_indices(fam0.k, m)
    if taus0 is None:
        taus0 = np.geomspace(1.5, 1.5 + 4.0 * m, m)
    best = [math.inf, None, None, 0]

    def unpack(z):
        if tune_widths:
            fam = Family(d, s, _widths_from_phi(z[m:], s))
        else:
            fam = fam0
        return fam, _taus_from_theta(z[:m])

    def obj(z):
        best[3] += 1
        try:
            fam, taus = unpack(z)
        except ValueError:
            return 1e3
        r = rho_of(fam, taus, n_scan, pinned=pinned, cond_max=cond_max)
        if not math.isfinite(r):
            return 1e3
        if r < best[0]:
            best[0], best[1], best[2] = r, taus.copy(), fam
            if verbose:
                print(f"    rho = {r:.10f}   taus = "
                      f"{np.array2string(taus, precision=4)}")
        return r

    z0 = _theta_from_taus(taus0)
    if tune_widths:
        z0 = np.concatenate([z0, _phi_from_widths(fam0.scales, s)])

    rng = np.random.default_rng(0)
    for attempt in range(restarts):
        z = z0 if attempt == 0 else best_z + 0.25 * rng.standard_normal(len(z0))
        res = minimize(obj, z, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-9,
                                "fatol": 1e-13, "adaptive": True})
        best_z = res.x if best[1] is None else np.concatenate(
            [_theta_from_taus(best[1]),
             _phi_from_widths(best[2].scales, s)] if tune_widths
            else [_theta_from_taus(best[1])])

    if best[1] is None:
        return None
    fam, taus = best[2], best[1]
    return CollocationResult(fam, taus, collocate_xp(fam, taus, pinned=pinned),
                             best[0], best[3])


def seed_from_lp(d: int, s: int, scales, **lp_kw):
    """
    Use the LP to find the contact structure, then hand it to collocation.

    The LP is globally reliable but coarse and its coefficients are unusable
    at the precision we need; collocation is precise but only converges from a
    decent starting configuration and can fall off the last-sign-change
    discontinuity.  Composing them plays to both.
    """
    fam = Family(d, s, scales)
    sol = min_rho(fam, **lp_kw)
    if sol is None:
        return None, None
    r = analyse_roots(fam, sol["coeffs"], min(sol["u_max"], 12 * sol["u0"] + 40))
    return sol, np.array(r["double_roots_u"], dtype=float)


# ---------------------------------------------------------------------------
# mpmath polish: the same system, 50 digits, so the answer can be checked
# ---------------------------------------------------------------------------

def polish_mp(fam: Family, taus, dps: int = 50, verify: bool = True,
               eta=None, pinned=None):
    if mp is None:
        raise RuntimeError("mpmath not available")
    mp.mp.dps = dps
    a = [mp.mpf(repr(float(x))) for x in fam.scales]
    nu = 1 / a[-1]
    d, s = mp.mpf(fam.d), mp.mpf(fam.s)
    w = [ai ** (-d / 2) for ai in a]

    def phi(j, u):
        return mp.e ** (-(a[j] - nu) * u) + s * w[j] * mp.e ** (-(1 / a[j] - nu) * u)

    def dphi(j, u):
        return (-(a[j] - nu) * mp.e ** (-(a[j] - nu) * u)
                - s * w[j] * (1 / a[j] - nu) * mp.e ** (-(1 / a[j] - nu) * u))

    k = fam.k
    T = [mp.mpf(repr(float(t))) for t in taus]
    if pinned is None:
        pinned = free_indices(k, len(T))
    if eta is None:
        eta = [0.0] * len(pinned)
    rows = [[phi(j, mp.mpf(0)) for j in range(k)]]
    rows += [[phi(j, t) for j in range(k)] for t in T]
    rows += [[dphi(j, t) for j in range(k)] for t in T]
    rows += [[(s * w[j] if j == k - 1 else mp.mpf(0)) for j in range(k)]]
    rhs = [mp.mpf(0)] * (1 + 2 * len(T)) + [mp.mpf(1)]
    for idx, val in zip(pinned, eta):
        rows.append([(mp.mpf(1) if j == idx else mp.mpf(0)) for j in range(k)])
        rhs.append(mp.mpf(repr(float(val))))
    c = mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))

    def g(u):
        u = mp.mpf(u)
        return mp.fsum(c[j] * phi(j, u) for j in range(k))

    out = {"coeffs_mp": [mp.nstr(c[j], dps) for j in range(k)],
           "coeffs_str": [mp.nstr(c[j], 25) for j in range(k)],
           "g_at_0": mp.nstr(g(0), 8),
           "max_abs_coeff": mp.nstr(max(abs(c[j]) for j in range(k)), 6)}
    if not verify:
        return c, g, out

    u_dom = tail_dominance_xp(fam, collocate_xp(fam, taus, eta=eta, pinned=pinned))
    u_hi = max(1.5 * u_dom, 10.0) if math.isfinite(u_dom) else 200.0
    us = scan_grid(u_hi, 3000)
    vals = [g(u) for u in us]
    sg = [mp.sign(v) for v in vals]
    idx = [i for i in range(len(us) - 1) if sg[i] * sg[i + 1] < 0]
    if idx:
        i = idx[-1]
        # findroot's tol is on |g|, not on the bracket, and g near a simple
        # root of a well-scaled exponential sum bottoms out around
        # eps * sum|gamma|.  Demanding 10^(-2dps/3) therefore throws
        # "Could not find root within given tolerance" at dps >= 100 even
        # though the bracket is exact to full precision.  Bisect on the
        # bracket instead and only then polish.
        lo_b, hi_b = mp.mpf(us[i]), mp.mpf(us[i + 1])
        s_lo = mp.sign(g(lo_b))
        for _ in range(4 * dps):
            mid_b = (lo_b + hi_b) / 2
            if mp.sign(g(mid_b)) == s_lo:
                lo_b = mid_b
            else:
                hi_b = mid_b
            if hi_b - lo_b < mp.mpf(10) ** (-dps + 5) * max(hi_b, 1):
                break
        u0 = (lo_b + hi_b) / 2
    else:
        u0 = mp.mpf(0)
    dips = [i for i in range(1, len(us) - 1)
            if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1] and us[i] > float(u0)]
    worst = None
    for i in dips:
        try:
            r = mp.findroot(lambda t: mp.diff(g, t), mp.mpf(us[i]),
                            solver="secant",
                            tol=mp.mpf(10) ** (-min(30, dps // 3)))
        except Exception:
            continue
        v = g(r)
        if worst is None or v < worst[0]:
            worst = (v, r)
    out.update(u0=mp.nstr(u0, dps // 2),
               rho=mp.nstr(mp.sqrt(u0 / mp.pi), dps // 2),
               rho_float=float(mp.sqrt(u0 / mp.pi)),
               n_interior_minima=len(dips),
               worst_dip=(mp.nstr(worst[0], 8) if worst else None),
               worst_dip_u=(mp.nstr(worst[1], 12) if worst else None),
               tail_sign=mp.nstr(g(u_hi), 6))
    return c, g, out


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--k", type=int, default=8, help="number of widths (even)")
    ap.add_argument("--spread", type=float, default=1.3)
    ap.add_argument("--scales", default=None)
    ap.add_argument("--tune-widths", action="store_true")
    ap.add_argument("--seed", choices=("geom", "lp"), default="geom")
    ap.add_argument("--taus", default=None,
                    help="comma-separated seed double roots (overrides --seed)")
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--m", type=int, default=None,
                    help="number of double roots; default k/2-1. Smaller m pins "
                         "k-2m-2 coefficients at 0, which embeds the smaller "
                         "solution exactly and lets k grow at fixed contact count.")
    ap.add_argument("--pinned", default=None,
                    help="comma-separated coefficient indices to pin")
    ap.add_argument("--cond-max", type=float, default=COND_MAX)
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scales = (np.array([float(v) for v in args.scales.split(",")])
              if args.scales else make_scales(args.k, args.sign, args.spread))
    taus0 = (None if args.taus is None
             else np.array([float(v) for v in args.taus.split(",")]))
    if taus0 is None and args.seed == "lp":
        sol, taus0 = seed_from_lp(args.d, args.sign, scales, bisect=24)
        if sol is not None:
            print(f"LP seed: rho = {sol['rho']:.8f}, "
                  f"{0 if taus0 is None else len(taus0)} double roots")
        m = args.m if args.m is not None else n_double_roots(len(scales))
        if taus0 is None or len(taus0) != m:
            print(f"  (LP gave {0 if taus0 is None else len(taus0)} roots, "
                  f"need {m}; falling back to a geometric seed)")
            taus0 = None

    pinned = (None if args.pinned is None
              else [int(v) for v in args.pinned.split(",")])
    m_eff = args.m if args.m is not None else n_double_roots(len(scales))
    need = len(scales) - 2 * m_eff - 2
    if pinned is not None and len(pinned) != need:
        ap.error(f"k={len(scales)}, m={m_eff} needs exactly nu = k-2m-2 = {need} "
                 f"pinned indices, got {len(pinned)}")
    res = optimise(args.d, args.sign, scales, taus0=taus0,
                   tune_widths=args.tune_widths, maxiter=args.maxiter,
                   restarts=args.restarts, m=args.m, pinned=pinned,
                   cond_max=args.cond_max, verbose=not args.quiet)
    if res is None:
        print("no admissible collocation configuration found")
        return

    pub = published_upper(args.d, args.sign)
    lb, lb_name = best_lower_bound(args.d, args.sign)
    print("=" * 74)
    print(f"collocation   d = {args.d}  s = {args.sign:+d}  k = {res.fam.k}  "
          f"m = {len(res.taus)} double roots   ({res.n_evals} evaluations)")
    print(f"widths : {np.array2string(res.fam.scales, precision=8)}")
    print(f"taus   : {np.array2string(res.taus, precision=8)}")
    print(f"rho (longdouble) : {res.rho:.12f}")
    if pub:
        print(f"published        : {pub:.6f}   "
              f"({'BEATS IT' if res.rho < pub else 'above it'} by "
              f"{abs(res.rho - pub):.6f})")
    print(f"best lower bound : {lb:.6f}  ({lb_name})")
    print(f"rho*sqrt(2pi/d)  : {res.rho / cdg_asymptote(args.d):.6f}")

    if mp is not None:
        _, _, info = polish_mp(res.fam, res.taus, dps=args.dps, pinned=pinned)
        print("-" * 74)
        print(f"mpmath polish at {args.dps} digits")
        print(f"  rho        : {info['rho']}")
        print(f"  |c|_max    : {info['max_abs_coeff']}   g(0) = {info['g_at_0']}")
        print(f"  interior minima past rho: {info['n_interior_minima']}, "
              f"worst dip {info['worst_dip']} at u = {info['worst_dip_u']}")
        print(f"  tail sign  : {info['tail_sign']}")
        agree = abs(info["rho_float"] - res.rho) / max(res.rho, 1e-30)
        print(f"  cond(M)    : {collocation_cond(res.fam, res.taus, pinned):.3e}")
        print(f"  longdouble vs mpmath rho: relative {agree:.2e}"
              + ("   *** DISAGREEMENT -- the longdouble number is noise; "
                 "trust the mpmath value ***" if agree > 1e-8 else ""))
        if args.json:
            # Write the same high-precision interchange schema consumed by
            # verify_mp.py.  Do not truncate to 25 digits: double contacts are
            # exactly where serialization error can manufacture a tiny negative
            # lobe.
            doc = {
                "format": "sign-uncertainty-candidate/1",
                "d": args.d,
                "s": args.sign,
                "widths": [repr(float(x)) for x in res.fam.scales],
                "taus": [repr(float(x)) for x in res.taus],
                "coeffs": info["coeffs_mp"],
                "u0": info["u0"],
                "rho": info["rho"],
                "precision_digits": args.dps,
                "source": "collocate+mpmath-polish",
                "pinned": ([] if pinned is None else list(pinned)),
                "notes": ("geometry discovered by collocate.py; coefficients re-solved "
                          f"in mpmath at {args.dps} digits before serialization"),
            }
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1)
            print(f"  written to {args.json}")


if __name__ == "__main__":
    main()
