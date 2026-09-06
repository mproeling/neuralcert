#!/usr/bin/env python3
"""
laguerre_lp.py -- GLOBAL solver for the classical (Cohn-Goncalves) family.

THE MISSING LINK

Every plateau in this project so far -- 0.5785 for pure Gaussians, 0.5929 /
0.5850 for pure Laguerre -- came out of a NONCONVEX optimisation over contact
points tau (Nelder-Mead, Adam), which finds a basin, not the optimum.  We kept
treating that as a numerical nuisance.  It is the whole problem, and for the
classical family it is entirely avoidable, because of one structural fact we
had verified but never exploited:

    For polynomial x Gaussian, the rescaled function g(u) = e^u f(u) IS A
    POLYNOMIAL, and the eigenfunction constraint fhat = f is LINEAR: it says
    P lies in V = span{ L_{2m}^{alpha}(2u) : m = 0..N }, alpha = d/2 - 1.
    (Hermite: H_{4m}(sqrt(2pi) r) e^{-pi r^2} are the +1 eigenfunctions, and
    H_{2n}(y) = (-1)^n 2^{2n} n! L_n^{(-1/2)}(y^2) turns them into exactly
    these Laguerre polynomials in u = pi r^2.)

    So for FIXED u0 the question "is there P in V, P(0) = 0, P >= 0 on
    [u0, oo), P != 0" is a CONVEX feasibility problem in the coefficients.
    No tau.  No basins.  Nothing to seed.  Feasibility is monotone in u0, so
    bisection on u0 gives the GLOBAL optimum of the family at each degree.

    And verification is EXACT: nonnegativity of a univariate polynomial on a
    ray is decidable (all roots via mpmath polyroots on exact dyadic
    coefficients; sign at midpoints), so the "grid slip" that killed the
    naive LP is closed by cutting planes at true minimisers, with the
    authority being root isolation, not any grid.

Why this should beat 0.572990 at d = 1, 2: Cohn-Goncalves state their d <= 2
entries are NOT converged -- their Newton method fails on the last-sign-change
discontinuity there.  A convex method has no such failure mode, and the family
itself keeps improving with degree at fixed d (the CDG obstruction is about
degree o(d) as d -> oo, it says nothing about fixed small d).

Pipeline per degree:
    1. LP + exact-minimiser cutting planes, bisection on u0  -> global
       structure: u0*, the tangency locations tau*.
       (The LP number is a grid RELAXATION: an estimate, not a bound.)
    2. Square collocation at tau* in mpmath (exact double roots)  -> a
       genuinely admissible f, verified by polyroots  -> a QUOTABLE upper
       bound for A_s(d).
    3. Both numbers printed; they should sandwich tightly.

The LP dual is also mathematically meaningful: an infeasibility certificate
is a nonnegative measure on [u0, oo) orthogonal to V through P(0) -- a
Krein-type quadrature -- i.e. a proof of optimality WITHIN the family.  We
print the dual multipliers' support for inspection.

Usage
    neuralcert sign laguerre --d 1 --n-basis 12       # CG-comparable, deg 22
    neuralcert sign laguerre --d 1 --n-basis 24 --json out.json
    neuralcert sign laguerre --d 2 --n-basis 20
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from .laguerre_basis import lag_coeffs_frac, orders_for

try:
    import mpmath as mp
except ImportError:                                          # pragma: no cover
    mp = None

XP = np.longdouble


# ---------------------------------------------------------------------------
# Exact Laguerre coefficients (rational for every d: alpha = (d-2)/2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stable evaluation by the three-term recurrence (never monomials)
# ---------------------------------------------------------------------------

def lag_table(u, top: int, alpha: float, dtype=XP) -> np.ndarray:
    """L_k^{alpha}(2u) for k = 0..top; shape (len(u), top+1)."""
    u = np.atleast_1d(np.asarray(u, dtype=dtype))
    x = dtype(2) * u
    T = np.empty((len(u), top + 1), dtype=dtype)
    T[:, 0] = dtype(1)
    if top >= 1:
        T[:, 1] = dtype(1 + alpha) - x
    for k in range(1, top):
        T[:, k + 1] = ((dtype(2 * k + 1 + alpha) - x) * T[:, k]
                       - dtype(k + alpha) * T[:, k - 1]) / dtype(k + 1)
    return T


def basis_vals(u, orders, alpha, dtype=XP) -> np.ndarray:
    T = lag_table(u, max(orders), alpha, dtype)
    return T[:, orders]


def basis_dvals(u, orders, alpha, dtype=XP) -> np.ndarray:
    """d/du L_n^{alpha}(2u) = -2 L_{n-1}^{alpha+1}(2u)."""
    top = max(orders)
    T1 = lag_table(u, max(top - 1, 0), alpha + 1.0, dtype)
    u = np.atleast_1d(np.asarray(u, dtype=dtype))
    out = np.zeros((len(u), len(orders)), dtype=dtype)
    for i, n in enumerate(orders):
        if n >= 1:
            out[:, i] = dtype(-2) * T1[:, n - 1]
    return out


# ---------------------------------------------------------------------------
# The LP oracle with exact-minimiser cutting planes
# ---------------------------------------------------------------------------

def mass_vector(u0: float, orders, d: int) -> np.ndarray:
    """m_n = e^{u0} * int_{u0}^oo L_n(2u) e^{-u} du -- strictly positive on the
    feasible cone minus {0} (P >= 0 on the ray and P analytic).  Using it as
    the normalisation removes the degenerate-leading-coefficient sweep
    entirely; any positive normalisation of a cone gives the same u0*."""
    out = []
    for n in orders:
        coef = lag_coeffs_frac(n, d)
        m = 0.0
        for j, a in enumerate(coef):
            # int_{u0}^oo u^j e^{-u} du = j! e^{-u0} sum_{i<=j} u0^i / i!
            tail = sum(u0 ** i / math.factorial(i) for i in range(j + 1))
            m += float(a) * math.factorial(j) * tail
        out.append(m)
    v = np.array(out)
    return v / np.linalg.norm(v)


def scan_and_cut(c: np.ndarray, u0: float, orders, alpha,
                 tol_rel: float = 3e-9, n_scan: int = 24000):
    """
    Locate every local minimiser of P on [u0, u_end] by sign changes of P'
    (recurrence evaluation, longdouble -- the Laguerre basis is
    near-orthogonal, so cancellation is |c|-scale, not monomial-scale).

    Returns (violations, tangencies, worst_rel, worst_u).  Beyond
    u_end = 4*top + 60 every basis element is past its largest zero
    (largest zero of L_k(2u) sits below u ~ 2k+2), which we double-check by
    requiring P and P' positive at the end of the scan.
    """
    top = max(orders)
    u_end = 4.0 * top + 60.0
    u = np.unique(np.concatenate([
        np.linspace(u0, u_end, n_scan),
        np.geomspace(max(u0, 1e-9), u_end, n_scan),
        np.array([u0, u_end])])).astype(XP)
    V = basis_vals(u, orders, alpha)
    Vd = basis_dvals(u, orders, alpha)
    cx = np.asarray(c, dtype=XP)
    P = V @ cx
    mag = np.maximum(np.abs(V) @ np.abs(cx), XP(1e-4930))
    Pd = Vd @ cx

    crits = []
    sg = np.sign(Pd)
    idx = np.where(sg[1:] * sg[:-1] < 0)[0]
    for i in idx:
        lo, hi, s0 = u[i], u[i + 1], sg[i]
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.sign(float(basis_dvals(mid, orders, alpha)[0] @ cx)) == s0:
                lo = mid
            else:
                hi = mid
        crits.append(float((lo + hi) / 2))

    pts = np.array([float(u0)] + crits + [float(u_end)], dtype=XP)
    Vp = basis_vals(pts, orders, alpha)
    vals = Vp @ cx
    mags = np.maximum(np.abs(Vp) @ np.abs(cx), XP(1e-4930))
    rel = np.asarray(vals / mags, dtype=float)

    worst = int(np.argmin(rel))
    violations = [float(pts[i]) for i in range(len(pts))
                  if rel[i] < -tol_rel]          # includes the endpoint u0
    tangencies = [float(pts[i]) for i in range(1, len(pts) - 1)
                  if abs(rel[i]) <= 1e-6]
    minima = sorted(((float(rel[i]), float(pts[i]))
                     for i in range(1, len(pts) - 1)), key=lambda t: t[0])
    # Tail: P is a polynomial, so "eventually positive" is decidable by
    # probing far beyond every Laguerre zero (all zeros of the basis sit
    # below u ~ 2*top + 2).  Far anchors as LP ROWS were a disaster -- their
    # entries reach 1e29 and HiGHS fails outright on the coefficient range --
    # so the far probes live only here, in longdouble, where 1e29 is nothing.
    far = np.array([u_end, 3.0 * u_end, 10.0 * u_end], dtype=XP)
    fvals = basis_vals(far, orders, alpha) @ cx
    tail_ok = (float(Pd[-1]) > 0 and all(float(v) > 0 for v in fvals))
    return (violations, tangencies, float(rel[worst]), float(pts[worst]),
            tail_ok, minima)


T_CAP = 1e3


def lp_oracle(u0: float, orders, d: int, cutpool: list,
              n_grid: int = 1400, max_cuts: int = 70, tol_rel: float = 3e-9,
              tol_margin: float = 1e-9, tol_stale: float = 1e-6):
    """
    Feasibility of {P in V : P(0)=0, mass=1, P >= 0 on [u0, oo)} by MARGIN
    MAXIMISATION (max t s.t. P(u_j) >= t), with three scaling decisions that
    are each load-bearing:

    * columns scaled over the oscillatory window only -- whole-grid scaling is
      dominated by the far end and pushes genuine solutions outside any
      finite variable bound (that produced a bogus 0.6010 "optimum");
    * EVERY inequality row equilibrated to unit sup-norm -- P(u_j) >= 0 is
      scale-free, and this is what lets far anchors (raw entries ~1e29) sit
      in the LP without wrecking HiGHS;
    * far anchors at 1.5x, 3x, 10x the window pin the tail sign, killing the
      degenerate-leading pathology (c_top parked at 0, next order negative,
      P -> -infinity just past the window) with no auxiliary constraint.

    Margin rather than bare feasibility because an objective-0 vertex hugs
    zero at many grid points and dips between them, and the cut loop crawls;
    for u0 strictly above the optimum an interior point exists, so t* > 0
    iff u0 > u0*, and the margin solution passes the scan almost at once.
    """
    alpha = d / 2.0 - 1.0
    top = max(orders)
    u_end = 4.0 * top + 60.0
    base = np.unique(np.concatenate([
        np.linspace(u0, u_end, n_grid),
        np.geomspace(max(u0, 1e-9), u_end, n_grid),
        np.array([u0, u_end])]))
    z = np.asarray(basis_vals(0.0, orders, alpha), dtype=float)[0]
    mvec = mass_vector(u0, orders, d)
    k = len(orders)

    for it in range(max_cuts + 1):
        pts = np.unique(np.concatenate(
            [base, np.array([x for x in cutpool if x >= u0], dtype=float)]))
        B = np.asarray(basis_vals(pts, orders, alpha), dtype=float)
        win = pts <= 2.0 * top + 10.0
        # NO column scaling.  The raw Laguerre-coefficient representation is
        # already the right coordinate system: measured optimal solutions
        # have c_n = O(0.1) uniformly.  Window-sup column scaling inflated
        # the same solutions to |c_scaled| ~ 3e7, and HiGHS's feasibility
        # tolerances -- relative to the solution scale -- then admitted
        # points violating the u0 row by 7e-3 and the P(0)=0 equality by
        # 4e-3 in absolute terms, which is exactly the "solution crosses
        # zero at 1.1095 regardless of the constraint" behaviour that pinned
        # three successive runs.  Row equilibration alone keeps every matrix
        # entry <= 1 with solutions O(0.1): tolerances then mean what they
        # say.
        colscale = np.ones(len(orders))
        Bs = B
        rowscale = np.maximum(np.max(np.abs(Bs), axis=1), 1e-300)
        Bs = Bs / rowscale[:, None]
        r0 = z / colscale
        r0 = r0 / max(np.linalg.norm(r0), 1e-300)
        r1 = mvec / colscale
        r1 = r1 / max(np.linalg.norm(r1), 1e-300)
        # Margin applies ONLY inside the oscillatory window.  A uniform
        # margin over every row silently excludes the degenerate-top corner:
        # for a solution with c_top = 0 (the previous rung embedded in this
        # one -- e.g. the verified deg-6 optimum inside dim 6), P at a far
        # anchor is positive but ~u^-2 of the row's top-column magnitude, so
        # "P/rowmax >= t" forces t ~ 0 and the margin test calls a feasible
        # u0 infeasible.  That is precisely what pinned the last run at
        # 1.1095 when the family provably reaches 1.10609.  Far rows keep
        # plain P >= 0, which every genuine solution satisfies with room.
        marg = win.astype(float)[:, None]
        # Tail sign as a STRUCTURAL row, c_top >= 0, not as far anchor rows.
        # Far anchors (u = 150/300/1000) are, after row normalisation, nearly
        # parallel to e_top and mutually near-duplicate; measured effect:
        # HiGHS reports "optimal" while its solution violates the u0 row by
        # 7e-3 in normalised units, and no cut can fix a violation AT an
        # existing constraint point -- that is what the 70-iteration crawl
        # was.  The clean row costs nothing and says the same thing: the sign
        # of P at +infinity is the sign of c_top (top order is even).  The
        # degenerate corner c_top = 0 is the previous rung of the ladder,
        # which is run separately and folded in by min.
        tailrow = np.zeros((1, k + 1))
        tailrow[0, k - 1] = -1.0
        A_ub = np.vstack([np.hstack([-Bs, marg]), tailrow])
        A_eq = np.hstack([np.vstack([r0, r1]), np.zeros((2, 1))])
        res = linprog(c=np.concatenate([np.zeros(k), [-1.0]]),
                      A_ub=A_ub, b_ub=np.zeros(len(pts) + 1),
                      A_eq=A_eq, b_eq=np.array([0.0, 1.0]),
                      # The grid relaxation can have a nonempty recession
                      # cone at high degree: a polynomial may be nonnegative
                      # at every grid point while dipping between them. Cap
                      # the margin so HiGHS returns a candidate that the
                      # continuous scan can cut instead of status=unbounded.
                      bounds=[(None, None)] * k + [(None, T_CAP)],
                      method="highs",
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10})
        if not res.success:
            if res.status == 3:
                cutpool.extend(list(np.linspace(u0, 2.0 * top + 10.0, 400)))
                continue
            return None, it
        if res.x[-1] <= tol_margin:
            return None, it
        c = res.x[:k] / colscale
        viols, tangs, worst_rel, worst_u, tail_ok, minima = scan_and_cut(
            c, u0, orders, alpha, tol_rel)
        import os
        if os.environ.get("LLP_DEBUG"):
            print(f"      [oracle u0={u0:.6f} it={it}] t*={res.x[-1]:.3e} "
                  f"nviol={len(viols)} worst={worst_rel:+.2e}@u={worst_u:.6f} "
                  f"tail_ok={tail_ok} npts={len(pts)}", flush=True)
        if not tail_ok:
            cutpool.extend([2.0 * u_end, 6.0 * u_end])
            continue
        if not viols:
            t_star = float(res.x[-1])
            # near the feasibility boundary the true tangencies are the
            # window minima sitting at the margin level, not at 1e-6
            tangs = sorted({round(u, 9) for r, u in minima
                            if r <= max(10.0 * t_star, 1e-6)})
            return {"coeffs": c, "tangencies": tangs, "min_rel": worst_rel,
                    "min_u": worst_u, "cuts_used": it, "minima": minima,
                    "margin": t_star}, it
        stale = [v for v in viols
                 if np.min(np.abs(pts - v)) < 1e-9 * max(v, 1.0)]
        if stale and worst_rel < -tol_stale:
            # A deep violation at an existing row is a solver failure and
            # cannot be repaired by adding the same constraint again.
            import sys
            print(f"      WARNING: LP violates its own row at u={stale[0]:.9f} "
                  f"by {worst_rel:.2e}; treating u0 as infeasible",
                  file=sys.stderr, flush=True)
            return None, it
        if stale:
            # At high degree, shallow residuals at constrained points are at
            # the float64 solver's accuracy floor. Accept the LP structure;
            # collocation and complete-root verification independently decide
            # whether the resulting polynomial is valid.
            t_star = float(res.x[-1])
            tangs = sorted({round(u, 9) for r, u in minima
                            if r <= max(10.0 * t_star, 1e-6)})
            return {"coeffs": c, "tangencies": tangs, "min_rel": worst_rel,
                    "min_u": worst_u, "cuts_used": it, "margin": t_star,
                    "lp_noise": True}, it
        for v in viols[:24]:
            h = 1e-3 * max(v, 1.0)
            cutpool.extend([v - h, v, v + h])
    return None, max_cuts       # cut budget exhausted counts as infeasible


def minimise_u0(orders, d: int, u_lo: float, u_hi: float, bisect: int = 42,
                verbose: bool = True, **kw):
    cutpool: list = []
    sol, _ = lp_oracle(u_hi, orders, d, cutpool, **kw)
    if sol is None:
        return None, None
    lo, hi, best = u_lo, u_hi, sol
    for step in range(bisect):
        mid = 0.5 * (lo + hi)
        s, cuts = lp_oracle(mid, orders, d, cutpool, **kw)
        if s is None:
            lo = mid
        else:
            hi, best = mid, s
        if verbose and step % 6 == 0:
            print(f"    bisect[{step:2d}] u0 in ({lo:.9f}, {hi:.9f}]  "
                  f"rho <= {math.sqrt(hi / math.pi):.9f}  "
                  f"(cut pool {len(cutpool)})", flush=True)
    best["u0"] = hi
    best["rho"] = math.sqrt(hi / math.pi)
    best["hit_lower_bracket"] = hi <= u_lo * 1.01
    return best, cutpool


# ---------------------------------------------------------------------------
# Step 2: exact collocation at the LP tangencies -> a quotable upper bound
# ---------------------------------------------------------------------------

def collocate_exact(taus, orders, d: int, dps: int = 60):
    """
    Solve, in mpmath, for P in V with P(0) = 0, double roots at taus, and the
    top-order coefficient fixed.  Requires len(taus) = (n_basis - 2) / 2 --
    the square count.  Returns (c_mp, poly_coeffs_mp ascending).
    """
    mp.mp.dps = dps
    coef = {n: [mp.mpf(x.numerator) / mp.mpf(x.denominator)
                for x in lag_coeffs_frac(n, d)] for n in orders}

    def ev(n, t):
        return mp.polyval(list(reversed(coef[n])), t)

    def dv(n, t):
        cc = coef[n]
        return mp.polyval(list(reversed([cc[i] * i for i in range(1, len(cc))])),
                          t) if len(cc) > 1 else mp.mpf(0)

    T = [mp.mpf(repr(float(t))) for t in taus]
    rows = [[ev(n, mp.mpf(0)) for n in orders]]
    rows += [[ev(n, t) for n in orders] for t in T]
    rows += [[dv(n, t) for n in orders] for t in T]
    nmax = max(orders)
    rows.append([(coef[n][-1] if n == nmax else mp.mpf(0)) for n in orders])
    rhs = [mp.mpf(0)] * (1 + 2 * len(T)) + [mp.mpf(1)]
    if len(rows) != len(orders):
        raise ValueError(f"square collocation needs {(len(orders)-2)//2} taus, "
                         f"got {len(T)}")
    # row equilibration (solution unchanged)
    A = mp.matrix(rows)
    b = mp.matrix(rhs)
    for i in range(A.rows):
        sc = mp.sqrt(mp.fsum(A[i, j] ** 2 for j in range(A.cols)))
        if sc > 0:
            for j in range(A.cols):
                A[i, j] /= sc
            b[i] /= sc
    c = mp.lu_solve(A, b)
    D = nmax
    poly = [mp.mpf(0)] * (D + 1)
    for cj, n in zip(c, orders):
        for i, pc in enumerate(coef[n]):
            poly[i] += cj * pc
    return list(c), poly


def verify_poly(poly, dps: int = 60, root_extraprec: int = 260):
    """
    EXACT verdict for a polynomial: all roots via polyroots, cluster into
    multiplicities, find the last odd-multiplicity positive root u0, and check
    the sign of P at midpoints beyond it.  This is the completeness statement
    no scan or box scheme could give.
    """
    mp.mp.dps = dps
    # trim trailing (numerically) zero leading coefficients
    scale = max(abs(x) for x in poly)
    D = len(poly) - 1
    while D > 0 and abs(poly[D]) < scale * mp.mpf(10) ** (-dps + 8):
        D -= 1
    p = poly[:D + 1]
    roots = mp.polyroots(list(reversed(p)), maxsteps=300,
                         extraprec=root_extraprec)
    tol = mp.mpf(10) ** (-dps // 3)
    real = sorted(mp.re(r) for r in roots
                  if abs(mp.im(r)) < tol * (1 + abs(r)) and mp.re(r) > tol)
    # cluster
    clusters: list[list] = []
    for r in real:
        if clusters and abs(r - clusters[-1][-1]) < tol * (1 + abs(r)):
            clusters[-1].append(r)
        else:
            clusters.append([r])
    info = [(sum(cl) / len(cl), len(cl)) for cl in clusters]
    odd = [r for r, mlt in info if mlt % 2 == 1]
    u0 = odd[-1] if odd else mp.mpf(0)
    # sign check at midpoints beyond u0 and at the far end
    def P(u):
        return mp.polyval(list(reversed(p)), u)
    pts = [u0 + 1] if not info else []
    beyond = [r for r, _ in info if r > u0]
    seq = [u0] + beyond + [(beyond[-1] if beyond else u0) + 10]
    pts = [(seq[i] + seq[i + 1]) / 2 for i in range(len(seq) - 1)]
    ok = all(P(t) > 0 for t in pts) and p[D] > 0
    return {"u0": u0, "rho": mp.sqrt(u0 / mp.pi), "roots": info,
            "lead_positive": bool(p[D] > 0), "verdict": "PASS" if ok else "REJECT",
            "degree_effective": D}


def compare_published(reference: float, published: float, digits: int = 6) -> str:
    """Classify a result at the precision of the published decimal.

    A difference smaller than half a unit in the last published place is a
    reproduction, not evidence for a strict improvement.
    """
    half_ulp = 0.5 * 10 ** (-digits)
    delta = published - reference
    if delta > half_ulp:
        return "*** BELOW PUBLISHED (genuine improvement) ***"
    if abs(delta) <= half_ulp:
        return ("MATCHES published to its reported precision "
                "(reproduction, not an improvement)")
    return f"above by {-delta:.6f}"


# ---------------------------------------------------------------------------

def lb_cdg(d: int, degree: int) -> float:
    """Return the finite-degree lower bound from CDG Theorem 2.1.

    For ``f = p(2u)e^-u`` with ``deg(p) <= degree`` and ``fhat(0) = 0``, the
    last sign change satisfies ``u0 >= lambda``, where ``lambda`` is the
    smallest root of ``L_m^(d/2-1)`` and ``m = floor(degree/2) + 1``.
    """
    if d < 1:
        raise ValueError("d must be positive")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if mp is None:
        return 0.0
    m = degree // 2 + 1
    alpha = mp.mpf(d) / 2 - 1
    coefficients = [
        mp.binomial(m + alpha, m - j) * mp.mpf(-1) ** j / mp.factorial(j)
        for j in range(m + 1)
    ]
    roots = mp.polyroots(list(reversed(coefficients)), maxsteps=200,
                         extraprec=300)
    return float(min(mp.re(root) for root in roots))


def lb_bck_d1() -> float:
    """[BCK] Theorem 1: A_+(1) >= 1 / (2(1+lambda)), lambda = -min sinc."""
    from scipy.optimize import brentq
    xs = brentq(lambda x: x * math.cos(x) - math.sin(x), 3.6, 4.7)
    lam = -math.sin(xs) / xs
    return 1.0 / (2.0 * (1.0 + lam))


def run(d: int, s_sign: int, n_basis: int, bisect: int, dps: int,
        json_path: str | None, verbose: bool,
        u_lo_cli: float | None = None, u_hi_cli: float | None = None):
    orders = orders_for(n_basis, s_sign)
    deg = max(orders)
    print(f"family: L_n^({d}/2-1)(2u) e^-u,  orders {orders[0]}..{deg} "
          f"(dim {n_basis}, degree {deg}),  d={d}, s={s_sign:+d}")
    lb = lb_bck_d1() if (d == 1 and s_sign == 1) else None
    lam = lb_cdg(d, deg)
    if lb:
        print(f"BCK Thm 1 lower bound : rho >= {lb:.7f}   "
              f"(u0 >= {math.pi*lb*lb:.6f})")
    print(f"CDG Thm 2.1 lower bound: u0 >= lambda = {lam:.8f}   "
          f"(rho >= {math.sqrt(lam/math.pi):.7f})   "
          f"[smallest root of L_{deg//2+1}^({d}/2-1)]")

    u_lo = max(lam, math.pi * lb ** 2 if lb else 0.0) * 0.999
    if u_lo <= 0:
        u_lo = 0.35
    if u_lo_cli is not None:
        u_lo = u_lo_cli
    # The sublinear-degree asymptote has u0 approximately d/2. Keep a
    # comfortable default upper bracket, while allowing exact reproduction
    # runs to override either endpoint.
    u_hi = (u_hi_cli if u_hi_cli is not None else
            (1.35 if d == 1 else (2.01 if d == 2 else 0.85 * d)))

    best = None
    for _attempt in range(5):
        best, cutpool = minimise_u0(orders, d, u_lo, u_hi, bisect=bisect,
                                    verbose=verbose)
        if best is not None:
            break
        u_hi *= 1.8
        print(f"    infeasible at the upper bracket; retrying with "
              f"u_hi = {u_hi:.4f}")
    if best is None:
        print(f"LP infeasible even at u_hi = {u_hi:.4f} -- pass --u-hi "
              f"explicitly, or the degree may be too high for the grid")
        return
    print("-" * 74)
    if best.get("hit_lower_bracket"):
        print("*** LP BREAKDOWN: the bisection reached its rigorous lower "
              "bracket. The float64 LP and scan cannot decide feasibility "
              "at this degree; discard this estimate. ***")
    print(f"[1] LP global structure  (grid relaxation -- ESTIMATE, not a bound)")
    print(f"    u0* = {best['u0']:.10f}   rho ~ {best['rho']:.10f}")
    print(f"    tangencies: {np.round(np.array(best['tangencies']), 6)}")
    print(f"    scan worst rel = {best['min_rel']:+.2e} at u = {best['min_u']:.6f}")

    quote = None
    tangs = list(best["tangencies"])
    # The LP optimum need not use the full dimension: c_top = 0 means the
    # optimum lives in a smaller rung, and the honest polish is the SQUARE
    # collocation of that rung -- n_eff = 2*len(tangs) + 2 basis elements.
    # Deep minima are NOT tangencies; inventing contacts from them produced a
    # "verified" 0.708 in an earlier run, a valid function that is nowhere
    # near the optimum.  Better no polish than a misleading one.
    n_eff = 2 * len(tangs) + 2
    if mp is None:
        print("mpmath unavailable; cannot polish")
    elif not tangs:
        print("[2] no tangencies detected; polish skipped")
    else:
        # n_eff may EXCEED the LP's dimension: near a rung boundary the
        # optimum announces the next rung's contact structure early, and the
        # collocation is then simply done in that larger space.  The verified
        # object stands on its own through the complete root list, so
        # polishing above the LP's dimension is legitimate -- refusing it
        # left the n=20 run without a quotable number.
        orders_eff = (orders[:n_eff] if n_eff <= n_basis
                      else orders_for(n_eff, s_sign))
        if n_eff != n_basis:
            print(f"    (tangency count {len(tangs)} -> polishing in rung "
                  f"n_basis = {n_eff}, degree {max(orders_eff)})")
        c_mp, poly = collocate_exact(tangs, orders_eff, d, dps=dps)
        v = verify_poly(poly, dps=dps)
        print(f"[2] exact collocation at the LP tangencies  ({dps} digits)")
        print(f"    roots (value, multiplicity): "
              + ", ".join(f"({mp.nstr(r, 8)},{m})" for r, m in v["roots"]))
        print(f"    last sign change u0 = {mp.nstr(v['u0'], 14)}")
        print(f"    rho = {mp.nstr(v['rho'], 14)}   verdict: {v['verdict']}")

        # The margin LP can report one or two shallow minima as contacts near
        # a rung boundary. If collocating every candidate disagrees with the
        # LP estimate, try all square-rung subsets and retain the best one
        # that independently passes the complete-root verification.
        if (v["verdict"] != "PASS"
                or abs(float(v["rho"]) - best["rho"]) > 1e-4 * best["rho"]):
            square_contacts = (n_basis - 2) // 2
            excess = len(tangs) - square_contacts
            if 0 < excess <= 2:
                print(f"    polish disagrees with the LP estimate; trying "
                      f"leave-{excess}-out in the square rung "
                      f"n_basis = {n_basis}")
                candidates = []
                for keep in combinations(range(len(tangs)), square_contacts):
                    subset = [tangs[index] for index in keep]
                    try:
                        coeffs, candidate_poly = collocate_exact(
                            subset, orders, d, dps=dps)
                    except Exception:
                        continue
                    candidate_verdict = verify_poly(candidate_poly, dps=dps)
                    if candidate_verdict["verdict"] == "PASS":
                        candidates.append((float(candidate_verdict["rho"]),
                                           subset, coeffs, candidate_verdict))
                if candidates:
                    candidates.sort(key=lambda candidate: candidate[0])
                    rho, subset, coeffs, candidate_verdict = candidates[0]
                    dropped = [tau for tau in tangs if tau not in subset]
                    tied = sum(abs(candidate[0] - rho) < 1e-9
                               for candidate in candidates)
                    print(f"    dropped {np.round(np.array(dropped), 6)}; "
                          f"{tied} of {len(candidates)} subsets tie at this value")
                    print("    roots: " + ", ".join(
                        f"({mp.nstr(root, 8)},{multiplicity})"
                        for root, multiplicity in candidate_verdict["roots"]))
                    print(f"    rho = {mp.nstr(candidate_verdict['rho'], 14)}   "
                          f"verdict: {candidate_verdict['verdict']}")
                    tangs, c_mp, v = subset, coeffs, candidate_verdict
                else:
                    print("    no leave-one-out subset verified")

        if v["verdict"] == "PASS":
            quote = float(v["rho"])

    ref_rho = quote if quote is not None else best["rho"]
    print("-" * 74)
    print(f"asymptotic normalisation: rho*sqrt(2pi/d) = "
          f"{ref_rho*math.sqrt(2*math.pi/d):.6f}   "
          f"[CDG Thm 1.2 sublinear asymptote 1.000000; "
          f"AJCHT Conj 3.2 target {math.sqrt(2/math.pi):.6f}]")
    print(f"degree/d = {deg/d:.4f}   (CDG Thm 1.2 governs degree/d -> 0)")

    from .gaussian_mixture import published_upper
    pub = published_upper(d, s_sign)
    print("-" * 74)
    if pub:
        ref = quote if quote is not None else best["rho"]
        tag = "VERIFIED upper bound" if quote is not None else "LP estimate"
        print(f"published A_{'+' if s_sign==1 else '-'}({d}) = {pub:.6f}   "
              f"this run ({tag}) = {ref:.9f}")
        print(f"   {compare_published(ref, pub)}")
    if json_path and quote is not None:
        doc = {"format": "sign-uncertainty-candidate/2-laguerre",
               "d": d, "s": s_sign, "alpha": f"{d-2}/2",
               "laguerre_orders": orders,
               "coeffs": [mp.nstr(x, 25) for x in c_mp],
               "taus": [repr(float(t)) for t in tangs],
               "u0": mp.nstr(v["u0"], 30), "rho": mp.nstr(v["rho"], 30),
               "precision_digits": dps, "source": "laguerre-lp+collocation",
               "verdict": v["verdict"]}
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        print(f"written to {json_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--n-basis", type=int, default=12,
                    help="dimension of the eigenspace slice; degree = 2(n-1)")
    ap.add_argument("--bisect", type=int, default=42)
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument("--json", default=None)
    ap.add_argument("--u-lo", type=float, default=None,
                    help="lower bracket for u0 (default: best rigorous bound)")
    ap.add_argument("--u-hi", type=float, default=None,
                    help="upper bracket for u0 (default: 0.85*d for d > 2)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--polish", default=None,
                    help="comma-separated taus: skip the LP, collocate+verify "
                         "in rung n_basis = 2*len(taus)+2")
    a = ap.parse_args()
    if a.polish:
        taus = [float(v) for v in a.polish.split(",")]
        rung = 2 * len(taus) + 2
        ro = orders_for(rung, a.sign)
        print(f"polish-only: {len(taus)} taus -> rung n_basis={rung}, "
              f"degree {max(ro)}, d={a.d}")
        c_mp, poly = collocate_exact(taus, ro, a.d, dps=a.dps)
        v = verify_poly(poly, dps=a.dps)
        print("roots: " + ", ".join(f"({mp.nstr(r,8)},{m})" for r, m in v["roots"]))
        print(f"u0 = {mp.nstr(v['u0'],14)}   rho = {mp.nstr(v['rho'],14)}   "
              f"verdict: {v['verdict']}")
        from .gaussian_mixture import published_upper
        pub = published_upper(a.d, a.sign)
        if pub and v["verdict"] == "PASS":
            r = float(v["rho"])
            print(f"published {pub:.6f}   verified {r:.9f}")
            print(f"   {compare_published(r, pub)}")
        return
    run(a.d, a.sign, a.n_basis, a.bisect, a.dps, a.json, not a.quiet,
        u_lo_cli=a.u_lo, u_hi_cli=a.u_hi)


if __name__ == "__main__":
    main()
