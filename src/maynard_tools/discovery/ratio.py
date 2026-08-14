"""
maynard_cluster_v3.py -- clustered confluent basis for M_k discovery.

BASIS
=====
Base points c_1 < ... < c_r with multiplicities mu_1..mu_r; the channels are

    phi_{a,j}(t) = (c_a + n t)^{-j},   j = 1..mu_a,   n = k-1,

i.e. divided differences WITHIN a cluster (which for this family collapse
exactly onto powers, since d^j/dc^j (c+nt)^{-1} = (-1)^j j! (c+nt)^{-(j+1)})
and distinct scales BETWEEN clusters.  mu = (1,1,...,1) recovers the plain
multi-c family; r = 1 recovers the pure confluent family.  The point is that
lambda_min(A) no longer collapses when two channels approach: coalescence is
absorbed into a multiplicity instead of falling off the rank_tol cliff.

STABLE F_p   (the v2 bug)
=========================
    F_p(th) = int_0^1 e^{i th t}(c+nt)^{-p} dt,
    F_p = -[e^{i th}(c+n)^{1-p} - c^{1-p} - i th F_{p-1}] / (n(p-1)),  F_1 = cf.

Upward recursion multiplies the error by |th|/(n(p-1)) per step, so the total
amplification from p=1 to pmax is bounded by

    Amp(th) = |th|^{pmax-1} / (n^{pmax-1} (pmax-1)!)      (when > 1).

v2 evaluated this at |th| ~ 2e4 with n = 50, where Amp ~ 1e11 -- hence the
mass = -4e41 blow-up at q = 8.  The repair is not a cleverer recursion but
the observation that phi = (cf/m0)^n is ZERO to machine precision there:
|phi| > 1e-16 needs |cf|/m0 > exp(-37/n), i.e. th within O(1/sigma_S) of the
origin, and sigma_S ~ 1/log k.  So we

    * solve Amp(th) <= AMP_BUDGET for th_stable,
    * evaluate F_p only on |th| <= th_stable and set phi = 0 beyond,
    * CHECK |phi| at the mask edge is below PHI_EDGE_TOL, and raise if not.

That is a gate, not an assumption: if the needed th range ever reaches into
the unstable region the evaluator refuses rather than returning noise.

CROSS-CLUSTER PRODUCT
=====================
With u = c_a + n t and d = c_b - c_a, expanding (u+d)^{-l} about u = 0,

    1/(u^j (u+d)^l) = sum_{r=1}^{j} A_r u^{-r} + sum_{s=1}^{l} B_s (u+d)^{-s},
    A_r = (-1)^{j-r} C(l+j-r-1, j-r) / d^{l+j-r},
    B_s = (-1)^{l-s} C(j+l-s-1, l-s) / (-d)^{j+l-s},

so every cross product is a short combination of F_p at the two base points,
and H and M1 likewise.  The coefficients carry d^{-(j+l-r)}, so cancellation
grows as (c/d)^{j+l-1}: clusters must stay separated and multiplicities
modest.  This is measured per pair and gated (CROSS_AMP_MAX), not hoped for.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np
from scipy.optimize import minimize
from scipy.special import sici

AMP_BUDGET = 1e4          # digits we are willing to lose in the F_p recursion
PHI_EDGE_TOL = 1e-13      # |phi| allowed at the masked edge
CROSS_AMP_MAX = 1e10      # partial-fraction cancellation budget


# ---------------------------------------------------------------------------
def _cf_simple(th: np.ndarray, c: float, n: int, L: float = 1.0) -> np.ndarray:
    """F_1 = int_0^L e^{i th t}/(c + n t) dt,  L = 1 + eps."""
    th = np.asarray(th, float)
    out = np.empty(th.shape, dtype=complex)
    z = np.abs(th)
    small = z < 1e-12
    if np.any(small):
        out[small] = math.log1p(n * L / c) / n
    big = ~small
    if np.any(big):
        lam = z[big] / n
        Sib, Cib = sici(lam * (c + n * L))
        Sia, Cia = sici(lam * c)
        val = np.exp(-1j * lam * c) / n * ((Cib - Cia) + 1j * (Sib - Sia))
        out[big] = np.where(th[big] < 0, np.conj(val), val)
    return out


def theta_stable(n: int, pmax: int, budget: float = AMP_BUDGET) -> float:
    """Largest |th| for which upward recursion to pmax loses < budget."""
    if pmax <= 1:
        return np.inf
    e = pmax - 1
    return float(n * (budget * math.factorial(e)) ** (1.0 / e))


def F_powers(th: np.ndarray, c: float, n: int, pmax: int, L: float = 1.0):
    """F_1..F_pmax on [0, L]; caller must mask to |th| <= theta_stable.

      F_p = -[e^{i th L}(c+nL)^{1-p} - c^{1-p} - i th F_{p-1}] / (n(p-1)).
    """
    F = [None, _cf_simple(th, c, n, L)]
    e1, cn = np.exp(1j * np.asarray(th, float) * L), c + n * L
    for p in range(2, pmax + 1):
        F.append(-(e1 * cn ** (1 - p) - c ** (1 - p)
                   - 1j * th * F[p - 1]) / (n * (p - 1)))
    return F


def _pow_int(c: float, n: int, y, s: int):
    """int_0^y (c + n t)^{-s} dt."""
    y = np.asarray(y, float)
    if s == 1:
        return np.log1p(n * y / c) / n
    return (c ** (1 - s) - (c + n * y) ** (1 - s)) / (n * (s - 1))


def _pf_coeffs(j: int, l: int, d: float):
    """(A_1..A_j at c_a, B_1..B_l at c_b) for 1/(u^j (u+d)^l)."""
    A = [(-1) ** (j - r) * comb(l + j - r - 1, j - r) / d ** (l + j - r)
         for r in range(1, j + 1)]
    B = [(-1) ** (l - s) * comb(j + l - s - 1, l - s) / (-d) ** (j + l - s)
         for s in range(1, l + 1)]
    return np.array(A), np.array(B)


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    N: int = 1 << 16
    P: float = 8.0

    @property
    def ds(self) -> float:
        return self.P / self.N

    def s(self):
        return np.arange(self.N) * self.ds

    def theta(self):
        return 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.ds)


def _pair_functions(ca, cb, j, l, n, L: float = 1.0):
    """(cf(th), H(y), M1) builders for phi_{a,j} phi_{b,l} on [0, L]."""
    one = np.array([L])
    if ca == cb:
        q = j + l
        cf = lambda th: F_powers(th, ca, n, q, L)[q]
        H = lambda y: _pow_int(ca, n, y, q)
        M1 = float((_pow_int(ca, n, one, q - 1) - ca * _pow_int(ca, n, one, q))[0]) / n
        amp = 1.0
        return cf, H, M1, amp, q
    d = cb - ca
    A, B = _pf_coeffs(j, l, d)
    scale = abs(ca ** (-(j)) * cb ** (-(l)))
    amp = float(max(np.max(np.abs(A) * ca ** -np.arange(1, j + 1)),
                    np.max(np.abs(B) * cb ** -np.arange(1, l + 1))) / scale)

    def cf(th):
        Fa, Fb = F_powers(th, ca, n, j, L), F_powers(th, cb, n, l, L)
        return sum(A[r - 1] * Fa[r] for r in range(1, j + 1)) + \
            sum(B[s - 1] * Fb[s] for s in range(1, l + 1))

    def H(y):
        return sum(A[r - 1] * _pow_int(ca, n, y, r) for r in range(1, j + 1)) + \
            sum(B[s - 1] * _pow_int(cb, n, y, s) for s in range(1, l + 1))

    def m1term(c, q):
        return float((_pow_int(c, n, one, q - 1) - c * _pow_int(c, n, one, q))[0]) / n

    M1 = sum(A[r - 1] * m1term(ca, r) for r in range(1, j + 1)) + \
        sum(B[s - 1] * m1term(cb, s) for s in range(1, l + 1))
    return cf, H, M1, amp, max(j, l)


@lru_cache(maxsize=1 << 14)
def _pair_entry(ca, cb, j, l, k, N, P, eps=0.0):
    """One Gram pair for the eps-enlarged problem.

    F lives on (1+eps)R_k.  With L = 1+eps, w = (c+nt)^{-2} on [0,L],
    m0 = int_0^L w, and S a sum of n i.i.d. draws from w/m0:

        A_jl = m0^n E[ H_jl(L-S) 1_{S <= L} ]                 (denominator)
        B_jl = m0^n E[ G_j(L-S) G_l(L-S) 1_{S <= 1-eps} ]     (numerator)

    The only structural change from eps = 0 is the asymmetry of the two
    indicators: F may occupy the enlarged simplex, but prime-detecting
    credit is collected only where the other coordinates satisfy
    s <= 1-eps = L-2eps.  In rho = L-S coordinates the numerator integral
    therefore starts at rho = 2eps.  The m0^n factor still cancels.
    """
    n = k - 1
    L = 1.0 + eps
    cf_f, H_f, M1, amp, pmax = _pair_functions(ca, cb, j, l, n, L)
    if amp > CROSS_AMP_MAX:
        raise FloatingPointError(
            f"cross-cluster cancellation {amp:.2e} for ({j},{l}) at "
            f"c={ca:.4g},{cb:.4g}: separate the clusters or lower mu")
    m0 = float(np.real(cf_f(np.zeros(1))[0]))
    if m0 <= 0:
        raise FloatingPointError("non-positive pair mass")
    mean_exact = n * M1 / m0
    f = 1
    while P * f < 4.0 * mean_exact:
        f *= 2
    grid = Grid(N=N * f, P=P * f)
    th = grid.theta()
    th_lim = theta_stable(n, pmax)
    act = np.abs(th) <= th_lim
    if not act.any():
        raise FloatingPointError("no stable theta window")
    phi = np.zeros(grid.N, dtype=complex)
    phi[act] = (cf_f(th[act]) / m0) ** n
    edge = float(np.max(np.abs(phi[act & (np.abs(th) > 0.9 * th_lim)]))) \
        if np.any(act & (np.abs(th) > 0.9 * th_lim)) else 0.0
    if edge > PHI_EDGE_TOL:
        raise FloatingPointError(
            f"|phi|={edge:.2e} at the stability edge th={th_lim:.3g}: the "
            f"needed theta range reaches into the unstable recursion region")
    p = np.real(np.fft.fft(phi)) / grid.P
    sg = grid.s()
    mass = float(np.trapezoid(p, sg))
    meanp = float(np.trapezoid(p * sg, sg))
    if abs(mass - 1.0) > 1e-6 or \
            abs(meanp - mean_exact) > 1e-6 * max(1.0, mean_exact):
        raise FloatingPointError(
            f"moment check: mass={mass:.8f}, mean={meanp:.6f} vs {mean_exact:.6f}")
    # NOISE FLOOR, applied to BOTH domains.  The FFT reconstructs p_S to an
    # absolute accuracy ~ eps_mach * max|p|; once the surviving probability
    # mass falls to that level, IA and IB are round-off and their RATIO is
    # arbitrary -- which is exactly how a large-c configuration can report a
    # Rayleigh quotient far above anything admissible.  Guarding only the
    # denominator domain is not enough: at eps > 0 the numerator lives on the
    # strictly smaller region S <= 1-eps and fails first.
    floor = 1e-11 * float(np.max(np.abs(p))) * L

    def _mass(mask, label):
        s_, p_ = sg[mask], p[mask]
        pk_ = np.clip(p_, 0.0, None)
        neg_ = float(np.trapezoid(np.clip(-p_, 0.0, None), s_))
        pos_ = float(np.trapezoid(pk_, s_))
        if pos_ <= floor or neg_ > 1e-6 * pos_:
            raise FloatingPointError(
                f"noise floor ({label}): mass~{pos_:.3e} <= floor {floor:.3e}"
                f" or negative mass {neg_:.3e}")
        return s_, pk_

    keep = sg <= L                        # denominator: S <= 1 + eps
    if not keep.any():
        raise FloatingPointError("empty denominator domain")
    s1, pk = _mass(keep, "A")
    y = L - s1
    IA = float(np.trapezoid(pk * H_f(y), s1))

    keepB = sg <= L - 2.0 * eps           # numerator: S <= 1 - eps
    if not keepB.any():
        raise FloatingPointError("empty numerator domain (eps too large)")
    s1B, pkB = _mass(keepB, "B")
    yB = L - s1B
    IB = float(np.trapezoid(
        pkB * _pow_int(ca, n, yB, j) * _pow_int(cb, n, yB, l), s1B))
    if IA <= 0 or IB <= 0:
        raise FloatingPointError("non-positive pair integral")
    return n * math.log(m0) + math.log(IA), n * math.log(m0) + math.log(IB)


def channels(cs, mus):
    return [(float(c), j) for c, mu in zip(cs, mus) for j in range(1, mu + 1)]


def gram(cs, mus, k, grid, eps=0.0):
    ch = channels(cs, mus)
    M = len(ch)
    lA = np.empty((M, M)); lB = np.empty((M, M))
    for i in range(M):
        for j2 in range(i, M):
            (ca, ja), (cb, lb) = ch[i], ch[j2]
            if ca > cb:
                (ca, ja), (cb, lb) = (cb, lb), (ca, ja)
            a, b = _pair_entry(ca, cb, ja, lb, k, grid.N, grid.P, eps)
            lA[i, j2] = lA[j2, i] = a
            lB[i, j2] = lB[j2, i] = b
    return lA, lB


def ceiling(k):
    return k / (k - 1.0) * math.log(k)


def _precond(lA, lB):
    d = np.diag(lA).copy()
    sc = 0.5 * (d[:, None] + d[None, :])
    A, B = np.exp(lA - sc), np.exp(lB - sc)
    return 0.5 * (A + A.T), 0.5 * (B + B.T)


def rayleigh(cs, mus, k, grid=None, rank_tol=1e-10, gate=True, eps=0.0):
    grid = grid or Grid()
    A, B = _precond(*gram(cs, mus, k, grid, eps))
    lam, Q = np.linalg.eigh(A)
    keep = lam > rank_tol * lam.max()
    W = Q[:, keep] / np.sqrt(lam[keep])
    Br = W.T @ B @ W
    ev, vr = np.linalg.eigh(0.5 * (Br + Br.T))
    i = int(np.argmax(ev))
    R = k * float(ev[i])
    # The tight ceiling M_k < k/(k-1) log k holds only at eps = 0; the
    # enlarged problem legitimately exceeds it -- that is the point of the
    # trick.  For eps > 0 fall back to the crude but valid R <= k.
    lim = ceiling(k) if eps == 0.0 else float(k)
    if gate and R > lim * (1 + 1e-9):
        raise FloatingPointError(f"R={R:.6f} exceeds bound {lim:.6f}")
    v = W @ vr[:, i]
    return R, v / np.max(np.abs(v)), dict(
        rank=int(keep.sum()), M=len(lam),
        lam_min_kept=float(lam[keep].min()),
        cond_kept=float(lam[keep].max() / lam[keep].min()))


def rayleigh_at(cs, mus, v, k, grid, eps=0.0):
    A, B = _precond(*gram(cs, mus, k, grid, eps))
    v = np.asarray(v, float)
    return k * float(v @ B @ v) / float(v @ A @ v)


# ---------------------------------------------------------------------------
def x_to_c(x):
    x = np.asarray(x, float)
    return np.exp(np.concatenate([[x[0]], x[0] + np.cumsum(np.logaddexp(0.0, x[1:]))]))


def c_to_x(cs):
    cs = np.sort(np.asarray(cs, float))
    return np.concatenate([[math.log(cs[0])],
                           np.log(np.expm1(np.clip(np.diff(np.log(cs)), 1e-8, None)))])


def _scalar_min(f, u0, half=1.6, npts=41, verbose=False):
    """1-D minimisation on an objective that is a valid basin plus a flat
    penalty plateau.

    method="bounded" (golden section) assumes unimodality and has no notion
    of feasibility: if its first probes land on the 1e3 plateau it contracts
    inside the plateau and returns a penalty point.  Instead: scan a grid,
    keep only feasible samples, and hand Brent a bracket that is VERIFIED
    (f(b) < f(a), f(b) < f(c)) by construction, so scipy's precondition
    cannot be violated either.
    """
    us = np.linspace(u0 - half, u0 + half, npts)
    vs = np.array([f(u) for u in us])
    ok = vs < 100.0
    if not ok.any():
        raise FloatingPointError("no feasible point on the scalar scan")
    i = int(np.argmin(np.where(ok, vs, np.inf)))
    if verbose:
        print(f"          scan: {ok.sum()}/{npts} feasible, best at "
              f"c={math.exp(us[i]):.6g}")
    # a verified interior bracket, restricted to feasible neighbours
    if 0 < i < npts - 1 and ok[i - 1] and ok[i + 1] \
            and vs[i] < vs[i - 1] and vs[i] < vs[i + 1]:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(f, bracket=(us[i - 1], us[i], us[i + 1]),
                              method="brent", options=dict(xtol=1e-9))
        if res.fun <= vs[i]:
            return float(res.x), float(res.fun)
    # otherwise refine locally by repeated bisection inside the feasible span
    a = us[max(i - 1, 0)]
    b = us[min(i + 1, npts - 1)]
    xb, fb = float(us[i]), float(vs[i])
    for _ in range(60):
        for u in (0.5 * (a + xb), 0.5 * (xb + b)):
            v = f(u)
            if v < fb:
                xb, fb = u, v
        a, b = 0.5 * (a + xb), 0.5 * (xb + b)
    return xb, fb


def _grad(x, mus, k, grid, h, eps=0.0):
    """R and dR/dx with PER-COORDINATE degradation.

    v3 returned (1e3, zeros) if ANY of the 2r perturbed evaluations tripped a
    gate, so L-BFGS-B stopped at x0 and reported the initial grid as if it
    were an optimum.  Here a failed side falls back to a one-sided
    difference, and only a coordinate whose both sides fail contributes zero.
    """
    R0 = rayleigh(x_to_c(x), mus, k, grid, eps=eps)[0]
    g = np.zeros(len(x))
    nfail = 0
    for p in range(len(x)):
        vals = {}
        for sgn in (+1, -1):
            xs = np.array(x, float); xs[p] += sgn * h
            try:
                vals[sgn] = rayleigh(x_to_c(xs), mus, k, grid, eps=eps)[0]
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                pass
        if +1 in vals and -1 in vals:
            g[p] = (vals[+1] - vals[-1]) / (2 * h)
        elif +1 in vals:
            g[p] = (vals[+1] - R0) / h
        elif -1 in vals:
            g[p] = (R0 - vals[-1]) / h
        else:
            nfail += 1
    return R0, g, nfail


def optimise(k, mus, grid=None, spreads=(4.0, 2.0, 8.0), maxiter=400,
             h=2e-3, verbose=True, eps=0.0):
    """Maximise R over the cluster base points.  Multiplicities are fixed."""
    grid = grid or Grid()
    r = len(mus)
    c0 = 1.0 / (math.log(k - 1) - 0.13)
    best = None
    for sp in (spreads if r > 1 else (1.0,)):
        cs0 = c0 * np.geomspace(1.0 / sp, sp, r) if r > 1 else np.array([c0])
        x0 = c_to_x(cs0)
        stats = dict(nfail=0, calls=0, last_ok=None)

        def negf(x):
            stats["calls"] += 1
            x = np.asarray(x, float)
            try:
                R0, g, nf = _grad(x, mus, k, grid, h, eps)
                stats["nfail"] += nf
                stats["last_ok"] = x.copy()
                return -R0, -g
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                # NOT a flat 1e3: a flat penalty has zero gradient, so a
                # quasi-Newton step into the infeasible region has no way
                # back.  Return a quadratic well centred on the last
                # feasible point, whose gradient points home.
                ref = stats["last_ok"] if stats["last_ok"] is not None else x
                dxv = x - ref
                return 1e3 + float(dxv @ dxv), 2.0 * dxv

        if r == 1:
            ub, _ = _scalar_min(lambda u: negf(np.array([u]))[0],
                                math.log(c0), verbose=verbose)
            xb, nit, msg = np.array([ub]), stats["calls"], "scan+brent"
        else:
            res = minimize(negf, x0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=maxiter, ftol=1e-14,
                                        gtol=1e-11))
            xb, nit, msg = res.x, int(res.nit), str(res.message)
        try:
            cs = x_to_c(xb)
            R, v, dg = rayleigh(
                cs, mus, k, Grid(N=grid.N * 4, P=grid.P), eps=eps)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        moved = float(np.max(np.abs(np.log(cs / cs0))))
        rec = dict(R=R, cs=cs, v=v, dg=dg, nit=nit, msg=msg, moved=moved,
                   spread=sp, nfail=stats["nfail"], calls=stats["calls"])
        if best is None or R > best["R"]:
            best = rec
    if best is None:
        raise FloatingPointError("every start was rejected by the gates")
    if verbose:
        dg = best["dg"]
        limit = ceiling(k) if eps == 0.0 else float(k)
        print(f"  k={k:7d} mu={list(mus)} eps={eps:g}  M={dg['M']}  "
              f"R={best['R']:.6f}  bound {limit:.6f}  "
              f"log k - R = {math.log(k)-best['R']:+.5f}")
        print(f"          rank {dg['rank']}/{dg['M']}  cond={dg['cond_kept']:.2e}"
              f"  c={[float(f'{x:.5g}') for x in best['cs']]}")
        print(f"          nit={best['nit']}  moved={best['moved']:.3f} in log c"
              f"  gate-failed coords={best['nfail']}  [{best['msg'][:40]}]")
        if best["moved"] < 1e-3:
            print("          *** WARNING: optimiser did not move; this is the "
                  "starting grid, not an optimum ***")
    return best["R"], best["cs"], best["v"], best["dg"], best


# ---------------------------------------------------------------------------
SINGLE_TARGETS = {51: 3.9178180, 201: 5.1903220, 1001: 6.7337888,
                  3601: 7.9853367}


def unit_test():
    ok = True
    print("1. single channel (mu=[1]) against stored targets")
    for k, tgt in SINGLE_TARGETS.items():
        c = 1.0 / (math.log(k - 1) - 0.3)
        R = rayleigh([c], [1], k)[0]
        good = abs(R - tgt) < 5e-6
        ok &= good
        print(f"   k={k:6d}  R={R:.7f}  target {tgt:.7f}  {'ok' if good else 'FAIL'}")

    print("2. cross-cluster partial fractions against quadrature")
    from scipy.integrate import quad
    n, ca, cb = 200, 0.19, 0.55
    for (j, l) in ((1, 1), (2, 1), (2, 3), (3, 3)):
        cf, H, M1, amp, _ = _pair_functions(ca, cb, j, l, n)
        for th in (0.0, 7.3, 91.0):
            got = cf(np.array([th]))[0]
            re = quad(lambda t: math.cos(th*t)/((ca+n*t)**j*(cb+n*t)**l), 0, 1,
                      limit=400)[0]
            im = quad(lambda t: math.sin(th*t)/((ca+n*t)**j*(cb+n*t)**l), 0, 1,
                      limit=400)[0]
            rel = abs(got - complex(re, im)) / abs(complex(re, im))
            ok &= rel < 1e-9
            print(f"   (j,l)=({j},{l}) th={th:6.1f}  rel={rel:.2e}  amp={amp:.2e}"
                  f"  {'ok' if rel < 1e-9 else 'FAIL'}")

    print("3. F_p stability window")
    for n_ in (50, 3600):
        for pm in (2, 6, 12):
            print(f"   n={n_:5d} pmax={pm:3d}  theta_stable={theta_stable(n_, pm):.4g}")

    print("4. confluent vs multi-c: mu=[1,1] must equal two separate channels")
    k = 201
    cs = np.array([0.15, 0.45])
    print(f"   mu=[1,1] R={rayleigh(cs, [1, 1], k)[0]:.8f}")

    print("5. coalescence is now smooth (mu=[1,1] with delta -> 0 vs mu=[2])")
    c = 1.0 / (math.log(200) - 0.13)
    for e in (2, 3, 4, 5, 6):
        d = c * 10.0 ** -e
        try:
            R2, _, dg2 = rayleigh(np.array([c, c + d]), [1, 1], k)
            print(f"   delta/c=1e-{e}  R={R2:.8f}  rank {dg2['rank']}/2  "
                  f"cond={dg2['cond_kept']:.2e}")
        except FloatingPointError as ex:
            print(f"   delta/c=1e-{e}  REJECTED: {str(ex)[:60]}")
    Rc, _, dgc = rayleigh([c], [2], k)
    print(f"   mu=[2]        R={Rc:.8f}  rank {dgc['rank']}/2  "
          f"cond={dgc['cond_kept']:.2e}")
    return ok


def export(path, k, mus, cs, v, R, grid, prune=1e-12, eps=0.0):
    """Write an exact, hashable trial function for the certifier.

    Contract: for ANY fixed physical weight vector u,
        k u^T B u / u^T A u <= M_k.
    So the certifier must never redo the eigensolve -- rounding u only
    weakens the bound, it cannot invalidate it.

    IMPORTANT: pruning is done in the diagonally normalised Ritz coordinates
    v, where diag(A)=1.  There |v_j| measures the channel contribution on a
    common energy scale.  Only AFTER selecting channels do we reconstruct the
    corresponding physical coefficients u_j = exp(-d_j/2) v_j.  Pruning on
    |u_j| is wrong because the basis functions can have wildly different
    norms, exactly as encoded by d = diag(log A).
    """
    from fractions import Fraction
    import hashlib
    # v is obtained from the final refined eigensolve in optimise(), which
    # uses Grid(N=grid.N * 4, P=grid.P).  Reconstruct the unpreconditioned
    # physical Ritz weights on that exact same grid; otherwise the diagonal
    # preconditioner would differ slightly between discovery and export.
    export_grid = Grid(N=grid.N * 4, P=grid.P)
    lA, _ = gram(cs, mus, k, export_grid, eps)
    d = np.diag(lA)

    v = np.asarray(v, float)
    vmax = float(np.max(np.abs(v)))
    if vmax == 0.0 or not np.isfinite(vmax):
        raise FloatingPointError("cannot export a zero/non-finite Ritz vector")

    # Select channels by their contribution in the diag(A)=1 basis.
    keep = np.abs(v) >= prune * vmax
    if not np.any(keep):
        raise FloatingPointError("all channels were pruned in Ritz coordinates")

    # Convert ONLY the retained channels to physical coefficients.  Work in
    # log space, then apply one common scaling (irrelevant to the Rayleigh
    # quotient) so the largest retained |u_j| is 1.
    lu = -0.5 * d[keep] + np.log(np.abs(v[keep]))
    lu -= np.max(lu)
    u_keep = np.sign(v[keep]) * np.exp(lu)

    ch = channels(cs, mus)
    ch_keep = [ch[i] for i in np.flatnonzero(keep)]
    # Convert the physical coefficients to exact decimal rationals WITHOUT
    # passing through float underflow or limit_denominator().  The physical
    # coefficients can legitimately span hundreds of orders of magnitude
    # because the basis functions have wildly different A-norms.
    from decimal import Decimal, localcontext

    lu_rel = lu - np.max(lu)
    u_fracs = []
    with localcontext() as decctx:
        decctx.prec = 50
        for sgn, ell in zip(np.sign(v[keep]), lu_rel):
            mag = Decimal(str(float(ell))).exp()
            q = Fraction(mag)
            u_fracs.append(q if sgn > 0 else -q)

    items = [(Fraction(c).limit_denominator(10 ** 12), j, uu)
             for (c, j), uu in zip(ch_keep, u_fracs)]
    eps_q = Fraction(eps).limit_denominator(10 ** 12)
    eps_tag = "" if eps_q == 0 else f"epsilon={eps_q}|"
    canon = f"k={k}|{eps_tag}" + "|".join(
        f"{c}^-{j}*{w}" for c, j, w in items)
    h = hashlib.sha256(canon.encode()).hexdigest()
    # Exact integers are stored as decimal Unicode, never dtype=object.
    # NumPy otherwise promotes sufficiently large Python integers to object
    # arrays, which require pickle on load even though the container is NPZ.
    exact_strings = lambda values: np.asarray([str(value) for value in values])
    np.savez(path, k=k, epsilon_num=str(eps_q.numerator),
             epsilon_den=str(eps_q.denominator), mu=np.asarray(mus, dtype=np.int64),
             c_num=exact_strings(it[0].numerator for it in items),
             c_den=exact_strings(it[0].denominator for it in items),
             power=[it[1] for it in items],
             w_num=exact_strings(it[2].numerator for it in items),
             w_den=exact_strings(it[2].denominator for it in items),
             R_discovery=R, ceiling=ceiling(k), sha256=h, canonical=canon)
    kept_idx = list(map(int, np.flatnonzero(keep)))
    print(f"  export keep indices (by |v|): {kept_idx}")
    print(f"  exported {len(items)} channels (pruned {len(ch)-len(items)} on |v|) "
          f"to {path}")
    print(f"  sha256 = {h}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--mu", type=str, default="1,1,1",
                    help="multiplicities per cluster, e.g. 2,2,1")
    ap.add_argument("--n-fft", type=int, default=1 << 16)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--epsilon", "--eps", type=float, default=0.0,
                    help="simplex enlargement epsilon (0 <= epsilon < 1)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--export", type=str, default=None,
                    help="write an exact .npz for the certifier")
    ap.add_argument("--export-prune", type=float, default=1e-12,
                    help="relative |v| threshold used only when exporting")
    a = ap.parse_args()
    if not 0.0 <= a.epsilon < 1.0:
        ap.error("--epsilon must lie in [0, 1)")
    mus = [int(z) for z in a.mu.split(",") if z.strip()]
    grid = Grid(N=a.n_fft)
    t0 = time.time()
    R, cs, v, dg, rec = optimise(
        a.k, mus, grid=grid, maxiter=a.maxiter, verbose=False, eps=a.epsilon)
    lk, cl = math.log(a.k), ceiling(a.k)
    print(f"k = {a.k}   mu = {mus}   epsilon = {a.epsilon:g}   "
          f"M = {dg['M']}   [{time.time()-t0:.1f}s]")
    print(f"  R          = {R:.8f}")
    print(f"  log k      = {lk:.8f}")
    if a.epsilon == 0.0:
        print(f"  ceiling    = {cl:.8f}   (headroom {cl - R:+.6f})")
    else:
        print(f"  bound      = {float(a.k):.8f}   (epsilon problem; M_k ceiling not used)")
    print(f"  log k - R  = {lk - R:+.6f}")
    print(f"  c          = {[float(f'{x:.8g}') for x in cs]}")
    print(f"  weights    = {[float(f'{x:.6g}') for x in v]}")
    print(f"  rank       = {dg['rank']}/{dg['M']}   cond = {dg['cond_kept']:.3e}")
    print(f"  optimiser  = nit {rec['nit']}, moved {rec['moved']:.4f} in log c, "
          f"{rec['nfail']} gate-failed coords, spread {rec['spread']}")
    if rec["moved"] < 1e-3:
        print("  *** WARNING: optimiser did not move -- starting grid, not an "
              "optimum ***")
    if a.epsilon == 0.0 and R > cl:
        print("  *** ABOVE THE CEILING -- evaluator defect, not a discovery ***")
    if a.verify:
        print("  grid refinement at fixed c and v:")
        prev = None
        for f in (1, 2, 4, 8):
            g2 = Grid(N=grid.N * f, P=grid.P)
            Rv = rayleigh_at(cs, mus, v, a.k, g2, eps=a.epsilon)
            print(f"    N = {g2.N:9d}   R = {Rv:.10f}" +
                  ("" if prev is None else f"   delta {Rv - prev:+.3e}"))
            prev = Rv
    if a.export:
        export(a.export, a.k, mus, cs, v, R, grid, prune=a.export_prune,
               eps=a.epsilon)
    if a.out:
        json.dump(dict(k=a.k, mu=mus, epsilon=a.epsilon, R=R,
                       log_k=lk, ceiling=(cl if a.epsilon == 0.0 else None),
                       c=list(map(float, cs)), weights=list(map(float, v)),
                       rank=dg["rank"], M=dg["M"]), open(a.out, "w"), indent=2)
        print(f"  written to {a.out}")


if __name__ == "__main__":
    import sys
    main() if len(sys.argv) > 1 else unit_test()
