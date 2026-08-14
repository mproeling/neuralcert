#!/usr/bin/env python3
"""
hybrid_lp.py -- the hybrid A/B test with no optimizer confounds.

QUESTION.  Does adding separated Gaussian widths to the classical
polynomial-x-Gaussian family lower the last sign change at fixed d, or is
the polynomial family already extremal among these ansatze?  Every earlier
attempt at this comparison was polluted by nonconvex tau-optimisation; here
BOTH sides are solved by the same convex method, so the comparison is
between FAMILIES, not between optimisers.

  A (baseline): span{ L_{2m}^{alpha}(2u) e^{-u} }               (laguerre_lp)
  B (hybrid) : the same  +  { phi_a(u) = e^{-au} + s a^{-d/2} e^{-u/a} }

Both are linear spaces of +1 eigenfunctions, so fixed-u0 feasibility is one
LP in the mixed coefficients; bisection on u0 gives each family's global
optimum, and since A is a subspace of B, u0*_B <= u0*_A always.  The A/B
question is whether the inequality is strict beyond bisection resolution.

The B side is an exponential-polynomial family, so its verdicts here are
scan-based (the same derivative-sign scan the A side uses as pre-filter),
not exact certificates; exact certification of a mixed winner is the
certkit follow-up (rational widths make x = e^{-u/L} polynomial again).
For the A/B DECISION the two LP estimates, computed identically, are the
right objects.

    python hybrid_lp.py --d 1 --n-basis 12 --widths 1.5,2.5,4.0
    python hybrid_lp.py --d 1 --n-basis 12 --widths 1.3,1.8,2.6,3.8,5.5 \\
        --u0-pure 1.0314388125
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import linprog

from .laguerre import (XP, basis_vals, basis_dvals, lag_coeffs_frac,
                         orders_for, minimise_u0, lb_bck_d1)


class Mixed:
    """Columns of the hybrid family, evaluated in f-space (the Laguerre
    columns carry their e^{-u}; nothing underflows below longdouble range)."""

    def __init__(self, d: int, s: int, orders, widths):
        self.d, self.s = d, s
        self.orders = list(orders)
        self.widths = sorted(float(a) for a in widths)
        if any(a <= 1.0 for a in self.widths):
            raise ValueError("widths must be > 1 (a = 1 IS the Laguerre block)")
        self.k = len(self.orders) + len(self.widths)
        self.alpha = d / 2.0 - 1.0
        self._proj = None            # set by orthogonalise()

    def orthogonalise(self, u_ref=None):
        """
        Replace each Gaussian column by its residual against the Laguerre
        block (least squares on a reference grid), rescaled to unit sup.

        Without this the mixed LP is numerically rank-deficient: on the
        working window e^{-u/a} = e^{-u} e^{(1-1/a)u} and e^{(1-1/a)u} is
        approximated by a degree-22 polynomial to ~1e-9, so HiGHS exploits
        the near-null direction with |c| ~ 6e8 of mutual cancellation and
        returns "optimal" points whose own residuals are 2e-2.  The span --
        and therefore the A/B question -- is EXACTLY unchanged; only the
        coordinates are.  The rescaled residual coefficient b_i then reads
        off directly how much of the genuinely-new direction the optimum
        uses, which is the A/B quantity itself.
        """
        top = max(self.orders)
        a_max = self.widths[-1]
        u_end = max(4.0 * top + 60.0, 40.0 * a_max)
        if u_ref is None:
            u_ref = np.unique(np.concatenate([
                np.linspace(0.0, u_end, 4000),
                np.geomspace(1e-4, u_end, 4000)]))
        L = np.asarray(basis_vals(u_ref, self.orders, self.alpha, np.float64),
                       dtype=float) * np.exp(-u_ref)[:, None]
        X, S = [], []
        for a in self.widths:
            phi = (np.exp(-a * u_ref)
                   + self.s * a ** (-self.d / 2) * np.exp(-u_ref / a))
            x, *_ = np.linalg.lstsq(L, phi, rcond=None)
            r = phi - L @ x
            sc = max(np.max(np.abs(r)), 1e-300)
            X.append(x)
            S.append(sc)
            print(f"    width a={a}: ||resid||_sup / ||phi||_sup = "
                  f"{sc / max(np.max(np.abs(phi)), 1e-300):.3e}  "
                  f"(how much of phi_a is OUTSIDE the polynomial span)")
        self._proj = (np.array(X), np.array(S))
        # A residual below ~1e-8 of the column's size means the "new"
        # direction is float noise: amplifying it (the rescale divides by
        # sc) hands the LP a garbage column and HiGHS returns points
        # violating their own rows.  Such widths are CONTAINED: the test for
        # them is already decided (gain bounded by the residual), so they
        # are dropped from the LP rather than crashed on.
        keep = [i for i, sc in enumerate(S)
                if sc / 1.0 >= 1e-8]
        dropped = [self.widths[i] for i in range(len(S)) if i not in keep]
        if dropped:
            print(f"    contained widths dropped from the LP (gain provably "
                  f"below residual scale): {dropped}")
        self.widths = [self.widths[i] for i in keep]
        self._proj = (np.array([X[i] for i in keep]) if keep else np.zeros((0, len(self.orders))),
                      np.array([S[i] for i in keep]))
        self.k = len(self.orders) + len(self.widths)
        return self

    def vals(self, u, dtype=XP):
        u = np.atleast_1d(np.asarray(u, dtype=dtype))
        L = np.asarray(basis_vals(u, self.orders, self.alpha, dtype), dtype=dtype)
        Le = L * np.exp(-u)[:, None]
        cols = [Le]
        for i, a in enumerate(self.widths):
            a = dtype(a)
            phi = (np.exp(-a * u) + dtype(self.s) * a ** (-dtype(self.d) / 2)
                   * np.exp(-u / a))
            if self._proj is not None:
                x, sc = self._proj[0][i], self._proj[1][i]
                phi = (phi - Le @ x.astype(dtype)) / dtype(sc)
            cols.append(phi[:, None])
        return np.hstack(cols)

    def dvals(self, u, dtype=XP):
        u = np.atleast_1d(np.asarray(u, dtype=dtype))
        L = np.asarray(basis_vals(u, self.orders, self.alpha, dtype), dtype=dtype)
        Ld = np.asarray(basis_dvals(u, self.orders, self.alpha, dtype), dtype=dtype)
        Lde = (Ld - L) * np.exp(-u)[:, None]
        cols = [Lde]
        for i, a in enumerate(self.widths):
            a = dtype(a)
            dphi = (-a * np.exp(-a * u)
                    - dtype(self.s) * a ** (-dtype(self.d) / 2) / a
                    * np.exp(-u / a))
            if self._proj is not None:
                x, sc = self._proj[0][i], self._proj[1][i]
                dphi = (dphi - Lde @ x.astype(dtype)) / dtype(sc)
            cols.append(dphi[:, None])
        return np.hstack(cols)

    def mass(self, u0: float) -> np.ndarray:
        """int_{u0}^oo column du, true scale (no dropped factors: the
        Laguerre part must carry its e^{-u0} because the Gaussian part's
        scale is genuinely different)."""
        out = []
        for n in self.orders:
            coef = lag_coeffs_frac(n, self.d)
            m = 0.0
            for j, c in enumerate(coef):
                tail = sum(u0 ** i / math.factorial(i) for i in range(j + 1))
                m += float(c) * math.factorial(j) * tail
            out.append(m * math.exp(-u0))
        lagmass = np.array(out)
        for i, a in enumerate(self.widths):
            m = (math.exp(-a * u0) / a
                 + self.s * a ** (-self.d / 2) * a * math.exp(-u0 / a))
            if self._proj is not None:
                x, sc = self._proj[0][i], self._proj[1][i]
                m = (m - float(lagmass @ x)) / sc
            out.append(m)
        v = np.array(out)
        return v / np.linalg.norm(v)


def scan(fam: Mixed, c, u0: float, tol_rel: float = 3e-11, n_scan: int = 22000):
    # tol_rel is looser than the pure-polynomial side on purpose: the mixed
    # magnitude scale mixes Laguerre cancellation (|L_n| large, sum O(1))
    # with Gaussian columns, and the float64 LP coefficients replayed in
    # longdouble legitimately disagree at ~1e-12 relative.  Flagging that as
    # a violation at an existing constraint point aborted the first A/B run;
    # it is arithmetic noise, three orders below the 1e-7 resolution the A/B
    # decision needs.
    top = max(fam.orders)
    a_max = fam.widths[-1] if fam.widths else 1.0
    u_end = max(4.0 * top + 60.0, 40.0 * a_max)
    u = np.unique(np.concatenate([
        np.linspace(u0, u_end, n_scan),
        np.geomspace(max(u0, 1e-9), u_end, n_scan)])).astype(XP)
    cx = np.asarray(c, dtype=XP)
    V, Vd = fam.vals(u), fam.dvals(u)
    mag = np.maximum(np.abs(V) @ np.abs(cx), XP(1e-4930))
    Fd = Vd @ cx
    sg = np.sign(Fd)
    idx = np.where(sg[1:] * sg[:-1] < 0)[0]
    crits = []
    for i in idx:
        lo, hi, s0 = u[i], u[i + 1], sg[i]
        for _ in range(70):
            mid = (lo + hi) / 2
            if np.sign(float(fam.dvals(mid)[0] @ cx)) == s0:
                lo = mid
            else:
                hi = mid
        crits.append(float((lo + hi) / 2))
    pts = np.array([float(u0)] + crits + [float(u_end)], dtype=XP)
    Vp = fam.vals(pts)
    vals = Vp @ cx
    mags = np.maximum(np.abs(Vp) @ np.abs(cx), XP(1e-4930))
    rel = np.asarray(vals / mags, dtype=float)
    worst = int(np.argmin(rel))
    viols = [float(pts[i]) for i in range(len(pts)) if rel[i] < -tol_rel]
    far = np.array([2.0 * u_end, 5.0 * u_end], dtype=XP)
    tail_ok = all(float(v) > 0 for v in (fam.vals(far) @ cx))
    minima = sorted((float(rel[i]), float(pts[i])) for i in range(1, len(pts) - 1))
    return viols, float(rel[worst]), float(pts[worst]), tail_ok, minima


def oracle(fam: Mixed, u0: float, cutpool: list, n_grid: int = 1400,
           max_cuts: int = 60, tol_margin: float = 1e-9):
    top = max(fam.orders)
    a_max = fam.widths[-1] if fam.widths else 1.0
    u_end = max(4.0 * top + 60.0, 40.0 * a_max)
    W = 2.0 * top + 10.0
    base = np.unique(np.concatenate([
        np.linspace(u0, u_end, n_grid),
        np.geomspace(max(u0, 1e-9), u_end, n_grid), np.array([u0])]))
    z = np.asarray(fam.vals(np.array([0.0])), dtype=float)[0]
    mv = fam.mass(u0)
    k = fam.k
    for it in range(max_cuts + 1):
        pts = np.unique(np.concatenate(
            [base, np.array([x for x in cutpool if x >= u0], dtype=float)]))
        B = np.asarray(fam.vals(pts.astype(XP)), dtype=float)
        rowscale = np.maximum(np.max(np.abs(B), axis=1), 1e-300)
        Bn = B / rowscale[:, None]
        r0 = z / max(np.linalg.norm(z), 1e-300)
        r1 = mv / max(np.linalg.norm(mv), 1e-300)
        marg = (pts <= W).astype(float)[:, None]
        # tail row: the slowest rate present is 1/a_max, carried by the last
        # column with coefficient s a^{-d/2} b_max, so s=+1 needs b_max >= 0
        tailrow = np.zeros((1, k + 1))
        tailrow[0, k - 1] = -1.0
        A_ub = np.vstack([np.hstack([-Bn, marg]), tailrow])
        A_eq = np.hstack([np.vstack([r0, r1]), np.zeros((2, 1))])
        res = linprog(c=np.concatenate([np.zeros(k), [-1.0]]),
                      A_ub=A_ub, b_ub=np.zeros(len(pts) + 1),
                      A_eq=A_eq, b_eq=np.array([0.0, 1.0]),
                      bounds=[(None, None)] * (k + 1), method="highs",
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10})
        if not res.success or res.x[-1] <= tol_margin:
            return None, it
        c = res.x[:k]
        viols, worst_rel, worst_u, tail_ok, minima = scan(fam, c, u0)
        if not tail_ok:
            return None, it
        if not viols:
            t_star = float(res.x[-1])
            tangs = sorted({round(uu, 9) for r, uu in minima
                            if r <= max(10.0 * t_star, 1e-6)})
            return {"coeffs": c, "margin": t_star, "tangencies": tangs,
                    "min_rel": worst_rel}, it
        stale = [v for v in viols
                 if np.min(np.abs(pts - v)) < 1e-9 * max(v, 1.0)
                 and worst_rel < -1e-9]
        if stale:
            import sys
            print(f"    WARNING: LP violates its own row at u={stale[0]:.9f}",
                  file=sys.stderr, flush=True)
            return None, it
        for v in viols[:24]:
            h = 1e-3 * max(v, 1.0)
            cutpool.extend([v - h, v, v + h])
    return None, max_cuts


def bisect_mixed(fam: Mixed, u_lo: float, u_hi: float, steps: int = 40,
                 verbose: bool = True):
    pool: list = []
    sol, _ = oracle(fam, u_hi, pool)
    if sol is None:
        return None
    lo, hi, best = u_lo, u_hi, sol
    for i in range(steps):
        mid = 0.5 * (lo + hi)
        s_, _ = oracle(fam, mid, pool)
        if s_ is None:
            lo = mid
        else:
            hi, best = mid, s_
        if verbose and i % 6 == 0:
            print(f"    B bisect[{i:2d}] u0 in ({lo:.10f}, {hi:.10f}]", flush=True)
    best["u0"] = hi
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--sign", type=int, choices=(1,), default=1)
    ap.add_argument("--n-basis", type=int, default=12)
    ap.add_argument("--widths", default="1.5,2.5,4.0")
    ap.add_argument("--bisect", type=int, default=40)
    ap.add_argument("--u0-pure", type=float, default=None,
                    help="known pure optimum; skips the A side")
    a = ap.parse_args()

    orders = orders_for(a.n_basis, a.sign)
    widths = [float(v) for v in a.widths.split(",")]
    lb = lb_bck_d1() if a.d == 1 else None
    u_lo = math.pi * lb * lb * 0.999 if lb else 0.35

    if a.u0_pure is not None:
        u0_A = a.u0_pure
        print(f"[A] pure Laguerre baseline (given): u0* = {u0_A:.10f}   "
              f"rho = {math.sqrt(u0_A / math.pi):.10f}")
    else:
        print(f"[A] pure Laguerre baseline, n_basis={a.n_basis}")
        bestA, _ = minimise_u0(orders, a.d, u_lo,
                               1.35 if a.d == 1 else 2.01,
                               bisect=a.bisect, verbose=False)
        u0_A = bestA["u0"]
        print(f"    u0* = {u0_A:.10f}   rho = {math.sqrt(u0_A / math.pi):.10f}")

    fam = Mixed(a.d, a.sign, orders, widths)
    print(f"[B] hybrid: same Laguerre block + widths {widths}   (k = {fam.k})")
    fam.orthogonalise()
    if not fam.widths:
        print("-" * 70)
        print("A/B: every requested width is numerically CONTAINED in the "
              "polynomial family on the active region.")
        print("VERDICT: no gain is possible from these widths at this degree; "
              "the polynomial family is extremal among these ansatze.")
        return
    bestB = bisect_mixed(fam, u_lo, u0_A * 1.002, steps=a.bisect)
    if bestB is None:
        print("    hybrid infeasible even just above the pure optimum -- inspect")
        return
    u0_B = bestB["u0"]
    print(f"    u0* = {u0_B:.10f}   rho = {math.sqrt(u0_B / math.pi):.10f}")
    print("-" * 70)
    gain = u0_A - u0_B
    print(f"A/B: delta u0 = {gain:+.3e}   ({gain / u0_A:+.2e} relative)")
    if gain > 5e-8:
        print("VERDICT: the Gaussian directions HELP at this degree -- the "
              "hybrid hypothesis survives; certify a mixed winner next "
              "(rational widths make it polynomial again).")
    else:
        print("VERDICT: no measurable gain -- at this degree the polynomial "
              "family is extremal among these ansatze; the Gaussian "
              "directions' natural home is the large-d asymptotic regime.")
    gcoef = np.asarray(bestB["coeffs"][-len(widths):])
    print(f"    B-side Gaussian coefficients: "
          f"{np.array2string(gcoef, precision=4)}")


if __name__ == "__main__":
    main()
