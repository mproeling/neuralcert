"""
maynard_scaled_domain.py -- exact simplex integrals for SPIKE-LOCALISED
channels, via a rescaled coordinate with compact support.

THE PROBLEM
===========
At large k the optimising channels g(u) concentrate near u=0.  Represented as
polynomials on the full interval [0,1] they need degree ~245 at k=3000 and
400+ at k=12000, and the certifier's cost is driven by that degree twice
over: the powered polynomial bh^k has degree 2*d*k, and the CRT prime count
scales with the bit-width of top_a! = (k + k*smax)!.

THE FIX, AND WHAT IT IS NOT
===========================
An affine change of variable does NOT reduce polynomial degree: g(x* s) has
the same degree as g.  What reduces the degree is TRUNCATING THE SUPPORT to
[0, x*] and spending every degree there.  This is certificate-safe: R is a
lower bound on M_k for ANY explicit F, so restricting the ansatz can only
cost bound quality, never validity.

But truncation breaks the closed form.  With t = sigma*s, sigma = x*, the
region {sum t_i <= 1} becomes {s in [0,1]^k : sum s_i <= L}, L = 1/sigma --
a BOX-TRUNCATED simplex, not the unit simplex.  The unit-simplex identity
    int_{sum t <= 1} prod h(t_i) dt = sum_n [x^n] bh^k / (k+n)!
no longer applies.

THE IDENTITY
============
Take sigma = 1/L with L a positive integer, h supported on [0,sigma],
h(u) = H(u/sigma) with H a polynomial of degree 2d on [0,1].  Split

    H * 1_[0,1] (y) = H(y) Theta(y) - Ht(y-1) Theta(y-1),   Ht(z) = H(z+1)

Both pieces are (polynomial x Heaviside), whose Laplace transform is
B(1/p)/p with B the Borel weighting  B(x) = sum_r H_r r! x^r.  So

    F(p) = ( B(1/p) - e^{-p} Bt(1/p) ) / p

    F(p)^k = p^{-k} sum_j C(k,j) (-1)^j e^{-jp} B(1/p)^{k-j} Bt(1/p)^j

Inverting F^k/p at y=L (the antiderivative) turns p^{-(k+1+n)} e^{-jp} into
(y-j)_+^{k+n}/(k+n)!, giving

  A = sigma^k sum_{j=0}^{min(k,L-1)} (-1)^j C(k,j)
        sum_n [x^n](B^{k-j} Bt^j) (L-j)^{k+n} / (k+n)!            (*)

and, for the B-side (which is int_0^rho h^{*(k-1)}(s) T(s) ds with T the pair
transformed polynomial), substituting s = sigma*y with Lam = rho*L:

  B = sigma^{k-1} sum_{j<Lam} (-1)^j C(k-1,j)
        sum_R [x^R](B^{k-1-j} Bt^j) / (k-2+R)!
        sum_m qt_m sigma^m sum_{i=0}^m C(m,i) j^{m-i}
              (Lam-j)^{k-1+R+i} / (k-1+R+i)                       (**)

TWO PROPERTIES THAT MATTER
--------------------------
1.  L = 1, sigma = 1 REPRODUCES v7 EXACTLY.  Only j=0 survives, (L-j)=1, and
    j^{m-i} = 0^{m-i} kills every term but i=m.  (*) collapses to
    sum_n [x^n] bh^k/(k+n)! and (**) to the v7 B formula.  Same code path,
    same denominators -- the scaled certifier is a superset, not a fork.

2.  EVERYTHING STAYS INTEGRAL.  (L-j) is an integer.  (Lam-j) =
    (rho_n L - j rho_d)/rho_d, integer numerator over a power of rho_d that
    the existing D_B = top_b! Lam rho_d^E T_den ... already carries.  The CRT
    prime worker needs the j-loop and shifted integer powers; no new
    denominator, no new reconstruction bound structure.

THE TRANSFORMED CHANNEL CHANGES TOO
===================================
T_j(s) = G_j(1-s) with G_j(x) = int_0^x q_j.  Truncating q_j to [0,sigma]
makes G_j constant (= total mass) for x >= sigma.  Since s ranges over
[0,rho], the argument 1-s ranges over [1-rho, 1].  So:

    sigma <= 1-rho  ->  T_j is CONSTANT on the whole B-integration range
    sigma >  1-rho  ->  T_j is piecewise, with ONE breakpoint at s = 1-sigma

With eps = 1/25, 1-rho = 2eps/(1+eps) = 1/13, so L >= 13 lands in the first
case and the epsilon structure decouples from the B-integral entirely.  That
is consistent with the observed sign reversal of the epsilon trick at large
k, and it is why `b_scaled` takes an explicit (alpha, beta) window: a
piecewise T is summed over pieces by the caller, exactly.

VALIDATION
==========
`selftest()` checks three independent things, all in seconds:
  (a) L=1 against the v7 closed form, EXACT Fraction equality;
  (b) (*) and (**) against direct numerical convolution on a fine grid, at
      two grid resolutions, so you see the numeric error CONVERGING to the
      exact value rather than merely being small;
  (c) H = 1 against the classical box-simplex volume.
Run it after any edit.  A wrong binomial or a wrong shift here would look
exactly like a valid certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

# ---------------------------------------------------------------------------
# minimal exact polynomial arithmetic (reference layer; the CRT worker will
# use nmod_poly for the same operations)
# ---------------------------------------------------------------------------

def pmul(a: list, b: list) -> list:
    if not a or not b:
        return [Fraction(0)]
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, va in enumerate(a):
        if va:
            for j, vb in enumerate(b):
                if vb:
                    out[i + j] += va * vb
    return out


def ppow(a: list, e: int) -> list:
    result = [Fraction(1)]
    base = list(a)
    while e:
        if e & 1:
            result = pmul(result, base)
        e >>= 1
        if e:
            base = pmul(base, base)
    return result


def pshift(P: list, a) -> list:
    """Coefficients of z -> P(z + a)."""
    n = len(P)
    out = [Fraction(0)] * n
    for r, c in enumerate(P):
        if not c:
            continue
        for i in range(r + 1):
            out[i] += c * comb(r, i) * Fraction(a) ** (r - i)
    return out


def borel(P: list) -> list:
    """Borel weighting  bP_r = P_r * r!  (the certifier's `bh`)."""
    return [Fraction(c) * factorial(r) for r, c in enumerate(P)]


def _pos_pow(x: Fraction, e: int) -> Fraction:
    """(x)_+^e, with (x)_+^0 = 1 for x > 0 and 0 for x <= 0."""
    if x <= 0:
        return Fraction(0)
    return x ** e


# ---------------------------------------------------------------------------
# the two scaled integrals
# ---------------------------------------------------------------------------

def a_scaled(H: list, k: int, L: int) -> Fraction:
    """int_{sum t <= 1} prod_{i=1}^k h(t_i) dt   for h(u) = H(u*L) on [0,1/L].

    Identity (*) in the module docstring.  L = 1 is the v7 case.
    """
    if L < 1:
        raise ValueError("L must be a positive integer")
    bH = borel(H)
    bHt = borel(pshift(H, 1))
    sigma = Fraction(1, L)
    total = Fraction(0)
    for j in range(0, min(k, L - 1) + 1):
        prod = pmul(ppow(bH, k - j), ppow(bHt, j))
        inner = Fraction(0)
        base = Fraction(L - j)
        for n, c in enumerate(prod):
            if c:
                inner += c * base ** (k + n) / factorial(k + n)
        total += Fraction((-1) ** j * comb(k, j)) * inner
    return sigma ** k * total


def b_scaled(H: list, qt: list, k: int, L: int, rho: Fraction,
             alpha: Fraction = Fraction(0),
             beta: Fraction | None = None) -> Fraction:
    """int_alpha^beta h^{*(k-1)}(s) T(s) ds   for h(u) = H(u*L) on [0,1/L],
    T(s) = sum_m qt_m s^m.

    Defaults alpha=0, beta=rho, which is the full v7 B-integral.  Pass an
    explicit window to sum a PIECEWISE transformed channel over its pieces
    (see the docstring section on T).  Identity (**).
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if beta is None:
        beta = rho
    if not (0 <= alpha <= beta):
        raise ValueError("need 0 <= alpha <= beta")

    bH = borel(H)
    bHt = borel(pshift(H, 1))
    sigma = Fraction(1, L)
    # integration window in the stretched coordinate y = s / sigma
    ya, yb = Fraction(alpha) * L, Fraction(beta) * L
    # transformed channel re-expressed in y:  T(sigma*y)
    qt_s = [Fraction(c) * sigma ** m for m, c in enumerate(qt)]

    total = Fraction(0)
    jmax = min(k - 1, int(yb) if yb == int(yb) else int(yb))
    for j in range(0, jmax + 1):
        if Fraction(j) >= yb:
            break
        prod = pmul(ppow(bH, k - 1 - j), ppow(bHt, j))
        inner = Fraction(0)
        for R, cR in enumerate(prod):
            if not cR:
                continue
            acc = Fraction(0)
            for m, cm in enumerate(qt_s):
                if not cm:
                    continue
                for i in range(m + 1):
                    e = k - 1 + R + i
                    jm = Fraction(j) ** (m - i) if (m - i) or j else Fraction(1)
                    if j == 0 and m - i > 0:
                        continue
                    acc += (cm * comb(m, i) * jm
                            * (_pos_pow(yb - j, e) - _pos_pow(ya - j, e))
                            / e)
            inner += cR * acc / factorial(k - 2 + R)
        total += Fraction((-1) ** j * comb(k - 1, j)) * inner
    return sigma ** (k - 1) * total


# ---------------------------------------------------------------------------
# v7 reference formulas (L = 1), for the exact-equality check
# ---------------------------------------------------------------------------

def a_v7(h: list, k: int) -> Fraction:
    bh = borel(h)
    prod = ppow(bh, k)
    return sum((c / factorial(k + n) for n, c in enumerate(prod)),
               Fraction(0))


def b_v7(h: list, qt: list, k: int, rho: Fraction) -> Fraction:
    bh = borel(h)
    prod = ppow(bh, k - 1)
    tot = Fraction(0)
    for m, cm in enumerate(qt):
        if not cm:
            continue
        for R, cR in enumerate(prod):
            if not cR:
                continue
            tot += (Fraction(cm) * cR * rho ** (m + k - 1 + R)
                    / (factorial(k - 2 + R) * (m + k - 1 + R)))
    return tot


# ---------------------------------------------------------------------------
# scale / degree recommender
# ---------------------------------------------------------------------------

@dataclass
class ScaleAdvice:
    L: int
    sigma: float
    mass_kept: float          # fraction of int g^2 inside [0, 1/L]
    degree_needed: int        # shifted-Legendre degree for rel L2 err <= tol
    est_power_degree: int     # 2 * degree_needed * k, the powered-poly degree


def recommend_scale(u, g, k: int, candidate_L=(1, 2, 4, 8, 16, 32),
                    tol: float = 1e-10, max_degree: int = 400):
    """Given a channel sampled as g(u) on a fine grid u in [0,1], report for
    each candidate L how much mass truncation at 1/L costs and what degree
    the stretched channel then needs.

    This is a DIAGNOSTIC on the float side.  It picks nothing and certifies
    nothing; it tells you which L is worth building a payload for.
    """
    import numpy as np
    from numpy.polynomial import legendre as legmod

    u = np.asarray(u, dtype=float)
    g = np.asarray(g, dtype=float)
    total = float(np.trapezoid(g * g, u)) if hasattr(np, "trapezoid") \
        else float(np.trapz(g * g, u))

    out = []
    for L in candidate_L:
        sigma = 1.0 / L
        sel = u <= sigma
        if sel.sum() < 8:
            continue
        us, gs = u[sel], g[sel]
        kept = (float(np.trapezoid(gs * gs, us)) if hasattr(np, "trapezoid")
                else float(np.trapz(gs * gs, us))) / max(total, 1e-300)
        # stretched coordinate s = u/sigma in [0,1] -> Legendre domain [-1,1]
        x = 2.0 * (us / sigma) - 1.0
        norm = float(np.sqrt(np.mean(gs * gs))) or 1.0
        deg = max_degree
        for d in range(1, max_degree + 1):
            c = legmod.legfit(x, gs, d)
            err = float(np.sqrt(np.mean((legmod.legval(x, c) - gs) ** 2)))
            if err / norm <= tol ** 0.5:
                deg = d
                break
        out.append(ScaleAdvice(L=L, sigma=sigma, mass_kept=kept,
                               degree_needed=deg,
                               est_power_degree=2 * deg * k))
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _numeric_a(Hf, k: int, L: int, N: int) -> float:
    """int_0^1 h^{*k} by direct midpoint convolution, h supported [0,1/L]."""
    import numpy as np
    sigma = 1.0 / L
    dy = sigma / N
    y = (np.arange(N) + 0.5) * dy
    v = Hf(y / sigma)
    acc = v.copy()
    for _ in range(k - 1):
        acc = np.convolve(acc, v) * dy
    s = (np.arange(acc.size) + 0.5 * k) * dy
    return float(acc[s < 1.0].sum() * dy)


def _numeric_b(Hf, Tf, k: int, L: int, rho: float, N: int) -> float:
    import numpy as np
    sigma = 1.0 / L
    dy = sigma / N
    y = (np.arange(N) + 0.5) * dy
    v = Hf(y / sigma)
    acc = v.copy()
    for _ in range(k - 2):
        acc = np.convolve(acc, v) * dy
    s = (np.arange(acc.size) + 0.5 * (k - 1)) * dy
    m = s < rho
    return float((acc[m] * Tf(s[m])).sum() * dy)


def selftest(verbose: bool = True) -> bool:
    import numpy as np
    ok = True

    # ---- (a) L = 1 must reproduce v7 EXACTLY --------------------------
    if verbose:
        print("(a) L=1 vs v7 closed form, exact Fraction equality")
    for (k, H, qt, rho) in [
        (5, [Fraction(1)], [Fraction(1)], Fraction(23, 25)),
        (6, [Fraction(1), Fraction(-2), Fraction(3, 2)],
            [Fraction(2), Fraction(1, 3)], Fraction(12, 13)),
        (9, [Fraction(1, 2), Fraction(0), Fraction(-1), Fraction(4, 5)],
            [Fraction(1), Fraction(-1), Fraction(1, 7)], Fraction(1)),
    ]:
        aa, av = a_scaled(H, k, 1), a_v7(H, k)
        bb, bv = b_scaled(H, qt, k, 1, rho), b_v7(H, qt, k, rho)
        good = (aa == av) and (bb == bv)
        ok &= good
        if verbose:
            print(f"    k={k:2d} degH={len(H)-1} : A {'==' if aa==av else '!='}"
                  f"  B {'==' if bb==bv else '!='}   {'PASS' if good else 'FAIL'}")

    # ---- (b) scaled formulas vs direct convolution, two resolutions ----
    if verbose:
        print("(b) scaled A,B vs numeric convolution (error must shrink)")
    cases = [
        (3, 2, [Fraction(0), Fraction(1), Fraction(-1)],
         [Fraction(1), Fraction(1, 2)], Fraction(23, 25)),
        (4, 4, [Fraction(1), Fraction(-1)],
         [Fraction(2), Fraction(-1), Fraction(1, 4)], Fraction(9, 10)),
        (5, 3, [Fraction(0), Fraction(1), Fraction(-1)],
         [Fraction(1)], Fraction(23, 25)),
    ]
    for (k, L, H, qt, rho) in cases:
        Hc = [float(c) for c in H]
        Hf = lambda z, Hc=Hc: sum(c * z ** i for i, c in enumerate(Hc))
        Tc = [float(c) for c in qt]
        Tf = lambda z, Tc=Tc: sum(c * z ** i for i, c in enumerate(Tc))
        ax, bx = float(a_scaled(H, k, L)), float(b_scaled(H, qt, k, L, rho))
        row = []
        for N in (400, 1600):
            an = _numeric_a(Hf, k, L, N)
            bn = _numeric_b(Hf, Tf, k, L, rho, N)
            row.append((abs(an - ax) / max(abs(ax), 1e-300),
                        abs(bn - bx) / max(abs(bx), 1e-300)))
        # A quantity already at machine epsilon cannot shrink further; that
        # happens when the box constraint is inactive (k/L <= 1) and the
        # quadrature is exact for the integrand degree.  Accept it.
        def _conv(coarse, fine):
            return fine <= 1e-14 or (fine < coarse * 0.6 and fine < 2e-3)
        good = (_conv(row[0][0], row[1][0]) and _conv(row[0][1], row[1][1]))
        ok &= good
        if verbose:
            print(f"    k={k} L={L}: relerr A {row[0][0]:.2e}->{row[1][0]:.2e}"
                  f"   B {row[0][1]:.2e}->{row[1][1]:.2e}"
                  f"   {'PASS' if good else 'FAIL'}")

    # ---- (c) H == 1 vs classical box-simplex volume --------------------
    if verbose:
        print("(c) H=1 vs box-simplex volume  sum_j (-1)^j C(k,j)(L-j)_+^k/k!")
    for (k, L) in [(4, 2), (6, 3), (8, 5), (10, 4)]:
        vol = sum(Fraction((-1) ** j * comb(k, j)) * Fraction(max(L - j, 0)) ** k
                  for j in range(k + 1)) / factorial(k)
        got = a_scaled([Fraction(1)], k, L) * Fraction(L) ** k
        good = got == vol
        ok &= good
        if verbose:
            print(f"    k={k:2d} L={L}: {'PASS' if good else 'FAIL'}")

    return ok


if __name__ == "__main__":
    print("maynard_scaled_domain selftest")
    print("PASS" if selftest() else "FAIL")
