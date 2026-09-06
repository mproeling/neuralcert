#!/usr/bin/env python3
"""
certify_laguerre.py -- theorem-grade certification over Q.

Everything before this file ends in "PASS (numerical)": mpmath roots are
correct with overwhelming probability, but they are still floating point.
This file removes the qualifier.  The whole computation is exact rational
arithmetic; if it prints CERTIFIED, the statement

    A_{+}(d)  <=  sqrt(u_q / pi)

is proved, modulo only the classical facts that L_n^{d/2-1}(2u) e^{-u} is a
(-1)^n Fourier eigenfunction and that Sturm's theorem counts real roots.

HOW

1.  The contact points tau_i are taken as EXACT rationals (a float is an
    exact dyadic rational; no rounding happens anywhere).
2.  The square collocation system -- P(0) = 0, P(tau_i) = P'(tau_i) = 0,
    top coefficient = 1 -- is solved over Fraction.  The solution P in Q[u]
    then has EXACT double roots at every tau_i by construction.
3.  P is deflated exactly: P = prod_i (u - tau_i)^2 * R with remainder
    verified to be identically zero at each division.  On [u_q, oo),
    P >= 0 iff R >= 0 there.
4.  Sturm's theorem on R over [u_q, oo): the chain is computed over Q, sign
    variations are evaluated exactly at u_q and at +oo (leading signs).
    Zero variations difference + R(u_q) > 0 + positive leading coefficient
    proves R > 0 on the closed ray, hence P >= 0 there.
5.  f(0) = 0 holds exactly (row of the exact solve); f is a +1 eigenfunction
    exactly (rational combination of exact eigenfunctions); f is not >= 0
    everywhere (integral f = fhat(0) = f(0) = 0 and f != 0), so the last
    sign change of f is <= u_q, giving the bound.

u_q is any rational just above the simple root r_1; the script proposes
one at relative distance ~1e-9 above the numerical r_1 and then PROVES the
ray property for that exact rational.  The quoted constant loses nothing
visible at 9 digits.

    python certify_laguerre.py --d 1 --taus 1.816562,2.835718,16.549814,20.577419,25.491782
    python certify_laguerre.py --json llp_d1_n16.json
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction

from .laguerre_basis import lag_coeffs_frac, orders_for


# ---------------------------------------------------------------------------
# Exact polynomial arithmetic over Q (ascending coefficient lists)
# ---------------------------------------------------------------------------

def p_trim(p):
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def p_add(a, b):
    n = max(len(a), len(b))
    return p_trim([ (a[i] if i < len(a) else Fraction(0))
                  + (b[i] if i < len(b) else Fraction(0)) for i in range(n)])


def p_scal(a, s):
    return [x * s for x in a]


def p_mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return p_trim(out)


def p_divmod(a, b):
    """Exact division with remainder over Q."""
    a = list(a)
    q = [Fraction(0)] * max(len(a) - len(b) + 1, 1)
    while len(a) >= len(b) and any(a):
        a = p_trim(a)
        if len(a) < len(b):
            break
        k = len(a) - len(b)
        c = a[-1] / b[-1]
        q[k] = c
        for i, y in enumerate(b):
            a[i + k] -= c * y
        a = a[:-1]
    return p_trim(q), p_trim(a)


def p_der(a):
    return p_trim([a[i] * i for i in range(1, len(a))]) if len(a) > 1 else [Fraction(0)]


def p_eval(a, x: Fraction) -> Fraction:
    r = Fraction(0)
    for c in reversed(a):
        r = r * x + c
    return r


# ---------------------------------------------------------------------------
# Sturm
# ---------------------------------------------------------------------------

def _to_int(p):
    """Clear denominators and content: primitive integer polynomial, sign kept."""
    from math import gcd
    den = 1
    for c in p:
        den = den * c.denominator // gcd(den, c.denominator)
    ip = [int(c * den) for c in p]
    g = 0
    for c in ip:
        g = gcd(g, abs(c))
    g = g or 1
    return [c // g for c in ip]


def _prem_neg_primitive(a, b):
    """primitive(-(|lc(b)|^(delta+1) * a  mod  b)) over Z.

    The multiplier is POSITIVE, so this is the Sturm remainder up to a
    positive scalar, which leaves every sign count unchanged.  Doing the
    chain over Fractions is what made the degree-38 certificate hang: naive
    rational remainders square the bit-length at every step, while the
    primitive pseudo-remainder sequence keeps coefficients compact.
    """
    from math import gcd
    da, db = len(a) - 1, len(b) - 1
    delta = da - db
    lb = abs(b[-1])
    r = [c * lb ** (delta + 1) for c in a]
    for k in range(delta, -1, -1):
        if len(r) - 1 < db + k:
            continue
        q = r[db + k] // b[-1] if r[db + k] % b[-1] == 0 else None
        # exact pseudo-division: coefficient is divisible by construction
        q = r[db + k] // b[-1]
        for i in range(db + 1):
            r[i + k] -= q * b[i]
    while len(r) > 1 and r[-1] == 0:
        r.pop()
    r = [-c for c in r]
    g = 0
    for c in r:
        g = gcd(g, abs(c))
    g = g or 1
    return [c // g for c in r]


def sturm_chain(p):
    a = _to_int(p)
    b = _to_int(p_der(p))
    chain = [a, b]
    while len(chain[-1]) > 1 or chain[-1][0] != 0:
        r = _prem_neg_primitive(chain[-2], chain[-1])
        if not any(r):
            break
        chain.append(r)
        if len(r) == 1:
            break
    return chain


def sign_variations_at(chain, x):
    signs = []
    for p in chain:
        v = Fraction(0)
        for c in reversed(p):
            v = v * x + c
        if v != 0:
            signs.append(1 if v > 0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def sign_variations_at_inf(chain):
    signs = []
    for p in chain:
        lc = p[-1]
        if lc != 0:
            signs.append(1 if lc > 0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def count_roots_on_ray(R, u_q: Fraction) -> int:
    """Number of DISTINCT real roots of R in (u_q, +oo), by Sturm."""
    chain = sturm_chain(R)
    return sign_variations_at(chain, u_q) - sign_variations_at_inf(chain)


# ---------------------------------------------------------------------------
# Exact collocation
# ---------------------------------------------------------------------------

def frac_solve(A, b):
    """Gaussian elimination with partial (max-|.|) pivoting over Fraction."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[piv][col] == 0:
            raise ZeroDivisionError("singular exact system")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def exact_collocate(taus, orders, d: int):
    """P in Q[u], P(0)=0, exact double roots at the rational taus, c_top = 1."""
    coef = {n: lag_coeffs_frac(n, d) for n in orders}
    dcoef = {n: p_der(coef[n]) for n in orders}
    rows, rhs = [], []
    rows.append([coef[n][0] for n in orders]);            rhs.append(Fraction(0))
    for t in taus:
        rows.append([p_eval(coef[n], t) for n in orders]);  rhs.append(Fraction(0))
    for t in taus:
        rows.append([p_eval(dcoef[n], t) for n in orders]); rhs.append(Fraction(0))
    nmax = max(orders)
    rows.append([(coef[n][-1] if n == nmax else Fraction(0)) for n in orders])
    rhs.append(Fraction(1))
    if len(rows) != len(orders):
        raise ValueError(f"square collocation: {len(orders)} basis vs "
                         f"{len(rows)} rows -- need len(taus) = (n_basis-2)/2")
    c = frac_solve(rows, rhs)
    P = [Fraction(0)] * (nmax + 1)
    for cj, n in zip(c, orders):
        for i, pc in enumerate(coef[n]):
            P[i] += cj * pc
    return c, p_trim(P)


# ---------------------------------------------------------------------------

def certify(d: int, s: int, taus_float, r1_hint: float | None = None,
            tau_digits: int = 9, verbose: bool = True) -> dict:
    if s != 1:
        raise NotImplementedError("s=-1 uses odd orders; same code path, "
                                  "flip orders_for -- not wired yet")
    # Rationalise the taus to 10^-tau_digits rather than taking the full
    # 52-bit dyadic: the certificate is exact for WHATEVER rational taus we
    # pick, and short denominators keep the Fraction Gaussian elimination
    # from ballooning into 10^5-digit rationals (degree-38 with full dyadic
    # taus did not finish; with 9 digits it takes seconds).  The function
    # moves by ~1e-9, far inside the slack that u_q already allows.
    taus = [Fraction(round(float(t) * 10 ** tau_digits), 10 ** tau_digits)
            for t in taus_float]
    n_basis = 2 * len(taus) + 2
    orders = orders_for(n_basis, s)
    c, P = exact_collocate(taus, orders, d)

    # exact deflation by every double root
    R = P
    for t in taus:
        lin = [-t, Fraction(1)]
        for _ in range(2):
            R, rem = p_divmod(R, lin)
            if any(rem):
                return {"verdict": "FAIL",
                        "reason": f"P not exactly divisible by (u-{float(t)})^2"}
    if R[-1] <= 0:
        return {"verdict": "FAIL", "reason": "leading coefficient of R <= 0"}

    # numerical hint for the simple root, then a rational u_q just above it
    import numpy as np
    if r1_hint is None:
        rr = np.roots([float(x) for x in reversed(R)])
        real = sorted(r.real for r in rr if abs(r.imag) < 1e-9 and r.real > 0)
        if not real:
            return {"verdict": "FAIL", "reason": "no positive real root of R found "
                                                 "numerically; supply --r1"}
        r1_hint = real[0]

    for bump in (1e-9, 1e-8, 1e-7, 1e-6):
        u_q = Fraction(math.ceil(r1_hint * (1 + bump) * 10 ** 12), 10 ** 12)
        if p_eval(R, u_q) > 0:
            break
    else:
        return {"verdict": "FAIL", "reason": "could not place rational u_q with "
                                             "R(u_q) > 0"}

    n_ray = count_roots_on_ray(R, u_q)
    ok = (n_ray == 0)
    rho_sq = u_q / Fraction(math.pi)          # only for display; the exact
    out = {"verdict": "CERTIFIED" if ok else "FAIL",
           "d": d, "s": s,
           "n_basis": n_basis, "degree": len(P) - 1,
           "taus": [str(t) for t in taus],
           "u_q": str(u_q),
           "u_q_float": float(u_q),
           "rho_bound": math.sqrt(float(u_q) / math.pi),
           "deg_R": len(R) - 1,
           "roots_of_R_on_ray": n_ray,
           "R_at_uq_positive": True,
           "R_leading_positive": True,
           "coeffs_laguerre": [str(x) for x in c]}
    if not ok:
        out["reason"] = f"Sturm counts {n_ray} real roots of R in (u_q, oo)"
    if verbose:
        print(f"exact collocation: n_basis={n_basis}, degree {len(P)-1}, "
              f"deflated R degree {len(R)-1}")
        print(f"u_q = {float(u_q):.12f}  (rational {u_q.numerator}/{u_q.denominator})")
        print(f"Sturm: distinct real roots of R in (u_q, +oo) = {n_ray}")
        print(f"R(u_q) > 0 exactly; leading coefficient of R > 0 exactly")
        if ok:
            print(f"THEOREM:  A_+({d})  <=  sqrt(u_q/pi)  =  "
                  f"{out['rho_bound']:.12f}   [CERTIFIED over Q]")
        else:
            print(f"FAIL: {out['reason']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--sign", type=int, choices=(1,), default=1)
    ap.add_argument("--taus", default=None, help="comma-separated contact points")
    ap.add_argument("--json", default=None,
                    help="read taus (and d) from a laguerre_lp candidate JSON")
    ap.add_argument("--tau-digits", type=int, default=9)
    ap.add_argument("--r1", type=float, default=None,
                    help="numerical hint for the simple root (optional)")
    ap.add_argument("--out", default=None, help="write certificate JSON here")
    a = ap.parse_args()

    if a.json:
        doc = json.load(open(a.json, encoding="utf-8"))
        d = int(doc["d"])
        taus = [float(x) for x in doc["taus"]]
    elif a.taus:
        d = a.d
        taus = [float(v) for v in a.taus.split(",")]
    else:
        ap.error("need --taus or --json")

    cert = certify(d, 1, taus, r1_hint=a.r1, tau_digits=a.tau_digits)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(cert, fh, indent=1)
        print(f"certificate written to {a.out}")


if __name__ == "__main__":
    main()
