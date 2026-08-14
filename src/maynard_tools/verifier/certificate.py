"""
Unified standalone verifier for Maynard certificates.

Usage:
  maynard-verify cert.json --method poly
  maynard-verify cert.json --method ratio

The poly route is the legacy exact-rational verifier.
The ratio route independently verifies a single-rational-product certificate
using python-flint Arb ball arithmetic and does not import discovery/certifier code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Exact polynomial arithmetic — deliberately naive and auditable
# ---------------------------------------------------------------------------
def poly_mul(a: list[int], b: list[int]) -> list[int]:
    """Schoolbook product of integer polynomials (coefficient lists)."""
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj:
                out[i + j] += ai * bj
    return out


def poly_pow(a: list[int], e: int) -> list[int]:
    """a(x)^e by square-and-multiply."""
    result = [1]
    base = a
    while e:
        if e & 1:
            result = poly_mul(result, base)
        e >>= 1
        if e:
            base = poly_mul(base, base)
    return result


def parse_frac(s: str) -> Fraction:
    return Fraction(s) if "/" not in s else Fraction(*map(int, s.split("/")))


# ---------------------------------------------------------------------------
# One channel pair -> (A_{jl}, B_{jl}) exactly
# ---------------------------------------------------------------------------
def pair_AB(args):
    (nj, dj, nl, dl, k, rho_n, rho_d, cj_list_j, cj_list_l) = args
    # h = q_j * q_l as integers over denominator dj*dl
    h = poly_mul(nj, nl)
    den_h = dj * dl

    fact = [1]
    max_f = k + 2 * (len(h) - 1) * k + len(h) * 2 + k + 6
    for i in range(1, max_f + 1):
        fact.append(fact[-1] * i)

    bh = [h[r] * fact[r] for r in range(len(h))]

    P_km1 = poly_pow(bh, k - 1)
    P_k = poly_mul(P_km1, bh)

    # ---- A ----
    K = k + len(P_k) - 1
    A_num = sum(P_k[R] * (fact[K] // fact[k + R]) for R in range(len(P_k)))
    A = Fraction(A_num, fact[K] * den_h ** k)

    # ---- B ----
    # Q(y) = int_0^y q: coefficient of y^{p+1} is n_p / (d * (p+1))
    Qj = [Fraction(0)] + [Fraction(n, dj * (p + 1)) for p, n in enumerate(nj)]
    Ql = [Fraction(0)] + [Fraction(n, dl * (p + 1)) for p, n in enumerate(nl)]
    q = [Fraction(0)] * (len(Qj) + len(Ql) - 1)
    for pj, aj in enumerate(Qj):
        if aj == 0:
            continue
        for pl, al in enumerate(Ql):
            if al:
                q[pj + pl] += aj * al

    Mq = len(q) - 1
    qt = []
    for m_ in range(Mq + 1):
        acc = sum(q[mp] * math.comb(mp, m_) for mp in range(m_, Mq + 1)
                  if q[mp] != 0)
        qt.append(acc if m_ % 2 == 0 else -acc)

    rho = Fraction(rho_n, rho_d)
    rho_pow = [Fraction(1)]
    max_pow = Mq + (k - 1) + (len(P_km1) - 1) + 2
    for _ in range(max_pow):
        rho_pow.append(rho_pow[-1] * rho)

    w = [qt[m_] * rho_pow[m_] for m_ in range(Mq + 1)]
    nz = [m_ for m_ in range(Mq + 1) if w[m_] != 0]

    B = Fraction(0)
    den_km1 = den_h ** (k - 1)
    for R in range(len(P_km1)):
        NR = P_km1[R]
        if NR == 0:
            continue
        base = k - 1 + R
        inner = Fraction(0)
        for m_ in nz:
            inner += w[m_] / (m_ + base)
        B += Fraction(NR, fact[k - 2 + R]) * rho_pow[base] * inner
    B = B / den_km1
    return A, B


# ---------------------------------------------------------------------------
# Self checks (run every invocation, at the certificate's own k, eps)
# ---------------------------------------------------------------------------
def self_checks(k: int, eps: Fraction) -> None:
    print("  [self-check] re-deriving closed-form identities ...")
    rho = (1 - eps) / (1 + eps)

    # (1) q == 1: A = 1/k!, B = [rho^{k-1}/(k-1) - 2rho^k/k + rho^{k+1}/(k+1)]
    #                             / (k-2)!    -- from Dirichlet directly.
    A, B = pair_AB(([1], 1, [1], 1, k, rho.numerator, rho.denominator,
                    None, None))
    A_cf = Fraction(1, math.factorial(k))
    B_cf = (rho ** (k - 1) / (k - 1) - 2 * rho ** k / k
            + rho ** (k + 1) / (k + 1)) / math.factorial(k - 2)
    if A != A_cf or B != B_cf:
        sys.exit("  SELF-CHECK FAILED (q=1 closed form). Do not trust "
                 "this verifier build.")

    # (2) q = 1 - u at k=2, eps: independent Dirichlet double sum for A.
    A2, _ = pair_AB(([1, -1], 1, [1, -1], 1, 2, rho.numerator,
                     rho.denominator, None, None))
    b = [Fraction(1), Fraction(-2), Fraction(1)]        # (1-u)^2
    A2_direct = Fraction(0)
    for r1 in range(3):
        for r2 in range(3):
            A2_direct += (b[r1] * b[r2] * math.factorial(r1)
                          * math.factorial(r2)
                          * Fraction(1, math.factorial(2 + r1 + r2)))
    if A2 != A2_direct:
        sys.exit("  SELF-CHECK FAILED (Dirichlet double sum). Do not trust "
                 "this verifier build.")
    print("  [self-check] PASSED (closed forms hold as exact identities)")


# ---------------------------------------------------------------------------
# Verification driver
# ---------------------------------------------------------------------------
def verify_poly(path: str, jobs: int = 1, digits: int = 30) -> bool:
    with open(path) as f:
        cert = json.load(f)

    if cert.get("format") != "maynard-separable-certificate-v1":
        sys.exit(f"unrecognized certificate format: {cert.get('format')!r}")

    k = int(cert["k"])
    eps = Fraction(int(cert["epsilon"]["num"]), int(cert["epsilon"]["den"]))
    if not (k >= 2 and 0 <= eps < 1):
        sys.exit("certificate parameters out of range (need k>=2, "
                 "0 <= eps < 1)")
    L = 1 + eps
    rho = (1 - eps) / (1 + eps)
    channels = [[parse_frac(s) for s in ch] for ch in cert["channels"]]
    c = [parse_frac(s) for s in cert["c"]]
    claimed = Fraction(cert["claimed_lower_bound"])
    J = len(channels)
    if len(c) != J:
        sys.exit("channel / coefficient count mismatch")

    print(f"  quantity      : {cert.get('quantity', '(unspecified)')}")
    print(f"  k = {k}   eps = {eps}   L = {L}   rho = {rho}")
    print(f"  channels      : {J}   max degree "
          f"{max(len(ch) - 1 for ch in channels)}")
    print(f"  claimed bound : {cert['claimed_lower_bound']}")

    self_checks(k, eps)

    # integer form of each channel: numerators over one denominator
    int_ch, dens = [], []
    for ch in channels:
        den = 1
        for co in ch:
            den = den * co.denominator // math.gcd(den, co.denominator)
        int_ch.append([int(co * den) for co in ch])
        dens.append(den)

    tasks = [(int_ch[j], dens[j], int_ch[l], dens[l], k,
              rho.numerator, rho.denominator, None, None)
             for j in range(J) for l in range(j, J)]
    print(f"  recomputing A, B exactly: {len(tasks)} pairs, jobs={jobs} ...")
    t0 = time.time()
    if jobs > 1:
        with Pool(jobs) as pool:
            results = pool.map(pair_AB, tasks)
    else:
        results = [pair_AB(t) for t in tasks]
    print(f"  ... done in {time.time() - t0:.1f}s")

    A = [[Fraction(0)] * J for _ in range(J)]
    B = [[Fraction(0)] * J for _ in range(J)]
    idx = 0
    for j in range(J):
        for l in range(j, J):
            A[j][l], B[j][l] = results[idx]
            A[l][j], B[l][j] = A[j][l], B[j][l]
            idx += 1

    cAc = Fraction(0)
    cBc = Fraction(0)
    for j in range(J):
        if c[j] == 0:
            continue
        for l in range(J):
            if c[l] == 0:
                continue
            cAc += c[j] * A[j][l] * c[l]
            cBc += c[j] * B[j][l] * c[l]

    if cAc <= 0:
        print("\n  FAIL: c^T A c <= 0 (trial function is a.e. zero or "
              "arithmetic error); certificate invalid.")
        return False

    R = k * L * cBc / cAc

    # truncated decimal (rounds toward zero -> printed value is a bound)
    ip = R.numerator // R.denominator
    fr = ((R.numerator - ip * R.denominator) * 10 ** digits) // R.denominator
    R_dec = f"{ip}.{str(fr).zfill(digits)}"

    print(f"\n  recomputed R  = {R_dec}")
    print(f"  claimed bound = {cert['claimed_lower_bound']}")
    ok = R >= claimed
    margin = R - claimed
    if ok:
        print(f"\n  PASS: R >= claimed bound (exact rational comparison; "
              f"margin = {float(margin):.3e}).")
        print(f"  Therefore  {cert.get('quantity', 'the target quantity')}"
              f"  >=  {cert['claimed_lower_bound']}  is PROVED, modulo only")
        print(f"  the two classical facts stated in this file's header.")
        if float(R) > 4.0 and eps > 0:
            print(f"  NOTE: R > 4 with eps = {eps} < 1: sufficient input for "
                  f"DHL[{k},2] at theta = 1/2 (Polymath8b Thm 3.12(i)).")
    else:
        print(f"\n  FAIL: recomputed R is SMALLER than the claimed bound "
              f"(deficit = {float(-margin):.3e}). Certificate invalid.")
    return ok




# ---------------------------------------------------------------------------
# Independent rational-product verifier -- DIRECTED ROUNDING (v4)
# Mirrors certify_v6: ball assembly, exact-node trapezoid, index-space
# blocks, outward-rounded float exits.  Regenerates its own far field;
# stored ladders are informational only.
# ---------------------------------------------------------------------------
def _load_flint():
    try:
        from flint import arb, acb, ctx
    except ImportError as exc:
        raise RuntimeError('--method ratio requires python-flint') from exc
    return arb, acb, ctx


def _parse_frac(s):
    return Fraction(s) if '/' not in s else Fraction(*map(int, s.split('/')))


def verify_ratio(path: str) -> bool:
    arb, acb, ctx = _load_flint()
    import numpy as np

    def ub_f(ball):
        f = float(ball.upper())
        if not math.isfinite(f):
            raise FloatingPointError("non-finite ball")
        for _ in range(64):
            if arb(f) >= ball:
                return f
            f = math.nextafter(f, math.inf)
        raise FloatingPointError("ub_f failed")

    def lb_f(ball):
        f = float(ball.lower())
        if not math.isfinite(f):
            raise FloatingPointError("non-finite ball")
        for _ in range(64):
            if arb(f) <= ball:
                return f
            f = math.nextafter(f, -math.inf)
        raise FloatingPointError("lb_f failed")

    with open(path) as fjson:
        cert = json.load(fjson)
    fmt = cert.get("format")
    if fmt not in ("maynard-Mk-certificate/1", "maynard-Mk-certificate/2"):
        raise ValueError(f"unrecognized ratio format: {fmt!r}")
    k = int(cert["claim"]["k"])
    n = k - 1
    cfrac = _parse_frac(cert["trial_function"]["c_exact"])
    # exact binary value of the parsed float; avoids the decimal-repr trap
    claim = Fraction(cert["claim"]["M_k_lower_bound"])
    prm = cert.get("parameters", {})
    P = float(prm.get("P", 8.0))
    th_far = float(prm.get("theta_far", 120.0))
    prec = int(prm.get("prec", 200))
    tol_bits = int(prm.get("tol_bits", 45))
    rmax = int(prm.get("rmax", 12))
    npanels = int(prm.get("npanels", 80))
    ratio = float(prm.get("block_ratio", 2.0))
    if not (P > 2 and th_far > 1 and ratio > 1 and npanels >= 2
            and prec >= 64 and tol_bits > 0 and rmax >= 1 and k >= 2
            and cfrac > 0):
        raise ValueError("invalid parameters")
    ctx.prec = prec
    c = arb(cfrac.numerator) / arb(cfrac.denominator)

    print("RATIONAL PRODUCT CERTIFICATE (directed v4)")
    print("==========================================")
    print(f"  k = {k}   c = {cfrac}")
    print(f"  claimed bound : {float(claim):.15g}")

    # self-checks: theta = 0 limit; closed form vs direct integral at
    # +7 (Ci/Si form), -7 (branch cancellation), 1.5 (just above handover)
    cc, nn = acb(c), acb(n)

    def wm0(th):
        th = acb(th)
        if float(abs(th).upper()) < 1.0:
            f = lambda x, _: (1j * th * x).exp() / (cc + nn * x) ** 2
            return acb.integral(f, 0, 1) * (cc * (cc + nn))
        lam = th / nn
        a, b = lam * cc, lam * (cc + nn)
        F1 = (-1j * a).exp() * ((b.ci() - a.ci())
                                + 1j * (b.si() - a.si())) / nn
        w = -((1j * th).exp() / (cc + nn) - 1 / cc - 1j * th * F1) / nn
        return w * (cc * (cc + nn))

    z0 = wm0(acb(0))
    if not (z0.real.contains(1) and z0.imag.contains(0)):
        raise RuntimeError("self-check failed at theta = 0")
    for tv in (1.5, 7.0, -7.0):
        zf = wm0(acb(tv))
        f = lambda x, _: (1j * acb(tv) * x).exp() / (cc + nn * x) ** 2
        zd = acb.integral(f, 0, 1) * cc * (cc + nn)
        dd = zf - zd
        if not (dd.real.contains(0) and dd.imag.contains(0)):
            raise RuntimeError(f"closed form vs integral mismatch at {tv}")
    print("  [self-check] PASSED (theta = 0, +-7, 1.5; branch cut exercised)")

    def psi(th, kind):
        th = acb(th)
        if kind == "H":
            f = lambda x, _: (1 / cc - 1 / (cc + nn * x)) / nn \
                * (1j * th * x).exp()
        else:
            f = lambda x, _: (((1 + nn * x / cc).log() / nn) ** 2
                              * (1j * th * x).exp())
        split = float(c) / n
        pts = [0.0, split, min(50 * split, 0.5), 1.0]
        tot = acb(0)
        for a, b in zip(pts[:-1], pts[1:]):
            if b > a:
                tot += acb.integral(f, a, b, rel_tol=2.0 ** (-tol_bits))
        return tot

    dthb = 2 * arb.pi() / arb(P)
    dth_lo, dth_hi = float(dthb.lower()), ub_f(dthb)
    A_hi = ub_f(2 * (c + n) / c)
    Th_ball = 2.0 * A_hi
    nnode = int(math.ceil(th_far / dth_lo))
    M0 = max(int(math.floor(Th_ball / dth_lo)), nnode)

    def block_sup(lo, hi):
        worst = 0.0
        for a, b in zip(np.geomspace(lo, hi, npanels + 1)[:-1],
                        np.geomspace(lo, hi, npanels + 1)[1:]):
            z = wm0(acb(arb(0.5 * (a + b), 0.5 * (b - a))))
            worst = max(worst, ub_f(abs(z)))
        return worst

    blocks = []
    a = nnode + 1
    while a <= M0:
        b = min(max(int(math.ceil(a * ratio)), a + 1) - 1, M0)
        blocks.append((a, b, block_sup(a * dth_lo * (1 - 1e-13),
                                       b * dth_hi * (1 + 1e-13))))
        a = b + 1
    print(f"  far blocks    : {len(blocks)} independently generated "
          f"(indices {nnode+1}..{M0})")

    # aliasing
    Et = [None] + [(cc * (cc + nn))
                   * acb.integral(lambda x, _, jj=j: x ** jj
                                  / (cc + nn * x) ** 2, 0, 1)
                   for j in range(1, rmax + 1)]
    M = [acb(1)]
    pt = math.inf
    for r in range(1, rmax + 1):
        s = acb(0)
        for j in range(1, r + 1):
            s += acb(math.comb(r - 1, j - 1)) * acb(n) * Et[j] * M[r - j]
        M.append(s)
        pt = min(pt, ub_f(abs(M[r]) / arb(P - 1.0) ** r))
    print(f"  aliasing      : P(S>{P-1:g}) <= {pt:.3e}")

    def trap(kind):
        tot = acb(0)
        for m in range(-nnode, nnode + 1):
            th = arb(m) * dthb
            tot += wm0(acb(th)) ** n * (-1j * acb(th)).exp() * psi(th, kind)
        return (tot * acb(dthb) / (2 * acb.pi())).real

    N = trap("G2")
    D = trap("H")
    XintN = psi(arb(0), "G2").real
    XintD = psi(arb(0), "H").real
    XmaxN = ((1 + arb(n) / c).log() / arb(n)) ** 2
    XmaxD = (1 / c - 1 / (c + n)) / arb(n)

    def errs(Xint, Xmax):
        bb = arb(0)
        for aa, b2, sup in blocks:
            bb += arb(b2 - aa + 1) * arb(sup) ** n
        band = bb * dthb / arb.pi() * Xint
        tail = (dthb / arb.pi()) * Xint * (arb(A_hi) / dthb) ** n \
            * arb(M0) ** (1 - n) / arb(n - 1)
        return band + tail + Xmax * arb(pt)

    N_lo = lb_f(N - errs(XintN, XmaxN))
    D_hi = ub_f(D + errs(XintD, XmaxD))
    if not (N_lo > 0 and D_hi > 0):
        print("  FAIL: non-positive certified bounds")
        return False
    R_lo = lb_f(arb(k) * arb(N_lo) / arb(D_hi))
    print(f"  N ball        : {N}")
    print(f"  D ball        : {D}")
    print(f"  recomputed LB : {R_lo:.12f}")
    print(f"  claimed bound : {float(claim):.12f}")
    ok = Fraction(R_lo) >= claim          # exact rational comparison
    print()
    print("  PASS: independently recomputed rigorous lower bound >= claim."
          if ok else
          "  FAIL: independently recomputed lower bound is below claim.")
    return ok


# ---------------------------------------------------------------------------
# Unified CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p=argparse.ArgumentParser(description='Standalone Maynard certificate verifier')
    p.add_argument('certificate',type=str)
    p.add_argument('--method', choices=('poly', 'ratio'), required=True,
                   help='poly = legacy exact polynomial route; ratio = rational-product Arb route')
    p.add_argument('--jobs',type=int,default=1,help='poly route only')
    p.add_argument('--digits',type=int,default=30,help='poly route only')
    args=p.parse_args()
    try:
        if args.method == 'poly':
            ok=verify_poly(args.certificate,jobs=args.jobs,digits=args.digits)
        else:
            ok=verify_ratio(args.certificate)
    except Exception as exc:
        print(f'\nVERIFIER ERROR: {type(exc).__name__}: {exc}')
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
