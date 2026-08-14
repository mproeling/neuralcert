"""
certify_v6.py -- rigorous lower bound on M_k for a single-channel rational
Maynard trial function, with DIRECTED ROUNDING end to end.

Changes over v5:
  * every error bound (band, p-series tail, aliasing) is assembled in Arb
    ball arithmetic and only leaves it through provably outward-rounded
    floats (ub_f / lb_f below), closing the audit's item 1;
  * Fourier nodes are exact multiples of 2 pi / P: each node is the Arb ball
    arb(m) * (2 arb.pi() / P), so the Poisson-summation aliasing argument
    applies verbatim at the intended period P;
  * far-field blocks live in NODE-INDEX space, so the assignment of omitted
    nodes to blocks is exact integer bookkeeping with no edge cases;
  * the p-series tail starts at max(index(Theta_ball), nnode) -- the small-k
    fix: the tail begins where the computed sum ends, never before it.

The mathematical chain (reduction, Fourier identity, one-sided aliasing,
compound-Poisson moment domination, IBP tail bound) is unchanged and is
documented in the accompanying LaTeX note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction

import numpy as np
from flint import arb, acb, ctx

CERT_FORMAT = "maynard-Mk-certificate/2"


# ---------------------------------------------------------------------------
# directed-rounding helpers: the ONLY exits from ball arithmetic
# ---------------------------------------------------------------------------
def ub_f(ball: arb) -> float:
    """Smallest float provably >= every member of `ball` (nudged upward)."""
    f = float(ball.upper())
    if not math.isfinite(f):
        raise FloatingPointError("non-finite ball in ub_f")
    for _ in range(64):
        if arb(f) >= ball:
            return f
        f = math.nextafter(f, math.inf)
    raise FloatingPointError("ub_f failed to certify an upper float")


def lb_f(ball: arb) -> float:
    """Largest float provably <= every member of `ball` (nudged downward)."""
    f = float(ball.lower())
    if not math.isfinite(f):
        raise FloatingPointError("non-finite ball in lb_f")
    for _ in range(64):
        if arb(f) <= ball:
            return f
        f = math.nextafter(f, -math.inf)
    raise FloatingPointError("lb_f failed to certify a lower float")


# ---------------------------------------------------------------------------
# analytic pieces (identical mathematics to v5)
# ---------------------------------------------------------------------------
def what_over_m0(th: acb, c: arb, n: int) -> acb:
    """what(th)/m0 as a ball; th may be exact, a node ball, or a wide ball.

    Ci/Si closed form for |th| >= 1; direct rigorous integration below
    (Ci(0) = -inf although the ratio -> 1).  For th < 0 the principal-branch
    i*pi terms of the two Ci's cancel identically because both arguments
    share the sign of th; verified against the direct integral.
    """
    th = acb(th)
    cc, nn = acb(c), acb(n)
    if float(abs(th).upper()) < 1.0:
        f = lambda x, _: (1j * th * x).exp() / (cc + nn * x) ** 2
        return acb.integral(f, 0, 1) * (cc * (cc + nn))
    lam = th / nn
    a, b = lam * cc, lam * (cc + nn)
    F1 = (-1j * a).exp() * ((b.ci() - a.ci()) + 1j * (b.si() - a.si())) / nn
    what = -((1j * th).exp() / (cc + nn) - 1 / cc - 1j * th * F1) / nn
    return what * (cc * (cc + nn))


def psi(th, c: arb, n: int, kind: str, tol_bits: int) -> acb:
    """Psi_X(th) = int_0^1 X(rho) e^{i th rho} drho as a ball."""
    cc, nn = acb(c), acb(n)
    th = acb(th)
    if kind == "H":
        f = lambda x, _: (1 / cc - 1 / (cc + nn * x)) / nn * (1j * th * x).exp()
    elif kind == "G2":
        f = lambda x, _: (((1 + nn * x / cc).log() / nn) ** 2
                          * (1j * th * x).exp())
    else:
        raise ValueError(kind)
    split = float(c) / n
    pts = [0.0, split, min(50 * split, 0.5), 1.0]
    tot = acb(0)
    for a, b in zip(pts[:-1], pts[1:]):
        if b > a:
            tot += acb.integral(f, a, b, rel_tol=2.0 ** (-tol_bits))
    return tot


def moment_t(c: arb, n: int, j: int) -> acb:
    cc, nn = acb(c), acb(n)
    f = lambda x, _: x ** j / (cc + nn * x) ** 2
    return (cc * (cc + nn)) * acb.integral(f, 0, 1)


def markov_tail(c: arb, n: int, y: float, rmax: int) -> float:
    """Upper bound on P(S > y) via Markov + compound-Poisson moments.

    Any single r yields a valid bound, so min over r of certified uppers is
    itself a certified upper bound.
    """
    Et = [None] + [moment_t(c, n, j) for j in range(1, rmax + 1)]
    M = [acb(1)]
    best = math.inf
    ya = arb(y)
    for r in range(1, rmax + 1):
        s = acb(0)
        for j in range(1, r + 1):
            s += acb(math.comb(r - 1, j - 1)) * acb(n) * Et[j] * M[r - j]
        M.append(s)
        best = min(best, ub_f(abs(M[r]) / ya ** r))
    return best


# ---------------------------------------------------------------------------
# far field, in node-index space
# ---------------------------------------------------------------------------
def block_sup(c: arb, n: int, lo: float, hi: float, npanels: int) -> float:
    """Certified sup of |what/m0| over theta in [lo, hi] via wide balls."""
    edges = np.geomspace(lo, hi, npanels + 1)
    worst = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mid, rad = 0.5 * (a + b), 0.5 * (b - a)
        z = what_over_m0(acb(arb(mid, rad)), c, n)
        worst = max(worst, ub_f(abs(z)))
    return worst


def far_blocks_idx(c: arb, n: int, m_lo: int, m_hi: int, dth_lo: float,
                   dth_hi: float, ratio: float, npanels: int):
    """Blocks [a, b] of node indices with certified sups covering their hulls.

    The hull [a*dth_lo*(1-1e-13), b*dth_hi*(1+1e-13)] contains the EXACT
    theta of every node with index in [a, b], so the block sup bounds
    |what/m0| at each of those nodes."""
    blocks = []
    a = m_lo
    while a <= m_hi:
        b = min(max(int(math.ceil(a * ratio)), a + 1) - 1, m_hi)
        lo = a * dth_lo * (1 - 1e-13)
        hi = b * dth_hi * (1 + 1e-13)
        blocks.append((a, b, block_sup(c, n, lo, hi, npanels)))
        a = b + 1
    return blocks


# ---------------------------------------------------------------------------
# the certifier
# ---------------------------------------------------------------------------
def certify(c_frac: Fraction, k: int, P: float = 8.0, th_far: float = 120.0,
            prec: int = 200, tol_bits: int = 45, rmax: int = 12,
            npanels: int = 80, block_ratio: float = 2.0,
            band_tol: float = 1e-8, verbose: bool = True):
    ctx.prec = prec
    n = k - 1
    c = arb(c_frac.numerator) / arb(c_frac.denominator)

    dthb = 2 * arb.pi() / arb(P)                 # ball around exact 2 pi / P
    dth_lo, dth_hi = float(dthb.lower()), ub_f(dthb)

    A_ball = 2 * (c + n) / c
    A_hi = ub_f(A_ball)
    Th_ball = 2.0 * A_hi

    # ---- adaptive th_far: cheap probe on band only, tail handled by M0 fix
    XintN_ball = psi(arb(0), c, n, "G2", tol_bits).real
    XintD_ball = psi(arb(0), c, n, "H", tol_bits).real
    for _ in range(6):
        nnode = int(math.ceil(th_far / dth_lo))
        M0 = max(int(math.floor(Th_ball / dth_lo)), nnode)
        blocks = far_blocks_idx(c, n, nnode + 1, M0, dth_lo, dth_hi,
                                block_ratio, npanels) if M0 > nnode else []
        band_ball = arb(0)
        for a, b, sup in blocks:
            band_ball += arb(b - a + 1) * arb(sup) ** n
        band_ball = band_ball * dthb / arb.pi() * XintN_ball
        if ub_f(band_ball) <= band_tol * lb_f(XintN_ball) \
                or th_far >= Th_ball / 4:
            break
        th_far *= 2.0
    if verbose:
        print(f"  A = {A_hi:.6g}   Theta_ball = {Th_ball:.6g}   "
              f"theta_far = {th_far:g}   nodes = {2*nnode+1}   "
              f"far blocks = {len(blocks)}")

    # ---- aliasing
    pt = markov_tail(c, n, P - 1.0, rmax)
    if verbose:
        print(f"  aliasing: P(S > {P-1:g}) <= {pt:.3e}   (Markov, r <= {rmax})")

    # ---- computed trapezoid sums at EXACT nodes m * (2 pi / P)
    def trap(kind):
        tot = acb(0)
        for m in range(-nnode, nnode + 1):
            th = arb(m) * dthb
            ph = what_over_m0(acb(th), c, n) ** n
            tot += ph * (-1j * acb(th)).exp() * psi(th, c, n, kind, tol_bits)
        return tot * acb(dthb) / (2 * acb.pi())

    N = trap("G2").real
    D = trap("H").real

    # ---- error assembly, entirely in balls until the final exits
    def errors(Xint_ball, Xmax_ball):
        bb = arb(0)
        for a, b, sup in blocks:
            bb += arb(b - a + 1) * arb(sup) ** n
        band = bb * dthb / arb.pi() * Xint_ball
        tail = (dthb / arb.pi()) * Xint_ball * (arb(A_hi) / dthb) ** n \
            * arb(M0) ** (1 - n) / arb(n - 1)
        alias = Xmax_ball * arb(pt)
        return band, tail, alias

    XmaxN_ball = ((1 + arb(n) / c).log() / arb(n)) ** 2
    XmaxD_ball = (1 / c - 1 / (c + n)) / arb(n)
    bN, tN, aN = errors(XintN_ball, XmaxN_ball)
    bD, tD, aD = errors(XintD_ball, XmaxD_ball)

    N_lo = lb_f(N - (bN + tN + aN))
    D_hi = ub_f(D + (bD + tD + aD))
    if not (N_lo > 0 and D_hi > 0):
        raise FloatingPointError("enclosure did not produce positive bounds")
    R_ball = arb(k) * arb(N_lo) / arb(D_hi)
    R_lo = lb_f(R_ball)
    R_mid = k * float(N.mid()) / float(D.mid())

    if verbose:
        print(f"  N in {N}   err(band,tail,alias) = "
              f"({ub_f(bN):.2e}, {ub_f(tN):.2e}, {ub_f(aN):.2e})")
        print(f"  D in {D}   err = ({ub_f(bD):.2e}, {ub_f(tD):.2e}, "
              f"{ub_f(aD):.2e})")

    ceil_k = k / (k - 1.0) * math.log(k)
    if R_lo > ceil_k * (1 + 1e-9):
        raise AssertionError("certified bound exceeds the ceiling -- defect")

    info = dict(
        k=k, n=n, c=str(c_frac), c_float=float(c_frac),
        params=dict(P=P, theta_far=th_far, prec=prec, tol_bits=tol_bits,
                    rmax=rmax, block_ratio=block_ratio, npanels=npanels,
                    Theta_ball=Th_ball, A_upper=A_hi, nnode=nnode, M0=M0,
                    assembly="directed-arb/2"),
        N=dict(lower=float(N.lower()), upper=float(N.upper())),
        D=dict(lower=float(D.lower()), upper=float(D.upper())),
        err=dict(N_band=ub_f(bN), N_tail=ub_f(tN), N_alias=ub_f(aN),
                 D_band=ub_f(bD), D_tail=ub_f(tD), D_alias=ub_f(aD),
                 P_S_gt=pt, y=P - 1.0),
        far_blocks=[dict(m_lo=a, m_hi=b, sup=s,
                         lo=a * dth_lo, hi=b * dth_hi) for a, b, s in blocks],
        final=dict(N_lower_used=N_lo, D_upper_used=D_hi, R_lower=R_lo,
                   R_midpoint=R_mid, ceiling=ceil_k),
    )
    return R_lo, R_mid, info


# ---------------------------------------------------------------------------
# npz loading, MC diagnostic, certificate export
# ---------------------------------------------------------------------------
def load(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            d = {name: archive[name] for name in archive.files}
    except ValueError as exc:
        raise ValueError(
            f"{path!s} is not a pickle-free NumPy NPZ export; re-export it "
            "with `neuracert discover --method ratio --export ...`") from exc
    canon = str(d["canonical"])
    got = hashlib.sha256(canon.encode()).hexdigest()
    if got != str(d["sha256"]):
        raise ValueError("sha256 mismatch: the .npz has been altered")
    epsilon = (Fraction(int(d["epsilon_num"]), int(d["epsilon_den"]))
               if "epsilon_num" in d else Fraction(0))
    return dict(k=int(d["k"]), epsilon=epsilon,
                cs=[Fraction(int(a), int(b))
                    for a, b in zip(d["c_num"], d["c_den"])],
                ws=[Fraction(int(a), int(b))
                    for a, b in zip(d["w_num"], d["w_den"])],
                powers=[int(p) for p in d["power"]],
                canonical=canon, sha=got,
                R_discovery=float(d["R_discovery"]),
                ceiling=float(d["ceiling"]))


def monte_carlo(c_frac, k, nsamp=40000, seed=0, max_n=4_000_000):
    """Independent simulation of N, D (diagnostic; see v5 for full notes)."""
    n = k - 1
    if n > max_n:
        return dict(skipped=True, reason=f"n = {n} > max_n")
    c = float(c_frac)
    m0 = 1.0 / (c * (c + n))
    rng = np.random.default_rng(seed)
    chunk = max(1, min(4000, 40_000_000 // max(n, 1)))
    num = den = 0.0
    cnt_k = 0
    tot = 0
    t0 = time.time()
    while tot < nsamp:
        m = min(chunk, nsamp - tot)
        u = rng.random((m, n))
        T = (c / n) * (1.0 / (1.0 - u * n / (c + n)) - 1.0)
        Sn = T.sum(axis=1)
        ue = rng.random(m)
        Sk = Sn + (c / n) * (1.0 / (1.0 - ue * n / (c + n)) - 1.0)
        ok = Sn <= 1.0
        rho = 1.0 - Sn[ok]
        num += float(np.sum((np.log1p(n * rho / c) / n) ** 2))
        den += float(np.sum((1.0 / c - 1.0 / (c + n * rho)) / n))
        cnt_k += int((Sk <= 1.0).sum())
        tot += m
    N_mc, D_mc = num / tot, den / tot
    D_alt = m0 * cnt_k / tot
    return dict(skipped=False, samples=tot, seconds=time.time() - t0,
                N=N_mc, D=D_mc, D_alt=D_alt, R=k * N_mc / D_mc,
                rel_D_mismatch=abs(D_mc - D_alt) / D_mc)


def build_certificate(info, src, mc=None):
    k = info["k"]
    cert = {
        "format": CERT_FORMAT,
        "claim": {
            "statement": f"M_{k} >= "
                         f"{math.floor(info['final']['R_lower']*1e9)/1e9!r}",
            "k": k,
            # floored at the 9th decimal: leaves a robustness margin so an
            # independent re-run (different Arb version, different enclosure
            # radii) still certifies the stated claim
            "M_k_lower_bound": math.floor(info["final"]["R_lower"] * 1e9) / 1e9,
            "known_upper_bound_ceiling": info["final"]["ceiling"],
            "ceiling_source": "M_k < k/(k-1) log k  (Polymath8b)",
            "primes_implied": math.ceil(info["final"]["R_lower"] / 4),
            "primes_criterion": "r_k = ceil(theta M_k / 2), theta = 1/2 - eps "
                                "(Bombieri-Vinogradov; Maynard Prop. 4.2)",
        },
        "trial_function": {
            "form": "F = prod_{i=1..k} g(t_i) on the simplex",
            "g": "g(t) = 1/(c + n t),  n = k-1",
            "c_exact": info["c"],
            "u_exact": src.get("u_exact", "1"),
            "note": "R is invariant under scaling of the weight; any fixed "
                    "weight yields a valid lower bound.",
        },
        "parameters": info["params"],
        "certified_quantities": {"N_enclosure": info["N"],
                                 "D_enclosure": info["D"]},
        "error_terms": info["err"],
        "far_field_blocks": info["far_blocks"],
        "final_arithmetic": info["final"],
        "assembly_note": "All error bounds assembled in Arb ball arithmetic; "
                         "floats exit only through provably outward-rounded "
                         "conversions (nudge-verified against the ball).",
        "source": src,
    }
    if mc is not None:
        cert["monte_carlo_diagnostic"] = mc
    payload = json.dumps({key: cert[key] for key in cert},
                         sort_keys=True, separators=(",", ":"))
    cert["self_hash_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return cert


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_positional", nargs="?", metavar="NPZ",
                    help="discovery .npz export (positional compatibility form)")
    ap.add_argument("--npz", dest="npz_option",
                    help="discovery .npz export")
    ap.add_argument("--prec", type=int, default=200)
    ap.add_argument("--P", type=float, default=8.0)
    ap.add_argument("--theta-far", type=float, default=120.0)
    ap.add_argument("--tol-bits", type=int, default=45)
    ap.add_argument("--rmax", type=int, default=12)
    ap.add_argument("--band-tol", type=float, default=1e-8)
    ap.add_argument("--monte-carlo", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=40000)
    ap.add_argument("--cert-json", type=str, default=None)
    a = ap.parse_args(argv)
    if a.npz_option and a.npz_positional:
        ap.error("give the discovery export either positionally or with --npz, not both")
    npz_path = a.npz_option or a.npz_positional
    if not npz_path:
        ap.error("an NPZ discovery export is required (use --npz FILE)")

    d = load(npz_path)
    print(f"trial function : {d['canonical'][:88]}")
    print(f"  sha256       = {d['sha'][:16]}...  (verified)")
    print(f"  k            = {d['k']}   R_discovery = {d['R_discovery']:.10f}")
    if d["epsilon"] != 0:
        raise NotImplementedError(
            "ratio certification currently supports epsilon=0 only; "
            f"this discovery export has epsilon={d['epsilon']}")
    if len(d["cs"]) != 1 or d["powers"][0] != 1:
        raise NotImplementedError("certify_v6 handles one power-1 channel")

    t0 = time.time()
    R_lo, R_mid, info = certify(d["cs"][0], d["k"], P=a.P, th_far=a.theta_far,
                                prec=a.prec, tol_bits=a.tol_bits,
                                rmax=a.rmax, band_tol=a.band_tol)
    print(f"  R midpoint   = {R_mid:.10f}   "
          f"|mid - discovery| = {abs(R_mid - d['R_discovery']):.2e}")
    print(f"\n  CERTIFIED (directed):  M_{d['k']} >= {R_lo:.10f}   "
          f"[{time.time()-t0:.0f}s]")
    print(f"  => at least {math.ceil(R_lo/4)} primes infinitely often "
          f"(theta = 1/2 - eps)")

    mc = None
    if a.monte_carlo:
        mc = monte_carlo(d["cs"][0], d["k"], nsamp=a.mc_samples)
        if not mc.get("skipped"):
            print(f"  MC diagnostic: R = {mc['R']:.6f}   peeling rel diff "
                  f"{mc['rel_D_mismatch']:.2e}")
    if a.cert_json:
        src = dict(npz=npz_path, canonical=d["canonical"], sha256=d["sha"],
                   R_discovery=d["R_discovery"], u_exact=str(d["ws"][0]))
        cert = build_certificate(info, src, mc=mc)
        with open(a.cert_json, "w") as f:
            json.dump(cert, f, indent=2, sort_keys=True)
        print(f"  certificate written to {a.cert_json}")
        print(f"  self hash sha256 = {cert['self_hash_sha256'][:32]}...")


if __name__ == "__main__":
    main()
