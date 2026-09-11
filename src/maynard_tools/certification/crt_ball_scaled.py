"""
Multimodular (CRT) exact certification for the EPSILON-ENLARGED Maynard
problem M_{k,eps,1/2}, with c-weighted channel pruning and scale-safe channel recovery.  v3.1 backend for the
family maynard_project_certify.py (vanilla) / ..._epsilon_flint_streaming_v2.py.

Functional (theta = 1/2, 0 <= eps < 1, box caps inactive), L = 1+eps, U = 1-eps:

    I(F)   = int_{L*Delta_k} F^2
    J_1(F) = int_{U*Delta_{k-1}} ( int_0^{L-S} F dt_1 )^2,   S = sum_{i>=2} t_i
    R      = k J_1 / I   (lower-bounds M_{k,eps,1/2} for any explicit F)

Exact pair formulas are IDENTICAL to the v2 streaming certifier (rescaled
u-coordinates, rho := U/L exact, h = q_j q_l, h_hat its Borel weighting):

    A_{jl} = sum_n [x^n] h_hat^k     / (k+n)!            / den_h^k
    B_{jl} = sum_m qt_m sum_R [x^R] h_hat^{k-1}
                 * rho^{m+k-1+R} / ( (k-2+R)! (m+k-1+R) ) / (t_den den_h^{k-1})

What v3 changes
---------------

1.  MULTIMODULAR EXACT BACKEND (Route C).  The giant integer object
    h_hat^(k-1) -- ~ (k-1) log2||h_hat||_1 bits PER COEFFICIENT in v2 -- is
    never formed over Z.  For word-size primes p we compute, entirely in
    nmod arithmetic,

        S_A(p) = [ sum_{j<=l} kappa c_j c_l A-numerator_{jl} ]  mod p
        S_B(p) = [ sum_{j<=l} kappa c_j c_l B-numerator_{jl} ]  mod p

    where the numerators are the KNOWN-DENOMINATOR integer scalings

        S_A_int = sum kappa cn_j cn_l sum_n [x^n] bh^k * top_a!/(k+n)!
        S_B_int = sum kappa cn_j cn_l mult_t *
                  sum_R [x^R] bh^(k-1) * (top_b!/(k-2+R)!) *
                        rho_n^{k-1+R} rho_d^{rmax-R} *
                  sum_m t_m rho_n^m rho_d^{mmax-m} * (Lam/(k-1+R+m))

    with top_a = k + k*smax, top_b = k-2 + (k-1)*smax,
    Lam = lcm{ v : k-1 <= v <= k-1+rmax+mmax }, so that

        c^T A c = S_A_int / D_A,      D_A = top_a! den_h^k cden^2
        c^T B c = S_B_int / D_B,      D_B = top_b! Lam rho_d^E T_den
                                            den_h^{k-1} cden^2,
        E = k-1+rmax+mmax,  T_den = lcm of pair t-denominators
        (per-pair multiplier mult_t = T_den / t_den_pair).

    Reconstruction is DETERMINISTIC centered CRT: with a rigorous a priori
    bound |S| <= BND (computed exactly from ||bh||_1^{k-1} etc., see
    exact_bounds()) and prod p_i > 2*BND, the centered representative equals
    the true integer.  No floats, no rational reconstruction, no
    probabilistic step.  Held-out verification primes re-check the
    reconstructed integers as an implementation sanity test.

    Per (pair, prime) the work is: one nmod_poly power bh^(k-1), one short
    multiply for bh^k, two weight-table polynomial multiplies (the linear
    functionals are executed as single coefficient extractions against
    REVERSED per-prime weight polynomials), and O(mmax) scalar-poly ops to
    combine the pair-dependent B weights.  All O(kd)-length objects are
    machine-word residue polynomials (~100 kB), built per prime and reused
    across every pair.  Parallelism is over prime chunks.

2.  FLOAT PASS WITHOUT EXACT ARITHMETIC.  v2 ran the full exact engine twice
    (once merely to float the Gram entries).  v3 computes float A, B by the
    stable iterated-convolution DP that the discovery solver uses, via the
    exact identities

        A_{jl} = int_0^1   (h^{*k})(s) ds
        B_{jl} = int_0^rho (h^{*(k-1)})(s) * Q_j(1-s) Q_l(1-s) ds

    where h^{*p} is the p-fold [0,1]-truncated convolution power of the true
    (rational-coefficient) pair polynomial h, computed by renormalized
    binary powering on a grid with per-pair log-scale tracking.  (Truncated
    self-convolution is exact for the [0,1] restriction because supports lie
    in [0,inf).)  This pass only steers pruning and the Ritz vector; the
    certificate never depends on it.

3.  C-WEIGHTED CHANNEL PRUNING.  Greedy backward elimination on the float
    Rayleigh quotient: repeatedly drop the channel whose removal (after
    re-Ritz on the remaining set) costs the least, while the cumulative
    relative loss stays within --prune-tol.  Validity is untouched -- any
    explicit c on any channel subset certifies -- only the bound quality can
    move, and the float pass measures that before any exact work.  Pair
    count falls quadratically in dropped channels.

Self tests (--self-test):
    * g == 1 closed forms  A = 1/k!,
      B = [rho^{k-1}/(k-1) - 2 rho^k/k + rho^{k+1}/(k+1)]/(k-2)!  through the
      complete CRT machinery (several k, eps), exact equality;
    * random multi-channel case, MIXED channel degrees: CRT aggregate
      S_A/D_A, S_B/D_B versus the trusted v2 exact pair worker, exact
      Fraction equality;
    * float DP versus exact pair values, relative error check;
    * deterministic Miller-Rabin sanity and CRT round trip.

Usage:
    python maynard_project_certify_epsilon_crt.py --npz k500eps.npz \
           --epsilon 1/25 --degrees 22 26 --prune-tol 1e-7 --jobs 16
    python maynard_project_certify_epsilon_crt.py --self-test
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from maynard_tools.certification.frames import load_discovery_frame

try:
    from flint import nmod_poly
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "python-flint is required for the CRT certifier. Install it with:\n"
        "    python -m pip install python-flint"
    ) from exc

try:
    import scipy.linalg as _sla
    import scipy.signal as _ssig
except ImportError:  # pragma: no cover
    _sla = None
    _ssig = None

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x rename

from .karatsuba import (
    dyadic,
    eval_shifted_legendre_design,
    frac_to_decimal_truncated,
    project_channels,
    shifted_legendre_coeffs,
    write_certificate,
)
from .flint_streaming import (
    ChannelData,
    parse_fraction,
    preprocess_channels,
)

from .scaled_crt import (
    ScaledCRTAggregator,
    project_scaled_channels,
)


# ---------------------------------------------------------------------------
# Deterministic 64-bit primality (Miller-Rabin, proven base set)
# ---------------------------------------------------------------------------
# Deterministic for all n < 318665857834031151167461 ~ 3.19e23: the
# least strong pseudoprime to all twelve prime bases 2..37 (Sorenson &
# Webster, "Strong pseudoprimes to twelve prime bases", Math. Comp. 86
# (2017), 985-1003; sequence OEIS A014233).  All moduli generated here
# are < 2^62 ~ 4.6e18, five orders of magnitude inside the proven
# range.  The self-test additionally cross-checks generated primes
# against FLINT's proven fmpz.is_prime when available.
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime_u64(n: int) -> bool:
    if n < 2:
        return False
    for q in _MR_BASES:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def gen_primes(count: int, start: int = (1 << 62) - 1) -> list[int]:
    """`count` distinct primes descending from `start` (all > 2^61)."""
    out, cand = [], start | 1
    while len(out) < count:
        if is_prime_u64(cand):
            out.append(cand)
        cand -= 2
        if cand < (1 << 61):  # pragma: no cover
            raise RuntimeError("prime pool exhausted (impossible in practice)")
    return out


def gen_primes_for_bound(bound: int, extra: int = 0,
                         start: int = (1 << 62) - 1) -> tuple[list[int], int]:
    """Primes p_1 > p_2 > ... with prod p_i > 4*bound (a two-bit safety
    margin over the centered-CRT requirement prod > 2*|S|), determined by
    the ACTUAL product rather than a 61-bit-per-prime estimate, plus `extra`
    held-out verification primes.  Returns (primes, n_reconstruction)."""
    target = 4 * bound
    out, prod, cand = [], 1, start | 1
    while prod <= target or len(out) < 1:
        if is_prime_u64(cand):
            out.append(cand)
            prod *= cand
        cand -= 2
        if cand < (1 << 61):  # pragma: no cover
            raise RuntimeError("prime pool exhausted (impossible in practice)")
    n_rec = len(out)
    while len(out) < n_rec + extra:
        if is_prime_u64(cand):
            out.append(cand)
        cand -= 2
    return out, n_rec


# ---------------------------------------------------------------------------
# Pair payloads (exact integers, built once, shipped to prime workers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairPayload:
    j: int
    l: int
    bh: tuple[int, ...]      # Borel-weighted integer coefficients of q_j q_l
    tw: tuple[int, ...]      # t_num_m * rho_n^m * rho_d^(mmaxG-m) * mult_t
    cc: int                  # kappa * cnum_j * cnum_l  (kappa = 2 off-diagonal)
    bh_l1: int               # ||bh||_1  (for the a priori bound)
    tw_l1: int               # sum |tw|  (for the a priori bound)


@dataclass(frozen=True)
class GlobalDims:
    k: int
    smax: int                # max deg (q_j q_l) over surviving pairs
    rmax: int                # (k-1) * smax
    nmax: int                # k * smax
    mmax: int                # max deg of the pair t-polynomials
    top_a: int               # k + nmax
    top_b: int               # k - 2 + rmax
    E: int                   # k - 1 + rmax + mmax
    rho_n: int
    rho_d: int
    lam: int                 # lcm{ k-1 .. k-1+rmax+mmax }
    den_h: int               # common channel-product denominator (2^{2 bits})
    cden2: int               # cden^2
    t_den_glob: int          # lcm of pair t-denominators


def build_payloads(channels: list[ChannelData], c_q, k: int,
                   eps: Fraction, compute_lam: bool = True
                   ) -> tuple[GlobalDims, list[PairPayload]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    m = len(channels)
    dens = {ch.denominator for ch in channels}
    if len(dens) != 1:
        raise ValueError("channels must share one dyadic denominator")
    den = dens.pop()
    den_h = den * den

    rho = Fraction(1 - eps, 1) / Fraction(1 + eps, 1)
    rho_n, rho_d = rho.numerator, rho.denominator

    c_fr = [Fraction(int(v.numerator), int(v.denominator)) for v in c_q]
    cden = math.lcm(*[f.denominator for f in c_fr]) if c_fr else 1
    cnum = [int(f * cden) for f in c_fr]

    # -- pass 1: pair polynomials, global dims -----------------------------
    raw = []
    smax = mmax = 0
    t_dens = []
    for j in range(m):
        for l in range(j, m):
            nj, nl = channels[j].numerators, channels[l].numerators
            h = [0] * (len(nj) + len(nl) - 1)
            for a, va in enumerate(nj):
                if va:
                    for b, vb in enumerate(nl):
                        h[a + b] += va * vb
            while len(h) > 1 and h[-1] == 0:
                h.pop()
            bh = tuple(h[r] * math.factorial(r) for r in range(len(h)))
            tj, tl = channels[j].transformed_numerators, channels[l].transformed_numerators
            t = [0] * (len(tj) + len(tl) - 1)
            for a, va in enumerate(tj):
                if va:
                    for b, vb in enumerate(tl):
                        t[a + b] += va * vb
            while len(t) > 1 and t[-1] == 0:
                t.pop()
            t_den = channels[j].transformed_denominator * channels[l].transformed_denominator
            smax = max(smax, len(bh) - 1)
            mmax = max(mmax, len(t) - 1)
            t_dens.append(t_den)
            raw.append((j, l, bh, tuple(t), t_den))

    rmax = (k - 1) * smax
    nmax = k * smax
    E = k - 1 + rmax + mmax
    # Lam is needed only by the integer-scaled CRT backend.  The ball
    # backend evaluates the normalised ratio directly, so constructing the
    # enormous LCM would be pure overhead.
    lam = 1
    if compute_lam:
        for v in range(k - 1, k - 1 + rmax + mmax + 1):
            lam = math.lcm(lam, v)
    t_den_glob = math.lcm(*t_dens)

    dims = GlobalDims(k=k, smax=smax, rmax=rmax, nmax=nmax, mmax=mmax,
                      top_a=k + nmax, top_b=k - 2 + rmax, E=E,
                      rho_n=rho_n, rho_d=rho_d, lam=lam, den_h=den_h,
                      cden2=cden * cden, t_den_glob=t_den_glob)

    # -- pass 2: payloads with globally scaled B weights -------------------
    rn_pow = [1] * (mmax + 1)
    rd_pow = [1] * (mmax + 1)
    for e in range(1, mmax + 1):
        rn_pow[e] = rn_pow[e - 1] * rho_n
        rd_pow[e] = rd_pow[e - 1] * rho_d
    payloads = []
    for (j, l, bh, t, t_den) in raw:
        mult_t = t_den_glob // t_den
        tw = tuple(t[mm] * rn_pow[mm] * rd_pow[mmax - mm] * mult_t
                   for mm in range(len(t)))
        cc = cnum[j] * cnum[l] * (1 if j == l else 2)
        payloads.append(PairPayload(
            j=j, l=l, bh=bh, tw=tw, cc=cc,
            bh_l1=sum(abs(v) for v in bh),
            tw_l1=sum(abs(v) for v in tw)))
    return dims, payloads


def exact_bounds(dims: GlobalDims, payloads: list[PairPayload]) -> tuple[int, int]:
    """Rigorous |S_A_int| and |S_B_int| bounds (exact integers).

    |[x^n] bh^p| <= ||bh||_1^p (l1 submultiplicativity);
    top_a!/(k+n)!  <= ratio_max_a = prod_{v=1..nmax} (k+v);
    top_b!/(k-2+R)!<= ratio_max_b = prod_{v=1..rmax} (k-2+v);
    rho_n^{k-1+R} rho_d^{rmax-R} <= max(rho_n,rho_d)^{k-1+rmax};
    Lam/(k-1+R+m)  <= Lam.
    """
    k = dims.k
    ratio_a = 1
    for v in range(1, dims.nmax + 1):
        ratio_a *= (k + v)
    ratio_b = 1
    for v in range(1, dims.rmax + 1):
        ratio_b *= (k - 2 + v)
    mpow = max(dims.rho_n, dims.rho_d) ** (k - 1 + dims.rmax)
    bnd_a = bnd_b = 0
    for pl in payloads:
        cc = abs(pl.cc)
        bnd_a += cc * (pl.bh_l1 ** k) * (dims.nmax + 1) * ratio_a
        bnd_b += (cc * (pl.bh_l1 ** (k - 1)) * pl.tw_l1
                  * (dims.rmax + 1) * ratio_b * mpow * dims.lam)
    return bnd_a, bnd_b


# ---------------------------------------------------------------------------
# Per-prime worker
# ---------------------------------------------------------------------------
_G_DIMS: GlobalDims | None = None
_G_PAYLOADS: list[PairPayload] = []


def _crt_init(dims: GlobalDims, payloads: list[PairPayload]) -> None:
    global _G_DIMS, _G_PAYLOADS
    _G_DIMS, _G_PAYLOADS = dims, payloads


def _prime_tables(p: int, d: GlobalDims):
    """Per-prime global weight tables (reused across all pairs).

    WA_rev : nmod_poly, coeff j = top_a!/(k+(nmax-j))!  mod p, so that
             a_num = [x^nmax] ( bh^k * WA_rev ).
    INV    : nmod_poly, coeff v' = (k-1+v')^{-1} mod p, v' = 0..rmax+mmax.
             For each pair the m-sum  S_R = sum_m tw_m/(k-1+R+m)  for ALL R
             is ONE multiply:  S_R = [x^{R+mmax}] ( INV * TW_rev ).
    base   : Python list, base[R] = Lam * (top_b!/(k-2+R)!) *
             rho_n^{k-1+R} rho_d^{rmax-R}  mod p; the final
             b_int = sum_R P_R base[R] S_R is a short interpreter loop.

    (v3.0 built (mmax+1) per-m weight polynomials instead; at d=160 that is
    323 polys x 160k coeffs -- ~400 MB and ~5e7 Python ops PER PRIME.  The
    correlation form is degree-oblivious.)
    """
    k = d.k
    wa = [0] * (d.nmax + 1)
    w = 1
    for jj, n in enumerate(range(d.nmax, -1, -1)):
        wa[jj] = w
        w = w * ((k + n) % p) % p
    WA_rev = nmod_poly(wa, p)

    # batch inverses of v = k-1 .. k-1+rmax+mmax
    nv = d.rmax + d.mmax + 1
    pref = [0] * (nv + 1)
    pref[0] = 1
    for i in range(nv):
        pref[i + 1] = pref[i] * ((k - 1 + i) % p) % p
    inv_all = pow(pref[nv], p - 2, p)
    invv = [0] * nv
    run = inv_all
    for i in range(nv - 1, -1, -1):
        invv[i] = run * pref[i] % p
        run = run * ((k - 1 + i) % p) % p
    INV = nmod_poly(invv, p)

    lam_p = d.lam % p
    rn, rd = d.rho_n % p, d.rho_d % p
    inv_rn = pow(rn, p - 2, p)
    base = [0] * (d.rmax + 1)
    ratio = 1
    rnp = pow(rn, k - 1 + d.rmax, p)
    rdp = 1
    for R in range(d.rmax, -1, -1):
        base[R] = ratio * rnp % p * rdp % p * lam_p % p
        ratio = ratio * ((k - 2 + R) % p) % p
        rnp = rnp * inv_rn % p
        rdp = rdp * rd % p
    return WA_rev, INV, base


def _crt_prime_chunk(primes: list[int]) -> list[tuple[int, int, int]]:
    d, pls = _G_DIMS, _G_PAYLOADS
    out = []
    for p in primes:
        WA_rev, INV, base = _prime_tables(p, d)
        SA = SB = 0
        for pl in pls:
            bh_p = nmod_poly([c % p for c in pl.bh], p)
            P = bh_p ** (d.k - 1)
            Pk = P * bh_p
            a_num = int((Pk * WA_rev)[d.nmax])
            # B-functional: one correlation multiply, then a short combine.
            twrev = [0] * (d.mmax + 1)
            for mm, twm in enumerate(pl.tw):
                r = twm % p
                if r:
                    twrev[d.mmax - mm] = r
            IT = INV * nmod_poly(twrev, p)
            degP = P.degree()
            pc = [int(P[i]) for i in range(degP + 1)]
            itc = [int(IT[d.mmax + i]) for i in range(degP + 1)]
            b_int = 0
            for r in range(degP + 1):
                b_int = (b_int + pc[r] * base[r] % p * itc[r]) % p
            cc = pl.cc % p
            SA = (SA + cc * a_num) % p
            SB = (SB + cc * b_int) % p
        out.append((p, SA, SB))
    return out


def crt_combine(residues: list[tuple[int, int]], primes: list[int],
                bound: int) -> int:
    """Deterministic centered CRT; asserts prod(primes) > 2*bound."""
    x, M = 0, 1
    for r, p in zip(residues, primes):
        inv = pow(M % p, p - 2, p)
        t = (r - x) % p * inv % p
        x += M * t
        M *= p
    if M <= 2 * bound:
        raise RuntimeError("CRT modulus does not exceed 2*bound -- "
                           "prime budget insufficient (bug)")
    if x > M // 2:
        x -= M
    if abs(x) > bound:
        raise RuntimeError("reconstructed integer violates the a priori "
                           "bound -- arithmetic bug")
    return x


class CRTAggregator:
    """Exact S_A_int, S_B_int for a fixed rational c via multimodular sweep."""

    def __init__(self, channels: list[ChannelData], c_q, k: int,
                 eps: Fraction, verify_primes: int = 2) -> None:
        self.eps = Fraction(eps)
        self.dims, self.payloads = build_payloads(channels, c_q, k, eps)
        self.bnd_a, self.bnd_b = exact_bounds(self.dims, self.payloads)
        self.verify_primes = max(0, int(verify_primes))
        bnd = max(self.bnd_a, self.bnd_b)
        self.primes, self.n_primes = gen_primes_for_bound(
            bnd, extra=self.verify_primes)

    def run(self, jobs: int = 1, chunk: int = 4, verbose: bool = True,
            resume_path: str | None = None):
        d = self.dims
        n_rec = self.n_primes
        all_p = self.primes
        t0 = time.time()
        results: dict[int, tuple[int, int]] = {}
        fh = None
        if resume_path:
            import hashlib
            import os

            # Resume fingerprint without decimal string conversion.  At high
            # degree the exact payload contains integers with millions of
            # decimal digits; repr()/str() is both needlessly expensive and,
            # on Python >= 3.11, rejected by the int_max_str_digits safety
            # limit.  Hash the signed binary representation incrementally
            # instead.  This is deterministic and includes every datum that
            # affects the modular residues.
            hfp = hashlib.sha256()

            def _hash_int(v: int) -> None:
                v = int(v)
                neg = v < 0
                mag = -v if neg else v
                raw = mag.to_bytes(max(1, (mag.bit_length() + 7) // 8),
                                   byteorder="big", signed=False)
                hfp.update(b"-" if neg else b"+")
                hfp.update(len(raw).to_bytes(8, "big"))
                hfp.update(raw)

            def _hash_int_seq(seq) -> None:
                hfp.update(len(seq).to_bytes(8, "big"))
                for value in seq:
                    _hash_int(value)

            for name in self.dims.__dataclass_fields__:
                hfp.update(name.encode("ascii"))
                _hash_int(getattr(self.dims, name))
            hfp.update(len(self.payloads).to_bytes(8, "big"))
            for pl in self.payloads:  # build_payloads order is deterministic
                _hash_int(pl.j)
                _hash_int(pl.l)
                _hash_int_seq(pl.bh)
                _hash_int_seq(pl.tw)
                _hash_int(pl.cc)

            fp = hfp.hexdigest()[:32]
            header = f"# maynard-crt-residues {fp}"
            if os.path.exists(resume_path):
                with open(resume_path) as f:
                    first = f.readline().strip()
                    if first != header:
                        raise SystemExit(
                            f"    resume file {resume_path} was written for "
                            f"DIFFERENT payloads (channels/c/epsilon/k); its "
                            f"residues are invalid here. Delete or rename it.")
                    for line in f:
                        parts = line.split()
                        if len(parts) == 3:
                            results[int(parts[0])] = (int(parts[1]),
                                                      int(parts[2]))
                results = {p: v for p, v in results.items() if p in set(all_p)}
                if verbose and results:
                    print(f"    resume: {len(results)}/{len(all_p)} prime "
                          f"residues loaded from {resume_path}")
                fh = open(resume_path, "a", buffering=1)
            else:
                fh = open(resume_path, "a", buffering=1)
                fh.write(header + "\n")
        todo = [p for p in all_p if p not in results]

        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        def _absorb(rows):
            for p, sa, sb in rows:
                results[p] = (sa, sb)
                if fh is not None:
                    fh.write(f"{p} {sa} {sb}\n")
            if verbose and len(results) % max(1, 32) < chunk:
                self._progress(len(results), t0)

        if jobs <= 1:
            _crt_init(d, self.payloads)
            for rows in map(_crt_prime_chunk, chunks):
                _absorb(rows)
        else:
            import multiprocessing as mp
            with mp.Pool(jobs, initializer=_crt_init,
                         initargs=(d, self.payloads)) as pool:
                for rows in pool.imap_unordered(_crt_prime_chunk, chunks):
                    _absorb(rows)
        if fh is not None:
            fh.close()
        if verbose:
            self._progress(len(results), t0, final=True)

        rec_p = all_p[:n_rec]
        S_A = crt_combine([results[p][0] for p in rec_p], rec_p, self.bnd_a)
        S_B = crt_combine([results[p][1] for p in rec_p], rec_p, self.bnd_b)
        for p in all_p[n_rec:]:
            sa, sb = results[p]
            if S_A % p != sa or S_B % p != sb:
                raise RuntimeError(f"held-out prime {p} disagrees with "
                                   "reconstruction -- arithmetic bug")
        return S_A, S_B

    def _progress(self, done: int, t0: float, final: bool = False) -> None:
        n = len(self.primes)
        print(f"    CRT primes {done:6d}/{n} [{time.time()-t0:8.1f}s]"
              + ("  (done)" if final else ""), flush=True)

    def exact_quad_forms(self, S_A: int, S_B: int) -> tuple[Fraction, Fraction]:
        """(c^T B c, c^T A c) as exact Fractions."""
        d = self.dims
        D_A = math.factorial(d.top_a) * d.den_h ** d.k * d.cden2
        D_B = (math.factorial(d.top_b) * d.lam * d.rho_d ** d.E
               * d.t_den_glob * d.den_h ** (d.k - 1) * d.cden2)
        return Fraction(S_B, D_B), Fraction(S_A, D_A)




def project_channels_v3(u_fine: np.ndarray, g_fine: np.ndarray, d: int,
                        bits: int):
    """As karatsuba.project_channels, but also returns the dyadic LEGENDRE
    numerators.  The exact engine consumes the monomial integers (bigints
    cannot cancel); the float engine must NOT -- degree >~ 60 Legendre
    combinations have ~8^d-scale monomial coefficients and np.polyval loses
    every digit to cancellation.  Stable evaluation goes through the
    Legendre recurrence instead (|P~_n| <= 1 on [0,1] at any degree)."""
    m = g_fine.shape[1]
    Phi = eval_shifted_legendre_design(u_fine, d)
    coef, *_ = np.linalg.lstsq(Phi, g_fine, rcond=None)
    leg_int = shifted_legendre_coeffs(d)
    int_coeffs, leg_num, l2_res, linf = [], [], [], []
    for jj in range(m):
        ln = [round(float(coef[i, jj]) * (1 << bits)) for i in range(d + 1)]
        mono = [0] * (d + 1)
        for i in range(d + 1):
            if ln[i]:
                for pdeg, cf in enumerate(leg_int[i]):
                    mono[pdeg] += ln[i] * cf
        int_coeffs.append(mono)
        leg_num.append(ln)
        approx = Phi @ (np.array(ln, dtype=np.float64) / (1 << bits))
        err = approx - g_fine[:, jj]
        scale_l2 = float(np.sqrt(np.mean(g_fine[:, jj] ** 2))) or 1.0
        scale_li = float(np.max(np.abs(g_fine[:, jj]))) or 1.0
        l2_res.append(float(np.sqrt(np.mean(err ** 2))) / scale_l2)
        linf.append(float(np.max(np.abs(err))) / scale_li)
    return int_coeffs, leg_num, l2_res, linf


def _legendre_values_and_antider(leg_num, bits: int, pts: np.ndarray,
                                 pts_anti: np.ndarray):
    """(values q_j(pts), antiderivative values Q_j(pts_anti)) via the stable
    recurrence and the exact identity  int_0^y P~_n = (P~_{n+1} - P~_{n-1})
    / (2(2n+1))  (n >= 1; the parity of P~_n(0) = (-1)^n makes the constant
    vanish), int_0^y P~_0 = y."""
    a = np.array(leg_num, dtype=np.float64).T / (1 << bits)   # (d+1, m)
    d = a.shape[0] - 1
    qv = eval_shifted_legendre_design(pts, d) @ a
    Phi1 = eval_shifted_legendre_design(pts_anti, d + 1)
    anti = np.outer(pts_anti, a[0])
    for n in range(1, d + 1):
        anti = anti + np.outer(
            (Phi1[:, n + 1] - Phi1[:, n - 1]) / (2 * (2 * n + 1)), a[n])
    return qv, anti


# ---------------------------------------------------------------------------
# Float pass: truncated-convolution DP (no exact arithmetic)
# ---------------------------------------------------------------------------
def _tconv(f: np.ndarray, g: np.ndarray, dx: float) -> np.ndarray:
    """[0,1]-truncated trapezoid convolution (exact for the restriction).

    DIRECT convolution is mandatory here.  h^{*j} spans s^{j-1} inside one
    vector, so entries sit up to ~2^{-j} below the renormalized peak.  FFT
    convolution carries an ABSOLUTE error ~1e-16 * peak, which buries those
    entries in noise by j ~ 55 and then compounds through every subsequent
    squaring (the self-test's large-k closed-form check fails by e^{+1000}).
    Direct summation keeps per-entry RELATIVE accuracy: each output is its
    own sum of float64 products, small values stay small, underflowed tails
    are exact zeros, and the squaring integrand w(t)w(s-t) peaks at the
    interior t = s/2 -- representable down to j ~ 511, a float64 ceiling
    near k ~ 1000.
    """
    G = f.shape[0]
    full = np.convolve(f, g)[:G]
    full = full - 0.5 * (f[0] * g[:G] + g[0] * f[:G])
    return full * dx


def _conv_power(h: np.ndarray, e: int, dx: float) -> tuple[np.ndarray, float]:
    """(h^{*e} up to scale, log scale) by renormalized binary powering."""
    def norm(v, ls):
        s = float(np.max(np.abs(v)))
        if s == 0.0 or not np.isfinite(s):
            return v, ls
        return v / s, ls + math.log(s)
    base, lb = norm(h.copy(), 0.0)
    acc, la = None, 0.0
    ee = e
    while ee:
        if ee & 1:
            if acc is None:
                acc, la = base.copy(), lb
            else:
                acc, la = norm(_tconv(acc, base, dx), la + lb)
        ee >>= 1
        if ee:
            base, lb = norm(_tconv(base, base, dx), 2 * lb)
    return acc, la


class FloatPairEngine:
    """Float A, B via A = int_0^1 h^{*k},
    B = rho^{k-1} int_0^1 h_rho^{*(k-1)}(u) T(rho u) du,  h_rho(t) := h(rho t).

    The rescaling of the B-side is essential at large k: h^{*(k-1)}
    concentrates like s^{k-2} near s = 1, so on the raw grid the whole region
    s <= rho sits rho^k-scale below the renormalized peak -- beneath float64
    convolution accuracy once k is in the hundreds (at k = 500, rho = 12/13
    this gap is ~e^{-40}).  After substituting u = s/rho, both integrands
    peak at the top of their own [0,1] grids and are computed at full
    relative accuracy; the exact factor rho^{k-1} is carried in the log
    scale.  eps = 0 (rho = 1) degenerates to the unrescaled path.

    Quadrature bias is O((k/G)^2) and systematic across pairs; it largely
    cancels in Ritz/pruning comparisons, but raise --grid with k.
    """

    def __init__(self, channels: list[ChannelData], k: int, eps_f: float,
                 grid: int = 8192, leg=None) -> None:
        self.ch, self.k = channels, k
        self.rho = (1.0 - eps_f) / (1.0 + eps_f)
        self.G = grid
        self.s = np.linspace(0.0, 1.0, grid)
        self.dx = self.s[1] - self.s[0]
        rs = self.rho * self.s
        if leg is not None:
            # Stable path (mandatory beyond d ~ 60): evaluate from Legendre
            # numerators; monomial coefficients grow like 8^d and np.polyval
            # cancellation destroys every digit, then infs/NaNs cascade
            # through the DP.  T(rho u) = Q_j(1 - rho u) via the exact
            # antiderivative identity.
            leg_num, lbits = leg
            qv, _ = _legendre_values_and_antider(leg_num, lbits, self.s,
                                                 self.s[:1])
            qvr, anti = _legendre_values_and_antider(leg_num, lbits, rs,
                                                    1.0 - rs)
            m = qv.shape[1]
            self.qv = [qv[:, j] for j in range(m)]
            self.qvr = [qvr[:, j] for j in range(m)]
            self.tvr = [anti[:, j] for j in range(m)]
        else:
            self.qv = []       # q_j(s)            (A-side)
            self.qvr = []      # q_j(rho s)        (B-side, rescaled)
            self.tvr = []      # Q_j(1 - rho s)    (B-side weight)
            for c in channels:
                num = np.array(c.numerators, dtype=np.float64)
                self.qv.append(np.polyval(num[::-1], self.s) / c.denominator)
                self.qvr.append(np.polyval(num[::-1], rs) / c.denominator)
                tn = np.array(c.transformed_numerators, dtype=np.float64)
                self.tvr.append(np.polyval(tn[::-1], rs)
                                / c.transformed_denominator)
        bad = [j for j in range(len(self.qv))
               if not (np.all(np.isfinite(self.qv[j]))
                       and np.all(np.isfinite(self.qvr[j]))
                       and np.all(np.isfinite(self.tvr[j])))]
        if bad:
            raise RuntimeError(
                f"non-finite channel evaluations for channels {bad}; at high "
                f"degree this means monomial-basis float evaluation was used "
                f"-- supply Legendre numerators (leg=...)")

    def matrices(self, verbose: bool = True):
        m = len(self.ch)
        logA = np.full((m, m), -np.inf)
        logB = np.full((m, m), -np.inf)
        sgnA = np.zeros((m, m))
        sgnB = np.zeros((m, m))
        s, dx = self.s, self.dx
        log_rho_fac = (self.k - 1) * math.log(self.rho) if self.rho < 1.0 else 0.0
        t0 = time.time()
        n_pairs = m * (m + 1) // 2
        done = 0
        for j in range(m):
            for l in range(j, m):
                # A-side: unrescaled h; h^{*k} = h^{*(k-1)} * h.
                h = self.qv[j] * self.qv[l]
                wk1, lk1 = _conv_power(h, self.k - 1, dx)
                wk = _tconv(wk1, h, dx)
                sk = float(np.max(np.abs(wk))) or 1.0
                wk, lk = wk / sk, lk1 + math.log(sk)
                Ia = _trapz(wk, s)
                # B-side: rescaled h_rho; full-range integral against T(rho u).
                if self.rho < 1.0:
                    hr = self.qvr[j] * self.qvr[l]
                    wr, lr = _conv_power(hr, self.k - 1, dx)
                else:
                    wr, lr = wk1, lk1
                Ib = _trapz(wr * (self.tvr[j] * self.tvr[l]), s)
                lb = lr + log_rho_fac
                for (M, Ssgn, val, lsc) in ((logA, sgnA, Ia, lk),
                                            (logB, sgnB, Ib, lb)):
                    if val != 0.0 and np.isfinite(val):
                        M[j, l] = M[l, j] = math.log(abs(val)) + lsc
                        Ssgn[j, l] = Ssgn[l, j] = math.copysign(1.0, val)
                done += 1
                if verbose and (done % max(1, m) == 0 or done == n_pairs):
                    print(f"    float DP {done:4d}/{n_pairs} "
                          f"[{time.time()-t0:6.1f}s]", flush=True)
        # DIAGONAL PRECONDITIONING (mirrors the discovery npz convention).
        # A single global offset off=max(logA) underflows every entry whose
        # log-scale is < off-745 to exactly 0 in exp(); at large k the A
        # diagonal spans thousands of nats (k=1000: ~6900), so ~all but a
        # couple of channels vanish and the Ritz solve collapses to rank 2.
        # Instead factor A = D^{1/2} A_hat D^{1/2} with
        # D = diag(exp(logA_diag)): subtract (logA_jj + logA_ll)/2 from each
        # A entry, making A_hat's diagonal exactly 1 and every off-diagonal
        # O(1) by Cauchy-Schwarz -- nothing underflows regardless of k.  The
        # SAME congruence D is applied to B so the generalized quotient
        # c^T B c / c^T A c is preserved exactly; logA_diag carries the
        # absolute scale (needed only if an absolute c^T A c is ever wanted,
        # never for R which is a ratio).
        logA_diag = np.array([logA[j, j] for j in range(m)])
        finite = np.isfinite(logA_diag)
        if not np.any(finite):
            return (np.zeros((m, m)), np.zeros((m, m)), 0.0, 0.0,
                    logA_diag)
        half = np.where(finite, logA_diag, 0.0) / 2.0
        logA_hat = logA - half[:, None] - half[None, :]
        logB_hat = logB - half[:, None] - half[None, :]
        # dead channels (logA_jj = -inf): force their rows/cols to -inf so
        # _ritz's diagonal-positivity mask drops them cleanly.
        dead = ~finite
        if np.any(dead):
            logA_hat[dead, :] = logA_hat[:, dead] = -np.inf
            logB_hat[dead, :] = logB_hat[:, dead] = -np.inf
        offA = float(np.nanmax(np.where(np.isfinite(logA_hat), logA_hat,
                                        -np.inf)))
        offB = float(np.nanmax(np.where(np.isfinite(logB_hat), logB_hat,
                                        -np.inf)))
        if not np.isfinite(offA):
            offA = 0.0
        if not np.isfinite(offB):
            offB = 0.0
        A = sgnA * np.exp(logA_hat - offA)
        B = sgnB * np.exp(logB_hat - offB)
        return A, B, offA, offB, logA_diag


class ScaledFloatPairEngine:
    """Float A,B for compactly supported scaled channels at epsilon=0.

    The exact ansatz is q_j(u)=H_j(Lu) for 0<=u<=1/L and zero otherwise.
    H_j is evaluated stably from shifted-Legendre coefficients on [0,1].
    The primitive is

        Q_j(x) = L^{-1} int_0^{min(Lx,1)} H_j(z) dz.

    This float pass therefore represents exactly the same truncated-support
    functions as ScaledCRTAggregator; it is used only for pruning/Ritz.
    """

    def __init__(self, leg_num, bits: int, k: int, support_L: int,
                 grid: int = 8192) -> None:
        if support_L < 1:
            raise ValueError("support_L must be positive")
        self.k = int(k)
        self.L = int(support_L)
        self.G = int(grid)
        self.s = np.linspace(0.0, 1.0, self.G)
        self.dx = self.s[1] - self.s[0]

        leg_num = np.asarray(leg_num, dtype=object)
        if leg_num.ndim != 2:
            raise ValueError("leg_num must have shape (m, degree+1)")
        m, dp1 = leg_num.shape
        d = dp1 - 1
        a = np.asarray(leg_num, dtype=np.float64).T / (1 << bits)

        z = np.clip(self.L * self.s, 0.0, 1.0)
        Phi = eval_shifted_legendre_design(z, d)
        Hv = Phi @ a
        inside = self.s <= (1.0 / self.L)
        Hv[~inside, :] = 0.0

        xarg = 1.0 - self.s
        zanti = np.clip(self.L * xarg, 0.0, 1.0)
        Phi1 = eval_shifted_legendre_design(zanti, d + 1)
        anti = np.outer(zanti, a[0])
        for n in range(1, d + 1):
            anti = anti + np.outer(
                (Phi1[:, n + 1] - Phi1[:, n - 1]) / (2 * (2 * n + 1)),
                a[n])
        anti /= self.L

        self.qv = [Hv[:, j] for j in range(m)]
        self.tvr = [anti[:, j] for j in range(m)]
        bad = [j for j in range(m) if not (
            np.all(np.isfinite(self.qv[j])) and
            np.all(np.isfinite(self.tvr[j])))]
        if bad:
            raise RuntimeError(f"non-finite scaled channel evaluations: {bad}")

    def matrices(self, verbose: bool = True):
        m = len(self.qv)
        logA = np.full((m, m), -np.inf)
        logB = np.full((m, m), -np.inf)
        sgnA = np.zeros((m, m))
        sgnB = np.zeros((m, m))
        t0 = time.time()
        n_pairs = m * (m + 1) // 2
        done = 0
        for j in range(m):
            for l in range(j, m):
                h = self.qv[j] * self.qv[l]
                wk1, lk1 = _conv_power(h, self.k - 1, self.dx)
                wk = _tconv(wk1, h, self.dx)
                sk = float(np.max(np.abs(wk))) or 1.0
                wk, lk = wk / sk, lk1 + math.log(sk)
                Ia = _trapz(wk, self.s)
                Ib = _trapz(wk1 * (self.tvr[j] * self.tvr[l]), self.s)
                for M, Ssgn, val, lsc in ((logA, sgnA, Ia, lk),
                                           (logB, sgnB, Ib, lk1)):
                    if val != 0.0 and np.isfinite(val):
                        M[j, l] = M[l, j] = math.log(abs(val)) + lsc
                        Ssgn[j, l] = Ssgn[l, j] = math.copysign(1.0, val)
                done += 1
                if verbose and (done % max(1, m) == 0 or done == n_pairs):
                    print(f"    scaled float DP {done:4d}/{n_pairs} "
                          f"[{time.time()-t0:6.1f}s]", flush=True)

        logA_diag = np.array([logA[j, j] for j in range(m)])
        finite = np.isfinite(logA_diag)
        if not np.any(finite):
            return (np.zeros((m, m)), np.zeros((m, m)), 0.0, 0.0,
                    logA_diag)
        half = np.where(finite, logA_diag, 0.0) / 2.0
        logA_hat = logA - half[:, None] - half[None, :]
        logB_hat = logB - half[:, None] - half[None, :]
        dead = ~finite
        if np.any(dead):
            logA_hat[dead, :] = logA_hat[:, dead] = -np.inf
            logB_hat[dead, :] = logB_hat[:, dead] = -np.inf
        offA = float(np.nanmax(np.where(np.isfinite(logA_hat), logA_hat,
                                        -np.inf)))
        offB = float(np.nanmax(np.where(np.isfinite(logB_hat), logB_hat,
                                        -np.inf)))
        if not np.isfinite(offA):
            offA = 0.0
        if not np.isfinite(offB):
            offB = 0.0
        A = sgnA * np.exp(logA_hat - offA)
        B = sgnB * np.exp(logB_hat - offB)
        return A, B, offA, offB, logA_diag


# ---------------------------------------------------------------------------
# Ritz + greedy c-weighted channel pruning (float only; steers, never certifies)
# ---------------------------------------------------------------------------
def _ritz(A: np.ndarray, B: np.ndarray,
          rank_tol: float = 1e-12) -> tuple[float, np.ndarray]:
    """Max generalized eigenpair of (B, A) for numerically SEMIdefinite A.

    The float Gram A is PSD in exact arithmetic but at large k its entries
    span enough orders of magnitude that Cholesky-based generalized solvers
    (scipy.eigh(B, A)) fail on indefinite rounding.  Instead: equilibrate by
    the diagonal, eigendecompose A, restrict to its numerical range
    (eigenvalues > rank_tol * max), and solve the standard symmetric problem
    in whitened coordinates.  Restricting to range(A) is the correct
    regularization: null-space components contribute ~0 to c^T A c and would
    only inflate the quotient spuriously.
    """
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
    W = V[:, keep] / np.sqrt(w[keep])[None, :]      # whitener onto range(A)
    lams, U = np.linalg.eigh(W.T @ Bs @ W)
    u = U[:, -1]
    c = np.zeros(m)
    c[ia] = s * (W @ u)
    return float(lams[-1]), c


def greedy_prune(A: np.ndarray, B: np.ndarray, tol: float,
                 min_channels: int = 1, verbose: bool = True,
                 rank_tol: float = 1e-10):
    """Backward elimination under a cumulative relative budget on lambda."""
    m = A.shape[0]
    active = list(range(m))
    lam0, _ = _ritz(A, B, rank_tol)
    if tol <= 0.0 or lam0 <= 0.0:
        return active, _ritz(A, B, rank_tol)[1]
    while len(active) > max(1, min_channels):
        best_lam, best_i = -np.inf, None
        for i in range(len(active)):
            sub = active[:i] + active[i + 1:]
            lam_i, _ = _ritz(A[np.ix_(sub, sub)], B[np.ix_(sub, sub)],
                             rank_tol)
            if lam_i > best_lam:
                best_lam, best_i = lam_i, i
        if (lam0 - best_lam) / lam0 <= tol:
            dropped = active.pop(best_i)
            if verbose:
                print(f"    prune: drop channel {dropped:3d}  "
                      f"(rel. lambda loss {max(0.0,(lam0-best_lam)/lam0):.3e},"
                      f" {len(active)} remain)")
        else:
            break
    lam_f, c = _ritz(A[np.ix_(active, active)], B[np.ix_(active, active)],
                     rank_tol)
    if verbose:
        kept = len(active)
        print(f"    prune: kept {kept}/{m} channels  "
              f"(pairs {m*(m+1)//2} -> {kept*(kept+1)//2}), "
              f"float rel. lambda loss {max(0.0,(lam0-lam_f)/lam0):.3e}")
    return active, c


# ---------------------------------------------------------------------------
# Certification driver
# ---------------------------------------------------------------------------

def _scale_vector_by_logdiag_safe(values: np.ndarray, log_diag: np.ndarray,
                                  exponent: float,
                                  label: str = "vector") -> np.ndarray:
    """Apply values_j * exp(exponent * log_diag_j) without overflow.

    The result is normalized by one common positive scalar so max(abs(out))=1.
    Such a global rescaling leaves every Rayleigh quotient unchanged.  Zeros
    remain exact zeros; non-finite inputs are rejected rather than converted
    silently to NaN/inf.
    """
    v = np.asarray(values, dtype=np.float64)
    ld = np.asarray(log_diag, dtype=np.float64)
    if v.shape != ld.shape:
        raise ValueError(f"{label}: value/log-diagonal shape mismatch "
                         f"{v.shape} != {ld.shape}")
    if np.any(~np.isfinite(ld)):
        bad = np.where(~np.isfinite(ld))[0].tolist()
        raise ValueError(f"{label}: non-finite log diagonal at channels {bad}")
    if np.any(~np.isfinite(v)):
        bad = np.where(~np.isfinite(v))[0].tolist()
        raise ValueError(f"{label}: non-finite coefficients at channels {bad}")

    out = np.zeros_like(v)
    nz = v != 0.0
    if not np.any(nz):
        raise ValueError(f"{label}: all coefficients are zero")
    logabs = np.log(np.abs(v[nz])) + exponent * ld[nz]
    shift = float(np.max(logabs))
    out[nz] = np.sign(v[nz]) * np.exp(logabs - shift)
    if not np.all(np.isfinite(out)) or np.max(np.abs(out)) == 0.0:
        raise FloatingPointError(f"{label}: scale-safe frame conversion failed")
    return out



def _dyadic_balance_projected_channels(int_coeffs, active, logA_diag, k: int,
                                        base_bits: int, scale_bits: int = 48):
    """Return exactly dyadically rescaled active polynomial channels.

    FloatPairEngine works in the diagonally preconditioned basis
        Fhat_j = exp(-logA_jj/2) F_j.
    For a separable channel F_j = q_j tensor-power k, this basis change can be
    moved to the one-dimensional polynomial:
        qhat_j = exp(-logA_jj/(2k)) q_j.

    We approximate each positive root scale by an exact dyadic integer
    a_j / 2**scale_bits.  A COMMON logarithmic offset is removed before
    quantisation; it multiplies every k-dimensional basis vector by the same
    scalar and therefore has no effect on a Rayleigh quotient.  The resulting
    channels all share denominator 2**(base_bits + scale_bits), so the existing
    exact CRT backend is unchanged.
    """
    active = list(active)
    ld = np.asarray(logA_diag, dtype=np.float64)[np.asarray(active, dtype=int)]
    if np.any(~np.isfinite(ld)):
        bad = np.where(~np.isfinite(ld))[0].tolist()
        raise ValueError(f"basis balance: non-finite logA diagonal at active positions {bad}")
    root_logs = -0.5 * ld / float(k)
    root_logs -= float(np.max(root_logs))   # all scales in (0,1], common factor only
    scales = np.exp(root_logs)
    unit = 1 << scale_bits
    nums = np.rint(scales * unit).astype(object)
    nums = [max(1, int(v)) for v in nums]
    scaled = []
    for src_idx, a in zip(active, nums):
        scaled.append([int(v) * a for v in int_coeffs[src_idx]])
    rel = [abs(a / unit - sc) / sc for a, sc in zip(nums, scales)]
    print(f"  exact basis balance: dyadic root scales in "
          f"[{min(scales):.3e}, {max(scales):.3e}], "
          f"max rel. quantisation {max(rel):.2e}")
    return scaled, base_bits + scale_bits


def certify_epsilon_crt(npz_path: str, degrees: list[int], bits: int,
                        epsilon: Fraction | None, digits: int = 30,
                        prune_tol: float = 1e-7, prune_min: int = 1,
                        prune_cmag: float = 0.0, no_prune: bool = False,
                        use_nn_c: bool = False, grid: int = 8192,
                        force_float: bool = False, float_only: bool = False,
                        rank_tol_cli: float = 0.0,
                        emit_certificate: str | None = None,
                        jobs: int = 1, chunk: int = 4,
                        verify_primes: int = 2,
                        resume: str | None = None,
                        backend: str = "crt",
                        ball_prec: int = 128,
                        ball_target_rel: float = 1e-12,
                        ball_max_prec: int = 4096,
                        support_L: int = 1):
    data = np.load(npz_path, mmap_mode="r")
    frame = load_discovery_frame(data)
    k = frame.k
    R_nn = frame.rayleigh
    R_nn_ref = R_nn
    c_nn, x_fine, g_fine = frame.coefficients, frame.points, frame.channels
    if frame.channel_norms is not None:
        cnorms = frame.channel_norms
        rms_after = np.sqrt(np.mean(g_fine * g_fine, axis=0))
        peak_after = np.max(np.abs(g_fine), axis=0)
        print(f" channel shape normalisation: raw norm range "
              f"[{cnorms.min():.3e}, {cnorms.max():.3e}]")
        print(f" normalised sampled RMS range [{rms_after.min():.3e}, "
              f"{rms_after.max():.3e}], peak range "
              f"[{peak_after.min():.3e}, {peak_after.max():.3e}]")
        tiny = np.where(cnorms < 1e-3 * np.median(cnorms))[0]
        if tiny.size:
            print(" strongly collapsed raw channels restored by shape "
                  f"normalisation: {tiny.tolist()}")
    else:
        print(" WARNING: npz has no channel_norms; assuming g_fine is already "
              "in the discovery channel frame")

    if frame.logA_diag is not None:
        print(" discovery c frame conversion: scale-safe log-space mapping "
              "(max |c| normalised to 1)")
    eps_npz = float(data["epsilon"]) if "epsilon" in data else 0.0
    if epsilon is None:
        if eps_npz == 0.0:
            epsilon = Fraction(0)
        else:
            raise SystemExit("npz has eps > 0: supply exact --epsilon, e.g. 1/25")
    if abs(float(epsilon) - eps_npz) > 1e-9:
        print(f" WARNING: exact epsilon {epsilon} differs from npz float "
              f"{eps_npz:.12g}; certificate remains valid for {epsilon}.")
    if support_L < 1:
        raise SystemExit("--support-L must be a positive integer")
    scaled_mode = support_L > 1
    if scaled_mode and epsilon != 0:
        raise SystemExit("scaled compact-support certification currently "
                         "requires --epsilon 0")
    if scaled_mode and backend != "crt":
        raise SystemExit("scaled compact-support certification currently "
                         "supports --backend crt only")
    if scaled_mode and emit_certificate:
        raise SystemExit("--emit-certificate is not yet available for the "
                         "scaled-support format; the ordinary certificate "
                         "writer cannot encode support_L")
    L_frac = Fraction(1) + epsilon
    m = g_fine.shape[1]
    print("=" * 76)
    print(f" CRT (v3) certification M_{{{k},eps,1/2}}, eps={epsilon}")
    print(f" m={m}  bits={bits}  discovery R_NN={R_nn:.12f}  jobs={jobs}")
    if scaled_mode:
        print(f" compact support: sigma=1/{support_L} "
              f"(epsilon=0 scaled CRT path)")
    if backend == "ball":
        print(" rigorous backend: Arb ball arithmetic (adaptive precision)")
    else:
        print(" exact backend: multimodular, deterministic centered CRT")
    print("=" * 76)

    if prune_cmag > 0.0:
        keep = np.abs(c_nn) > prune_cmag * np.max(np.abs(c_nn))
        n_dead = int(m - keep.sum())
        if n_dead:
            print(f" |c|-magnitude pre-drop: {n_dead}/{m} channels")
            g_fine = g_fine[:, keep]
            c_nn = c_nn[keep]
            m = int(keep.sum())

    u_fine = np.clip(x_fine / (1.0 + eps_npz), 0.0, 1.0)
    summaries, best = [], None
    for d in degrees:
        print(f"\n--- degree d = {d} ---")
        if scaled_mode:
            int_coeffs, leg_num, l2_res, linf, kept_mass = (
                project_scaled_channels(
                    u_fine, g_fine, degree=d, bits=bits,
                    support_L=support_L))
            print(f"  scaled projection support    : [0, 1/{support_L}]")
            print(f"  retained channel L2 mass     : min {min(kept_mass):.6e}   "
                  f"median {sorted(kept_mass)[m//2]:.6e}")
        else:
            int_coeffs, leg_num, l2_res, linf = project_channels_v3(
                u_fine, g_fine, d, bits)
        print(f"  projection rel. L2 residuals : max {max(l2_res):.2e}   "
              f"median {sorted(l2_res)[m//2]:.2e}")
        print(f"  projection rel. Linf errors  : max {max(linf):.2e}")
        channels = preprocess_channels(int_coeffs, bits)

        if float_only and use_nn_c:
            raise SystemExit("--float-only surveys the float pass; it is "
                             "incompatible with --use-nn-c")
        if use_nn_c and no_prune:
            active = list(range(m))
            c_float = c_nn / max(np.max(np.abs(c_nn)), 1e-300)
            c_source = "NN mixing (float pass skipped)"
        else:
            print(f"  float pass: truncated-convolution DP on "
                  f"{m*(m+1)//2} pairs (grid {grid}) ...")
            if scaled_mode:
                fe = ScaledFloatPairEngine(leg_num, bits, k, support_L,
                                           grid=grid)
            else:
                fe = FloatPairEngine(channels, k, eps_npz, grid=grid,
                                     leg=(leg_num, bits))
            A_f, B_f, offA, offB, logA_diag_f = fe.matrices()
            # The whitened Ritz keeps eigen-directions of A above
            # rank_tol * lambda_max, but at higher channel degrees the Gram
            # entries are cancellation-limited to ~1e-8 relative accuracy:
            # eigenvalues above a 1e-12 cut yet below the accuracy floor are
            # noise, and whitening by 1/sqrt(noise) produces absurd lambda
            # (the true pencil is bounded by M_{k,eps} -- single digits).
            # Escalate the truncation until the predicted R lands on a
            # physical scale, using R_NN as the anchor.
            R_cap = max(5.0 * abs(R_nn_ref), 100.0)
            tol_ladder = ([rank_tol_cli] if rank_tol_cli > 0
                          else [1e-10, 1e-8, 1e-6, 1e-4])
            R_float = math.inf
            for rt in tol_ladder:
                if no_prune:
                    active = list(range(m))
                    _, c_float = _ritz(A_f, B_f, rt)
                    c_source = "re-Ritz on float DP Gram (no pruning)"
                else:
                    active, c_float = greedy_prune(A_f, B_f, prune_tol,
                                                   min_channels=prune_min,
                                                   rank_tol=rt)
                    c_source = (f"greedy c-weighted pruning + re-Ritz "
                                f"({len(active)}/{m} channels)")
                lam_f, _ = _ritz(A_f[np.ix_(active, active)],
                                 B_f[np.ix_(active, active)], rt)
                R_float = k * float(L_frac) * lam_f * math.exp(offB - offA)
                if np.isfinite(R_float) and 0.0 < R_float <= R_cap:
                    if rt != tol_ladder[0]:
                        print(f"  (rank truncation escalated to {rt:.0e}: "
                              f"smaller tolerances admitted noise "
                              f"eigen-directions of the float Gram)")
                    break
                print(f"  rank truncation {rt:.0e}: unphysical float R "
                      f"{R_float:.3e} (noise direction); escalating")
            print(f"  float-predicted R ~= {R_float:.12f}")
            rel = abs(R_float - R_nn_ref) / max(abs(R_nn_ref), 1e-300)
            if float_only:
                print(f"  [float-only] d={d}: Linf {max(linf):.2e}, "
                      f"float R {R_float:.9g}, deviation from R_NN {rel:.1%}")
                summaries.append((d, R_float, R_nn - R_float))
                continue
            if scaled_mode:
                print(f"  scaled-support loss versus discovery R_NN: {rel:.1%} "
                      f"(diagnostic only; different ansatz)")
            else:
                if rel > 0.5 and not force_float:
                    raise SystemExit(
                        f"  ABORT: float-predicted R deviates from discovery R_NN "
                        f"by {rel:.1%}.\n"
                        f"  The float pass is unreliable at this configuration; a "
                        f"c frozen from it\n"
                        f"  would certify a valid but worthless bound.  Try a "
                        f"larger --grid, or\n"
                        f"  --use-nn-c --no-prune to bypass the float pass, or "
                        f"--force-float to override.")
                if rel > 0.05:
                    print(f"  WARNING: float R deviates from R_NN by {rel:.1%}; "
                          f"pruning/Ritz decisions may be degraded "
                          f"(consider a larger --grid)")

        c_float = np.asarray(c_float, dtype=np.float64)
        if not (use_nn_c and no_prune):
            # Do NOT map the Ritz vector into the true frame.  At large k that
            # vector can span thousands of nats; any float representation then
            # underflows small but essential components to exact zero.  Move
            # the diagonal basis change into the 1-D polynomial channels
            # instead: q_j -> A_jj^{-1/(2k)} q_j, approximated by an exact
            # dyadic scale.  The well-conditioned preconditioned-frame Ritz
            # vector can then be frozen directly without information loss.
            int_act, exact_bits = _dyadic_balance_projected_channels(
                int_coeffs, active, logA_diag_f, k, bits, scale_bits=48)
            ch_act = preprocess_channels(int_act, exact_bits)
            c_float = c_float / max(np.max(np.abs(c_float)), 1e-300)
        else:
            # NN-c path already carries a true-frame vector and uses the
            # unbalanced projected channels.
            int_act = [int_coeffs[i] for i in active]
            exact_bits = bits
            ch_act = [channels[i] for i in active]
            c_float = c_float / max(np.max(np.abs(c_float)), 1e-300)
        # Exact dyadic embedding: Fraction(float) preserves the full 53-bit
        # mantissa PER COMPONENT.  Uniform absolute rounding (dyadic(v,bits))
        # is fatal at large k: the whitened Ritz c_j ~ A_jj^{-1/2} spans
        # ~(sup ratio)^k orders of magnitude, and killing the small
        # components collapses the balanced combination (k=500 post-mortem:
        # float R 4.19 -> certified 9e-5).
        c_q = [Fraction(float(v)) for v in c_float]
        if not (use_nn_c and no_prune):
            # c_float remains in exactly the same preconditioned frame as A_f
            # and B_f.  Fraction(float) is exact for the binary64 value, so the
            # frozen check should agree to roundoff; no unstable round trip
            # through the true frame is performed.
            ch_hat = np.array([float(f) for f in c_q])
            Af_s = A_f[np.ix_(active, active)]
            Bf_s = B_f[np.ix_(active, active)]
            qa = float(ch_hat @ Af_s @ ch_hat)
            qb = float(ch_hat @ Bf_s @ ch_hat)
            if qa <= 0:
                raise SystemExit("  ABORT: frozen c gives non-positive "
                                 "float c^T A c; rationalization pathology")
            R_frozen = k * float(L_frac) * (qb / qa) * math.exp(offB - offA)
            drift = abs(R_frozen - R_float) / max(abs(R_float), 1e-300)
            print(f"  frozen-c float check: R = {R_frozen:.12f} "
                  f"(drift {drift:.2e})")
            if drift > 0.01:
                raise SystemExit(
                    "  ABORT: freezing c moved the float prediction by "
                    f"{drift:.1%}; the rationalized c does not represent "
                    "the Ritz optimum. Not spending the prime budget.")
            del A_f, B_f

        if backend == "ball":
            # Build exactly the same integer payload polynomials as CRT, but
            # skip Lam: Arb evaluates the normalised ratio directly and all
            # giant integer scaling constants cancel.
            from maynard_ball_engine import certify_ball_adaptive
            gd, payloads = build_payloads(ch_act, c_q, k, epsilon,
                                          compute_lam=False)
            print(f"  ball pass: {len(payloads)} pairs, "
                  f"deg bh^(k-1) = {gd.rmax}, "
                  f"initial precision = {ball_prec} bits, "
                  f"target rel. width = {ball_target_rel:.1e}")
            res = certify_ball_adaptive(
                gd, payloads, prec0=ball_prec,
                target_rel=ball_target_rel, max_prec=ball_max_prec,
                verbose=True)
            if not res.contracted and res.rel_width > ball_target_rel:
                print("  WARNING: Arb enclosure did not contract to the "
                      "requested width; no certificate emitted for this degree.")
                continue
            R_exact = res.R_lower   # exact dyadic lower endpoint
            dec = frac_to_decimal_truncated(R_exact, digits)
            gap = R_nn - float(R_exact)
            hi = frac_to_decimal_truncated(res.R_upper, digits)
            print(f"  c from: {c_source}")
            print(f"  rigorous Arb enclosure: [{dec}, {hi}]")
            print(f"  CERTIFIED M_{{{k},{epsilon},1/2}} >= {dec}")
            print(f"  R_NN - R_certified = {gap:+.3e}")
        else:
            if scaled_mode:
                agg = ScaledCRTAggregator(
                    H_num=int_act, den=(1 << exact_bits), c_q=c_q, k=k,
                    support_L=support_L, verify_primes=verify_primes)
            else:
                agg = CRTAggregator(ch_act, c_q, k, epsilon,
                                    verify_primes=verify_primes)
            gd = agg.dims
            print(f"  exact pass: {len(agg.payloads)} pairs, "
                  f"deg bh^(k-1) = {gd.rmax}, "
                  f"bound bits (A,B) = ({agg.bnd_a.bit_length()}, "
                  f"{agg.bnd_b.bit_length()}), primes = {agg.n_primes}"
                  f" (+{verify_primes} verify)")
            suffix = f".d{d}.L{support_L}" if scaled_mode else f".d{d}"
            rp = (f"{resume}{suffix}" if resume else None)
            S_A, S_B = agg.run(jobs=jobs, chunk=chunk, resume_path=rp)
            num, den = agg.exact_quad_forms(S_A, S_B)
            if den <= 0:
                print("  WARNING: c^T A c <= 0; skipping")
                continue
            R_exact = Fraction(k) * L_frac * num / den
            dec = frac_to_decimal_truncated(R_exact, digits)
            gap = R_nn - float(R_exact)
            print(f"  c from: {c_source}")
            print(f"  CERTIFIED M_{{{k},{epsilon},1/2}} >= {dec}")
            print(f"  R_NN - R_certified = {gap:+.3e}")

        if float(R_exact) > 4.0:
            print(f"  *** exceeds 4: sufficient for DHL[{k},2] at theta=1/2 ***")
        summaries.append((d, R_exact, gap))
        if best is None or R_exact > best["R_exact"]:
            best = {"d": d, "R_exact": R_exact, "gap": gap,
                    "int_coeffs": int_act, "c_q": c_q,
                    "bits": exact_bits}

    if emit_certificate and best is not None:
        qty = f"M_{{{k},{epsilon},1/2}}" if epsilon else f"M_{k}"
        write_certificate(emit_certificate, qty, k, epsilon,
                          best["int_coeffs"], best.get("bits", bits), best["c_q"],
                          best["R_exact"], digits=digits)
    if len(summaries) >= 2:
        tag = "float-predicted R" if float_only else "R_certified"
        print("\n--- degree sweep summary ---")
        print(f"  {'d':>4} {tag:>34} {'R_NN - R':>15}")
        for d, rr, gap in summaries:
            txt = (f"{rr:.12g}" if float_only
                   else frac_to_decimal_truncated(rr, 24))
            print(f"  {d:>4} {txt:>34} {gap:>+15.3e}")
    return summaries


# ---------------------------------------------------------------------------
# Self tests
# ---------------------------------------------------------------------------
def _crt_reference(channels, c_q, k, eps):
    """(c^T B c, c^T A c) through the trusted v2 exact pair worker."""
    from .flint_streaming import (
        _init_worker, _pair_worker)
    rho = Fraction(1 - eps, 1) / Fraction(1 + eps, 1)
    _init_worker(k, rho.numerator, rho.denominator)
    cf = [Fraction(int(v.numerator), int(v.denominator)) for v in c_q]
    num = den = Fraction(0)
    for j in range(len(channels)):
        for l in range(j, len(channels)):
            _, _, an, ad, bn, bd = _pair_worker((j, l, channels[j], channels[l]))
            w = cf[j] * cf[l] * (1 if j == l else 2)
            den += w * Fraction(an, ad)
            num += w * Fraction(bn, bd)
    return num, den


def self_test() -> None:
    print("=== self test: CRT (v3) epsilon certifier ===")
    ok = True

    # 0. primality + CRT round trip
    good = all(is_prime_u64(v) for v in (2, 3, 61, (1 << 61) - 1)) and \
        not any(is_prime_u64(v) for v in (1, 341, 561, (1 << 62) - 1))
    ps = gen_primes(5)
    try:
        from flint import fmpz
        good &= all(bool(fmpz(p).is_prime()) for p in ps)
        print("  generated primes cross-checked against FLINT is_prime: True")
    except (ImportError, AttributeError):
        pass
    import random
    rng = random.Random(0)
    for _ in range(20):
        s = rng.randrange(-10**80, 10**80)
        rec = crt_combine([s % p for p in ps], ps, abs(s) + 1)
        good &= (rec == s)
    ok &= good
    print(f"  Miller-Rabin + centered CRT round trip: {good}")

    # 1. g == 1 closed forms through the full CRT machinery
    for k, eps in ((2, Fraction(1, 4)), (5, Fraction(3, 10)),
                   (10, Fraction(1, 25)), (12, Fraction(0))):
        bits = 8
        channels = preprocess_channels([[1 << bits]], bits)
        c_q = [Fraction(1)]
        agg = CRTAggregator(channels, c_q, k, eps, verify_primes=1)
        S_A, S_B = agg.run(jobs=1, verbose=False)
        num, den = agg.exact_quad_forms(S_A, S_B)
        rho = Fraction(1 - eps, 1) / Fraction(1 + eps, 1)
        A_cf = Fraction(1, math.factorial(k))
        B_cf = (rho**(k-1)/(k-1) - 2*rho**k/k + rho**(k+1)/(k+1)) \
            / math.factorial(k - 2)
        good = (den == A_cf and num == B_cf)
        ok &= good
        print(f"  k={k:3d} eps={str(eps):>5}: closed forms exact via CRT: {good}")

    # 2. mixed-degree random channels: CRT aggregate == v2 exact worker
    rng = np.random.default_rng(0)
    bits = 12
    coeffs = [[int(v) for v in rng.integers(-(1 << bits), 1 << bits, size=6)],
              [int(v) for v in rng.integers(-(1 << bits), 1 << bits, size=4)],
              [int(v) for v in rng.integers(-(1 << bits), 1 << bits, size=6)]]
    coeffs[1] += [0, 0]                       # exercise trimming / mixed degree
    k, eps = 7, Fraction(1, 25)
    channels = preprocess_channels(coeffs, bits)
    c_q = [dyadic(v, bits) for v in (1.0, -0.25, 0.5)]
    agg = CRTAggregator(channels, c_q, k, eps, verify_primes=1)
    S_A, S_B = agg.run(jobs=1, verbose=False)
    num, den = agg.exact_quad_forms(S_A, S_B)
    num_ref, den_ref = _crt_reference(channels, c_q, k, eps)
    good = (num == num_ref and den == den_ref)
    ok &= good
    print(f"  mixed-degree m=3, k=7: CRT aggregate == v2 exact worker: {good}")
    good = den > 0 and abs(S_A) <= agg.bnd_a and abs(S_B) <= agg.bnd_b
    ok &= good
    print(f"  a priori bounds dominate reconstructed integers:        {good}")

    # 2b. strong cancellation: channels q and q + tiny perturbation with
    #     c = (1, -1).  Every per-prime residue is full-size and per-pair
    #     contributions carry both signs, but the reconstructed aggregate is
    #     orders of magnitude below the a priori bound -- exercising the
    #     centered CRT on small signed integers and the modular sign
    #     handling, verified against the v2 exact worker.
    pert = [c + (1 if i == 0 else 0) for i, c in enumerate(coeffs[0])]
    ch_c = preprocess_channels([coeffs[0], pert], bits)
    cq_c = [dyadic(1.0, bits), dyadic(-1.0, bits)]
    agg_c = CRTAggregator(ch_c, cq_c, k, eps, verify_primes=1)
    S_Ac, S_Bc = agg_c.run(jobs=1, verbose=False)
    num_c, den_c = agg_c.exact_quad_forms(S_Ac, S_Bc)
    numr_c, denr_c = _crt_reference(ch_c, cq_c, k, eps)
    slack_a = agg_c.bnd_a.bit_length() - max(1, abs(S_Ac)).bit_length()
    good = (num_c == numr_c and den_c == denr_c and den_c > 0
            and slack_a > 30)
    ok &= good
    print(f"  cancellation (c=(1,-1), near-equal channels): exact match, "
          f"|S_A| {slack_a} bits under bound: {good}")

    # 3. float DP versus exact values
    fe = FloatPairEngine(channels, k, float(eps), grid=4096)
    A_f, B_f, offA, offB, logA_diag = fe.matrices(verbose=False)
    from .flint_streaming import (
        _init_worker, _pair_worker)
    rho = Fraction(1 - eps, 1) / Fraction(1 + eps, 1)
    _init_worker(k, rho.numerator, rho.denominator)
    half = logA_diag / 2.0            # undo per-channel diagonal precond.
    max_rel = 0.0
    for j in range(3):
        for l in range(j, 3):
            _, _, an, ad, bn, bd = _pair_worker((j, l, channels[j], channels[l]))
            a_ex, b_ex = an / ad, bn / bd
            hh = half[j] + half[l]
            a_fl = (A_f[j, l] * math.exp(offA + hh) if A_f[j, l] != 0 else 0.0)
            b_fl = (B_f[j, l] * math.exp(offB + hh) if B_f[j, l] != 0 else 0.0)
            for ex, fl in ((a_ex, a_fl), (b_ex, b_fl)):
                if ex != 0:
                    max_rel = max(max_rel, abs(fl - ex) / abs(ex))
    good = max_rel < 5e-4
    ok &= good
    print(f"  float DP vs exact pairs: max rel err {max_rel:.2e} < 5e-4: {good}")

    # 3b. stable Legendre evaluation == monomial evaluation at low degree,
    #     and remains finite/physical at d = 160 where monomial-basis
    #     np.polyval loses ~150 digits to cancellation (the k=500 NaN
    #     cascade).  Projection of smooth synthetic channels.
    Pg = 1500
    ug = np.linspace(0.0, 1.0, Pg)
    gsyn = np.stack([np.exp(-3 * ug), 1.0 / (1.0 + 5 * ug),
                     np.cos(2.0 * ug)], axis=1)
    ic_lo, ln_lo, _, _ = project_channels_v3(ug, gsyn, 8, 24)
    ch_lo = preprocess_channels(ic_lo, 24)
    fe_m = FloatPairEngine(ch_lo, 9, float(Fraction(1, 25)), grid=2048)
    fe_s = FloatPairEngine(ch_lo, 9, float(Fraction(1, 25)), grid=2048,
                           leg=(ln_lo, 24))
    Am, Bm, oAm, oBm, _ = fe_m.matrices(verbose=False)
    As2, Bs2, oAs, oBs, _ = fe_s.matrices(verbose=False)
    dev = max(float(np.max(np.abs(Am * math.exp(oAm) - As2 * math.exp(oAs)))
                    / np.max(np.abs(Am * math.exp(oAm)))),
              float(np.max(np.abs(Bm * math.exp(oBm) - Bs2 * math.exp(oBs)))
                    / np.max(np.abs(Bm * math.exp(oBm)))))
    good = dev < 1e-9
    ok &= good
    print(f"  stable Legendre eval == monomial eval (d=8): "
          f"max rel dev {dev:.2e}: {good}")
    ic_hi, ln_hi, _, linf_hi = project_channels_v3(ug, gsyn, 160, 48)
    ch_hi = preprocess_channels(ic_hi, 48)
    fe_hi = FloatPairEngine(ch_hi, 40, 0.04, grid=2048, leg=(ln_hi, 48))
    A_hi, B_hi, oA_hi, oB_hi, _ = fe_hi.matrices(verbose=False)
    lam_hi, _ = _ritz(A_hi, B_hi, 1e-8)
    good = (np.all(np.isfinite(A_hi)) and np.all(np.isfinite(B_hi))
            and np.isfinite(lam_hi) and max(linf_hi) < 1e-9)
    ok &= good
    print(f"  d=160 stable path finite through DP+Ritz "
          f"(proj Linf {max(linf_hi):.1e}):    {good}")

    # 3c. END-TO-END BRIDGE at d=160 (k=7): float Gram quad forms with the
    #     exactly-rationalized frozen c versus the CRT exact value.  This is
    #     the link that failed at k=500: uniform-absolute dyadic rounding of
    #     a wide-dynamic-range Ritz c.  Fraction(float) embedding must keep
    #     per-component relative precision.
    ch160 = preprocess_channels(ic_hi[:2], 48)
    fe160 = FloatPairEngine(ch160, 7, float(Fraction(1, 25)), grid=4096,
                            leg=(ln_hi[:2], 48))
    A160, B160, oA160, oB160, _ = fe160.matrices(verbose=False)
    _, c160 = _ritz(A160, B160, 1e-10)
    c160 = c160 / np.max(np.abs(c160))
    cq160 = [Fraction(float(v)) for v in c160]
    good = all(float(cq160[i]) == float(c160[i]) for i in range(2))
    wide = Fraction(float(1e-40))
    good &= (float(wide) == 1e-40)
    agg160 = CRTAggregator(ch160, cq160, 7, Fraction(1, 25), verify_primes=1)
    S_A160, S_B160 = agg160.run(jobs=1, verbose=False)
    num160, den160 = agg160.exact_quad_forms(S_A160, S_B160)
    R_ex = float(Fraction(7) * (1 + Fraction(1, 25)) * num160 / den160)
    qa = float(c160 @ A160 @ c160) * math.exp(oA160)
    qb = float(c160 @ B160 @ c160) * math.exp(oB160)
    R_fl = 7 * (1 + 1 / 25) * qb / qa
    rel160 = abs(R_fl - R_ex) / abs(R_ex)
    good &= rel160 < 1e-3
    ok &= good
    print(f"  d=160 bridge: frozen-c float vs CRT exact rel {rel160:.2e}, "
          f"exact dyadic embedding: {good}")

    # 4. LARGE-k float DP against exact closed forms (g == 1).  This is the
    #    k=500 failure class: without the B-side rescaling, the region
    #    s <= rho of h^{*(k-1)} lies rho^k below the renormalized peak and
    #    the float B is FFT noise.
    for klc in (200, 500):
        epsl = Fraction(1, 25)
        bits_l = 8
        ch_l = preprocess_channels([[1 << bits_l]], bits_l)
        fe_l = FloatPairEngine(ch_l, klc, float(epsl), grid=16384)
        A_l, B_l, oA, oB, _ = fe_l.matrices(verbose=False)
        rho_l = Fraction(1 - epsl, 1) / Fraction(1 + epsl, 1)
        A_cf = Fraction(1, math.factorial(klc))
        B_cf = (rho_l**(klc-1)/(klc-1) - 2*rho_l**klc/klc
                + rho_l**(klc+1)/(klc+1)) / math.factorial(klc - 2)
        ratio_ex = float(Fraction(klc) * (1 + epsl) * B_cf / A_cf)
        ratio_fl = klc * float(1 + epsl) * (B_l[0, 0] / A_l[0, 0]) \
            * math.exp(oB - oA)
        rel = abs(ratio_fl - ratio_ex) / abs(ratio_ex)
        good = rel < 5e-3
        ok &= good
        print(f"  k={klc} g==1 float DP R-ratio rel err {rel:.2e} < 5e-3:"
              f"     {good}")

    # 4b. DIAGONAL-PRECONDITIONING regression (the k=1000 collapse).
    #     Channels with genuinely different sup-norms give an A diagonal
    #     spanning hundreds/thousands of nats after the k-th power.  A single
    #     global exp() offset underflows all but a couple of channels to 0
    #     and the Ritz solve collapses to rank ~2 (observed: 2/32 at k=1000).
    #     Per-channel log-diagonal preconditioning must keep every channel
    #     alive and the Rayleigh quotient must match exact CRT.
    rngp = np.random.default_rng(11)
    bits_p = 40
    cps = [[int(v) for v in rngp.integers(-(1 << bits_p), 1 << bits_p,
                                          size=6)] for _ in range(6)]
    kp, epsp = 40, Fraction(1, 25)
    chp = preprocess_channels(cps, bits_p)
    fep = FloatPairEngine(chp, kp, float(epsp), grid=4096)
    Ap, Bp, oAp, oBp, logDp = fep.matrices(verbose=False)
    spread = float(logDp.max() - logDp.min())
    alive = int(np.sum(np.diag(Ap) > 0))
    lamp, _ = _ritz(Ap, Bp, 1e-10)
    good = (alive == 6 and spread > 100.0 and np.isfinite(lamp))
    # Cross-check float vs exact at a SINGLE FIXED c (not the ritz vector,
    # whose float/exact quotients differ near-degenerately).  In the
    # preconditioned frame c_hat, the quotient is
    #   R = k L (c_hat^T Bp c_hat e^{oBp}) / (c_hat^T Ap c_hat e^{oAp});
    # exact CRT uses the UNconditioned c = exp(-half) c_hat.
    halfp = logDp / 2.0
    c_hat = np.ones(6)
    qb = float(c_hat @ Bp @ c_hat) * math.exp(oBp)
    qa = float(c_hat @ Ap @ c_hat) * math.exp(oAp)
    R_flp = kp * float(1 + epsp) * qb / qa
    c_true = np.exp(-halfp) * c_hat
    c_true = c_true / np.max(np.abs(c_true))
    cqp = [Fraction(float(v)) for v in c_true]
    aggp = CRTAggregator(chp, cqp, kp, epsp, verify_primes=1)
    S_Ap, S_Bp = aggp.run(jobs=1, verbose=False)
    nump, denp = aggp.exact_quad_forms(S_Ap, S_Bp)
    R_exp = float(Fraction(kp) * (1 + epsp) * nump / denp)
    relp = abs(R_flp - R_exp) / abs(R_exp) if R_exp else 1.0
    good &= relp < 5e-3
    ok &= good
    print(f"  diagonal-precond regression ({spread:.0f}-nat spread): "
          f"{alive}/6 channels alive, float vs CRT rel {relp:.2e}: {good}")

    # 5. whitened Ritz == scipy generalized eigh (PD case), and survives
    #    a numerically semidefinite metric (the k=500 failure mode)
    rngl = np.random.default_rng(7)
    X = rngl.standard_normal((8, 5))
    Apd = X.T @ X + 5.0 * np.eye(5)
    Y = rngl.standard_normal((8, 5))
    Bsym = 0.5 * ((Y.T @ Y) + (Y.T @ Y).T)
    lam_w, c_w = _ritz(Apd, Bsym)
    good = True
    if _sla is not None:
        lam_ref = float(_sla.eigh(Bsym, Apd, eigvals_only=True)[-1])
        good &= abs(lam_w - lam_ref) < 1e-10 * max(1.0, abs(lam_ref))
    Asemi = Apd.copy()
    Asemi[:, 4] = Asemi[4, :] = 0.0          # dead channel: singular metric
    lam_s, c_s = _ritz(Asemi, Bsym)
    good &= np.isfinite(lam_s) and c_s.shape == (5,) and c_s[4] == 0.0
    ok &= good
    print(f"  whitened Ritz vs scipy eigh + semidefinite metric:      {good}")

    # 5. greedy pruning is a no-op at tol=0 and monotone in the budget
    active0, _ = greedy_prune(A_f, B_f, 0.0, verbose=False)
    active1, _ = greedy_prune(A_f, B_f, 0.5, verbose=False)
    good = active0 == [0, 1, 2] and set(active1) <= {0, 1, 2} and len(active1) >= 1
    ok &= good
    print(f"  greedy prune sanity (tol=0 keeps all):                  {good}")

    # 6. Export-frame regression: discovery constructs the Gram from
    #    one-dimensional channels normalised by nrm.  Rebuilding g_raw/nrm
    #    therefore leaves the normalised-basis coefficient unchanged apart
    #    from diagonal-Gram depreconditioning.  A spurious linear nrm factor
    #    changes the represented k-fold separable function by nrm and is not
    #    frame invariant (the true raw-channel conversion would involve
    #    nrm**(-k), illustrating why the linear factor is categorically wrong).
    kf = 11
    nrmf = np.array([2.0**-7, 2.0**3, 1.25])
    chatf = np.array([0.3, -0.7, 1.1])
    logDf = np.array([-4.0, 2.0, 0.5])
    c_norm = chatf * np.exp(-0.5 * logDf)
    # Same F represented on raw channels would use c_norm / nrm**k.
    c_raw = c_norm / (nrmf ** kf)
    # Mapping back to the normalised basis recovers c_norm exactly.
    recovered = c_raw * (nrmf ** kf)
    good = np.allclose(recovered, c_norm, rtol=2e-15, atol=0.0)
    # The old linear multiplication must demonstrably fail for unequal norms.
    old_wrong = c_norm * nrmf
    good &= not np.allclose(old_wrong, c_norm, rtol=1e-12, atol=0.0)
    ok &= good
    print(f"  exported channel-frame / k-fold scaling regression:     {good}")

    print(f"\n  self test {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Rigorous CRT/Arb certification of the epsilon-enlarged "
                    "Maynard problem M_{k,eps,1/2} with channel pruning")
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--epsilon", type=str, default=None,
                   help="EXACT rational eps, e.g. 1/25")
    p.add_argument("--degrees", type=int, nargs="+", default=[18, 22, 26])
    p.add_argument("--bits", type=int, default=48)
    p.add_argument("--digits", type=int, default=30)
    p.add_argument("--prune-tol", type=float, default=1e-7,
                   help="cumulative relative float-lambda budget for greedy "
                        "channel elimination (0 disables)")
    p.add_argument("--prune-min-channels", type=int, default=1)
    p.add_argument("--prune-cmag", type=float, default=0.0,
                   help="legacy |c|-magnitude pre-drop threshold (v2 rule)")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--use-nn-c", action="store_true")
    p.add_argument("--grid", type=int, default=8192,
                   help="float DP grid size (raise with k; quadrature bias ~ (k/grid)^2)")
    p.add_argument("--rank-tol", type=float, default=0.0,
                   help="fixed rank-truncation tolerance for the whitened Ritz\n(0 = auto-escalate 1e-10 -> 1e-4 until float R is physical)")
    p.add_argument("--float-only", action="store_true",
                   help="run projection + float DP + pruning survey per degree and stop\nbefore any exact/CRT work (cheap degree scouting)")
    p.add_argument("--force-float", action="store_true",
                   help="proceed even if the float pass disagrees grossly with R_NN")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--emit-certificate", type=str, default=None)
    p.add_argument("--jobs", type=int, default=1,
                   help="processes for the CRT prime sweep")
    p.add_argument("--chunk", type=int, default=4,
                   help="primes per worker task")
    p.add_argument("--verify-primes", type=int, default=2,
                   help="held-out primes re-checked after reconstruction")
    p.add_argument("--support-L", type=int, default=1,
                   help="compact support [0,1/L] using the scaled epsilon=0 "
                        "CRT backend; L=1 keeps the ordinary certifier")
    p.add_argument("--resume", type=str, default=None,
                   help="residue checkpoint file; completed (prime, S_A, S_B)\ntriples are appended and skipped on restart (per-degree suffix added)")
    p.add_argument("--backend", choices=("crt", "ball"), default="crt",
                   help="rigorous backend: exact multimodular CRT or adaptive Arb balls")
    p.add_argument("--ball-prec", type=int, default=128,
                   help="initial Arb precision in bits")
    p.add_argument("--ball-target-rel", type=float, default=1e-12,
                   help="target relative width of the rigorous Arb enclosure")
    p.add_argument("--ball-max-prec", type=int, default=4096,
                   help="maximum adaptive Arb precision in bits")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.npz is None:
        raise SystemExit("provide --npz (or run --self-test)")
    eps = parse_fraction(args.epsilon) if args.epsilon is not None else None
    certify_epsilon_crt(args.npz, degrees=args.degrees, bits=args.bits,
                        epsilon=eps, digits=args.digits,
                        prune_tol=args.prune_tol,
                        prune_min=args.prune_min_channels,
                        prune_cmag=args.prune_cmag, no_prune=args.no_prune,
                        use_nn_c=args.use_nn_c, grid=args.grid,
                        force_float=args.force_float,
                        float_only=args.float_only,
                        rank_tol_cli=args.rank_tol,
                        emit_certificate=args.emit_certificate,
                        jobs=args.jobs, chunk=args.chunk,
                        verify_primes=args.verify_primes,
                        resume=args.resume, backend=args.backend,
                        ball_prec=args.ball_prec,
                        ball_target_rel=args.ball_target_rel,
                        ball_max_prec=args.ball_max_prec,
                        support_L=args.support_L)


if __name__ == "__main__":
    main()
