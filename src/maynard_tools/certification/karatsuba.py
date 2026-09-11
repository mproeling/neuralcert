"""
Projection + exact evaluation ("certification") for separable Maynard trial
functions discovered by maynard_separable_rayleigh.py.

Input:  the .npz exported by the discovery script (learned channels g_j on a
        fine grid, channel norms, mixing c, k).
Output: an EXACT rational lower bound  M_k >= R  for the explicit trial
        function
            F(t) = sum_j c_j prod_{i=1}^k p_j(t_i),
        where p_j in Q[x] is the degree-d projection of the learned channel
        g_j, and c in Q^m. Both p_j and c are explicit rationals, so R is an
        exact rational number: the bound is theorem-grade for this F.

Mathematics (no (d+1)^k expansion ever happens):

    Dirichlet:  int_{Delta_k} prod t_i^{r_i} dt = prod r_i! / (k + sum r_i)!

    With h_{jl} = p_j p_l = sum_r b_r x^r and the Borel-weighted generating
    polynomial  h_hat(x) = sum_r b_r r! x^r :

        A_{jl} = int_{Delta_k} prod_i h_{jl}(t_i) dt
               = sum_R  [x^R] h_hat^k  /  (k + R)!

        B_{jl} = int_{Delta_{k-1}} G_j(1-S) G_l(1-S) prod_{i>=2} h_{jl}(t_i) dt
               = sum_m q_m m!  sum_R  [x^R] h_hat^{k-1} / (m + k - 1 + R)!

        where G_j(y) = int_0^y p_j,  q(y) = G_j(y) G_l(y) = sum_m q_m y^m,
        S = sum_{i>=2} t_i.

    (These are the exact-arithmetic counterparts of the iterated 1D
    convolutions in the discovery solver; the identity
    int_{Delta_p} prod h(t_i) dt = sum over compositions of Dirichlet
    integrals = coefficient extraction from the p-th power of h_hat.)

    Polynomial powers over Z use Kronecker substitution: pack integer
    coefficients into one big integer, multiply with CPython's subquadratic
    bigint arithmetic, unpack. Signed polynomials are split into positive /
    negative parts (four nonnegative products).

Certification chain:

    1. project normalized channels onto shifted Legendre (integer monomial
       coefficients!) by least squares on the exported fine grid;
    2. dyadically rationalize the Legendre coefficients (denominator 2^bits)
       -> each p_j is EXACT in Q[x] with denominator 2^bits;
    3. assemble A, B exactly (Fractions);
    4. re-solve the m x m Ritz problem in float on the exact A, B, then
       dyadically rationalize the resulting c  (ANY explicit c yields a
       valid bound; the eig only needs to be approximately right);
    5. R = k * (c^T B c) / (c^T A c) as an exact Fraction. Decimal output is
       TRUNCATED (rounded toward zero), so every printed decimal is itself
       a valid lower bound.

Usage:
    python maynard_project_certify.py --npz k10_channels.npz --degrees 8 12 16
    python maynard_project_certify.py --npz k10_channels.npz --degrees 12 \
           --use-nn-c            # skip the re-Ritz, use the NN's mixing
    python maynard_project_certify.py --self-test
    python maynard_project_certify.py --npz k10.npz --degrees 10 --cross-check
"""

from __future__ import annotations

import argparse
import math
import time
from fractions import Fraction

import numpy as np

from maynard_tools.certification.frames import load_discovery_frame

try:                                    # exact GMP-backed arithmetic (fast)
    import gmpy2
    from gmpy2 import mpq as RAT
    _HAVE_GMPY2 = True
except ImportError:                     # stdlib fallback (slow, identical)
    RAT = Fraction
    _HAVE_GMPY2 = False

try:  # optional, only for --cross-check
    import scipy.linalg as _sla
except ImportError:  # pragma: no cover
    _sla = None


# ---------------------------------------------------------------------------
# Exact integer polynomial arithmetic (Kronecker substitution)
# ---------------------------------------------------------------------------
def _mul_nonneg(a: list[int], b: list[int]) -> list[int]:
    """Multiply polynomials with NONNEGATIVE integer coefficients via
    Kronecker substitution (byte-aligned packing)."""
    if not a or not b:
        return []
    amax = max(a)
    bmax = max(b)
    if amax == 0 or bmax == 0:
        return [0] * (len(a) + len(b) - 1)
    bound = min(len(a), len(b)) * amax * bmax
    W = ((bound.bit_length() + 1 + 7) // 8) * 8          # bits per word
    Wb = W // 8
    A = int.from_bytes(b"".join(c.to_bytes(Wb, "little") for c in a), "little")
    B = int.from_bytes(b"".join(c.to_bytes(Wb, "little") for c in b), "little")
    if _HAVE_GMPY2:
        C = int(gmpy2.mpz(A) * gmpy2.mpz(B))
    else:
        C = A * B
    n = len(a) + len(b) - 1
    raw = C.to_bytes(n * Wb + 16, "little")
    return [int.from_bytes(raw[i * Wb:(i + 1) * Wb], "little")
            for i in range(n)]


def poly_mul(a: list[int], b: list[int]) -> list[int]:
    """Multiply integer polynomials (signed coefficients allowed)."""
    if not a or not b:
        return []
    ap = [x if x > 0 else 0 for x in a]
    an = [-x if x < 0 else 0 for x in a]
    bp = [x if x > 0 else 0 for x in b]
    bn = [-x if x < 0 else 0 for x in b]
    pp = _mul_nonneg(ap, bp)
    nn = _mul_nonneg(an, bn)
    pn = _mul_nonneg(ap, bn)
    np_ = _mul_nonneg(an, bp)
    return [pp[i] + nn[i] - pn[i] - np_[i] for i in range(len(pp))]


def poly_pow(a: list[int], e: int) -> list[int]:
    """a(x)^e by binary exponentiation, exact integer arithmetic."""
    if e == 0:
        return [1]
    result: list[int] | None = None
    base = a
    while e:
        if e & 1:
            result = base if result is None else poly_mul(result, base)
        e >>= 1
        if e:
            base = poly_mul(base, base)
    return result if result is not None else [1]


# ---------------------------------------------------------------------------
# Shifted Legendre basis (integer monomial coefficients)
# ---------------------------------------------------------------------------
def shifted_legendre_coeffs(d: int) -> list[list[int]]:
    """
    Integer monomial coefficients of shifted Legendre polynomials
    P~_n(x) = P_n(2x - 1) on [0, 1]:

        P~_n(x) = sum_{p=0}^n (-1)^{n+p} C(n, p) C(n+p, p) x^p
    """
    out = []
    for n_ in range(d + 1):
        coeffs = [
            (-1) ** (n_ + p) * math.comb(n_, p) * math.comb(n_ + p, p)
            for p in range(n_ + 1)
        ]
        out.append(coeffs)
    return out


def eval_shifted_legendre_design(x: np.ndarray, d: int) -> np.ndarray:
    """(len(x), d+1) design matrix of P~_0 .. P~_d via the stable recurrence."""
    y = 2.0 * x - 1.0
    cols = [np.ones_like(y)]
    if d >= 1:
        cols.append(y)
    for n_ in range(1, d):
        cols.append(((2 * n_ + 1) * y * cols[n_] - n_ * cols[n_ - 1]) / (n_ + 1))
    return np.stack(cols[: d + 1], axis=1)


# ---------------------------------------------------------------------------
# Dyadic rationalization
# ---------------------------------------------------------------------------
def dyadic(x: float, bits: int):
    """Nearest fraction with denominator 2^bits (exact float -> Q map)."""
    return RAT(round(x * (1 << bits)), 1 << bits)


# ---------------------------------------------------------------------------
# Exact A, B assembly for polynomial channels
# ---------------------------------------------------------------------------
def _build_fact(n: int) -> list[int]:
    f = [1] * (n + 1)
    for i in range(1, n + 1):
        f[i] = f[i - 1] * i
    return f


def pair_vanilla(nj: list[int], nl: list[int], bits: int, k: int,
                 fact: list[int] | None = None):
    """A_{jl}, B_{jl} for the vanilla problem, exact (RAT)."""
    den_ch = 1 << bits
    h = poly_mul(nj, nl)
    if fact is None:
        d2 = max(len(nj), len(nl)) * 2
        fact = _build_fact((2 * d2 + 2) + k + 2 * d2 * k + 4)
    bh = [h[r] * fact[r] for r in range(len(h))]

    P_km1 = poly_pow(bh, k - 1)
    P_k = poly_mul(P_km1, bh)

    # ---- A:  sum_R P_k[R] * (K! / (k+R)!)  via a RUNNING PRODUCT ----
    Rmax = len(P_k) - 1
    K = k + Rmax
    A_num = 0
    ratio = 1                                   # K!/(k+R)! at R = Rmax
    for R in range(Rmax, -1, -1):
        if P_k[R]:
            A_num += P_k[R] * ratio
        ratio *= (k + R)                        # -> value for R-1
    A = RAT(A_num, fact[K] * den_ch ** (2 * k))

    # ---- B ----
    Gj = [RAT(0)] + [RAT(cf, den_ch * (p + 1)) for p, cf in enumerate(nj)]
    Gl = [RAT(0)] + [RAT(cf, den_ch * (p + 1)) for p, cf in enumerate(nl)]
    q = [RAT(0)] * (len(Gj) + len(Gl) - 1)
    for pj, aj in enumerate(Gj):
        if aj == 0:
            continue
        for pl, al in enumerate(Gl):
            if al != 0:
                q[pj + pl] += aj * al

    # B = sum_m q_m m! sum_R N_R / (m+k-1+R)!  with running products per m
    Rm1 = len(P_km1) - 1
    B_acc = RAT(0)
    for m_, qm in enumerate(q):
        if qm == 0:
            continue
        top = m_ + k - 1 + Rm1
        ratio = 1                               # top!/(m+k-1+R)! at R = Rm1
        S = 0
        for R in range(Rm1, -1, -1):
            if P_km1[R]:
                S += P_km1[R] * ratio
            ratio *= (m_ + k - 1 + R)
        B_acc += qm * RAT(fact[m_] * S, fact[top])
    B = B_acc / (den_ch ** (2 * (k - 1)))
    return A, B


def _vanilla_worker(args):
    j, l, nj, nl, bits, k = args
    A, B = pair_vanilla(nj, nl, bits, k)
    return (j, l, int(A.numerator), int(A.denominator),
            int(B.numerator), int(B.denominator))


class ExactSeparable:
    """
    Channels p_j in Q[x] given as INTEGER coefficient lists with one common
    power-of-two denominator 2^bits per channel:

        p_j(x) = (1 / 2^bits) * sum_p int_coeffs[j][p] x^p
    """

    def __init__(self, int_coeffs: list[list[int]], bits: int, k: int) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.bits = bits
        self.int_coeffs = int_coeffs
        self.m = len(int_coeffs)
        max_needed = 0
        d2 = max(len(c) for c in int_coeffs) * 2          # generous
        max_needed = (2 * d2 + 2) + k + 2 * d2 * k + 4
        self._fact = [1] * (max_needed + 1)
        for i in range(1, max_needed + 1):
            self._fact[i] = self._fact[i - 1] * i

    # -- one pair --------------------------------------------------------
    def _pair(self, j: int, l: int):
        return pair_vanilla(self.int_coeffs[j], self.int_coeffs[l],
                            self.bits, self.k, self._fact)

    def gram_matrices(self, verbose: bool = True, jobs: int = 1):
        m = self.m
        A = [[RAT(0)] * m for _ in range(m)]
        B = [[RAT(0)] * m for _ in range(m)]
        if jobs > 1:
            from multiprocessing import Pool
            tasks = [(j, l, self.int_coeffs[j], self.int_coeffs[l],
                      self.bits, self.k)
                     for j in range(m) for l in range(j, m)]
            with Pool(jobs) as pool:
                for (j, l, An, Ad, Bn, Bd) in pool.map(_vanilla_worker,
                                                       tasks):
                    A[j][l] = A[l][j] = RAT(An, Ad)
                    B[j][l] = B[l][j] = RAT(Bn, Bd)
            return A, B
        n_pairs = m * (m + 1) // 2
        done = 0
        t0 = time.time()
        for j in range(m):
            for l in range(j, m):
                A[j][l], B[j][l] = self._pair(j, l)
                A[l][j], B[l][j] = A[j][l], B[j][l]
                done += 1
            if verbose:
                print(f"    pairs {done:4d}/{n_pairs}   "
                      f"[{time.time() - t0:6.1f}s]", flush=True)
        return A, B


# ---------------------------------------------------------------------------
# Exact quadratic forms and decimal display
# ---------------------------------------------------------------------------
def quad_form(M: list[list[Fraction]], c: list[Fraction]) -> Fraction:
    total = Fraction(0)
    m = len(c)
    for j in range(m):
        if c[j] == 0:
            continue
        for l in range(m):
            if c[l] == 0 or M[j][l] == 0:
                continue
            total += c[j] * M[j][l] * c[l]
    return total


def frac_to_decimal_truncated(fr: Fraction, digits: int) -> str:
    """Decimal expansion truncated toward zero -> printed value is itself a
    valid lower bound when fr is a lower bound."""
    sign = "-" if fr < 0 else ""
    fr = abs(fr)
    ip = fr.numerator // fr.denominator
    rem = fr.numerator - ip * fr.denominator
    frac_digits = (rem * 10 ** digits) // fr.denominator
    return f"{sign}{ip}.{str(frac_digits).zfill(digits)}"


# ---------------------------------------------------------------------------
# Certificate emission (verified by standalone verify_maynard_certificate.py)
# ---------------------------------------------------------------------------
def write_certificate(path: str, quantity: str, k: int, eps,
                      int_coeffs, bits: int, c_q, R_exact,
                      digits: int = 30) -> None:
    import json
    from fractions import Fraction as _F
    eps = _F(eps)
    cert = {
        "format": "maynard-separable-certificate-v1",
        "quantity": quantity,
        "k": int(k),
        "epsilon": {"num": eps.numerator, "den": eps.denominator},
        "channels": [[f"{n}/{1 << bits}" for n in ch] for ch in int_coeffs],
        "c": [f"{f.numerator}/{f.denominator}" for f in c_q],
        "claimed_lower_bound": frac_to_decimal_truncated(R_exact, digits),
    }
    with open(path, "w") as f:
        json.dump(cert, f, indent=1)
    print(f"  certificate written to {path}  "
          f"(verify: python verify_maynard_certificate.py {path})")


# ---------------------------------------------------------------------------
# Projection of learned channels
# ---------------------------------------------------------------------------
def project_channels(x_fine: np.ndarray, g_fine: np.ndarray, d: int,
                     bits: int):
    """
    Least-squares projection of each channel onto shifted Legendre of degree
    <= d, then dyadic rationalization of the LEGENDRE coefficients and EXACT
    conversion to integer monomial coefficients with denominator 2^bits.

    Returns (int_coeffs, l2_residuals, linf_errors).
    """
    m = g_fine.shape[1]
    Phi = eval_shifted_legendre_design(x_fine, d)        # (P, d+1)
    coef, *_ = np.linalg.lstsq(Phi, g_fine, rcond=None)  # (d+1, m) float

    leg_int = shifted_legendre_coeffs(d)                 # exact ints
    int_coeffs: list[list[int]] = []
    l2_res: list[float] = []
    linf: list[float] = []
    for jj in range(m):
        # dyadic Legendre coefficients (numerators over 2^bits)
        leg_num = [round(float(coef[i, jj]) * (1 << bits))
                   for i in range(d + 1)]
        # exact monomial numerators: sum_i leg_num[i] * P~_i coeffs
        mono = [0] * (d + 1)
        for i in range(d + 1):
            if leg_num[i] == 0:
                continue
            for p, cf in enumerate(leg_int[i]):
                mono[p] += leg_num[i] * cf
        int_coeffs.append(mono)

        # diagnostics against the fine grid (float, informational only)
        p_vals = Phi @ coef[:, jj]
        r = g_fine[:, jj] - p_vals
        scale = max(float(np.max(np.abs(g_fine[:, jj]))), 1e-300)
        l2_res.append(float(np.sqrt(np.mean(r ** 2))) / scale)
        linf.append(float(np.max(np.abs(r))) / scale)
    return int_coeffs, l2_res, linf


# ---------------------------------------------------------------------------
# Certification driver
# ---------------------------------------------------------------------------
def certify(npz_path: str, degrees: list[int], bits: int,
            use_nn_c: bool = False, digits: int = 30,
            cross_check: bool = False, prune_tol: float = 0.0,
            emit_certificate: str | None = None, jobs: int = 1):
    data = np.load(npz_path)
    frame = load_discovery_frame(data)
    k, R_nn = frame.k, frame.rayleigh
    c_nn, x_fine, g_fine = frame.coefficients, frame.points, frame.channels
    m = g_fine.shape[1]

    if prune_tol > 0.0:
        keep = np.abs(c_nn) > prune_tol * np.max(np.abs(c_nn))
        n_dead = int(m - keep.sum())
        if n_dead > 0:
            print(f" pruning {n_dead}/{m} channels with |c_j| < "
                  f"{prune_tol:g} * max|c|  "
                  f"(pairs: {m*(m+1)//2} -> "
                  f"{int(keep.sum())*(int(keep.sum())+1)//2})")
            g_fine = g_fine[:, keep]
            c_nn = c_nn[keep]
            m = g_fine.shape[1]

    print("=" * 72)
    print(f" Exact certification   k = {k}   m = {m} channels   "
          f"bits = {bits}")
    print(f" discovery (float64) R_NN = {R_nn:.12f}")
    print("=" * 72)

    results = []
    for d in degrees:
        print(f"\n--- degree d = {d} ---")
        int_coeffs, l2_res, linf = project_channels(x_fine, g_fine, d, bits)
        print(f"  projection rel. L2 residuals : "
              f"max {max(l2_res):.2e}   median {sorted(l2_res)[m // 2]:.2e}")
        print(f"  projection rel. Linf errors  : max {max(linf):.2e}")

        ex = ExactSeparable(int_coeffs, bits=bits, k=k)
        print(f"  assembling exact A, B ({m * (m + 1) // 2} pairs) ...")
        A, B = ex.gram_matrices(verbose=False, jobs=jobs)

        # float copies for the (approximate) Ritz step
        A_f = np.array([[float(A[j][l]) for l in range(m)] for j in range(m)])
        B_f = np.array([[float(B[j][l]) for l in range(m)] for j in range(m)])

        if use_nn_c:
            c_float = c_nn
            c_source = "NN mixing (exported c)"
        else:
            if _sla is None:
                raise RuntimeError("scipy required for the re-Ritz step")
            jit = 1e-12 * np.max(np.abs(np.diag(A_f)))
            lams, V = _sla.eigh(B_f, A_f + jit * np.eye(m))
            c_float = V[:, -1]
            c_source = "re-Ritz on exact A, B (float eig, then rationalized)"
        # normalize scale before rationalizing (R is scale-invariant)
        c_float = c_float / max(np.max(np.abs(c_float)), 1e-300)
        c_q = [dyadic(float(v), bits) for v in c_float]

        num = quad_form(B, c_q)
        den = quad_form(A, c_q)
        if den <= 0:
            print("  WARNING: c^T A c <= 0 after rationalization — skipping")
            continue
        R_exact = k * num / den

        dec = frac_to_decimal_truncated(R_exact, digits)
        gap = R_nn - float(R_exact)
        print(f"  c from: {c_source}")
        print(f"  CERTIFIED  M_{k} >= {dec}   (exact rational, truncated)")
        print(f"  R_NN - R_certified = {gap:+.3e}")
        results.append({"d": d, "R_exact": R_exact, "gap": gap,
                        "int_coeffs": int_coeffs, "c_q": c_q})

        if cross_check:
            _do_cross_check(int_coeffs, bits, k, R_exact)

    if emit_certificate and results:
        best = max(results, key=lambda r: r["R_exact"])
        write_certificate(emit_certificate, f"M_{k}", k, 0,
                          best["int_coeffs"], bits, best["c_q"],
                          best["R_exact"], digits=digits)
    if len(results) >= 2:
        print("\n--- degree sweep summary ---")
        print(f"  {'d':>4} {'R_certified':>34} {'R_NN - R_cert':>15}")
        for r in results:
            print(f"  {r['d']:>4} {frac_to_decimal_truncated(r['R_exact'], 24):>34} "
                  f"{r['gap']:>+15.3e}")
        gaps = [abs(r["gap"]) for r in results]
        if gaps[-1] < 1e-9:
            print("  projection gap closed to <1e-9: the certificate has "
                  "reached the discovery value.")
        else:
            # geometric contraction test: per-degree-step improvement ratio
            ratios = [gaps[i] / max(gaps[i + 1], 1e-300)
                      for i in range(len(gaps) - 1)]
            if min(ratios) > 1.5:
                print(f"  projection gap contracting geometrically "
                      f"(x{min(ratios):.1f}-{max(ratios):.1f} per degree "
                      f"step): raise d further to shrink the remaining "
                      f"{gaps[-1]:.1e}.")
            else:
                print("  projection gap stalled: check channel smoothness "
                      "(kinks resist polynomial projection), raise bits, or "
                      "use a poly*exp basis for boundary-layer channels.")
    return results


def _do_cross_check(int_coeffs, bits, k, R_exact):
    """Evaluate the SAME rational polynomials through the float discovery
    solver; generalized eigenvalues are invariant under per-channel scaling,
    so the two R values must agree to ~FP64."""
    try:
        import torch
        from maynard_separable_rayleigh import SeparableMaynard
    except ImportError:
        print("  cross-check skipped (torch / discovery module unavailable)")
        return

    class PolyChannels(torch.nn.Module):
        def __init__(self, coeffs, bits_):
            super().__init__()
            self.co = [[c / (1 << bits_) for c in ch] for ch in coeffs]

        def forward(self, x):
            cols = []
            for ch in self.co:
                acc = torch.zeros_like(x)
                for c in reversed(ch):
                    acc = acc * x + c
                cols.append(acc)
            return torch.stack(cols, dim=1)

    n_chk = max(48, int(6 * math.sqrt(k)) + 8)
    prob = SeparableMaynard(k=k, n=n_chk)
    rep = prob.report(PolyChannels(int_coeffs, bits))
    diff = rep.R - float(R_exact)
    print(f"  cross-check: float conv solver on the same rational F: "
          f"R = {rep.R:.12f}   (exact - float = {-diff:+.2e})")


# ---------------------------------------------------------------------------
# Self test (closed forms)
# ---------------------------------------------------------------------------
def self_test() -> None:
    print("=== self test: exact identities ===")
    ok = True

    # g == 1 (single channel, d = 0):  A = 1/k!, B = 2/(k+1)!, R = 2k/(k+1)
    for k in (2, 3, 5, 10, 20):
        bits = 8
        ex = ExactSeparable([[1 << bits]], bits=bits, k=k)
        A, B = ex.gram_matrices(verbose=False, jobs=1)
        R = k * B[0][0] / A[0][0]
        expect_A = Fraction(1, math.factorial(k))
        expect_B = Fraction(2, math.factorial(k + 1))
        expect_R = Fraction(2 * k, k + 1)
        good = (A[0][0] == expect_A and B[0][0] == expect_B and R == expect_R)
        ok &= good
        print(f"  k={k:3d}: A == 1/k! {A[0][0] == expect_A}   "
              f"B == 2/(k+1)! {B[0][0] == expect_B}   "
              f"R == 2k/(k+1) {R == expect_R}")

    # g(x) = 1 - x  (d = 1):  h(x) = (1-x)^2, closed forms via Dirichlet.
    # A = int_{Delta_k} prod (1-t_i)^2 dt : verify against direct Dirichlet
    # expansion for k = 2 computed independently.
    bits = 4
    ex = ExactSeparable([[1 << bits, -(1 << bits)]], bits=bits, k=2)
    A, B = ex.gram_matrices(verbose=False)
    # direct: h = 1 - 2x + x^2; A = sum_{r1,r2} b_{r1} b_{r2} r1! r2!/(2+r1+r2)!
    b = [Fraction(1), Fraction(-2), Fraction(1)]
    A_direct = Fraction(0)
    for r1 in range(3):
        for r2 in range(3):
            A_direct += (b[r1] * b[r2] *
                         math.factorial(r1) * math.factorial(r2) *
                         Fraction(1, math.factorial(2 + r1 + r2)))
    good = A[0][0] == A_direct
    ok &= good
    print(f"  g=1-x, k=2: A matches direct Dirichlet double sum: {good} "
          f"(A = {A[0][0]})")

    # Kronecker polynomial arithmetic sanity vs naive convolution
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = [int(v) for v in rng.integers(-999, 1000, size=rng.integers(1, 30))]
        bpoly = [int(v) for v in rng.integers(-999, 1000,
                                              size=rng.integers(1, 30))]
        naive = [0] * (len(a) + len(bpoly) - 1)
        for i, ai in enumerate(a):
            for jdx, bj in enumerate(bpoly):
                naive[i + jdx] += ai * bj
        ok &= (poly_mul(a, bpoly) == naive)
    print(f"  Kronecker poly_mul == naive convolution on 20 random cases: "
          f"{ok}")

    print(f"\n  self test {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact rational certification of separable Maynard "
                    "trial functions")
    p.add_argument("--npz", type=str, default=None,
                   help="channels file exported by the discovery script")
    p.add_argument("--degrees", type=int, nargs="+", default=[8, 12, 16],
                   help="projection degrees to sweep")
    p.add_argument("--bits", type=int, default=48,
                   help="dyadic rationalization bits (denominator 2^bits)")
    p.add_argument("--digits", type=int, default=30,
                   help="decimal digits to print (truncated)")
    p.add_argument("--use-nn-c", action="store_true",
                   help="use the exported NN mixing instead of re-Ritz")
    p.add_argument("--prune-tol", type=float, default=0.0,
                   help="drop channels with |c_j| < tol * max|c| before "
                        "projection (e.g. 1e-8); 0 disables")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel workers over channel pairs")
    p.add_argument("--emit-certificate", type=str, default=None,
                   help="write a standalone-verifiable JSON certificate for "
                        "the best degree in the sweep")
    p.add_argument("--cross-check", action="store_true",
                   help="also evaluate the rational F through the float "
                        "discovery solver")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.npz is None:
        raise SystemExit("provide --npz (or run --self-test)")
    certify(args.npz, degrees=args.degrees, bits=args.bits,
            use_nn_c=args.use_nn_c, digits=args.digits,
            cross_check=args.cross_check, prune_tol=args.prune_tol,
            emit_certificate=args.emit_certificate, jobs=args.jobs)


if __name__ == "__main__":
    main()
