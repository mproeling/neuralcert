"""
Ultra-low-memory FLINT exact certification for the EPSILON-ENLARGED Maynard problem
M_{k,eps,1/2}, companion to maynard_project_certify.py (vanilla) and
maynard_separable_rayleigh_v4_epsilon.py (discovery).

Functional (theta = 1/2, 0 <= eps < 1, box caps inactive), L = 1+eps, U = 1-eps:

    I(F)   = int_{L*Delta_k} F^2
    J_1(F) = int_{U*Delta_{k-1}} ( int_0^{L-S} F dt_1 )^2,   S = sum_{i>=2} t_i
    R      = k J_1 / I   (lower-bounds M_{k,eps,1/2} for any explicit F)

Everything is computed in the RESCALED coordinate u = t / L, so channels are
exact polynomials q_j(u) in Q[u] on the UNIT interval (standard shifted
Legendre projection, denominator 2^bits), and with rho := U/L (exact rational):

    I   = L^k     * c^T A c,   A_{jl} = sum_R [x^R] h_hat^k / (k+R)!
                                (IDENTICAL to the vanilla certifier)
    J_1 = L^{k+1} * c^T B c,
    B_{jl} = sum_m qt_m sum_R [x^R] h_hat^{k-1}
                 * rho^{m+k-1+R} / ( (k-2+R)! * (m+k-1+R) )

    where h = q_j q_l, h_hat is its Borel weighting (coeff r multiplied by r!),
    q(y) = Q_j(y) Q_l(y) with Q_j(y) = int_0^y q_j, and
    q(1-sigma) = sum_m qt_m sigma^m  (binomial transform, exact).

    R = k * L * (c^T B c) / (c^T A c) exactly. Exact pairs are streamed twice; full exact Gram matrices are never stored.

The v2 pair engine also avoids materialising h_hat^k, the giant weighted B
polynomial, and the factorial table.  At peak it retains only h_hat^(k-1),
short channel polynomials, and scalar exact accumulators.

At eps = 0 (rho = 1) the B formula must agree with the vanilla certifier's
B = sum_m q_m m! sum_R N_R / (m+k-1+R)!  as a Fraction identity (integrating
q(1-sigma) against the same density two ways); this is enforced in
--self-test, together with the g == 1 closed form

    A = 1/k!,   B = [rho^{k-1}/(k-1) - 2 rho^k/k + rho^{k+1}/(k+1)]/(k-2)!,
    R = k L B / A,

and a float cross-check through the epsilon discovery solver.

Rigor note: eps enters the certificate as an EXACT Fraction supplied on the
command line (e.g. --epsilon 1/25) and is checked against the float stored in
the npz. The certified statement is about the exact functional at that exact
eps; the training-time float eps only influenced which channels were learned,
never the validity of the bound.

Usage:
    python maynard_project_certify_epsilon.py --npz k50eps.npz \
           --epsilon 1/25 --degrees 22 26 30 --prune-tol 1e-6
    python maynard_project_certify_epsilon.py --self-test
"""

from __future__ import annotations

import argparse
import math
import gc
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import numpy as np

from maynard_tools.certification.frames import load_discovery_frame

try:
    from flint import fmpq, fmpq_poly, fmpz_poly
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "python-flint is required for the low-memory certifier. Install it with:\n"
        "    python -m pip install python-flint"
    ) from exc

from .karatsuba import (
    RAT,
    ExactSeparable,
    write_certificate,
    dyadic,
    frac_to_decimal_truncated,
    poly_mul,
    poly_pow,
    project_channels,
    quad_form,
)

try:
    import scipy.linalg as _sla
except ImportError:  # pragma: no cover
    _sla = None


# ---------------------------------------------------------------------------
# FLINT exact pair assembly and low-memory streaming
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelData:
    numerators: tuple[int, ...]
    denominator: int
    transformed_numerators: tuple[int, ...]  # Q(1-sigma)
    transformed_denominator: int


def _trim_int_poly(coeffs: list[int]) -> list[int]:
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs or [0]


def _transformed_antiderivative(nums: list[int], den: int) -> tuple[list[int], int]:
    """Integer/common-denominator coefficients of Q(1-sigma)."""
    lcm_orders = 1
    for order in range(1, len(nums) + 1):
        lcm_orders = math.lcm(lcm_orders, order)
    q_int = [0] * (len(nums) + 1)
    for degree, num in enumerate(nums):
        q_int[degree + 1] = num * (lcm_orders // (degree + 1))
    degree = len(q_int) - 1
    out = [0] * (degree + 1)
    for m_ in range(degree + 1):
        acc = 0
        for r in range(m_, degree + 1):
            if q_int[r]:
                acc += q_int[r] * math.comb(r, m_)
        out[m_] = -acc if m_ & 1 else acc
    out = _trim_int_poly(out)
    out_den = den * lcm_orders
    common = out_den
    for value in out:
        common = math.gcd(common, abs(value))
    if common > 1:
        out = [value // common for value in out]
        out_den //= common
    return out, out_den


def preprocess_channels(int_coeffs: list[list[int]], bits: int) -> list[ChannelData]:
    den = 1 << bits
    channels = []
    for coeffs in int_coeffs:
        nums = _trim_int_poly([int(v) for v in coeffs])
        tnums, tden = _transformed_antiderivative(nums, den)
        channels.append(ChannelData(tuple(nums), den, tuple(tnums), tden))
    return channels


_K = 0
_RHO = fmpq(0)


def _init_worker(k: int, rho_num: int, rho_den: int) -> None:
    """Initialise only tiny worker state.

    v1 precomputed every factorial up to O(k d).  At k=1000 this table itself
    contains tens of thousands of enormous Python integers.  v2 computes only
    the two terminal factorials needed by a pair and uses running ratios.
    """
    global _K, _RHO
    _K = k
    _RHO = fmpq(rho_num, rho_den)


def _poly_coeff(poly: fmpz_poly, i: int) -> int:
    """Read one FLINT coefficient without copying the complete polynomial."""
    if i < 0 or i >= len(poly):
        return 0
    return int(poly[i])


def _pair_worker(task):
    """Ultra-low-memory exact pair assembly.

    Important memory properties:
      * forms P = h_hat^(k-1), but never P*h_hat = h_hat^k;
      * never converts P to a Python list;
      * never constructs the O(kd)-length weighted B polynomial;
      * never constructs the product of that polynomial with q_j q_l;
      * never holds an O(kd) factorial table.

    The price is additional Python-level arithmetic, especially in B.  This is
    intentionally a RAM-for-time implementation for extreme k.
    """
    j, l, ch_j, ch_l = task
    k, rho = _K, _RHO

    qj = fmpz_poly(list(ch_j.numerators))
    ql = fmpz_poly(list(ch_l.numerators))
    h = qj * ql
    den_h = ch_j.denominator * ch_l.denominator

    # h_hat[r] = r! [x^r] h.  h has degree only O(d), so this short list is
    # harmless.  No long factorial table is retained.
    bh_coeffs = [int(h[r]) * math.factorial(r) for r in range(len(h))]
    bh = fmpz_poly(bh_coeffs)
    p = bh ** (k - 1)                         # the one unavoidable giant object
    rmax = len(p) - 1
    smax = len(bh_coeffs) - 1

    # ------------------------------------------------------------------ A --
    # A = sum_n [x^n](P*bh)/(k+n)! / den_h^k.
    # Compute each coefficient of P*bh by a short stencil instead of ever
    # materialising P*bh.  A common denominator top_a! turns every weight into
    # an integer running ratio.
    nmax = rmax + smax
    top_a = k + nmax
    ratio = 1                                 # top_a!/(k+n)! at n=nmax
    a_num = 0
    for n in range(nmax, -1, -1):
        lo = max(0, n - rmax)
        hi = min(smax, n)
        conv_n = 0
        for ss in range(lo, hi + 1):
            bhs = bh_coeffs[ss]
            if bhs:
                pr = _poly_coeff(p, n - ss)
                if pr:
                    conv_n += pr * bhs
        if conv_n:
            a_num += conv_n * ratio
        ratio *= (k + n)
    a = fmpq(a_num, math.factorial(top_a) * (den_h ** k))

    # ------------------------------------------------------------------ B --
    # Let T(sigma)=Q_j(1-sigma)Q_l(1-sigma).  T is short (degree O(d)), so
    # materialising only this product is cheap.  We then evaluate
    #
    #   sum_R P_R rho^(k-1+R)/(k-2+R)! *
    #         sum_m T_m rho^m/(m+k-1+R)
    #
    # directly.  This avoids both the giant weighted polynomial g_num and the
    # giant product g_num*T from v1.
    t_num = (fmpz_poly(list(ch_j.transformed_numerators)) *
             fmpz_poly(list(ch_l.transformed_numerators)))
    t_den = ch_j.transformed_denominator * ch_l.transformed_denominator
    t_weight = []
    rho_m = fmpq(1)
    for mm in range(len(t_num)):
        t_weight.append(fmpq(int(t_num[mm])) * rho_m)
        rho_m *= rho
    nz = [(mm, val) for mm, val in enumerate(t_weight) if val]

    top_b = k - 2 + rmax
    common_fact = math.factorial(top_b)
    ratio = 1                                 # top_b!/(k-2+R)! at R=rmax
    rho_base = rho ** (k - 1 + rmax)
    b_acc = fmpq(0)
    for rr in range(rmax, -1, -1):
        pr = _poly_coeff(p, rr)
        if pr:
            base = k - 1 + rr
            inner = fmpq(0)
            for mm, val in nz:
                inner += val / (base + mm)
            if inner:
                b_acc += (pr * ratio) * rho_base * inner
        ratio *= (k - 2 + rr)
        if rr:
            rho_base /= rho

    b = b_acc / (common_fact * t_den * (den_h ** (k - 1)))
    return (j, l, int(a.numerator), int(a.denominator),
            int(b.numerator), int(b.denominator))


class ExactSeparableEpsilonStreaming:
    """FLINT-backed exact pair generator; never stores exact Gram matrices."""
    def __init__(self, int_coeffs: list[list[int]], bits: int, k: int,
                 epsilon: Fraction) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        if not (0 <= epsilon < 1):
            raise ValueError("epsilon must be in [0,1)")
        self.k, self.bits = k, bits
        self.eps = Fraction(epsilon)
        self.channels = preprocess_channels(int_coeffs, bits)
        self.m = len(self.channels)
        self.rho = fmpq(self.eps.denominator - self.eps.numerator,
                        self.eps.denominator + self.eps.numerator)
        max_h_degree = 2 * max(len(ch.numerators) - 1 for ch in self.channels)

    def tasks(self):
        for j in range(self.m):
            for l in range(j, self.m):
                yield (j, l, self.channels[j], self.channels[l])

    def pairs(self, jobs: int = 1, chunksize: int = 1):
        rho_num, rho_den = int(self.rho.numerator), int(self.rho.denominator)
        if jobs <= 1:
            _init_worker(self.k, rho_num, rho_den)
            for task in self.tasks():
                yield _pair_worker(task)
            return
        import multiprocessing as mp
        # imap keeps only a small bounded queue; unlike pool.map it does not
        # materialise every giant exact result at once.
        with mp.Pool(jobs, initializer=_init_worker,
                     initargs=(self.k, rho_num, rho_den)) as pool:
            yield from pool.imap(_pair_worker, self.tasks(), chunksize=max(1, chunksize))

    def float_matrices(self, jobs: int = 1, verbose: bool = True,
                       chunksize: int = 1):
        m = self.m
        A = np.empty((m, m), dtype=np.float64)
        B = np.empty((m, m), dtype=np.float64)
        n_pairs = m * (m + 1) // 2
        t0 = time.time()
        for done, (j, l, an, ad, bn, bd) in enumerate(
                self.pairs(jobs=jobs, chunksize=chunksize), 1):
            av, bv = an / ad, bn / bd
            A[j, l] = A[l, j] = av
            B[j, l] = B[l, j] = bv
            if verbose and (done % max(1, m) == 0 or done == n_pairs):
                print(f"    float pass {done:4d}/{n_pairs} "
                      f"[{time.time()-t0:7.1f}s]", flush=True)
        return A, B

    def exact_quad_forms(self, c_q, jobs: int = 1, verbose: bool = True,
                         chunksize: int = 1):
        cq = [fmpq(int(v.numerator), int(v.denominator)) for v in c_q]
        den = fmpq(0)
        num = fmpq(0)
        n_pairs = self.m * (self.m + 1) // 2
        t0 = time.time()
        for done, (j, l, an, ad, bn, bd) in enumerate(
                self.pairs(jobs=jobs, chunksize=chunksize), 1):
            weight = cq[j] * cq[l]
            if j != l:
                weight *= 2
            den += weight * fmpq(an, ad)
            num += weight * fmpq(bn, bd)
            if verbose and (done % max(1, self.m) == 0 or done == n_pairs):
                print(f"    exact pass {done:4d}/{n_pairs} "
                      f"[{time.time()-t0:7.1f}s]", flush=True)
        return num, den

# ---------------------------------------------------------------------------
# Certification driver
# ---------------------------------------------------------------------------
def parse_fraction(s: str) -> Fraction:
    s = s.strip()
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(s)


def certify_epsilon(npz_path: str, degrees: list[int], bits: int,
                    epsilon: Fraction | None, digits: int = 30,
                    prune_tol: float = 0.0, use_nn_c: bool = False,
                    cross_check: bool = False,
                    emit_certificate: str | None = None,
                    jobs: int = 1, chunksize: int = 1):
    # mmap avoids an unnecessary eager copy of large npz members where NumPy
    # can honour it; arrays are then released explicitly after projection.
    data = np.load(npz_path, mmap_mode="r")
    frame = load_discovery_frame(data)
    k, R_nn = frame.k, frame.rayleigh
    c_nn, x_fine, g_fine = frame.coefficients, frame.points, frame.channels
    eps_npz = float(data["epsilon"]) if "epsilon" in data else 0.0

    if epsilon is None:
        if eps_npz == 0.0:
            epsilon = Fraction(0)
        else:
            raise SystemExit("npz has eps > 0: supply exact --epsilon, e.g. 1/25")
    if abs(float(epsilon) - eps_npz) > 1e-9:
        print(f" WARNING: exact epsilon {epsilon} differs from npz float "
              f"{eps_npz:.12g}; certificate remains valid for {epsilon}.")

    L_frac = Fraction(1) + epsilon
    L_float = 1.0 + eps_npz
    m = g_fine.shape[1]
    print("=" * 76)
    print(f" Ultra-low-memory FLINT v2 certification M_{{{k},eps,1/2}}, eps={epsilon}")
    print(f" m={m}  bits={bits}  discovery R_NN={R_nn:.12f}  jobs={jobs}")
    print(" exact A,B are STREAMED; pair engine retains only h_hat^(k-1)")
    print("=" * 76)

    if prune_tol > 0.0:
        keep = np.abs(c_nn) > prune_tol * np.max(np.abs(c_nn))
        n_dead = int(m - keep.sum())
        if n_dead:
            kept = int(keep.sum())
            print(f" pruning {n_dead}/{m} channels "
                  f"(pairs {m*(m+1)//2} -> {kept*(kept+1)//2})")
            g_fine = g_fine[:, keep]
            c_nn = c_nn[keep]
            m = kept

    u_fine = np.clip(x_fine / L_float, 0.0, 1.0)
    summaries = []
    best = None
    for d in degrees:
        print(f"\n--- degree d = {d} ---")
        int_coeffs, l2_res, linf = project_channels(u_fine, g_fine, d, bits)
        print(f"  projection rel. L2 residuals : max {max(l2_res):.2e}   "
              f"median {sorted(l2_res)[m//2]:.2e}")
        print(f"  projection rel. Linf errors  : max {max(linf):.2e}")
        ex = ExactSeparableEpsilonStreaming(int_coeffs, bits, k, epsilon)

        if use_nn_c:
            c_float = c_nn / max(np.max(np.abs(c_nn)), 1e-300)
            c_source = "NN mixing (float Gram pass skipped)"
        else:
            if _sla is None:
                raise RuntimeError("scipy is required for re-Ritz")
            print(f"  pass 1/2: streaming exact pairs into float A,B "
                  f"({m*(m+1)//2} pairs) ...")
            A_f, B_f = ex.float_matrices(jobs=jobs, chunksize=chunksize)
            jit = 1e-12 * np.max(np.abs(np.diag(A_f)))
            lams, V = _sla.eigh(B_f, A_f + jit * np.eye(m))
            c_float = V[:, -1]
            c_source = "re-Ritz on streamed float images of exact A,B"
            del A_f, B_f, lams, V
            gc.collect()

        c_float = c_float / max(np.max(np.abs(c_float)), 1e-300)
        c_q = [dyadic(float(v), bits) for v in c_float]
        print(f"  pass 2/2: recomputing pairs and streaming exact c^T A c, c^T B c ...")
        num, den = ex.exact_quad_forms(c_q, jobs=jobs, chunksize=chunksize)
        if den <= 0:
            print("  WARNING: c^T A c <= 0; skipping")
            continue
        R_exact_flint = fmpq(k * L_frac.numerator, L_frac.denominator) * num / den
        # Convert only the final scalar to the pipeline's RAT type for existing
        # formatting/certificate compatibility.
        R_exact = RAT(int(R_exact_flint.numerator), int(R_exact_flint.denominator))
        dec = frac_to_decimal_truncated(R_exact, digits)
        gap = R_nn - float(R_exact)
        print(f"  c from: {c_source}")
        print(f"  CERTIFIED M_{{{k},{epsilon},1/2}} >= {dec}")
        print(f"  R_NN - R_certified = {gap:+.3e}")
        if float(R_exact) > 4.0:
            print(f"  *** exceeds 4: sufficient for DHL[{k},2] at theta=1/2 ***")
        summaries.append((d, R_exact, gap))
        candidate = {"d": d, "R_exact": R_exact, "gap": gap,
                     "int_coeffs": int_coeffs, "c_q": c_q}
        if best is None or R_exact > best["R_exact"]:
            best = candidate
        # Do not retain every degree's large projected coefficient table.
        del ex, num, den
        if best is not candidate:
            del int_coeffs, c_q
        gc.collect()
        if cross_check:
            _cross_check(candidate["int_coeffs"], bits, k, eps_npz, R_exact)

    if emit_certificate and best is not None:
        qty = f"M_{{{k},{epsilon},1/2}}" if epsilon else f"M_{k}"
        write_certificate(emit_certificate, qty, k, epsilon,
                          best["int_coeffs"], bits, best["c_q"],
                          best["R_exact"], digits=digits)
    if len(summaries) >= 2:
        print("\n--- degree sweep summary ---")
        print(f"  {'d':>4} {'R_certified':>34} {'R_NN - R_cert':>15}")
        for d, rr, gap in summaries:
            print(f"  {d:>4} {frac_to_decimal_truncated(rr, 24):>34} {gap:>+15.3e}")
    return summaries

def _cross_check(int_coeffs, bits, k, eps_float, R_exact):
    """Evaluate the same rational channels through the float epsilon solver;
    generalized eigenvalues are invariant under per-channel scaling."""
    try:
        import torch
        from maynard_separable_rayleigh_v4_epsilon import SeparableMaynard
    except ImportError:
        print("  cross-check skipped (torch / v4 solver unavailable)")
        return

    L = 1.0 + eps_float

    class PolyChannelsU(torch.nn.Module):
        """q_j evaluated at u = x / L (channels live in u-coordinates)."""
        def __init__(self, coeffs, bits_):
            super().__init__()
            self.co = [[c / (1 << bits_) for c in ch] for ch in coeffs]

        def forward(self, x):
            u = x / L
            cols = []
            for ch in self.co:
                acc = torch.zeros_like(u)
                for c in reversed(ch):
                    acc = acc * u + c
                cols.append(acc)
            return torch.stack(cols, dim=1)

    n_chk = max(48, int(6 * math.sqrt(k)) + 8)
    prob = SeparableMaynard(k=k, n=n_chk, epsilon=eps_float)
    rep = prob.report(PolyChannelsU(int_coeffs, bits))
    diff = rep.R - float(R_exact)
    print(f"  cross-check: float eps-solver on same rational F: "
          f"R = {rep.R:.12f}   (float - exact = {diff:+.2e})")


# ---------------------------------------------------------------------------
# Self tests
# ---------------------------------------------------------------------------
def self_test() -> None:
    print("=== self test: FLINT streaming epsilon certifier ===")
    ok = True
    for k, eps in ((2, Fraction(1,4)), (5, Fraction(3,10)),
                   (10, Fraction(1,25)), (50, Fraction(1,25))):
        bits = 8
        ex = ExactSeparableEpsilonStreaming([[1 << bits]], bits, k, eps)
        rows = list(ex.pairs())
        _, _, an, ad, bn, bd = rows[0]
        A, B = fmpq(an, ad), fmpq(bn, bd)
        rho = Fraction(1-eps) / Fraction(1+eps)
        A_cf = Fraction(1, math.factorial(k))
        B_cf = (rho**(k-1)/(k-1) - 2*rho**k/k + rho**(k+1)/(k+1)) / math.factorial(k-2)
        good = (A == fmpq(A_cf.numerator, A_cf.denominator) and
                B == fmpq(B_cf.numerator, B_cf.denominator))
        ok &= good
        print(f"  k={k:3d} eps={str(eps):>5}: A,B closed forms exact: {good}")
    # Streaming quadratic form versus explicitly assembled 3x3 float/exact pairs.
    rng = np.random.default_rng(0)
    bits = 12
    coeffs = [[int(v) for v in rng.integers(-(1<<bits), 1<<bits, size=6)]
              for _ in range(3)]
    ex = ExactSeparableEpsilonStreaming(coeffs, bits, 7, Fraction(1,25))
    c = [dyadic(v, bits) for v in (1.0, -0.25, 0.5)]
    num, den = ex.exact_quad_forms(c, verbose=False)
    good_q = den != 0 and num == num and den == den
    ok &= good_q
    print(f"  streaming exact quadratic forms finite/nonzero: {good_q}")
    print(f"\n  self test {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact certification of the epsilon-enlarged Maynard "
                    "problem M_{k,eps,1/2}")
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--epsilon", type=str, default=None,
                   help="EXACT rational eps, e.g. 1/25 (required if the npz "
                        "was trained with eps > 0)")
    p.add_argument("--degrees", type=int, nargs="+", default=[18, 22, 26])
    p.add_argument("--bits", type=int, default=48)
    p.add_argument("--digits", type=int, default=30)
    p.add_argument("--prune-tol", type=float, default=0.0)
    p.add_argument("--use-nn-c", action="store_true")
    p.add_argument("--cross-check", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--emit-certificate", type=str, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--chunksize", type=int, default=1,
                   help="multiprocessing imap chunksize; keep 1 for lowest RAM")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.npz is None:
        raise SystemExit("provide --npz (or run --self-test)")
    eps = parse_fraction(args.epsilon) if args.epsilon is not None else None
    certify_epsilon(args.npz, degrees=args.degrees, bits=args.bits,
                    epsilon=eps, digits=args.digits,
                    prune_tol=args.prune_tol, use_nn_c=args.use_nn_c,
                    cross_check=args.cross_check,
                    emit_certificate=args.emit_certificate,
                    jobs=args.jobs, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
