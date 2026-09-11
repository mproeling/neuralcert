"""
Exact rational certification for the EPSILON-ENLARGED Maynard problem
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

    R = k * L * (c^T B c) / (c^T A c)   as an exact Fraction.

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
import time
from fractions import Fraction

import numpy as np

from maynard_tools.certification.frames import load_discovery_frame

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
# Exact A, B assembly for the epsilon problem (rescaled u-coordinates)
# ---------------------------------------------------------------------------
def pair_epsilon(nj, nl, bits, k, eps: Fraction,
                 fact=None, rho_pow=None):
    """A_{jl}, B_{jl} for the epsilon problem, exact (RAT)."""
    den_ch = 1 << bits
    h = poly_mul(nj, nl)
    if fact is None:
        d2 = max(len(nj), len(nl)) * 2
        n = (2 * d2 + 2) + k + 2 * d2 * k + 4
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i
    bh = [h[r] * fact[r] for r in range(len(h))]
    P_km1 = poly_pow(bh, k - 1)
    P_k = poly_mul(P_km1, bh)

    # ---- A (identical to vanilla; running product) ----
    Rmax = len(P_k) - 1
    K = k + Rmax
    A_num = 0
    ratio = 1
    for R in range(Rmax, -1, -1):
        if P_k[R]:
            A_num += P_k[R] * ratio
        ratio *= (k + R)
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

    Mq = len(q) - 1
    qt = []
    for m_ in range(Mq + 1):
        acc = RAT(0)
        for mp in range(m_, Mq + 1):
            if q[mp] != 0:
                acc += q[mp] * math.comb(mp, m_)
        qt.append(acc if m_ % 2 == 0 else -acc)

    rho = RAT(1 - eps) / RAT(1 + eps)
    if rho_pow is None:
        max_pow = Mq + (k - 1) + (len(P_km1) - 1) + 2
        rho_pow = [RAT(1)]
        for _ in range(max_pow):
            rho_pow.append(rho_pow[-1] * rho)

    w = [qt[m_] * rho_pow[m_] for m_ in range(Mq + 1)]
    nz = [m_ for m_ in range(Mq + 1) if w[m_] != 0]

    # B = sum_R N_R rho^{k-1+R}/(k-2+R)! * sum_m w_m/(m+k-1+R)
    # with fact[k-2+R] handled by a running integer ratio to avoid per-R
    # gcd normalization of giant denominators.
    Rm1 = len(P_km1) - 1
    top = k - 2 + Rm1
    ratio = 1                                   # top!/(k-2+R)! at R = Rm1
    B_acc = RAT(0)
    for R in range(Rm1, -1, -1):
        NR = P_km1[R]
        if NR:
            base = k - 1 + R
            inner = RAT(0)
            for m_ in nz:
                inner += w[m_] / (m_ + base)
            B_acc += (NR * ratio) * rho_pow[base] * inner
        ratio *= (k - 2 + R)
    B = B_acc / (fact[top] * den_ch ** (2 * (k - 1)))
    return A, B


def _eps_worker(args):
    j, l, nj, nl, bits, k, en, ed = args
    A, B = pair_epsilon(nj, nl, bits, k, Fraction(en, ed))
    return (j, l, int(A.numerator), int(A.denominator),
            int(B.numerator), int(B.denominator))


class ExactSeparableEpsilon:
    """
    Channels q_j in Q[u] on [0,1], integer coefficient lists over a common
    denominator 2^bits, exactly as in the vanilla certifier; eps as an exact
    Fraction. A is identical to the vanilla case; B uses the rho-truncated
    powers-of-sigma formula.
    """

    def __init__(self, int_coeffs: list[list[int]], bits: int, k: int,
                 epsilon: Fraction) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        if not (0 <= epsilon < 1):
            raise ValueError("epsilon must be a Fraction in [0, 1)")
        self.k = k
        self.bits = bits
        self.int_coeffs = int_coeffs
        self.m = len(int_coeffs)
        self.eps = Fraction(epsilon)
        self.L = 1 + self.eps
        self.rho = (1 - self.eps) / (1 + self.eps)          # exact rational

        d2 = max(len(c) for c in int_coeffs) * 2
        max_fact = (2 * d2 + 2) + k + 2 * d2 * k + 4
        self._fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            self._fact[i] = self._fact[i - 1] * i

        # rho powers up to the maximal exponent m + k - 1 + R
        max_pow = (2 * d2 + 3) + (k - 1) + 2 * d2 * (k - 1) + 4
        rn, rd = self.rho.numerator, self.rho.denominator
        self._rho_pow = [RAT(1)]
        num, den = 1, 1
        for _ in range(max_pow):
            num *= rn
            den *= rd
            self._rho_pow.append(RAT(num, den))

    def _pair(self, j: int, l: int):
        return pair_epsilon(self.int_coeffs[j], self.int_coeffs[l],
                            self.bits, self.k, self.eps, self._fact,
                            self._rho_pow)

    def gram_matrices(self, verbose: bool = False, jobs: int = 1):
        m = self.m
        A = [[RAT(0)] * m for _ in range(m)]
        B = [[RAT(0)] * m for _ in range(m)]
        if jobs > 1:
            from multiprocessing import Pool
            tasks = [(j, l, self.int_coeffs[j], self.int_coeffs[l],
                      self.bits, self.k, self.eps.numerator,
                      self.eps.denominator)
                     for j in range(m) for l in range(j, m)]
            with Pool(jobs) as pool:
                for (j, l, An, Ad, Bn, Bd) in pool.map(_eps_worker, tasks):
                    A[j][l] = A[l][j] = RAT(An, Ad)
                    B[j][l] = B[l][j] = RAT(Bn, Bd)
            return A, B
        t0 = time.time()
        done = 0
        n_pairs = m * (m + 1) // 2
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
                    jobs: int = 1):
    data = np.load(npz_path)
    frame = load_discovery_frame(data)
    k, R_nn = frame.k, frame.rayleigh
    c_nn, x_fine, g_fine = frame.coefficients, frame.points, frame.channels
    eps_npz = float(data["epsilon"]) if "epsilon" in data else 0.0

    if epsilon is None:
        if eps_npz == 0.0:
            epsilon = Fraction(0)
        else:
            raise SystemExit(
                f"npz was trained at eps ~ {eps_npz:.12g}: supply the EXACT "
                f"rational via --epsilon (e.g. --epsilon 1/25); the certified "
                f"statement is about that exact functional.")
    if abs(float(epsilon) - eps_npz) > 1e-9:
        print(f" WARNING: --epsilon = {epsilon} = {float(epsilon):.12g} vs "
              f"npz eps = {eps_npz:.12g} (diff {abs(float(epsilon)-eps_npz):.2e}).")
        print(f"          The certificate is VALID for eps = {epsilon} — the "
              f"channels are merely suboptimal for it — but verify this is "
              f"what you intend.")

    L_frac = 1 + epsilon
    L_float = 1.0 + eps_npz
    m = g_fine.shape[1]

    print("=" * 72)
    print(f" Exact certification  M_{{{k},eps,1/2}}  with eps = {epsilon} "
          f"(rho = {(1-epsilon)/(1+epsilon)})")
    print(f" m = {m} channels   bits = {bits}   discovery R_NN = {R_nn:.12f}")
    print("=" * 72)

    if prune_tol > 0.0:
        keep = np.abs(c_nn) > prune_tol * np.max(np.abs(c_nn))
        n_dead = int(m - keep.sum())
        if n_dead > 0:
            kept = int(keep.sum())
            print(f" pruning {n_dead}/{m} channels with |c_j| < {prune_tol:g}"
                  f" * max|c|  (pairs: {m*(m+1)//2} -> {kept*(kept+1)//2})")
            g_fine = g_fine[:, keep]
            c_nn = c_nn[keep]
            m = kept

    # rescale the sample grid to u = x / L  in [0, 1]
    u_fine = x_fine / L_float
    u_fine = np.clip(u_fine, 0.0, 1.0)

    results = []
    for d in degrees:
        print(f"\n--- degree d = {d} ---")
        int_coeffs, l2_res, linf = project_channels(u_fine, g_fine, d, bits)
        print(f"  projection rel. L2 residuals : max {max(l2_res):.2e}   "
              f"median {sorted(l2_res)[m // 2]:.2e}")
        print(f"  projection rel. Linf errors  : max {max(linf):.2e}")

        ex = ExactSeparableEpsilon(int_coeffs, bits=bits, k=k,
                                   epsilon=epsilon)
        print(f"  assembling exact A, B ({m * (m + 1) // 2} pairs) ...")
        A, B = ex.gram_matrices(verbose=False, jobs=jobs)

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
        c_float = c_float / max(np.max(np.abs(c_float)), 1e-300)
        c_q = [dyadic(float(v), bits) for v in c_float]

        num = quad_form(B, c_q)
        den = quad_form(A, c_q)
        if den <= 0:
            print("  WARNING: c^T A c <= 0 after rationalization — skipping")
            continue
        R_exact = k * L_frac * num / den

        dec = frac_to_decimal_truncated(R_exact, digits)
        gap = R_nn - float(R_exact)
        print(f"  c from: {c_source}")
        print(f"  CERTIFIED  M_{{{k},{epsilon},1/2}} >= {dec}")
        print(f"  R_NN - R_certified = {gap:+.3e}")
        if float(R_exact) > 4.0:
            print(f"  *** exceeds 4: sufficient for DHL[{k},2] at theta=1/2 "
                  f"(Polymath8b Thm 3.12(i), eps < 1) ***")
        results.append({"d": d, "R_exact": R_exact, "gap": gap,
                        "int_coeffs": int_coeffs, "c_q": c_q})

        if cross_check:
            _cross_check(int_coeffs, bits, k, eps_npz, R_exact)

    if emit_certificate and results:
        best = max(results, key=lambda r: r["R_exact"])
        qty = f"M_{{{k},{epsilon},1/2}}" if epsilon else f"M_{k}"
        write_certificate(emit_certificate, qty, k, epsilon,
                          best["int_coeffs"], bits, best["c_q"],
                          best["R_exact"], digits=digits)
    if len(results) >= 2:
        print("\n--- degree sweep summary ---")
        print(f"  {'d':>4} {'R_certified':>34} {'R_NN - R_cert':>15}")
        for r in results:
            print(f"  {r['d']:>4} "
                  f"{frac_to_decimal_truncated(r['R_exact'], 24):>34} "
                  f"{r['gap']:>+15.3e}")
    return results


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
    print("=== self test: epsilon-certifier exact identities ===")
    ok = True

    # (1) g == 1 closed form, several (k, eps), Fraction equality
    for k, eps in ((2, Fraction(1, 4)), (5, Fraction(3, 10)),
                   (10, Fraction(1, 25)), (50, Fraction(1, 25))):
        bits = 8
        ex = ExactSeparableEpsilon([[1 << bits]], bits=bits, k=k, epsilon=eps)
        A, B = ex.gram_matrices()
        rho = (1 - eps) / (1 + eps)
        A_cf = Fraction(1, math.factorial(k))
        B_cf = (rho ** (k - 1) / (k - 1) - 2 * rho ** k / k
                + rho ** (k + 1) / (k + 1)) / math.factorial(k - 2)
        R = k * (1 + eps) * B[0][0] / A[0][0]
        R_cf = k * (1 + eps) * B_cf / A_cf
        good = (A[0][0] == A_cf and B[0][0] == B_cf and R == R_cf)
        ok &= good
        print(f"  k={k:3d} eps={str(eps):>5}: A exact {A[0][0]==A_cf}   "
              f"B exact {B[0][0]==B_cf}   R = {float(R):.10f}")

    # (2) eps = 0 must reproduce the vanilla certifier EXACTLY
    rng = np.random.default_rng(0)
    bits = 12
    coeffs = [[int(v) for v in rng.integers(-(1 << bits), 1 << bits, size=6)]
              for _ in range(3)]
    for k in (2, 3, 7):
        exv = ExactSeparable(coeffs, bits=bits, k=k)
        exe = ExactSeparableEpsilon(coeffs, bits=bits, k=k,
                                    epsilon=Fraction(0))
        Av, Bv = exv.gram_matrices(verbose=False)
        Ae, Be = exe.gram_matrices()
        same = all(Av[i][j] == Ae[i][j] and Bv[i][j] == Be[i][j]
                   for i in range(3) for j in range(3))
        ok &= same
        print(f"  eps=0 regression vs vanilla certifier, k={k}: "
              f"A,B Fraction-identical: {same}")

    # (3) float cross-check on random rational channels, k=3, eps=0.3
    try:
        import torch  # noqa: F401
        from maynard_separable_rayleigh_v4_epsilon import SeparableMaynard  # noqa: F401
        eps = Fraction(3, 10)
        exe = ExactSeparableEpsilon(coeffs, bits=bits, k=3, epsilon=eps)
        Ae, Be = exe.gram_matrices()
        A_f = np.array([[float(Ae[i][j]) for j in range(3)]
                        for i in range(3)])
        B_f = np.array([[float(Be[i][j]) for j in range(3)]
                        for i in range(3)])
        lams, V = _sla.eigh(B_f, A_f)
        c_q = [dyadic(float(v), 40) for v in V[:, -1] / np.max(np.abs(V[:, -1]))]
        R_ex = 3 * (1 + eps) * quad_form(Be, c_q) / quad_form(Ae, c_q)
        _cross_check(coeffs, bits, 3, float(eps), R_ex)
        print(f"  (exact R for that c: {float(R_ex):.12f})")
    except ImportError:
        print("  float cross-check skipped (torch unavailable)")

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
                    jobs=args.jobs)


if __name__ == "__main__":
    main()
