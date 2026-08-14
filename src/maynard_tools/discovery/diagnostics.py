"""Conditioning and dense-reference diagnostics.

These checks validate discovery computations but never claim a mathematical
certificate; rigorous exact arithmetic lives under ``certification``.
"""
from __future__ import annotations
import math
import numpy as np
import scipy.linalg
import torch

from .numerics import gauss_legendre_01


def ridge_sensitivity(A: np.ndarray, B: np.ndarray, k: int,
                      epsilons=(0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6),
                      trunc_tols=(1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)) -> dict:
    """
    (1) The residual against the UNREGULARISED A is algebraically forced:
        B c_eps - lambda_eps A c_eps = lambda_eps eps alpha c_eps, so it grows
        linearly in eps as a tautology and is NOT a stability signal. We
        report DeltaR, its first-order prediction, and the achieved backward
        error on the pencil actually solved.
    (2) There is no universal cond(A) * eps_mach noise floor for a symmetric-
        definite pencil; we report the ingredients and defer the precision
        claim to the empirical ridge-free <-> truncated agreement.
    """
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    m = A.shape[0]
    alpha = float(np.max(np.abs(np.diag(A)))) or 1.0
    w_A = np.linalg.eigvalsh(A)
    cond_A = float(w_A[-1] / max(w_A[0], 1e-300)) if w_A[0] > 0 else float("inf")
    print("\n--- Ridge sensitivity ---")
    print(f"  A: m = {m}   alpha = max|diag(A)| = {alpha:.3e}   cond(A) = {cond_A:.3e}")
    print(f"  A eigenvalues:  min = {w_A[0]:+.3e}   max = {w_A[-1]:.3e}\n")
    R_0 = None
    try:
        R_0 = k * float(scipy.linalg.eigh(B, A)[0][-1])
    except np.linalg.LinAlgError:
        pass
    print(f"  {'eps':>10} {'jitter':>11} {'R_eps':>16} {'Delta R':>12} "
          f"{'Delta R pred':>13} {'bwd err':>10} {'||c||_2':>10} {'lam1-lam2':>10}")
    rows = []
    for eps in epsilons:
        jitter = eps * alpha
        A_eps = A + jitter * np.eye(m)
        try:
            lams, V = scipy.linalg.eigh(B, A_eps)
        except np.linalg.LinAlgError as e:
            print(f"  {eps:10.0e} {jitter:11.3e}   eigh failed ({type(e).__name__})")
            continue
        lam_top = float(lams[-1])
        lam_2 = float(lams[-2]) if lams.size >= 2 else float("nan")
        c = V[:, -1]
        R_eps = k * lam_top
        cn2 = float(np.dot(c, c))
        Bc, Aec = B @ c, A_eps @ c
        bwd = float(np.linalg.norm(Bc - lam_top * Aec)) / max(
            float(np.linalg.norm(Bc)) + abs(lam_top) * float(np.linalg.norm(Aec)), 1e-300)
        dRp = -R_0 * eps * alpha * cn2 if R_0 is not None else float("nan")
        dR = (R_eps - R_0) if R_0 is not None else float("nan")
        print(f"  {eps:10.0e} {jitter:11.3e} {R_eps:16.10f} {dR:+12.2e} "
              f"{dRp:+13.2e} {bwd:10.2e} {math.sqrt(cn2):10.3e} {lam_top-lam_2:10.2e}")
        rows.append({"eps": eps, "R": R_eps, "bwd_err": bwd,
                     "cnorm": math.sqrt(cn2), "gap": lam_top - lam_2})
    print(f"\n  Truncated-basis reference (drop w_A / w_max < tol):")
    print(f"  {'tol':>10} {'rank':>6} {'R_trunc':>16}")
    wf, Uf = np.linalg.eigh(A)
    trunc = []
    for tol in trunc_tols:
        keep = wf > tol * wf.max()
        r = int(keep.sum())
        if r == 0:
            continue
        Uk = Uf[:, keep] / np.sqrt(np.clip(wf[keep], 1e-300, None))
        Bh = Uk.T @ B @ Uk
        R_tr = k * float(np.linalg.eigvalsh(0.5 * (Bh + Bh.T))[-1])
        print(f"  {tol:10.0e} {r:6d} {R_tr:16.10f}")
        trunc.append({"tol": tol, "rank": r, "R": R_tr})
    if not rows or not trunc:
        print("\n  (diagnostic inconclusive.)")
        return {"ridge": rows, "trunc": trunc}
    rf = rows[0]
    tight = next((t for t in trunc if t["rank"] == m), trunc[0])
    agree = abs(rf["R"] - tight["R"])
    print(f"\n  ridge-free R                                = {rf['R']:.12f}")
    print(f"  truncated reference R (tol={tight['tol']:.0e}, "
          f"rank={tight['rank']}/{m}) = {tight['R']:.12f}")
    print(f"  |ridge-free - truncated|                    = {agree:.2e}")
    print(f"  achieved LAPACK backward error              = {rf['bwd_err']:.2e}")
    print(f"  spectral gap lam_top - lam_second            = {rf['gap']:.2e}   "
          f"(isolated? {rf['gap'] > 1e-3})")
    rank12 = next((t["rank"] for t in trunc if t["tol"] == 1e-12), tight["rank"])
    print()
    if agree <= max(10.0 * rf["bwd_err"], 1e-12) and rf["gap"] > 1e-6:
        d = int(math.floor(-math.log10(max(agree, rf["bwd_err"], 1e-16))))
        print(f"  verdict: R = {rf['R']:.10f} — ridge-free and truncated "
              f"references agree to ~{d} digits.")
        print(f"           Empirical agreement, not a formal Stewart-Sun bound.")
        if rank12 < m:
            print(f"           NOTE: effective rank {rank12}/{m} — prune before "
                  f"certification.")
    elif agree < 1e-6 and rf["bwd_err"] < 1e-10 and rf["gap"] > 1e-6:
        d = int(math.floor(-math.log10(max(agree, 1e-16))))
        print(f"  verdict: R = {tight['R']:.10f} (truncated, rank {rank12}/{m}) "
              f"— reliable to ~{d} digits; the PARAMETERIZATION is degenerate "
              f"but the VALUE is not in doubt at ~{agree:.0e}.")
    else:
        print("  verdict: WARNING — references disagree beyond ~1e-6 or the top "
              "eigenvalue is not isolated. Orthogonalise channels, increase m, "
              "or retrain.")
    return {"ridge": rows, "trunc": trunc, "cond_A": cond_A, "agreement": agree}


class DenseCheck:
    """Tensor-product Duffy evaluation of I and J for the SAME separable
    model. Compares RATIOS only, since v9's A, B are preconditioned."""

    def __init__(self, k: int, n_quad: int, device, dtype, epsilon: float = 0.0):
        self.k, self.n_quad = k, n_quad
        self.L, self.U = 1.0 + epsilon, 1.0 - epsilon
        self.device, self.dtype = torch.device(device), dtype
        x_np, w_np = gauss_legendre_01(n_quad)
        self.x = torch.tensor(x_np, dtype=dtype, device=self.device)
        self.w = torch.tensor(w_np, dtype=dtype, device=self.device)

    def _duffy(self, u, dim):
        t = torch.empty_like(u)
        prod = torch.ones(u.shape[0], dtype=self.dtype, device=self.device)
        for i in range(dim):
            t[:, i] = prod * u[:, i]
            prod = prod * (1.0 - u[:, i])
        jac = torch.ones(u.shape[0], dtype=self.dtype, device=self.device)
        for i in range(dim - 1):
            jac = jac * (1.0 - u[:, i]) ** (dim - 1 - i)
        return t, jac

    def _grid(self, dim):
        us = torch.meshgrid(*([self.x] * dim), indexing="ij")
        ws = torch.meshgrid(*([self.w] * dim), indexing="ij")
        return (torch.stack(us, -1).reshape(-1, dim),
                torch.stack(ws, -1).prod(-1).reshape(-1))

    @staticmethod
    def _F(g, c, t):
        Bn, k = t.shape
        return g(t.reshape(-1)).reshape(Bn, k, -1).prod(dim=1) @ c

    @torch.no_grad()
    def I_and_J(self, g, c):
        k, L, U = self.k, self.L, self.U
        u, uw = self._grid(k)
        t, jac = self._duffy(u, k)
        F = self._F(g, c, L * t)
        I_val = (L ** k) * torch.dot(uw * jac, F.square())
        v, vw = self._grid(k - 1)
        ot, ojac = self._duffy(v, k - 1)
        ot = U * ot
        slack = L - ot.sum(dim=1)
        M, nq = v.shape[0], self.n_quad
        t1 = slack.unsqueeze(-1) * self.x.unsqueeze(0)
        iw = slack.unsqueeze(-1) * self.w.unsqueeze(0)
        pts = torch.cat([t1.unsqueeze(-1),
                         ot.unsqueeze(1).expand(M, nq, k - 1)], -1).reshape(-1, k)
        F = self._F(g, c, pts).reshape(M, nq)
        J_val = (U ** (k - 1)) * torch.dot(vw * ojac, (iw * F).sum(-1).square())
        return float(I_val), float(J_val)
