"""
Scalable neural Rayleigh optimisation for the Maynard / Polymath constant M_k
via SEPARABLE symmetric trial functions and 1D iterated convolutions.

v9 — the discretisation is rebuilt so that k = 1000 is a routine run with
     REALISTIC channels. All v7 stability work is preserved verbatim:
     ridge-free rank-truncated whitening; one and the same unregularised A
     in the eigensolve, the HF quotient and report(); report()-validated
     best-state snapshots; minimum effective-rank control; factorial-free
     closed form; the differentiated jittered pencil stays removed.

WHAT WAS ACTUALLY WRONG
=======================
The "positivity-preserving linear interpolation" was a symptom-level fix for
a stack of four independent defects. Measured, not guessed:

 (1) EXTRAPOLATION. Gauss-Legendre nodes are strictly interior, but the
     convolution needs nu_p at s_q u_r, whose smallest value is x_1^2 =
     O(n^-4), far below the smallest node x_1 = O(n^-2). At n = 256, 1.4% of
     targets lie below x_1 and are EXTRAPOLATED; the linear-interpolation
     matrix carries entries down to -2.3e-1. Linear interpolation removes
     Gibbs oscillation but not extrapolation, so the sign leak survives.

 (2) ACCURACY. Piecewise-linear interpolation is O(h^2), and the error
     compounds over the k convolutions. Shape error of nu_{k-1} measured
     against s^{p-1} at n = 256: 0.5% at k = 50, 7% at k = 200, 24% at
     k = 500. A is a Gram matrix only in exact arithmetic; at that error it
     is not PSD, which surfaces as "A is numerically zero after rank
     truncation".

 (3) THE RECURSION IS UNSTABLE IN THIS VARIABLE — no grid fixes this.
     A perturbation eps*s^j/j! injected at step p reappears at step K as
     eps*s^{j+K-p}/(j+K-p)!, amplified relative to the true answer by
     (K-1)!/(j+K-p)!. For K = 500 a relative 1e-16 perturbation at degree
     j = 0 injected at p = 100 arrives with relative weight 1e+246.
     Spectral interpolation only delays it: barycentric-on-Gauss is exact
     to 3e-16 at k = 10, 4e-8 at k = 50, and O(1) wrong by k = 200.
     Chebyshev-Lobatto instead of Gauss changes the constant, not the
     factorial. THE VARIABLE has to change, not the grid.

 (4) SCALE. Two outright bugs and one representational limit:
       * softplus^{-1}(r) = log(expm1(r)) overflows to +inf for r >= 710,
         i.e. for every k >= 355 with rate_cap = 2k. All parameters NaN.
       * the per-step rescaling took a GLOBAL max over all m(m+1)/2 pairs,
         so a pair whose nu is e^{-200} below the largest pair was flushed
         to round-off before it was ever integrated. Measured diag(A) at
         k = 50, m = 8: spread 8e+268, with NEGATIVE entries — for a
         quantity that is provably a positive Gram diagonal.
       * even per-pair, h = e^{-a x} spans e^{-aL}. With the physical scale
         rate ~ k, the mass of nu_{k-1} sits where phi/max = e^{-(k-1)}.
         No polynomial represents that at relative accuracy.

THE FIX: FACTOR BOTH SINGULAR STRUCTURES OUT
============================================
All the dynamic range and all the amplification live in an explicit
algebraic-times-exponential prefactor. Write h = g_j g_l = raw_j raw_l
e^{-a x} with a = rho_j + rho_l the (exactly known) envelope rates, and

    nu_p(s) = s^{p-1} e^{-a s} e^{sigma_p} psi_p(s),   |psi_p|_inf = 1,

with sigma_p and psi_p carried PER PAIR. Substituting into the truncated
convolution, the s-power comes out, the exponential comes out EXACTLY
(e^{-a s u} e^{-a s(1-u)} = e^{-a s}), and the u-dependence of the sharp
part becomes an exact Beta weight:

    psi_{p+q}(s) = B(p,q) * sum_r what_r psi_p(s u_r) psi_q(s(1-u_r)),

where (u_r, what_r) is the GAUSS-JACOBI rule for u^{p-1}(1-u)^{q-1} on
[0,1], normalised to sum what_r = 1, and log B(p,q) = lgamma(p) + lgamma(q)
- lgamma(p+q) is accumulated into sigma. Nothing overflows; psi is O(1);
and decisively the node count no longer scales with k, because the Jacobi
weight absorbs the entire k-dependent peak. The nodes migrate on their own:
for (p,q) = (998,1) they sit in [0.87, 1], for (500,500) in [0.33, 0.67].

Verified end-to-end against the closed form R = 2k/(k+1) with N = 48
representation nodes and nq = 32 Jacobi nodes:

    k        10       50      100      200      500      999     1000
    rel err  2e-16   7e-15    8e-15    4e-14    4e-14    1e-13   2e-14

and the epsilon-enlarged closed form to <= 2e-11 for eps in [0, 0.05].

TWO GRIDS, AS THEY SHOULD BE
============================
Representation and quadrature are separate objects — the distinction v7
collapsed:

  * REPRESENTATION: Chebyshev-Lobatto on [0, L], ENDPOINT-INCLUSIVE. psi is
    stored here. Every convolution target s_i u_r and s_i(1-u_r) lies in
    [0, s_i] subset [0, L], so there is NO extrapolation anywhere, and s = 0
    is a node rather than a guess. Chebyshev clustering (spacing ~
    (pi/N)^2/2 near the ends) matches the residual boundary layer of the
    raw (envelope-free) channel factors.
  * CONVOLUTION QUADRATURE: Gauss-Jacobi, nq nodes, k-independent.
  * OUTER INTEGRALS: per-pair ADAPTIVE log-space Gauss-Legendre. The
    integrand x^{k-2} e^{-a x} psi(x) W(L-x) is log-concave in its
    dominant factor; the code locates the mode x* = min(X, (k-2)/a),
    brackets it where the log-integrand has fallen by --outer-drop (default
    60, i.e. e^-60 ~ 1e-26), and lays a Gauss-Legendre rule on that window
    only. This is what lets a and k be independent: a fixed Jacobi rule for
    x^{k-2} would put its nodes near x = L while the true mode sits at
    (k-2)/a ~ 1/2 when a ~ 2k.
  * INNER integrals G(y) = int_0^y g and H(y) = int_0^y h: plain
    Gauss-Legendre, evaluated at the representation nodes and interpolated
    to the outer windows (both are smooth and saturating).
  * Clenshaw-Curtis on the representation grid, used only for the channel
    L2 normalisation.

CONDITIONING: DIAGONAL PRECONDITIONING
======================================
diag(A) spans hundreds of orders of magnitude across channels of different
envelope rate (A_jj ~ a^k/k! for slow channels, ~1 for fast ones). Since
lambda(B, A) is exactly invariant under congruence A -> D A D, B -> D B D
for any invertible diagonal D, the code forms A and B in LOG space per pair
and applies D = diag(A)^{-1/2}, so diag(A) = 1 by construction. D is
detached; because the invariance is exact, both the value and the gradient
are unchanged. This is a pure conditioning transform, not an approximation,
and it is what turns rank 1/8 into full rank at k >= 100.

Cost per Rayleigh, dominant term:
    v7:  ~ 2 log2(k) * n^3 * m(m+1)/2,          n >~ k        (and wrong)
    v9:  ~ 2 log2(k) * N^2 * nq * m(m+1)/2,     N ~ 6 sqrt(rate_raw), nq ~ 32
plus one (P, n_out, N) batched interpolation for the outer windows.

VALIDATION LADDER (nothing here replaces the exact rational certification
pipeline; this remains the discovery model):
    preflight g==1        runs on EVERY invocation, aborts on failure
    --grad-check          finite-difference test of the HF gradient
    --check-dense         dense Duffy cross-check, k <= 4
    --conv-mode sequential   k-2 multiplies instead of ~2log2(k), same psi
    --n-rep-check         representation-grid refinement on trained channels
    --outer-check         outer-window refinement (drop and node count)

Usage:
    python v9.py --k 100  --m-schedule 16,32 --iters 3000
    python v9.py --k 1000 --m-schedule 16,32 --iters 3000 --export k1000.npz
"""

from __future__ import annotations

import argparse
import math
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.special import gammaln  # roots_jacobi replaced by Golub-Welsch (overflow-safe)
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# Shared discovery building blocks.  Keeping these imports inside the discovery
# namespace enforces the architectural boundary with exact certification.
from .numerics import (
    addition_chain, barycentric_matrix, cheb_lobatto_01,
    cheb_lobatto_bary_weights, clenshaw_curtis_01, gauss_jacobi_01,
    gauss_legendre_01, linear_interp_pairs, log_beta, parse_torch_dtype,
    resolve_device, sequential_chain, torch_barycentric,
)
from .models import (
    ConstantOne, FixedDictionary, LinearX, ValidationError, channel_log_parts,
    channel_parts, inv_softplus,
)
from .diagnostics import DenseCheck, ridge_sensitivity
from .scheduling import (
    closed_form_R, make_warmup_cosine, parse_int_list, split_iters_cost_balanced,
)








# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------






















# ---------------------------------------------------------------------------
# Channel functions.  Protocol: parts(x) -> (raw, rates) with
#     g_j(x) = raw_j(x) * exp(-rates_j * x)
# The split is what lets the exponential be factored out of the convolution
# analytically instead of being resolved on a grid.
# ---------------------------------------------------------------------------




class ChannelNet(nn.Module):
    """m channel functions from one shared MLP:

        g_j(x) = net_j( x, e^{-k x} ) * exp( -softplus(rho_j) * x )

    * the feature e^{-k x} resolves the O(1/k) boundary layer where the
      optimal F concentrates at large k;
    * envelope rates are initialised log-spaced in [0.5, rate_cap];
    * grow(new_m) supports the m-continuation schedule: the first m rows of
      the grown head are EXACT copies (the grown span contains the old span,
      so the Ritz value cannot drop), the appended rows are perturbed
      duplicates.
    """

    def __init__(self, k: int, m: int = 12, hidden: int = 64, depth: int = 3,
                 rate_cap: float | None = None, positive: bool = True) -> None:
        super().__init__()
        if m < 1 or hidden < 1 or depth < 1:
            raise ValueError("m, hidden, depth must be positive")
        self.k, self.m, self.hidden = k, m, hidden
        self.positive = bool(positive)
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, m))
        self.net = nn.Sequential(*layers)
        cap = float(rate_cap) if rate_cap is not None else max(2.0 * k, 4.0)
        self.rate_cap = cap
        rates = np.geomspace(0.5, max(cap, 4.0), m)
        self.rho = nn.Parameter(torch.tensor(inv_softplus(rates)))

    def _net_out(self, x: torch.Tensor):
        feats = torch.stack([x, torch.exp(-self.k * x)], dim=-1)
        return self.net(feats)

    def log_parts(self, x: torch.Tensor):
        """Positive mode: raw_j = exp(net_j), so log raw = net directly — no
        exp/log round trip and no overflow."""
        if not self.positive:
            raise TypeError("log_parts requires positive=True")
        return self._net_out(x), nn.functional.softplus(self.rho)

    def parts(self, x: torch.Tensor):
        out = self._net_out(x)
        raw = torch.exp(out.clamp(-700.0, 700.0)) if self.positive else out
        return raw, nn.functional.softplus(self.rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, rates = self.parts(x)
        return raw * torch.exp(-x.unsqueeze(-1) * rates.unsqueeze(0))

    @torch.no_grad()
    def grow(self, new_m: int, noise: float = 0.05) -> None:
        if new_m <= self.m:
            raise ValueError(f"grow: new_m={new_m} must exceed m={self.m}")
        head = self.net[-1]
        W, b = head.weight, head.bias
        src = torch.arange(new_m, device=W.device) % self.m
        Wn, bn, rn = W[src].clone(), b[src].clone(), self.rho[src].clone()
        extra = slice(self.m, new_m)
        row_scale = Wn[extra].std(dim=1, keepdim=True).clamp_min(1e-3)
        Wn[extra] += noise * row_scale * torch.randn_like(Wn[extra])
        bn[extra] += noise * bn[extra].abs().clamp_min(1e-3) * torch.randn_like(bn[extra])
        rn[extra] += noise * torch.randn_like(rn[extra])
        new_head = nn.Linear(self.hidden, new_m, device=W.device, dtype=W.dtype)
        new_head.weight.copy_(Wn)
        new_head.bias.copy_(bn)
        self.net[-1] = new_head
        self.rho = nn.Parameter(rn)
        self.m = new_m








# ---------------------------------------------------------------------------
# The separable Maynard problem
# ---------------------------------------------------------------------------


@dataclass
class RayleighReport:
    R: float
    c: np.ndarray
    A: np.ndarray          # diagonally preconditioned: diag(A) == 1
    B: np.ndarray
    logA_diag: np.ndarray  # log of the true (unpreconditioned) diag(A)
    rank: int
    xi_span: float         # max(xi) - min(xi) over rep nodes, in nats
    xi_jump: float         # max |xi_{i+1} - xi_i| between adjacent nodes


class SeparableMaynard:
    """
    Quadratic forms of the (optionally epsilon-enlarged) Maynard problem
    M_{k,eps,1/2} with L = 1+eps, U = 1-eps:

        A_{jl} = int_0^L nu_{k-1}(x) H_{jl}(L-x) dx,  H_{jl}(y) = int_0^y g_j g_l
        B_{jl} = int_0^U nu_{k-1}(x) G_j(L-x) G_l(L-x) dx,  G_j(y) = int_0^y g_j

    with R = k lambda_max(B, A), nu_{k-1} = h^{*(k-1)} truncated to [0,L],
    h = g_j g_l. Everything is carried per pair in the fully factored variable

        nu_p(s) = s^{p-1} e^{-a s} e^{sigma_p} psi_p(s),   a = rho_j + rho_l,

    see the module docstring. A and B are returned diagonally preconditioned
    (diag(A) = 1) by a DETACHED congruence, under which lambda and its
    gradient are exactly invariant.
    """

    def __init__(self, k: int, n_rep: int = 128, n_jacobi: int = 32,
                 n_inner: int = 40, n_outer: int = 64, outer_drop: float = 60.0,
                 epsilon: float = 0.0, device="cpu",
                 dtype: torch.dtype = torch.float64,
                 ritz_dtype: torch.dtype | None = None,
                 conv_mode: str = "doubling", pair_chunk: int = 0,
                 trunc_tol: float = 1e-12,
                 channel_sign: str = "positive",
                 interp: str = "positive", a_ref: float | None = None) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        if not (0.0 <= epsilon < 1.0):
            raise ValueError("epsilon must be in [0, 1)")
        if conv_mode not in ("doubling", "sequential"):
            raise ValueError("conv_mode must be 'doubling' or 'sequential'")
        self.k, self.N, self.nq = k, n_rep, n_jacobi
        self.ng, self.no = n_inner, n_outer
        self.outer_drop = float(outer_drop)
        self.epsilon = epsilon
        self.L, self.U = 1.0 + epsilon, 1.0 - epsilon
        self.conv_mode = conv_mode
        self.pair_chunk = int(pair_chunk)
        self.trunc_tol = float(trunc_tol)
        if channel_sign not in ("positive", "free"):
            raise ValueError("channel_sign must be 'positive' or 'free'")
        self.channel_sign = channel_sign
        if interp not in ("positive", "spectral"):
            raise ValueError("interp must be 'positive' or 'spectral'")
        self.interp = interp
        self._interp_mode = interp
        self.a_ref = float(a_ref) if a_ref is not None else 4.0 * k
        self.trunc_floor = 0.0
        self.stability_tol = 1e-3
        self.stability_rtol = 2.5e-1
        self.psd_tol = 1e-8
        self.R_tol = 1e-3
        # Rigorous upper bound. The tight Polymath8b bound M_k <= k/(k-1) log k
        # applies to the VANILLA problem (eps = 0) only. The epsilon-enlarged
        # M_{k,eps,1/2} is genuinely larger — that is the whole point of the
        # enlargement — so the tight bound would reject legitimate values
        # (e.g. 4.005 at k=50, eps=1/25). For eps > 0 we fall back to the
        # Cauchy-Schwarz bound R <= k, which holds unconditionally: each
        # J_m <= I by C-S on the inner integral, so sum_m J_m <= k I.
        if epsilon > 0.0:
            self.R_bound = float(k)
        else:
            self.R_bound = (k / (k - 1.0)) * math.log(k) if k >= 2 else float("inf")
        self.max_panels = 512
        self.device = resolve_device(device) if isinstance(device, str) else torch.device(device)
        self.dtype = dtype
        self.ritz_dtype = ritz_dtype or dtype
        if self.device.type == "mps" and self.ritz_dtype == torch.float64:
            raise ValueError("Apple MPS has no float64; use --dtype float32 --ritz-dtype float32")

        L, N, nq, ng = self.L, self.N, self.nq, self.ng
        interp = self.interp
        t_np = cheb_lobatto_01(N)
        wb_np = cheb_lobatto_bary_weights(N)
        self.t_frac = torch.tensor(t_np, dtype=dtype, device=self.device)
        self.Xrep = torch.tensor(L * t_np, dtype=dtype, device=self.device)
        self.wb = torch.tensor(wb_np, dtype=dtype, device=self.device)
        self.w_cc = torch.tensor(L * clenshaw_curtis_01(N), dtype=dtype,
                                 device=self.device)

        x_gl, w_gl = gauss_legendre_01(ng)
        self.x_gl = torch.tensor(x_gl, dtype=dtype, device=self.device)
        self.w_gl = torch.tensor(w_gl, dtype=dtype, device=self.device)
        xo, wo = gauss_legendre_01(n_outer)
        self.x_out = torch.tensor(xo, dtype=dtype, device=self.device)
        self.w_out = torch.tensor(wo, dtype=dtype, device=self.device)

        # ---- convolution chain and its Jacobi rules ------------------------
        chain = (addition_chain(k - 1) if conv_mode == "doubling"
                 else sequential_chain(k - 1))
        self.chain = chain

        pt_blocks: list[np.ndarray] = [L * t_np]          # block 0: rep nodes
        cursor = N
        self._steps: list[dict] = []
        for idx, (kind, a, b) in enumerate(chain):
            u, wh = gauss_jacobi_01(nq, a, b)
            step: dict = {"kind": kind, "a": a, "b": b, "logB": log_beta(a, b),
                          "wh": torch.tensor(wh, dtype=dtype, device=self.device)}
            T1 = (t_np[:, None] * u[None, :]).reshape(-1)
            if interp == "spectral":
                step["M1"] = torch.tensor(barycentric_matrix(t_np, wb_np, T1),
                                          dtype=dtype, device=self.device)
            step["L1"] = self._lin_pack(L * T1, dtype, self.device)
            if kind == "sq":
                # Jacobi nodes for (a,a) are symmetric about 1/2, so the second
                # factor's targets are the first factor's with r reversed.
                step["sym"] = bool(np.allclose(u + u[::-1], 1.0, atol=1e-10))
                if not step["sym"]:
                    T2 = (t_np[:, None] * (1.0 - u)[None, :]).reshape(-1)
                    if interp == "spectral":
                        step["M2"] = torch.tensor(
                            barycentric_matrix(t_np, wb_np, T2),
                            dtype=dtype, device=self.device)
                    step["L2"] = self._lin_pack(L * T2, dtype, self.device)
            else:
                # 'mul': the second factor is the BASE psi_1 = raw_j raw_l.
                # Evaluate the channels exactly there rather than interpolating.
                pt_blocks.append((L * t_np[:, None] * (1.0 - u)[None, :]).reshape(-1))
                step["base_slice"] = slice(cursor, cursor + N * nq)
                cursor += N * nq
                step["sym"] = False
            self._steps.append(step)

        if chain and chain[0][0] == "sq":
            u0, _ = gauss_jacobi_01(nq, chain[0][1], chain[0][2])
            pt_blocks.append((L * t_np[:, None] * u0[None, :]).reshape(-1))
            self._steps[0]["first_a"] = slice(cursor, cursor + N * nq); cursor += N * nq
            pt_blocks.append((L * t_np[:, None] * (1.0 - u0)[None, :]).reshape(-1))
            self._steps[0]["first_b"] = slice(cursor, cursor + N * nq); cursor += N * nq
            self._steps[0]["first_exact"] = True

        # inner integrals G, H at the representation nodes:
        # G(Y_i) = Y_i sum_r w_r g(Y_i x_r); same point set serves H.
        pt_blocks.append(((L * t_np)[:, None] * x_gl[None, :]).reshape(-1))
        self._sl_inner = slice(cursor, cursor + N * ng); cursor += N * ng

        self.pts_all = torch.tensor(np.concatenate(pt_blocks), dtype=dtype,
                                    device=self.device)
        self._sl_rep = slice(0, N)
        self._pair_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # fixed scan grid for locating the outer-integral mode (detached)
        scan = np.concatenate([np.geomspace(1e-12, 1.0, 400), np.linspace(0.0, 1.0, 200)])
        scan = np.unique(np.clip(scan, 1e-14, 1.0))
        self.scan = torch.tensor(scan, dtype=dtype, device=self.device)

    # ---- pair bookkeeping ---------------------------------------------------
    def _pairs(self, m: int):
        if m not in self._pair_cache:
            iu = torch.triu_indices(m, m, device=self.device)
            self._pair_cache[m] = (iu[0], iu[1])
        return self._pair_cache[m]

    # ---- channels, L2-normalised -------------------------------------------
    def _channels(self, g: nn.Module):
        raw, rates = channel_parts(g, self.pts_all)
        env = torch.exp(-self.pts_all.unsqueeze(-1) * rates.unsqueeze(0))
        gr = raw[self._sl_rep] * env[self._sl_rep]
        norm = torch.sqrt(
            torch.einsum("q,qj->j", self.w_cc, gr.square()).clamp_min(1e-300))
        return raw / norm.unsqueeze(0), rates, env

    @staticmethod
    def _rescale(psi: torch.Tensor):
        """Divide by a DETACHED max taken PER PAIR (dim 0 only).

        v7 took a global max over all pairs, which flushed any pair that was
        orders of magnitude below the largest one to round-off before it was
        ever integrated. Detachment is what makes the scale drop out of the
        gradient exactly."""
        s = psi.detach().abs().amax(dim=0, keepdim=True).clamp_min(
            torch.finfo(psi.dtype).tiny)
        return psi / s, s.squeeze(0).log().double()

    # ---- psi recursion ------------------------------------------------------
    def _psi_power(self, raw: torch.Tensor, iu0, iu1):
        """Returns (psi at rep nodes (N,P), sigma (P,)) with
        nu_{k-1}(s) = s^{k-2} e^{-a s} e^{sigma} psi(s)."""
        N, nq = self.N, self.nq
        base = raw[self._sl_rep]
        psi1 = base[:, iu0] * base[:, iu1]
        psi1, sig1 = self._rescale(psi1)
        if self.k == 2:
            return psi1, sig1

        cur, cur_sig = psi1, sig1
        P = psi1.shape[1]
        chunk = self.pair_chunk if self.pair_chunk > 0 else P

        # psi1 is STORED rescaled: true psi_1 = e^{sig1} * psi1. Wherever the
        # base is re-evaluated exactly (the first square, and every multiply-
        # by-base) the raw channel product carries the TRUE scale, so it must
        # be divided by the same per-pair e^{sig1} before entering the
        # recursion — otherwise sig1 is counted both in the stored value and
        # again in new_sig.
        #
        # This is not a cosmetic scale slip. The spurious factor is
        # e^{sig1_{jl}} with sig1_{jl} = log max_s |raw_j(s) raw_l(s)|, which
        # is NOT of the separable form d_j + d_l. An entrywise product of a
        # Gram matrix with a positive matrix that is not rank-one is in
        # general NOT positive semidefinite — which is exactly the
        # lambda_min = -2.93 against lambda_max = +12.09 that showed up.
        inv1 = torch.exp(-sig1).to(psi1.dtype)                # (P,)

        for si, step in enumerate(self._steps):
            kind, wh = step["kind"], step["wh"]
            outs = []
            for s0 in range(0, P, chunk):
                sl = slice(s0, min(s0 + chunk, P))
                if si == 0 and step.get("first_exact", False):
                    ga = raw[step["first_a"]].reshape(N, nq, -1)
                    gb = raw[step["first_b"]].reshape(N, nq, -1)
                    f1 = (ga[..., iu0[sl]] * ga[..., iu1[sl]]) * inv1[sl]
                    f2 = (gb[..., iu0[sl]] * gb[..., iu1[sl]]) * inv1[sl]
                else:
                    f1 = (step["M1"] @ cur[:, sl]).reshape(N, nq, -1)
                    if kind == "sq":
                        f2 = f1.flip(1) if step["sym"] else \
                            (step["M2"] @ cur[:, sl]).reshape(N, nq, -1)
                    else:
                        gb = raw[step["base_slice"]].reshape(N, nq, -1)
                        f2 = (gb[..., iu0[sl]] * gb[..., iu1[sl]]) * inv1[sl]
                outs.append(torch.einsum("r,qrp,qrp->qp", wh, f1, f2))
            new = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            new_sig = (2.0 * cur_sig if kind == "sq" else cur_sig + sig1) + step["logB"]
            new, ls = self._rescale(new)
            cur, cur_sig = new, new_sig + ls
        return cur, cur_sig



    def _lin_pack(self, targets, dtype, device):
        """(i0, i1, log w0, log w1) tensors for positive linear interpolation."""
        i0, i1, w0, w1 = linear_interp_pairs(self.Xrep.detach().cpu().numpy(),
                                             np.asarray(targets, float))
        tiny = 1e-300
        return (torch.tensor(i0, dtype=torch.long, device=device),
                torch.tensor(i1, dtype=torch.long, device=device),
                torch.tensor(np.log(np.maximum(w0, tiny)), dtype=dtype, device=device),
                torch.tensor(np.log(np.maximum(w1, tiny)), dtype=dtype, device=device))

    @staticmethod
    def _lin_apply(pack, xi):
        """log( sum_i M_i psi_i ) for the two-point positive stencil.
        Linear in psi -> multilinear in (g_j, g_l) -> Gram structure kept."""
        i0, i1, lw0, lw1 = pack
        return torch.logaddexp(lw0.unsqueeze(-1) + xi[i0],
                               lw1.unsqueeze(-1) + xi[i1])

    # ---- SHARED graded outer mesh (Gram structure) -------------------------
    def _shared_outer_mesh(self, X: float):
        """One composite Gauss-Legendre rule on a geometrically graded mesh of
        [x_lo, X], used by EVERY pair.

        This is not a performance choice, it is a correctness one. A_{jl} is
        an inner product <Phi_j, Phi_l>, so A must be PSD. With the shared
        chain the discrete entry is

            A_{jl} = sum_alpha c_alpha prod_i g_j(t_i^alpha) g_l(t_i^alpha)
                   = sum_alpha c_alpha u_alpha(j) u_alpha(l),

        i.e. Gram form, and any PSD violation is bounded by the interpolation
        error and converges away under refinement. The per-pair adaptive
        windows of the first v9 draft broke exactly this: every entry used a
        different rule, so A was the Gram matrix of nothing, and the observed
        lambda_min/lambda_max sat at -1 and did NOT improve with n_rep, nq,
        n_out or drop — the signature of a structural defect rather than an
        unconverged one.

        Grading is geometric because the integrand x^{k-2} e^{-a x} has its
        mode at (k-2)/a, and a ranges over [2 min rho, 2 max rho] across
        pairs; a geometric mesh from x_lo = mode(a_ref)/32 to X resolves every
        pair's peak with the same positive-weight rule.
        """
        key = ("mesh", X)
        if key in self._pair_cache:
            return self._pair_cache[key]
        k = self.k
        # a_ref is FIXED (2 * rate_cap), NOT the current channel rates. A mesh
        # that moves with theta while autograd treats it as constant makes the
        # Hellmann-Feynman gradient inconsistent with the value — the source of
        # the ~3% finite-difference discrepancy.
        x_star = min(X, (k - 2) / max(self.a_ref, 1e-12))
        x_lo = max(x_star / 32.0, X * 1e-14)
        # Panel count must scale with k. Each panel carries the factor
        # x^(k-2), which across a panel of ratio r varies by r^(k-2); an
        # n_out-point Gauss-Legendre rule resolves exponential variation
        # e^c only for n_out >~ c/2. With a fixed 8-panel mesh (r = 1.85)
        # the variation is e^30 at k = 50 (fine), e^152 at k = 250 and
        # e^610 at k = 1000 — both far beyond a 64-point rule, and the old
        # max_panels = 24 cap prevented refinement. This error is invisible
        # to the dN gate, because dN varies only the REPRESENTATION grid.
        span = math.log(X / x_lo)
        npan_res = int(math.ceil((self.k - 2) * span / max(self.no, 1)))
        npan = max(4, int(math.ceil(span / math.log(2.0))), npan_res)
        npan = min(npan, self.max_panels)
        if npan >= self.max_panels:
            print(f"   WARNING: outer mesh hit max_panels = {self.max_panels}; "
                  f"needed {npan_res}. Raise --max-panels or --n-outer, or the "
                  f"outer quadrature is under-resolved.")
        edges = np.geomspace(x_lo, X, npan + 1)
        xg, wg = gauss_legendre_01(self.no)
        nodes = np.concatenate([e0 + (e1 - e0) * xg
                                for e0, e1 in zip(edges[:-1], edges[1:])])
        wts = np.concatenate([(e1 - e0) * wg
                              for e0, e1 in zip(edges[:-1], edges[1:])])
        # the first panel starts at x_lo > 0; add one panel covering [0, x_lo]
        nodes = np.concatenate([x_lo * xg, nodes])
        wts = np.concatenate([x_lo * wg, wts])
        out = (torch.tensor(nodes, dtype=self.dtype, device=self.device),
               torch.tensor(wts, dtype=self.dtype, device=self.device),
               self._lin_pack(nodes, self.dtype, self.device),
               self._lin_pack(np.clip(self.L - nodes, 0.0, self.L),
                              self.dtype, self.device))
        self._pair_cache[key] = out
        return out

    def _outer_integral_log_shared(self, xi, a, X, logW, iu0, iu1, pairwise,
                                   ):
        """log int_0^X x^{k-2} e^{-a x} psi(x) W(L-x) dx on the SHARED mesh."""
        k, L = self.k, self.L
        nodes, wts, packx, packy = self._shared_outer_mesh(X)
        if self.interp == "positive":
            xi_w = self._lin_apply(packx, xi)
            lWraw = self._lin_apply(packy, logW)
        else:
            xi_w = torch_barycentric(self.Xrep, self.wb, nodes) @ xi
            lWraw = torch_barycentric(self.Xrep, self.wb,
                                      (L - nodes).clamp(0.0, L)) @ logW
        ly = (L - nodes).clamp_min(1e-300).log().unsqueeze(-1)   # (Q,1)
        if pairwise:
            lW = lWraw + ly                      # H_{jl}(L-x) = (L-x) Hhat
        else:
            lW = lWraw[:, iu0] + lWraw[:, iu1] + 2.0 * ly
        ell = ((k - 2) * nodes.clamp_min(1e-300).log().unsqueeze(-1)
               - nodes.unsqueeze(-1) * a.unsqueeze(0)
               + xi_w + lW + wts.clamp_min(1e-300).log().unsqueeze(-1))
        return torch.logsumexp(ell, dim=0)

    # ---- log-space psi recursion (sign-definite channels) ------------------
    def _xi_power(self, log_raw: torch.Tensor, iu0, iu1):
        """Returns xi = log psi at the representation nodes, (N, P).

        Identical mathematics to _psi_power, carried in the log. The Jacobi
        weights what_r are strictly positive, so

            xi_{p+q}(s) = log B(p,q)
                        + logsumexp_r [ log what_r + xi_p(s u_r) + xi_q(s(1-u_r)) ]

        has no cancellation. This is what removes the (min psi_1/max psi_1)^k
        collapse: psi_{p+q}(0) = B psi_p(0) psi_q(0) means RATIOS MULTIPLY, so
        any non-constant psi_1 drives psi past the float64 relative floor by
        k ~ 50-100. What is left there is round-off, and since it enters A
        with random sign it destroys the Gram structure (measured lam_min =
        -2.93 vs lam_max = +12.09 at k=100). In the log there is no floor:
        xi simply becomes large and negative, and interpolating xi to
        absolute accuracy 1e-16*|xi| IS relative accuracy on psi.

        No rescaling is needed or performed — a log cannot overflow — so the
        returned sigma is identically zero and the scale lives in xi itself.
        """
        N, nq = self.N, self.nq
        lr = log_raw[self._sl_rep]
        xi = lr[:, iu0] + lr[:, iu1]
        if self.k == 2:
            return xi
        P = xi.shape[1]
        chunk = self.pair_chunk if self.pair_chunk > 0 else P

        for si, step in enumerate(self._steps):
            kind = step["kind"]
            logw = step["wh"].clamp_min(1e-300).log()
            outs = []
            for s0 in range(0, P, chunk):
                sl = slice(s0, min(s0 + chunk, P))
                if si == 0 and step.get("first_exact", False):
                    la = log_raw[step["first_a"]].reshape(N, nq, -1)
                    lb = log_raw[step["first_b"]].reshape(N, nq, -1)
                    e1 = la[..., iu0[sl]] + la[..., iu1[sl]]
                    e2 = lb[..., iu0[sl]] + lb[..., iu1[sl]]
                else:
                    if self.interp == "positive":
                        e1 = self._lin_apply(step["L1"], xi[:, sl]).reshape(N, nq, -1)
                    else:
                        e1 = (step["M1"] @ xi[:, sl]).reshape(N, nq, -1)
                    if kind == "sq":
                        if step["sym"]:
                            e2 = e1.flip(1)
                        elif self.interp == "positive":
                            e2 = self._lin_apply(step["L2"], xi[:, sl]).reshape(N, nq, -1)
                        else:
                            e2 = (step["M2"] @ xi[:, sl]).reshape(N, nq, -1)
                    else:
                        lb = log_raw[step["base_slice"]].reshape(N, nq, -1)
                        e2 = lb[..., iu0[sl]] + lb[..., iu1[sl]]
                outs.append(torch.logsumexp(
                    logw.view(1, nq, 1) + e1 + e2, dim=1) + step["logB"])
            xi = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return xi

    def _outer_integral_log(self, xi, a, X, logW, iu0, iu1, pairwise):
        """log int_0^X x^{k-2} e^{-a x} psi(x) W(L-x) dx, per pair.

        Every factor is positive, so the integrand is carried in the log and
        summed by logsumexp: no cancellation. Note this makes A positive
        ENTRYWISE, which does NOT imply PSD ([[1,2],[2,1]] is positive and
        indefinite). PSD comes only from the Gram structure, i.e. from using
        one shared rule with NONNEGATIVE interpolation weights."""
        k, L = self.k, self.L
        nodes, wts = self._outer_window(a, X, xi=xi)
        Mx = torch_barycentric(self.Xrep, self.wb, nodes)
        xi_w = torch.einsum("pqi,ip->pq", Mx, xi)
        My = torch_barycentric(self.Xrep, self.wb, (L - nodes).clamp(0.0, L))
        if pairwise:
            lW = torch.einsum("pqi,ip->pq", My, logW)
        else:
            lWj = torch.einsum("pqi,ij->pqj", My, logW)
            lW = lWj.gather(2, iu0.view(-1, 1, 1).expand(-1, self.no, 1)).squeeze(2) + \
                lWj.gather(2, iu1.view(-1, 1, 1).expand(-1, self.no, 1)).squeeze(2)
        ell = ((k - 2) * nodes.clamp_min(1e-300).log() - a.unsqueeze(1) * nodes
               + xi_w + lW + wts.clamp_min(1e-300).log())
        return torch.logsumexp(ell, dim=1)

    def _log_inner(self, log_raw, rates, iu0, iu1):
        """log G_j and log H_jl at the representation nodes, in log space.
        G_j(Y) = Y sum_v w_v raw_j(Y x_v) e^{-rho_j Y x_v}, all terms > 0."""
        N, ng = self.N, self.ng
        lr = log_raw[self._sl_inner].reshape(N, ng, -1)
        pts = self.pts_all[self._sl_inner].reshape(N, ng)
        lw = self.w_gl.clamp_min(1e-300).log().view(1, ng, 1)
        lY = self.Xrep.clamp_min(1e-300).log().unsqueeze(-1)
        expo = -pts.unsqueeze(-1) * rates.view(1, 1, -1)
        # Return log Ghat and log Hhat with the LINEAR ZERO FACTORED OUT:
        #     G(y) = y * Ghat(y),   Ghat(y) = int_0^1 g(y v) dv,
        # and likewise for H. This matters because these get interpolated to
        # the outer mesh: log G has a logarithmic singularity at y = 0 (G(0)=0
        # -> the clamp puts -690 at the first representation node), and
        # polynomial interpolation through that oscillates wildly across the
        # whole interval. Ghat and Hhat are smooth, positive and O(1), so
        # their logs interpolate cleanly; the log y is re-attached
        # analytically at the outer nodes, where y = L - x > 0.
        del lY
        logGhat = torch.logsumexp(lw + lr + expo, dim=1)
        lh = lr[..., iu0] + lr[..., iu1]
        eh = -pts.unsqueeze(-1) * (rates[iu0] + rates[iu1]).view(1, 1, -1)
        logHhat = torch.logsumexp(lw + lh + eh, dim=1)
        return logGhat, logHhat

    def _gram_log(self, g: nn.Module):
        """A, B assembled entirely in log space from sign-definite channels."""
        k, L, U = self.k, self.L, self.U
        log_raw, rates = channel_log_parts(g, self.pts_all)
        # L2 normalisation of g_j over [0,L], applied in the log
        gr2 = 2.0 * (log_raw[self._sl_rep]
                     - self.Xrep.unsqueeze(-1) * rates.unsqueeze(0))
        lognorm = 0.5 * torch.logsumexp(
            self.w_cc.clamp_min(1e-300).log().unsqueeze(-1) + gr2, dim=0)
        log_raw = log_raw - lognorm.unsqueeze(0)

        m = log_raw.shape[1]
        iu0, iu1 = self._pairs(m)
        a = rates[iu0] + rates[iu1]
        xi = self._xi_power(log_raw, iu0, iu1)
        logG, logH = self._log_inner(log_raw, rates, iu0, iu1)

        logA = self._outer_integral_log_shared(xi, a, L, logH, iu0, iu1, True)
        logB_ = self._outer_integral_log_shared(xi, a, U, logG, iu0, iu1, False)

        diag_pos = self._diag_positions(m, iu0, iu1)
        logAd = logA[diag_pos]
        half = 0.5 * (logAd[iu0] + logAd[iu1]).detach()
        Avec = torch.exp((logA - half).clamp(-700.0, 700.0))
        Bvec = torch.exp((logB_ - half).clamp(-700.0, 700.0))

        Af = Avec.new_zeros(m, m); Bf = Bvec.new_zeros(m, m)
        Af[iu0, iu1] = Avec; Bf[iu0, iu1] = Bvec
        A = Af + Af.T - torch.diag_embed(Af.diagonal())
        B = Bf + Bf.T - torch.diag_embed(Bf.diagonal())
        return A, B, logAd.detach()

    def _diag_positions(self, m, iu0, iu1):
        key = ("diag", m)
        if key not in self._pair_cache:
            pos = torch.zeros(m, dtype=torch.long, device=self.device)
            for j in range(m):
                pos[j] = ((iu0 == j) & (iu1 == j)).nonzero()[0, 0]
            self._pair_cache[key] = pos
        return self._pair_cache[key]

    # ---- adaptive outer window ---------------------------------------------
    def _outer_window(self, a: torch.Tensor, X: float, xi=None):
        """Per-pair Gauss-Legendre window for int_0^X x^{k-2} e^{-a x} (...) dx.

        The log of the dominant factor, ell(x) = (k-2) ln x - a x, is concave
        with mode x* = min(X, (k-2)/a). The window is the bracket where ell
        has fallen by outer_drop from its maximum, located on a fixed scan
        grid. Everything here is DETACHED: it selects quadrature nodes, it
        does not enter the value.
        """
        k = self.k
        xs = (self.scan * X).clamp_min(1e-300)                    # (S,)
        ell = (k - 2) * xs.log().unsqueeze(0) - a.unsqueeze(1) * xs.unsqueeze(0)
        if xi is not None:
            # include psi in the mode search: for stiff channels xi shifts the
            # peak appreciably, and a window centred on the wrong x* silently
            # integrates the tail instead of the bulk.
            Ms = torch_barycentric(self.Xrep, self.wb,
                                   xs.unsqueeze(0).expand(a.shape[0], -1))
            ell = ell + torch.einsum("pqi,ip->pq", Ms, xi.detach())
        ell = ell.detach()
        emax = ell.amax(dim=1, keepdim=True)
        ok = ell > emax - self.outer_drop
        idx = torch.arange(xs.numel(), device=xs.device).unsqueeze(0)
        lo_i = torch.where(ok, idx, torch.full_like(idx, xs.numel() - 1)).amin(dim=1)
        hi_i = torch.where(ok, idx, torch.zeros_like(idx)).amax(dim=1)
        lo = xs[lo_i]
        hi = xs[hi_i]
        # widen by one scan cell each way, clip to [0, X]
        lo = (lo * 0.5).clamp_min(0.0)
        hi = torch.minimum(hi * 1.5, torch.full_like(hi, X))
        nodes = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * self.x_out.unsqueeze(0)
        wts = (hi - lo).unsqueeze(1) * self.w_out.unsqueeze(0)
        return nodes.detach(), wts.detach()

    def _outer_integral(self, psi, sigma, a, X, W_at_rep, iu0, iu1, pairwise):
        """int_0^X x^{k-2} e^{-a x} psi(x) W(L-x) dx, per pair, in log space.

        Returns (mantissa (P,), log_shift (P,)) with the true value
        = mantissa * exp(sigma + log_shift). The shift is the detached
        maximum of the log-integrand core, so the mantissa is O(1)."""
        k, L = self.k, self.L
        nodes, wts = self._outer_window(a, X)                     # (P, no)
        ell = (k - 2) * nodes.clamp_min(1e-300).log() - a.unsqueeze(1) * nodes
        shift = ell.detach().amax(dim=1)                          # (P,)
        core = torch.exp(ell - shift.unsqueeze(1))                # (P, no) O(1)

        Mx = torch_barycentric(self.Xrep, self.wb, nodes)         # (P, no, N)
        psi_w = torch.einsum("pqi,ip->pq", Mx, psi)
        My = torch_barycentric(self.Xrep, self.wb, (L - nodes).clamp(0.0, L))
        if pairwise:
            W_w = torch.einsum("pqi,ip->pq", My, W_at_rep)
        else:
            Wj = torch.einsum("pqi,ij->pqj", My, W_at_rep)
            W_w = Wj.gather(2, iu0.view(-1, 1, 1).expand(-1, self.no, 1)).squeeze(2) * \
                Wj.gather(2, iu1.view(-1, 1, 1).expand(-1, self.no, 1)).squeeze(2)
        val = (wts * core * psi_w * W_w).sum(dim=1)
        return val, shift

    # ---- A, B assembly ------------------------------------------------------
    def gram_matrices(self, g: nn.Module):
        """Returns (A, B, logA_diag) with A, B DIAGONALLY PRECONDITIONED so
        diag(A) = 1. The congruence is detached, and lambda(B,A) is exactly
        invariant under it, so both value and gradient are exact."""
        if self.channel_sign == "positive":
            return self._gram_log(g)
        k, N, ng, L, U = self.k, self.N, self.ng, self.L, self.U
        raw, rates, env = self._channels(g)
        m = raw.shape[1]
        iu0, iu1 = self._pairs(m)
        a = rates[iu0] + rates[iu1]                                # (P,)

        psi, sigma = self._psi_power(raw, iu0, iu1)                # (N,P), (P,)

        # inner integrals at the representation nodes
        gi = (raw * env)[self._sl_inner].reshape(N, ng, -1)        # (N, ng, m)
        Grep = self.Xrep.unsqueeze(-1) * torch.einsum("v,qvj->qj", self.w_gl, gi)
        hi_ = gi[..., iu0] * gi[..., iu1]
        Hrep = self.Xrep.unsqueeze(-1) * torch.einsum("v,qvp->qp", self.w_gl, hi_)

        A_val, A_shift = self._outer_integral(psi, sigma, a, L, Hrep, iu0, iu1, True)
        B_val, B_shift = self._outer_integral(psi, sigma, a, U, Grep, iu0, iu1, False)

        logA = sigma + A_shift + A_val.detach().abs().clamp_min(1e-300).log()
        # diagonal preconditioner from the true log-diagonal of A
        diag_pos = torch.zeros(m, dtype=torch.long, device=self.device)
        for j in range(m):
            diag_pos[j] = ((iu0 == j) & (iu1 == j)).nonzero()[0, 0]
        logAd = logA[diag_pos]                                     # (m,)
        half = 0.5 * (logAd[iu0] + logAd[iu1])                     # (P,)

        fA = torch.exp((sigma + A_shift - half).clamp(-700.0, 700.0)).detach()
        fB = torch.exp((sigma + B_shift - half).clamp(-700.0, 700.0)).detach()
        Avec = A_val * fA
        Bvec = B_val * fB

        Afull = Avec.new_zeros(m, m)
        Bfull = Bvec.new_zeros(m, m)
        Afull[iu0, iu1] = Avec
        Bfull[iu0, iu1] = Bvec
        A = Afull + Afull.T - torch.diag_embed(Afull.diagonal())
        B = Bfull + Bfull.T - torch.diag_embed(Bfull.diagonal())
        return A, B, logAd.detach()

    # ---- ridge-free rank-truncated top eigenpair (no autograd) -------------
    def _solve_top(self, A: torch.Tensor, B: torch.Tensor, ridge: float = 0.0):
        """Ridge-free rank-truncated whitening for B v = lambda A v.
        The same unregularised A is used by the eigensolve, the HF quotient
        and report(). Directions below trunc_tol * lambda_max(A) are dropped."""
        del ridge  # kept for CLI compatibility; v7 removed the ridge entirely
        As = 0.5 * (A + A.T)
        Bs = 0.5 * (B + B.T)
        tiny = torch.finfo(As.dtype).tiny
        wA, UA = torch.linalg.eigh(As)
        wmax = wA.max()
        if not torch.isfinite(wmax) or wmax <= 0:
            raise FloatingPointError("A has no positive numerical direction")
        keep = wA > max(self.trunc_tol, self.trunc_floor) * wmax
        r = int(keep.sum().item())
        if r == 0:
            raise FloatingPointError("A is numerically zero after rank truncation")
        Uk = UA[:, keep] / wA[keep].clamp_min(tiny).sqrt().unsqueeze(0)
        Bh = Uk.T @ Bs @ Uk
        Bh = 0.5 * (Bh + Bh.T)
        lams, V = torch.linalg.eigh(Bh)
        v = Uk @ V[:, -1]
        den = v @ (As @ v)
        if not torch.isfinite(den) or den <= 0:
            raise FloatingPointError("non-positive A norm of Ritz vector")
        return lams[-1], v / den.sqrt(), r

    # ---- Rayleigh value + gradient -----------------------------------------
    def rayleigh(self, g: nn.Module, ridge: float = 0.0, grad_mode: str = "hf",
                 frozen_v: torch.Tensor | None = None) -> torch.Tensor:
        """R = k lambda_max(B, A) by Hellmann-Feynman / Danskin: solve the
        m x m pencil outside autograd, detach v, return k (v'Bv)/(v'Av).
        Value exactly k*lambda at the current point; gradient exact by the
        envelope theorem; a minorant of k*lambda_max for every theta, hence
        safe under a line search."""
        if grad_mode == "backprop":
            raise NotImplementedError(
                "the differentiated jittered pencil is inconsistent with the "
                "ridge-free HF route and was removed in v7; use --grad-mode hf")
        A, B, _ = self.gram_matrices(g)
        if frozen_v is None:
            with torch.no_grad():
                _, v, _ = self._solve_top(A.detach().to(self.ritz_dtype),
                                          B.detach().to(self.ritz_dtype))
            v = v.to(dtype=A.dtype)
        else:
            v = frozen_v
        v = v.detach()
        return self.k * (v @ (B @ v)) / (v @ (A @ v))

    @torch.no_grad()
    def report(self, g: nn.Module, ridge: float = 0.0) -> RayleighReport:
        A, B, logAd = self.gram_matrices(g)
        Ar, Br = A.to(self.ritz_dtype), B.to(self.ritz_dtype)
        # --- gate 1: A must be numerically PSD. A_{jl} = <Phi_j, Phi_l>, so
        # any eigenvalue materially below zero means the assembly is not a
        # Gram matrix. Rank truncation can DISCARD small directions; it cannot
        # repair a spectrum with a genuine negative branch.
        wA = torch.linalg.eigvalsh(0.5 * (Ar + Ar.T))
        if wA[-1] <= 0 or wA[0] < -self.psd_tol * wA[-1]:
            raise ValidationError(
                f"A is not numerically PSD: lambda_min/lambda_max = "
                f"{float(wA[0]/wA[-1]):.3e} (tolerance -{self.psd_tol:.0e}). "
                f"A is an inner-product Gram matrix, so this is a defect in "
                f"the discretisation, not round-off. With --interp spectral "
                f"the barycentric weights are signed and the Gram structure "
                f"is not preserved; use --interp positive.")
        # --- gate 1b: PER-PAIR bound. F = prod_i g_j(t_i) is itself an
        # admissible trial function, so k B_jj / A_jj obeys the same bound as
        # R. This LOCALIZES a violation: if some diagonal is absurd the fault
        # is in that pair's assembly (convolution / outer integral); if every
        # diagonal is sane but the global R is not, the fault is in the
        # pencil, i.e. off-diagonal entries or A's near-null directions.
        dR = self.k * Br.diagonal() / Ar.diagonal().clamp_min(1e-300)
        j_bad = int(torch.argmax(dR))
        if float(dR[j_bad]) > self.R_bound * (1.0 + self.R_tol):
            raise ValidationError(
                f"diagonal pair {j_bad} gives k B_jj/A_jj = "
                f"{float(dR[j_bad]):.6f} > bound {self.R_bound:.6f}. That "
                f"single-channel trial function is admissible, so this is a "
                f"defect in the assembly of THIS pair, not in the pencil.")
        self.last_diag_R = float(dR.max())
        _, v, rank = self._solve_top(Ar, Br)
        R = self.k * (v @ (Br @ v)) / (v @ (Ar @ v))
        # --- gate 1c: RANK STABILITY. trunc_tol is calibrated to machine
        # epsilon, but A and B are only as accurate as the discretisation —
        # measured 1.1e-3 for the k=250 excursion state. Whitening divides by
        # sqrt(lambda), so a direction with lambda/lambda_max ~ 1e-6 amplifies
        # B's assembly error by ~1e3 and the Rayleigh quotient becomes a
        # statement about noise. Dissection of that state: R = 56.32 at
        # trunc_tol 1e-6 (rank 28) and R = 2.04 at 1e-4 (rank 19), while every
        # DIAGONAL k B_jj/A_jj stayed at 1.99 — the assembly was fine, the
        # pencil was not. A healthy state loses only 0.04% under the same
        # raise. So: require the value to survive truncation at stability_tol.
        # Compare the WORKING (floored) value R against a coarser truncation
        # at stability_tol. A genuine excursion into A's near-null space shows
        # a GROSS gap here (measured 28x and 10x on the two bad k=250/k=1000
        # states); an honest weak-channel state wobbles ~10%. So fire only on
        # a large relative discrepancy, and always surface the gap so it is
        # visible in the log before it becomes fatal.
        wS = torch.linalg.eigvalsh(0.5 * (Ar + Ar.T))
        keepS = wS > self.stability_tol * wS.max()
        R_stab = float(R)
        if int(keepS.sum()) >= 1:
            _, US = torch.linalg.eigh(0.5 * (Ar + Ar.T))
            Uk = US[:, keepS] / wS[keepS].clamp_min(
                torch.finfo(Ar.dtype).tiny).sqrt().unsqueeze(0)
            Bh = Uk.T @ (0.5 * (Br + Br.T)) @ Uk
            R_stab = self.k * float(torch.linalg.eigvalsh(0.5 * (Bh + Bh.T))[-1])
        self.last_R_stab = R_stab
        rel_gap = abs(float(R) - R_stab) / max(abs(R_stab), 1e-30)
        self.last_rank_gap = rel_gap
        if rel_gap > self.stability_rtol:
            raise ValidationError(
                f"R is not rank-stable: {float(R):.6f} at floor "
                f"{max(self.trunc_tol, self.trunc_floor):.0e} (rank {rank}) "
                f"vs {R_stab:.6f} at {self.stability_tol:.0e} "
                f"(rank {int(keepS.sum())}); relative gap {rel_gap:.0%} > "
                f"{self.stability_rtol:.0%}. The value leans on directions "
                f"where A is below its own assembly accuracy — noise, not "
                f"M_k. Raise --trunc-floor or --n-rep.")
        # --- gate 2: M_k <= k for EVERY admissible F. By Cauchy-Schwarz on the
        # inner integral, (int F dt_m)^2 <= (1-Sigma) int F^2 dt_m <= int F^2,
        # so J^{(m)} <= I and sum_m J^{(m)} <= k I. R > k is therefore
        # impossible and proves the pencil has stopped representing the
        # variational problem.
        if not math.isfinite(float(R)) or float(R) > self.R_bound * (1.0 + self.R_tol):
            raise ValidationError(
                f"R = {float(R):.6f} exceeds the rigorous upper bound "
                f"M_k <= (k/(k-1)) log k = {self.R_bound:.6f} (Polymath8b; "
                f"tol {self.R_tol:.0e}). Every admissible trial function "
                f"satisfies R <= M_k, so the discretisation is being "
                f"exploited, not the variational problem solved. Raise "
                f"--n-rep and rerun.")
        if self.channel_sign == "positive":
            log_raw, _ = channel_log_parts(g, self.pts_all)
            iu0, iu1 = self._pairs(log_raw.shape[1])
            xi = self._xi_power(log_raw, iu0, iu1)
        else:
            raw, rates, _ = self._channels(g)
            iu0, iu1 = self._pairs(raw.shape[1])
            psi, _ = self._psi_power(raw, iu0, iu1)
            xi = psi.abs().clamp_min(1e-300).log()
        # In the log-space formulation a large span of xi is EXPECTED and
        # harmless — the log is the representation. What measures resolution
        # is local smoothness of xi against the grid spacing, and above all
        # the refinement gate on R itself.
        xi_span = float(xi.amax() - xi.amin())
        xi_jump = float((xi[1:, :] - xi[:-1, :]).abs().amax()) if xi.shape[0] > 1 else 0.0
        return RayleighReport(R=float(R), c=v.cpu().numpy(),
                              A=Ar.cpu().numpy(), B=Br.cpu().numpy(),
                              logA_diag=logAd.cpu().numpy(), rank=rank,
                              xi_span=xi_span, xi_jump=xi_jump)


# ---------------------------------------------------------------------------
# Ridge sensitivity diagnostic (unchanged in substance; A, B arrive
# preconditioned, under which R, gaps and backward errors are invariant)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dense Duffy cross-check (small k only)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(problem: SeparableMaynard, g: nn.Module, iters: int, lr: float,
          warmup_frac: float, grad_clip: float, log_every: int,
          weight_decay: float = 0.0, grad_mode: str = "hf",
          stage_label: str = "", min_rank_frac: float = 0.25,
          problem_hi: SeparableMaynard | None = None,
          snap_rtol: float = 2e-2, max_violations: int = 3,
          dump_violation: str | None = None, patience: int = 0,
          excursion_factor: float = 3.0, excursion_floor: float = 2.0):
    """Adam with report()-validated best-state snapshots. A state is eligible
    as 'best' only after report() independently agrees with the training
    value and the retained numerical rank is adequate."""
    g.to(device=problem.device, dtype=problem.dtype)
    opt = torch.optim.Adam(g.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_warmup_cosine(max(int(iters * warmup_frac), 50), iters))
    best_R, best_iter = -math.inf, 0
    best_state = deepcopy(g.state_dict())
    best_any_R, best_any_iter = -math.inf, 0
    best_any_state = None
    last_good = deepcopy(g.state_dict())
    strikes = 0
    stale = 0
    t0 = time.time()

    # Validation happens BEFORE the optimizer step, so that the HF quotient
    # and report() are evaluated at the SAME parameters. v7/v8 read R_train
    # off the pre-step forward pass and compared it with a report() taken
    # after opt.step(): two different model states, so `agree` measured the
    # size of the Adam update rather than solver consistency, and almost
    # every state was rejected. That is why the k=100 run kept falling back
    # to iteration 700 — the one place the two states happened to coincide.
    for it in range(iters + 1):
        if it % max(log_every, 1) == 0 or it == iters:
            try:
                with torch.no_grad():
                    R_hf = float(problem.rayleigh(g, grad_mode=grad_mode))
                rep = problem.report(g)
            except ValidationError as e:
                # The optimizer maximizes R_discrete = R_true + error(theta)
                # and will find the channels where error(theta) is largest, so
                # a bound violation usually means the trajectory drifted into
                # discretisation-exploiting territory — not that the run is
                # void. Strike policy: no snapshot, keep training briefly in
                # case Adam wanders back, stop the STAGE (restoring the best
                # validated state) after max_violations consecutive strikes so
                # the pipeline still reaches the polish and refinement gate.
                strikes += 1
                print(f"  adam{stage_label} iter {it:5d}   BOUND VIOLATION "
                      f"({strikes}/{max_violations}): {e}")
                if dump_violation and strikes == 1:
                    try:
                        torch.save({"state_dict": g.state_dict(),
                                    "iter": it, "m": g.m,
                                    "message": str(e)}, dump_violation)
                        print(f"  offending channel state written to "
                              f"{dump_violation} for offline dissection")
                    except Exception as ex:
                        print(f"  (could not write dump: {ex})")
                if strikes >= max_violations:
                    g.load_state_dict(best_state if best_R > -math.inf
                                      else last_good)
                    print(f"  adam{stage_label} stopping stage early; restored "
                          f"best validated R = {best_R:.10f}. Raise --n-rep to "
                          f"push the exploitable error below the bound margin.")
                    break
                rep = None
            except FloatingPointError as e:
                print(f"  adam{stage_label} iter {it:5d}   validation failed: {e}")
                rep = None
            if rep is not None:
                # NB: do NOT reset strikes here. Resetting before the
                # excursion check below made the counter read 1/3 forever, so
                # max_violations never fired and stages ran their full budget
                # inside an excursion. strikes is cleared only when a point is
                # actually ACCEPTED (below).
                # Excursion guard. For eps > 0 the only rigorous bound is
                # R <= k, which at k = 250 is far too loose to catch a 5 -> 70
                # discretisation excursion. This is an explicitly HEURISTIC
                # relative guard: a genuine optimisation step does not
                # multiply the validated best several-fold.
                abs_cap = problem.R_bound * (1.0 + problem.R_tol)
                over_abs = rep.R > abs_cap
                # the relative multiple is only meaningful once the incumbent
                # is itself a sane value; anchoring it to a bad early R (e.g.
                # 0.38 at rank 10/16, still converging) rejects legitimate
                # climbs. Gate it on best_R exceeding the dictionary baseline.
                rel_anchor = max(best_R, excursion_floor)
                over_rel = (best_R > excursion_floor and excursion_factor > 0
                            and rep.R > excursion_factor * rel_anchor)
                if over_abs or over_rel:
                    strikes += 1
                    why = (f"R = {rep.R:.4f} > bound {abs_cap:.4f}" if over_abs
                           else f"R = {rep.R:.4f} is {rep.R/rel_anchor:.1f}x the "
                                f"validated best {best_R:.6f}")
                    print(f"  adam{stage_label} iter {it:5d}   EXCURSION "
                          f"({strikes}/{max_violations}): {why}.")
                    if dump_violation and strikes == 1:
                        try:
                            torch.save({"state_dict": g.state_dict(),
                                        "iter": it, "m": g.m,
                                        "R": rep.R, "best_R": best_R},
                                       dump_violation)
                            print(f"  offending state written to {dump_violation}")
                        except Exception as ex:
                            print(f"  (could not write dump: {ex})")
                    if strikes >= max_violations:
                        g.load_state_dict(best_state)
                        print(f"  adam{stage_label} stopping stage; restored "
                              f"best validated R = {best_R:.10f}.")
                        break
                    continue
                strikes = 0          # this point is plausible; clear strikes
                agree = abs(R_hf - rep.R) <= 1e-4 * max(1.0, abs(rep.R))
                rank_ok = rep.rank >= max(1, math.ceil(min_rank_frac * g.m))
                # refinement stability: a state is snapshot-eligible only if
                # its R is reproduced on a 1.5x finer representation grid.
                # This is what prevents "best" from ever meaning "the state
                # whose discretisation error the optimizer liked most".
                # snap_rtol is a LOOSE anti-exploitation cap, not a
                # convergence criterion: at N = 512, k = 50 the honest
                # trained-channel error is a few 1e-3, and demanding dN below
                # that rejects every legitimate state (which is exactly what
                # crashed the first N=512 run). Convergence is judged once, at
                # the end, by Richardson extrapolation — not per snapshot.
                refine_ok, ref_note = True, ""
                score = rep.R
                if problem_hi is not None:
                    try:
                        R_hi = problem_hi.report(g).R
                        drel = abs(R_hi - rep.R) / max(abs(R_hi), 1e-30)
                        refine_ok = drel <= snap_rtol
                        # rank states by the WORSE of the two grids: inflating
                        # either grid's discretisation error cannot then
                        # improve a state's score.
                        score = min(rep.R, R_hi)
                        ref_note = f"   dN {drel:.1e}"
                    except (ValidationError, FloatingPointError):
                        refine_ok = False
                        ref_note = "   dN FAIL"
                ok = math.isfinite(rep.R) and rank_ok
                if ok:
                    last_good = deepcopy(g.state_dict())
                if ok and agree and score > best_any_R:
                    best_any_R, best_any_iter = score, it
                    best_any_state = deepcopy(g.state_dict())
                if ok and agree and refine_ok and score > best_R:
                    best_R, best_iter = score, it
                    best_state = deepcopy(g.state_dict())
                    stale = 0
                else:
                    stale += 1
                if patience > 0 and stale >= patience:
                    g.load_state_dict(best_state)
                    print(f"  adam{stage_label} early stop: {stale} validation "
                          f"points without improvement; restored best "
                          f"validated R = {best_R:.10f}.")
                    break
                flag = "" if (ok and agree and refine_ok) else \
                    ("  [NOT SNAPSHOTTED: refine]" if (ok and agree)
                     else "  [NOT SNAPSHOTTED]")
                print(f"  adam{stage_label} iter {it:5d}   hf {R_hf:.10f}   "
                      f"report {rep.R:.10f}   rank {rep.rank}/{g.m}   "
                      f"best {best_R:.10f}{ref_note}   "
                      f"[{time.time()-t0:6.1f}s]{flag}")

        if it == iters:
            break
        opt.zero_grad(set_to_none=True)
        R = problem.rayleigh(g, grad_mode=grad_mode)
        loss = -R
        if not torch.isfinite(loss):
            g.load_state_dict(last_good)
            raise FloatingPointError(
                f"non-finite loss at iter {it}; restored last validated state")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=grad_clip)
        opt.step()
        sched.step()

    if best_R == -math.inf:
        if best_any_R > -math.inf and math.isfinite(best_any_R):
            print(f"  WARNING: no state passed the dN cap ({snap_rtol:.0e}); "
                  f"falling back to the best rank/agree-validated state "
                  f"(score {best_any_R:.10f} at iter {best_any_iter}). Its "
                  f"discretisation error exceeds the cap — the Richardson "
                  f"gate at the end quantifies it.")
            g.load_state_dict(best_any_state)
            return best_any_R, best_any_iter
        g.load_state_dict(last_good)
        raise FloatingPointError("no numerically validated Adam state was found")
    g.load_state_dict(best_state)
    return best_R, best_iter


def polish(problem: SeparableMaynard, g: nn.Module, n_steps: int, *,
           lr: float = 1.0, max_iter: int = 100, max_eval: int | None = None,
           history_size: int = 50, tolerance_grad: float = 1e-14,
           tolerance_change: float = 1e-16, log_every: int = 1,
           min_rank_frac: float = 0.25,
           problem_hi: SeparableMaynard | None = None,
           snap_rtol: float = 2e-2, excursion_factor: float = 3.0):
    """Block minorize-maximize LBFGS polish. Per outer step the Ritz vector v
    is re-solved (no autograd) and FROZEN; the inner run maximises the
    fixed-v quotient, a minorant touching k*lambda at the current point:
        R(th_{s+1}) >= q_v(th_{s+1}) >= q_v(th_s) = R(th_s),
    monotone by construction, with a deterministic objective inside the line
    search and no eigensolve in the closure. Snapshots are report()-validated."""
    if n_steps <= 0:
        with torch.no_grad():
            return float(problem.rayleigh(g))
    if max_eval is None:
        max_eval = max_iter * 5 // 4
    # Seed the incumbent with the state we were handed. Starting at -inf let
    # the first LBFGS step install ANY value as "best" — which is how a
    # polish that began from a validated R = 5.04 snapshotted R = 70.5.
    best_state = deepcopy(g.state_dict())
    try:
        _r0 = problem.report(g)
        best_R = _r0.R if math.isfinite(_r0.R) else -math.inf
        if problem_hi is not None:
            try:
                best_R = min(best_R, problem_hi.report(g).R)
            except (ValidationError, FloatingPointError):
                pass
        print(f"  lbfgs incumbent R = {best_R:.10f}")
    except (ValidationError, FloatingPointError):
        best_R = -math.inf

    for step in range(n_steps):
        with torch.no_grad():
            A, B, _ = problem.gram_matrices(g)
            _, v, _ = problem._solve_top(A.to(problem.ritz_dtype),
                                         B.to(problem.ritz_dtype))
        v_frozen = v.to(dtype=problem.dtype).detach()
        opt = torch.optim.LBFGS(
            g.parameters(), lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = -problem.rayleigh(g, grad_mode="hf", frozen_v=v_frozen)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite LBFGS loss")
            loss.backward()
            return loss

        prev = deepcopy(g.state_dict())
        try:
            opt.step(closure)
        except (RuntimeError, FloatingPointError) as e:
            g.load_state_dict(prev)
            print(f"  lbfgs step {step:3d}   error: {e} — restoring & stopping")
            break
        try:
            rep = problem.report(g)
        except ValidationError as e:
            g.load_state_dict(best_state)
            print(f"  lbfgs step {step:3d}   {e}")
            print("  restoring best validated state and stopping the polish.")
            break
        rank_ok = rep.rank >= max(1, math.ceil(min_rank_frac * g.m))
        refine_ok, ref_note = True, ""
        score = rep.R
        if problem_hi is not None:
            try:
                R_hi = problem_hi.report(g).R
                drel = abs(R_hi - rep.R) / max(abs(R_hi), 1e-30)
                refine_ok = drel <= snap_rtol
                score = min(rep.R, R_hi)
                ref_note = f"   dN {drel:.1e}"
            except (ValidationError, FloatingPointError):
                refine_ok = False
                ref_note = "   dN FAIL"
        if (excursion_factor > 0 and best_R > -math.inf
                and rep.R > excursion_factor * best_R):
            g.load_state_dict(best_state)
            print(f"  lbfgs step {step:3d}   EXCURSION: R = {rep.R:.6f} is "
                  f"{rep.R/max(best_R,1e-30):.1f}x the incumbent "
                  f"{best_R:.6f}; restoring and stopping the polish. "
                  f"(MM-LBFGS is monotone in the DISCRETE objective, so it "
                  f"exploits discretisation error very efficiently.)")
            break
        if math.isfinite(rep.R) and rank_ok and refine_ok and score > best_R:
            best_R = score
            best_state = deepcopy(g.state_dict())
        if step % max(log_every, 1) == 0 or step == n_steps - 1:
            flag = "" if (rank_ok and refine_ok) else \
                ("  [NOT SNAPSHOTTED: refine]" if rank_ok
                 else "  [NOT SNAPSHOTTED]")
            print(f"  lbfgs step {step:3d}   R {rep.R:.10f}   "
                  f"rank {rep.rank}/{g.m}   best {best_R:.10f}{ref_note}{flag}")
    g.load_state_dict(best_state)
    return best_R






# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Separable-DeepSets Maynard M_k solver, v9 "
                    "(factored nu = s^{p-1} e^{-as} psi; Chebyshev-Lobatto "
                    "representation + Gauss-Jacobi convolution)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--m-schedule", type=str, default=None,
                   help="comma list, e.g. 16,32: continuation in m")
    p.add_argument("--iters-schedule", type=str, default=None)
    p.add_argument("--n-schedule", type=str, default=None,
                   help="comma list of representation-grid sizes per stage "
                        "(grid continuation); default ramps 0.6*n_rep -> "
                        "n_rep across stages when --m-schedule has >1 stage")
    p.add_argument("--grow-noise", type=float, default=0.05)

    p.add_argument("--n-rep", type=int, default=None,
                   help="Chebyshev-Lobatto representation nodes; default "
                        "max(64, ceil(8*sqrt(k))). Set by the RAW (envelope-"
                        "free) channel structure, not by k directly.")
    p.add_argument("--n", type=int, default=None, help="alias for --n-rep")
    p.add_argument("--n-jacobi", type=int, default=32,
                   help="Gauss-Jacobi nodes per convolution; k-INDEPENDENT")
    p.add_argument("--n-inner", type=int, default=40,
                   help="Gauss-Legendre nodes for G(y)=int_0^y g, H(y)=int_0^y h")
    p.add_argument("--n-outer", type=int, default=64,
                   help="Gauss-Legendre nodes on the adaptive outer window")
    p.add_argument("--max-panels", type=int, default=512,
                   help="cap on geometric panels in the shared outer mesh; the "
                        "required count scales like (k-2)*ln(X/x_lo)/n_outer")
    p.add_argument("--outer-drop", type=float, default=60.0,
                   help="log-integrand drop defining the outer window "
                        "(60 => e^-60 ~ 1e-26 of the peak)")
    p.add_argument("--rate-cap", type=float, default=None,
                   help="cap on the initial channel envelope rates (default 2k)")

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--min-rank-frac", type=float, default=0.25)
    p.add_argument("--init-npz", type=str, default=None)

    p.add_argument("--conv-mode", type=str, default="doubling",
                   choices=("doubling", "sequential"))
    p.add_argument("--grad-mode", type=str, default="hf", choices=("hf",))
    p.add_argument("--pair-chunk", type=int, default=0)
    p.add_argument("--trunc-tol", type=float, default=1e-12)
    p.add_argument("--trunc-floor", type=float, default=1e-6,
                   help="PREVENTION: hard floor on the A-eigenvalue "
                        "truncation, applied in the eigensolve, the HF "
                        "gradient and report(). trunc_tol alone is calibrated "
                        "to machine epsilon; the matrices are only as accurate "
                        "as the discretisation (~1e-3 for the k=250 excursion "
                        "state). Raising the floor stops the optimizer from "
                        "steering into near-null directions at all.")
    p.add_argument("--stability-tol", type=float, default=1e-3,
                   help="DETECTION: second truncation level; R must survive it")
    p.add_argument("--stability-rtol", type=float, default=2.5e-1,
                   help="tolerated relative change in R between the working "
                        "floor and --stability-tol")
    p.add_argument("--interp", type=str, default="positive",
                   choices=("positive", "spectral"),
                   help="positive: two-point linear stencil, weights >= 0, "
                        "applied in log space by logsumexp. Linear in psi, "
                        "hence multilinear in (g_j,g_l), hence A is a Gram "
                        "matrix and PSD. O(h^2), so use a larger --n-rep; it "
                        "is also O(N) not O(N^2) per convolution, so that is "
                        "affordable. spectral: barycentric, signed weights — "
                        "higher order but NOT Gram-preserving; A comes out "
                        "genuinely indefinite. Diagnostic use only.")
    p.add_argument("--psd-tol", type=float, default=1e-8)
    p.add_argument("--channel-sign", type=str, default="positive",
                   choices=("positive", "free"),
                   help="positive: g_j = exp(net_j) e^{-rho_j x}, recursion in "
                        "log space, A positive entrywise and PSD by "
                        "construction. free: sign-changing channels, linear "
                        "recursion — mathematically more general but psi "
                        "underflows past k ~ 50 and A stops being PSD. Use "
                        "'free' only for small-k comparisons.")

    p.add_argument("--lbfgs-steps", type=int, default=20)
    p.add_argument("--lbfgs-max-iter", type=int, default=100)
    p.add_argument("--lbfgs-max-eval", type=int, default=None)
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-history-size", type=int, default=50)
    p.add_argument("--lbfgs-tol-grad", type=float, default=1e-14)
    p.add_argument("--lbfgs-tol-change", type=float, default=1e-16)
    p.add_argument("--lbfgs-log-every", type=int, default=1)

    p.add_argument("--preflight-rtol", type=float, default=1e-8)
    p.add_argument("--grad-rtol", type=float, default=1e-3,
                   help="gate on the finite-difference gradient check")
    p.add_argument("--n-rep-check", action="store_true",
                   help="(kept for compatibility; the refinement gate now "
                        "always runs)")
    p.add_argument("--refine-rtol", type=float, default=5e-3,
                   help="gate on the RICHARDSON-EXTRAPOLATED relative "
                        "uncertainty |R(1.5N)-R*|/|R*|; failure is recorded "
                        "as a warning but never suppresses an explicit export")
    p.add_argument("--snap-rtol", type=float, default=2e-2,
                   help="loose anti-exploitation cap on |R(1.5N)-R(N)|/R for "
                        "a state to be snapshot-eligible during training; NOT "
                        "a convergence criterion")
    p.add_argument("--excursion-floor", type=float, default=2.0,
                   help="the relative excursion multiple is only applied once "
                        "the validated best exceeds this (default 2.0, the "
                        "classical dictionary baseline). Prevents a bad early "
                        "R from anchoring the guard and rejecting real climbs.")
    p.add_argument("--excursion-factor", type=float, default=3.0,
                   help="HEURISTIC guard: reject a validation point whose R "
                        "exceeds this multiple of the validated best. Not a "
                        "theorem — for eps > 0 the only rigorous bound is "
                        "R <= k, too loose to catch 5 -> 70. 0 disables.")
    p.add_argument("--dump-violation", type=str, default=None,
                   help="write the channel state to this path the first time "
                        "a bound violation fires, for offline dissection")
    p.add_argument("--patience", type=int, default=0,
                   help="stop a stage after this many validation points with "
                        "no improvement in the validated best (0 = off). The "
                        "k=250 runs found their best state early and then "
                        "spent the remaining budget on excursions.")
    p.add_argument("--max-violations", type=int, default=3,
                   help="consecutive bound violations at validation points "
                        "before a training stage is stopped early")
    p.add_argument("--outer-check", action="store_true")
    p.add_argument("--grad-check", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-dict", action="store_true")
    p.add_argument("--skip-neural", action="store_true")
    p.add_argument("--check-dense", action="store_true")
    p.add_argument("--no-ridge-sweep", action="store_true")
    p.add_argument("--export", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float64",
                   choices=("float64", "float32"))
    p.add_argument("--ritz-dtype", type=str, default="float64",
                   choices=("float64", "float32"))
    p.add_argument("--cuda-matmul-precision", type=str, default="highest",
                   choices=("highest", "high", "medium"))
    return p.parse_args()




def build_problem(args, k, n_rep, eps, device, cdt, rdt):
    _p = SeparableMaynard(
        k=k, n_rep=n_rep, n_jacobi=args.n_jacobi, n_inner=args.n_inner,
        n_outer=args.n_outer, outer_drop=args.outer_drop, epsilon=eps,
        device=device, dtype=cdt, ritz_dtype=rdt, conv_mode=args.conv_mode,
        pair_chunk=args.pair_chunk, trunc_tol=args.trunc_tol,
        channel_sign=args.channel_sign, interp=args.interp)
    _p.max_panels = args.max_panels
    _p.trunc_floor = args.trunc_floor
    _p.stability_tol = args.stability_tol
    _p.stability_rtol = args.stability_rtol
    return _p  # noqa


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    cdt = parse_torch_dtype(args.dtype)
    rdt = parse_torch_dtype(args.ritz_dtype)
    torch.set_float32_matmul_precision(args.cuda_matmul_precision)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k, eps = args.k, args.epsilon
    rate_cap = args.rate_cap if args.rate_cap is not None else max(2.0 * k, 4.0)
    # The spectral default max(64, 8 sqrt k) is calibrated for spectral
    # accuracy. The positive (Gram-preserving) stencil is O(h^2), so it needs
    # a materially finer grid — and can afford one, since its convolution
    # cost is O(N nq), not O(N^2 nq).
    if args.interp == "positive":
        n_auto = max(256, int(math.ceil(24.0 * math.sqrt(k))))
    else:
        n_auto = max(64, int(math.ceil(8.0 * math.sqrt(k))))
    n_rep = args.n_rep or args.n or n_auto
    n_rep_src = "user-specified" if (args.n_rep or args.n) else \
        f"AUTO ({args.interp} heuristic — verify with the refinement gate)"

    ms = parse_int_list(args.m_schedule, "m-schedule") if args.m_schedule else [args.m]
    if any(b <= a for a, b in zip(ms, ms[1:])):
        raise ValueError("--m-schedule must be strictly increasing")
    if args.iters_schedule:
        its = parse_int_list(args.iters_schedule, "iters-schedule")
        if len(its) != len(ms):
            raise ValueError("--iters-schedule length must match --m-schedule")
    else:
        its = split_iters_cost_balanced(args.iters, ms) if len(ms) > 1 else [args.iters]

    chain = (addition_chain(k - 1) if args.conv_mode == "doubling"
             else sequential_chain(k - 1))
    mf = ms[-1]
    Pf = mf * (mf + 1) // 2

    print("=" * 78)
    label = f"M_{{{k},eps={eps:g},1/2}}" if eps > 0 else f"M_{k}"
    print(f" Separable Maynard problem  k = {k}   target {label}")
    print(f"   factorisation  : nu_p(s) = s^(p-1) e^(-a s) e^sigma psi_p(s), per pair")
    print(f"   representation : Chebyshev-Lobatto, N = {n_rep} on [0,{1+eps:g}] "
          f"(endpoint-inclusive; {n_rep_src})")
    print(f"   convolution    : Gauss-Jacobi, nq = {args.n_jacobi} (k-independent)")
    print(f"   outer integral : adaptive log-space window, "
          f"n_out = {args.n_outer}, drop = {args.outer_drop:g}")
    print(f"   inner integral : Gauss-Legendre, ng = {args.n_inner}")
    print(f"   chain          : {args.conv_mode}, {len(chain)} convolutions "
          f"(vs {max(k-2,1)} sequential)")
    print(f"   pairs          : m(m+1)/2 = {Pf} at final m = {mf}")
    print(f"   m schedule     : {ms}   adam iters = {its}")
    print("=" * 78)
    print(f" seed={args.seed}  device={device}  eps={eps:g}  rate_cap={rate_cap:g}")
    print(f" dtype={args.dtype}  ritz={args.ritz_dtype}  trunc_tol={args.trunc_tol:.0e}"
          f"  min_rank_frac={args.min_rank_frac:g}")
    if device.type == "cuda":
        pr = torch.cuda.get_device_properties(device)
        print(f" GPU: {pr.name}  VRAM={pr.total_memory/2**30:.1f} GiB")

    problem = build_problem(args, k, n_rep, eps, device, cdt, rdt)
    problem.psd_tol = args.psd_tol
    # 1.5x-finer twin used ONLY for snapshot validation during training/polish
    n_hi = int(math.ceil(1.5 * n_rep))
    problem_hi = build_problem(args, k, n_hi, eps, device, cdt, rdt)
    problem_hi.psd_tol = args.psd_tol

    # ---------------- preflight: g == 1 against the closed form -------------
    R_exact = closed_form_R(k, eps)
    with torch.no_grad():
        A1, B1, ld1 = problem.gram_matrices(ConstantOne())
        a1, b1 = float(A1[0, 0]), float(B1[0, 0])
        R_num = k * b1 / a1 if a1 != 0.0 else float("nan")
    rel1 = abs(R_num - R_exact) / max(abs(R_exact), 1e-300)
    print(f"\n preflight g==1 : A={a1:+.6e}  B={b1:+.6e}  logA_diag={float(ld1[0]):+.3f}")
    print(f"   R_num={R_num:.12f}   closed form={R_exact:.12f}   rel.err={rel1:.2e}")
    print(f"   channel-sign = {args.channel_sign}"
          + ("   (log-space recursion, sign-definite channels)"
             if args.channel_sign == "positive" else
             "   (linear recursion; psi underflows past k ~ 50 — small k only)"))
    # second control, g = x: non-constant xi, so unlike g == 1 this actually
    # probes the interpolation resolution (vanilla eps = 0 only; the closed
    # form is for the unit simplex).
    if eps == 0.0:
        with torch.no_grad():
            Ax, Bx, _ = problem.gram_matrices(LinearX())
        R_x = k * float(Bx[0, 0]) / float(Ax[0, 0])
        R_x_exact = 3.0 * k / (3.0 * k + 1.0)
        rel_x = abs(R_x - R_x_exact) / R_x_exact
        print(f" preflight g==x : R_num={R_x:.12f}   closed form 3k/(3k+1)="
              f"{R_x_exact:.12f}   rel.err={rel_x:.2e}")
        # Calibration (measured): the error is O(h^2) — a factor ~4 per
        # doubling of N — and grows ~linearly in k at fixed N. This control
        # is deliberately harsher than trained channels (xi = 2p log s is
        # singular at 0; MLP channels have raw = exp(net), bounded away from
        # zero), so it warns early rather than gates hard. The binding gate
        # is the refinement gate on the ACTUAL trained channels at the end.
        if rel_x > 1e-3:
            n_sugg = int(math.ceil(n_rep * math.sqrt(rel_x / 1e-3)))
            print(f"   note: g==x error is O(h^2); N ~ {n_sugg} would bring "
                  f"it to ~1e-3.")
        if rel_x > 1e-1:
            raise ValidationError(
                f"g == x preflight failed at {rel_x:.2e}: the representation "
                f"grid grossly under-resolves a non-constant xi. (The g == 1 "
                f"control is exact for ANY N in positive mode and cannot see "
                f"this.) Raise --n-rep.")
        if rel_x > 1e-2:
            print(f"   WARNING: g==x preflight at {rel_x:.2e} — marginal "
                  f"resolution; the refinement gate must pass before any "
                  f"result is used.")
    if not (math.isfinite(R_num) and a1 > 0.0 and b1 > 0.0
            and rel1 <= args.preflight_rtol):
        raise FloatingPointError(
            "convolution preflight failed; refusing to train on an invalid "
            "discretisation. Raise --n-rep / --n-jacobi / --n-outer, use "
            "float64, or cross-check with --conv-mode sequential.")

    if args.grad_check:
        gc = ChannelNet(k, m=min(ms[0], 6), hidden=16, depth=2, rate_cap=rate_cap,
                        positive=(args.channel_sign == "positive")).to(device=problem.device, dtype=problem.dtype)
        problem.rayleigh(gc).backward()
        par = next(pp for pp in gc.parameters() if pp.grad is not None)
        idx = tuple(0 for _ in par.shape)
        ana = float(par.grad[idx])
        base = float(par[idx].item())
        # Sweep the step. dR/dtheta ~ 1e-3 here, so a step of 1e-6 puts the
        # difference Rp - Rm at ~1e-9 against R ~ 3.5 — i.e. 7 digits of
        # cancellation against a 1e-16 representation. The apparent "3%
        # gradient error" in the earlier run was entirely this: at h = 1e-3 the
        # same gradient agrees to 6e-6. Take the best step rather than a fixed
        # one, and report the whole sweep so the choice is auditable.
        best = (float("inf"), None, None)
        rows = []
        for e in (1e-2, 1e-3, 1e-4, 1e-5):
            h = e * max(1.0, abs(base))
            with torch.no_grad():
                par[idx] = base + h
                Rp = float(problem.rayleigh(gc))
                par[idx] = base - h
                Rm = float(problem.rayleigh(gc))
                par[idx] = base
            nm = (Rp - Rm) / (2 * h)
            rl = abs(ana - nm) / max(abs(nm), 1e-30)
            rows.append((h, nm, rl))
            if rl < best[0]:
                best = (rl, nm, h)
        for h_, nm_, rl_ in rows:
            print(f"   fd step {h_:.1e}: {nm_:+.10e}   rel {rl_:.2e}")
        num = best[1]
        rel = best[0]
        print(f"\n HF gradient check: analytic {ana:+.8e}  finite-diff {num:+.8e}  "
              f"rel {rel:.2e}")
        if rel > args.grad_rtol:
            raise ValidationError(
                f"HF gradient disagrees with finite differences by {rel:.2e} "
                f"(gate {args.grad_rtol:.0e}). The optimiser would be "
                f"following a direction that is not the gradient of the "
                f"quantity being reported.")

    if not args.skip_dict:
        dic = FixedDictionary(k).to(device=problem.device, dtype=problem.dtype)
        try:
            rd = problem.report(dic)
            print(f"\n dictionary baseline (x^p e^-rx, m = {dic.m}) : "
                  f"R = {rd.R:.10f}   rank {rd.rank}/{dic.m}")
        except FloatingPointError as e:
            print(f"\n dictionary baseline skipped: {e}")

    if args.check_dense:
        if k > 4:
            print("\n --check-dense skipped: only feasible for k <= 4")
        else:
            gk = ChannelNet(k, m=min(ms[0], 6), hidden=32, depth=2, rate_cap=rate_cap,
                            positive=(args.channel_sign == "positive")).to(device=problem.device,
                                                  dtype=problem.dtype)
            rc = problem.report(gk)
            ct = torch.tensor(rc.c, dtype=problem.dtype, device=problem.device)
            I_conv = float(ct @ torch.tensor(rc.A) @ ct)
            J_conv = float(ct @ torch.tensor(rc.B) @ ct)
            # v9's A, B are preconditioned; compare the RATIO, which is invariant
            with torch.no_grad():
                gr = gk(problem.Xrep)
                nrm = torch.sqrt(torch.einsum("q,qj->j", problem.w_cc, gr.square()))
            D = torch.exp(torch.tensor(-0.5 * rc.logA_diag, dtype=problem.dtype,
                                       device=problem.device))
            c_raw = (ct * D) / nrm.pow(k)
            dense = DenseCheck(k, n_quad=30, epsilon=eps, device=problem.device,
                               dtype=problem.dtype)
            I_d, J_d = dense.I_and_J(gk, c_raw)
            print(f"\n dense Duffy cross-check (k = {k}, random init model):")
            print(f"   J/I  conv {J_conv/I_conv:.12e}   dense {J_d/I_d:.12e}   "
                  f"rel {abs(J_conv/I_conv - J_d/I_d)/abs(J_d/I_d):.2e}")
            print(f"   (ratios only: v9 stores A, B diagonally preconditioned)")

    if args.skip_neural:
        return

    g = ChannelNet(k, m=ms[0], hidden=args.hidden, depth=args.depth,
                   rate_cap=rate_cap, positive=(args.channel_sign == "positive")).to(device=problem.device, dtype=problem.dtype)

    if args.init_npz is not None:
        print(f"\nLoading warm start from {args.init_npz}")
        data = np.load(args.init_npz)
        xf, gf = data["x_fine"], data["g_fine"]
        if gf.shape[1] != ms[0]:
            raise ValueError(f"warm start has {gf.shape[1]} channels, "
                             f"first stage expects {ms[0]}")
        xw = torch.tensor(xf, dtype=problem.dtype, device=problem.device)
        tgt = torch.tensor(gf, dtype=problem.dtype, device=problem.device)
        oi = torch.optim.Adam(g.parameters(), lr=1e-3)
        for it in range(1000):
            oi.zero_grad(set_to_none=True)
            loss = torch.mean((g(xw) - tgt) ** 2)
            loss.backward()
            oi.step()
            if (it + 1) % 250 == 0:
                print(f" warm-start fit {it+1:4d} MSE={loss.item():.3e}")

    print(f"\n ChannelNet: m={ms[0]}  hidden={args.hidden}  depth={args.depth}  "
          f"params={sum(p_.numel() for p_ in g.parameters())}")

    # Grid continuation: early stages train on coarser grids, only the final
    # stage sees the full n_rep. This limits how deeply theta can adapt to
    # any one grid's error pattern (the mechanism behind the +3.5e-2
    # training-grid inflation measured at k = 50), and makes early iterations
    # cheaper as a side effect.
    if args.n_schedule:
        ns = parse_int_list(args.n_schedule, "n-schedule")
        if len(ns) != len(ms):
            raise ValueError("--n-schedule length must match --m-schedule")
    elif len(ms) > 1:
        ns = [int(math.ceil(n_rep * (0.6 + 0.4 * i / (len(ms) - 1))))
              for i in range(len(ms))]
    else:
        ns = [n_rep]
    if len(ms) > 1:
        print(f"\n grid continuation: N schedule = {ns}")

    best_R = -math.inf
    for stage, (m_s, it_s, n_s) in enumerate(zip(ms, its, ns)):
        if n_s == n_rep:
            prob_s, prob_s_hi = problem, problem_hi
        else:
            prob_s = build_problem(args, k, n_s, eps, device, cdt, rdt)
            prob_s.psd_tol = args.psd_tol
            prob_s_hi = build_problem(args, k, int(math.ceil(1.5 * n_s)),
                                      eps, device, cdt, rdt)
            prob_s_hi.psd_tol = args.psd_tol
        if stage > 0:
            try:
                R_before = prob_s.report(g).R
            except ValidationError:
                R_before = float("nan")
            g.grow(m_s, noise=args.grow_noise)
            try:
                ra = prob_s.report(g)
                print(f"\n--- grow: m {ms[stage-1]} -> {m_s}   R before "
                      f"{R_before:.10f}  after {ra.R:.10f}   "
                      f"rank {ra.rank}/{m_s} ---")
            except ValidationError as e:
                print(f"\n--- grow: m {ms[stage-1]} -> {m_s}   R before "
                      f"{R_before:.10f}   post-grow state not yet rank-stable "
                      f"({str(e).split(';')[1].strip()}); training will "
                      f"proceed and re-validate ---")
        print(f"\n--- Adam stage {stage+1}/{len(ms)}: m={m_s}, iters={it_s}, "
              f"N={n_s} ---")
        best_R, best_it = train(
            prob_s, g, iters=it_s, lr=args.lr, warmup_frac=args.warmup_frac,
            grad_clip=args.grad_clip, log_every=args.log_every,
            weight_decay=args.weight_decay, grad_mode=args.grad_mode,
            stage_label=f"[s{stage+1}]" if len(ms) > 1 else "",
            min_rank_frac=args.min_rank_frac,
            problem_hi=prob_s_hi, snap_rtol=args.snap_rtol,
            max_violations=args.max_violations,
            dump_violation=args.dump_violation, patience=args.patience,
            excursion_factor=args.excursion_factor,
            excursion_floor=args.excursion_floor)
        print(f" best stage R = {best_R:.10f}  at iter {best_it}")

    if args.lbfgs_steps > 0:
        print("\n--- LBFGS polish (minorize-maximize) ---")
        best_R = polish(problem, g, n_steps=args.lbfgs_steps, lr=args.lbfgs_lr,
                        max_iter=args.lbfgs_max_iter, max_eval=args.lbfgs_max_eval,
                        history_size=args.lbfgs_history_size,
                        tolerance_grad=args.lbfgs_tol_grad,
                        tolerance_change=args.lbfgs_tol_change,
                        log_every=args.lbfgs_log_every,
                        min_rank_frac=args.min_rank_frac,
                        problem_hi=problem_hi, snap_rtol=args.snap_rtol,
                        excursion_factor=args.excursion_factor)

    # ---- representation refinement gate: Richardson with FITTED order ------
    # Certification at k = 50 measured the effective convergence order of
    # TRAINED channels as p ~ 1.6, not the stencil's nominal 2: theta adapts
    # to the training grid, so the fixed-c O(h^2) model is mildly violated.
    # The gate therefore fits R(N) = R* - c N^{-p} with p free (bisection on
    # the difference ratio), reports the fitted p, and extrapolates with it.
    # At k = 50 this fitted extrapolation reproduced the exact certified
    # value to 2.3e-3 — inside its own quoted bar.
    print("\n--- representation refinement gate (Richardson, fitted order) ---")
    Ns = (n_rep, int(math.ceil(1.25 * n_rep)), int(math.ceil(1.5 * n_rep)))
    R_grid, gate_ok = [], True
    for Nh in Ns:
        ph = build_problem(args, k, Nh, eps, device, cdt, rdt)
        ph.psd_tol = args.psd_tol
        try:
            rh = ph.report(g)
            R_grid.append(rh.R)
            print(f"   N = {Nh:5d}   R = {rh.R:.12f}   rank {rh.rank}/{g.m}   "
                  f"xi jump {rh.xi_jump:.3f}")
        except ValidationError as e:
            print(f"   N = {Nh:5d}   VALIDATION FAILURE: {e}")
            gate_ok = False
            break
    R_extrap, R_err, p_fit = None, None, None
    if gate_ok and len(R_grid) == 3:
        d1 = R_grid[1] - R_grid[0]
        d2 = R_grid[2] - R_grid[1]
        tinyd = 1e-12 * max(abs(R_grid[2]), 1.0)
        if abs(d2) <= tinyd and abs(d1) <= tinyd:
            R_extrap, R_err, p_fit = R_grid[2], abs(d2), 2.0
            print(f"   grid differences at round-off; R* = R(1.5N).")
        elif d1 * d2 <= 0 or abs(d2) >= abs(d1):
            # non-monotone or non-contracting: no power law fits; fall back to
            # the hold-out value with the raw difference as the bar.
            R_extrap, R_err, p_fit = R_grid[2], abs(d2) + abs(d1), None
            print(f"   d1 = {d1:+.3e}, d2 = {d2:+.3e}: not a contracting "
                  f"power law; using R(1.5N) with a conservative bar.")
            gate_ok = False
        else:
            target = d1 / d2

            def ratio(pp):
                return ((Ns[0] ** -pp - Ns[1] ** -pp)
                        / (Ns[1] ** -pp - Ns[2] ** -pp))
            lo, hi = 0.3, 5.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                # ratio(p) is increasing in p for N1<N2<N3
                if ratio(mid) < target:
                    lo = mid
                else:
                    hi = mid
            p_fit = 0.5 * (lo + hi)
            r_fac = (Ns[2] ** -p_fit) / (Ns[1] ** -p_fit - Ns[2] ** -p_fit)
            R_extrap = R_grid[2] + r_fac * d2
            R_err = abs(R_extrap - R_grid[2])
            print(f"   d(N->1.25N) = {d1:+.3e}   d(1.25N->1.5N) = {d2:+.3e}"
                  f"   fitted order p = {p_fit:.2f}   (stencil nominal 2; "
                  f"trained channels ~1.6 at k=50)")
            print(f"   Richardson(p): R* = {R_extrap:.12f}   "
                  f"|R(1.5N) - R*| = {R_err:.2e}  "
                  f"(rel {R_err/max(abs(R_extrap),1e-30):.2e})")
            if R_extrap > problem.R_bound * (1.0 + problem.R_tol):
                print(f"   R* exceeds the Polymath bound {problem.R_bound:.6f}"
                      f" — extrapolation invalid.")
                gate_ok = False
            elif not (0.8 <= p_fit <= 4.0):
                print(f"   fitted order p = {p_fit:.2f} outside [0.8, 4]: the "
                      f"asymptotic regime is not reached; R* unreliable.")
                gate_ok = False
            else:
                gate_ok = (R_err / max(abs(R_extrap), 1e-30)
                           <= args.refine_rtol)
        print(f"   gate (extrapolated uncertainty <= {args.refine_rtol:.0e}) "
              f"-> {'PASS' if gate_ok else 'FAIL'}")

        # -------------------------------------------------------------------
        # UNCONDITIONAL post-checks, applied to the final reported value no
        # matter which branch above produced it.  These close two holes that
        # let inflated values pass at k=500 (vanilla eps=0):
        #   (1) CEILING: R must not exceed the rigorous upper bound
        #       R_bound (= (k/(k-1)) log k for eps=0; = k for eps>0).  The
        #       in-branch ceiling test only ran inside the power-law-fit
        #       branch, so a value reaching the "converged"/non-contracting
        #       fallbacks bypassed it.  M_k CANNOT exceed R_bound; a value
        #       above it is near-null-space inflation (a single channel whose
        #       A-diagonal collapsed into the assembly-accuracy floor), never
        #       a real optimum.
        #   (2) NON-CONTRACTION: even below the ceiling, a monotone grid
        #       sequence whose successive changes do NOT shrink (|d2| >~ |d1|)
        #       is not converged -- "small but not contracting" is the
        #       signature of inflation being tracked as the grid refines, not
        #       of a settled value.  Small step size alone is NOT convergence.
        R_final = R_extrap if R_extrap is not None else (
            R_grid[-1] if R_grid else float("nan"))
        if gate_ok and R_final == R_final:  # not NaN
            if R_final > problem.R_bound * (1.0 + problem.R_tol):
                print(f"   CEILING VIOLATION: R = {R_final:.6f} exceeds the "
                      f"rigorous bound R_bound = {problem.R_bound:.6f} "
                      f"({'(k/(k-1))log k, vanilla' if eps == 0 else 'k, C-S'})"
                      f". M_k cannot exceed this; the value is near-null-space "
                      f"inflation, not an optimum. Raise --n-rep / "
                      f"--trunc-floor.")
                gate_ok = False
            elif len(R_grid) == 3:
                d1g = R_grid[1] - R_grid[0]
                d2g = R_grid[2] - R_grid[1]
                rr = max(abs(R_final), 1.0)
                # Only trip when the drift is BOTH non-contracting AND large
                # enough to matter: if |d2| is already within the gate's own
                # rtol target, the value is converged to tolerance regardless
                # of whether the ratio d2/d1 contracts (tiny wiggles at the
                # noise floor are not drift).  The k=500 vanilla 7.30 had
                # steps ~1.2e-2 (rel 1.6e-3, above refine_rtol) that did NOT
                # shrink -> real drift; the 4.21 run had steps ~1e-5
                # (rel 3e-6, far below tol) -> converged, must not trip.
                rel_d2 = abs(d2g) / rr
                if (rel_d2 > args.refine_rtol
                        and abs(d2g) >= 0.7 * abs(d1g)):
                    print(f"   NON-CONTRACTION: grid steps {d1g:+.2e}, "
                          f"{d2g:+.2e} (rel {rel_d2:.1e} > {args.refine_rtol:.0e})"
                          f" are not shrinking (|d2|>=0.7|d1|); the sequence is "
                          f"drifting with grid refinement, not converged. "
                          f"'Small step' != 'converged'. Raise --n-rep.")
                    gate_ok = False
            if not gate_ok:
                print(f"   gate (post-check) -> FAIL")
        # -------------------------------------------------------------------

    if not gate_ok:
        print("   VERDICT: the reported values carry unconverged "
              "discretisation error; use R* (if printed) only with its bar.")
        if args.export:
            print(f"   WARNING: exporting unconverged candidate to "
                  f"{args.export} because --export was explicitly requested.")

    rep = problem.report(g)
    print("\n================ RESULT (quote these, in this order) ================")
    if R_extrap is not None:
        print(f" R* (Richardson, fitted order)      = {R_extrap:.10f}   "
              f"+/- {R_err:.2e}")
    if len(R_grid) == 3:
        print(f" R  hold-out grid (N = {Ns[2]})        = {R_grid[2]:.10f}")
    print(f" R  training grid (N = {n_rep})        = {rep.R:.10f}   "
          f"<- inflated by grid-adaptive error (k=50 certification measured "
          f"+3.5e-2); NOT a lower-bound candidate")
    print(" The exact certification of the exported channels is the final "
          "arbiter.")
    print("=====================================================================")
    print(f" solver rank        = {rep.rank}/{g.m}")
    print(f" xi span            = {rep.xi_span:.1f} nats   (large is expected "
          f"in log space; NOT a resolution warning)")
    print(f" xi adjacent jump   = {rep.xi_jump:.3f} nats   (local smoothness; "
          f">~1 suggests the grid undersamples xi)")
    print(f" log diag(A) range  = [{rep.logA_diag.min():+.1f}, "
          f"{rep.logA_diag.max():+.1f}]  (spread {rep.logA_diag.max()-rep.logA_diag.min():.1f} "
          f"nats, removed by preconditioning)")
    print(f" |c| profile        = " + " ".join(f"{abs(v):.2e}" for v in rep.c))
    _dR = k * np.diag(rep.B) / np.clip(np.diag(rep.A), 1e-300, None)
    print(f" max diagonal k B_jj/A_jj = {_dR.max():.6f}   (each is an "
          f"admissible single-channel trial function; must be <= "
          f"{problem.R_bound:.4f})")
    print(f" global R / max diagonal  = {rep.R/max(_dR.max(),1e-30):.2f}   "
          f"(a large ratio means the value comes from off-diagonal mixing, "
          f"where near-null directions of A can inflate it)")
    _wA = np.linalg.eigvalsh(0.5 * (rep.A + rep.A.T))
    print(f" cond(A) estimate   = {np.linalg.cond(rep.A):.2e}")
    print(f" trunc floor        = {max(problem.trunc_tol, problem.trunc_floor):.0e}"
          f"   rank-stability check at {problem.stability_tol:.0e} passed")
    print(f" A spectrum         = [{_wA[0]:+.3e}, {_wA[-1]:.3e}]   "
          f"lam_min/lam_max = {_wA[0]/_wA[-1]:+.2e}   "
          f"({'PSD' if _wA[0] > -1e-10 * _wA[-1] else 'NOT PSD — A IS INVALID'})")

    if not args.no_ridge_sweep:
        ridge_sensitivity(rep.A, rep.B, k=k)

    if args.export:
        x_fine = np.linspace(0.0, problem.L, 2001)
        with torch.no_grad():
            g_fine = g(torch.tensor(x_fine, dtype=problem.dtype,
                                    device=problem.device)).cpu().numpy()
            gr = g(problem.Xrep)
            nrm = torch.sqrt(torch.einsum("q,qj->j", problem.w_cc,
                                          gr.square())).cpu().numpy()
            rates = torch.nn.functional.softplus(g.rho).cpu().numpy()
        np.savez(args.export, k=k, n_rep=n_rep, n_jacobi=args.n_jacobi,
                 n_inner=args.n_inner, n_outer=args.n_outer, m=g.m, R=rep.R,
                 c=rep.c, epsilon=eps, channel_norms=nrm, envelope_rates=rates,
                 x_fine=x_fine, g_fine=g_fine, A=rep.A, B=rep.B,
                 logA_diag=rep.logA_diag, rank=rep.rank,
                 N_grid=np.array(Ns), R_grid=np.array(R_grid),
                 R_extrap=(R_extrap if R_extrap is not None else np.nan),
                 R_extrap_err=(R_err if R_err is not None else np.nan),
                 p_fit=(p_fit if p_fit is not None else np.nan))
        print(f"\n exported channels + Ritz data to {args.export}")
        print("   (A, B stored DIAGONALLY PRECONDITIONED; logA_diag recovers "
              "the true scales: A_true = exp(logA_diag/2) A exp(logA_diag/2))")


if __name__ == "__main__":
    main()
