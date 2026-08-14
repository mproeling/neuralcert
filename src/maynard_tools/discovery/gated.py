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

v17 distributed v3 — MULTI-GPU / DISTRIBUTED, SHARDED OVER PAIRS
=================================================
The single-process numerics are UNCHANGED. Adam iterations are bit-identical
to v16; the only divergence is a ~1e-10 drift that appears after the first
LBFGS line search, and it is a rounding-order artefact, not a defect (see
CONDITIONING below).

WHY PAIRS, AND WHY NOT FSDP / ZeRO / TENSOR PARALLELISM
------------------------------------------------------
The VRAM here is ACTIVATIONS, not parameters. The dominant term is the xi
recursion,

    mem ~ 2 * (#chain steps) * N * nq * P * 8 bytes,     P = m(m+1)/2,

i.e. ~1.1 MB per pair at N=128, nq=32, k=1000: 0.6 GB at m=32, 2.3 GB at
m=64, 9.2 GB at m=128, 37 GB at m=256. The ChannelNet is ~1e4 doubles, i.e.
noise. Parameter-sharding schemes address the wrong resource entirely and buy
nothing. What binds is P, growing QUADRATICALLY in m — and P is a perfectly
parallel axis: for pair (j,l) the whole chain (_xi_power, _log_inner, the
outer integrals) touches only channels j and l, and produces exactly TWO
SCALARS. The compute-to-communication ratio is enormous, so this runs fine
over ordinary Ethernet with gloo, never mind NVLink.

DDP IS THE WRONG WRAPPER. nn.parallel.DistributedDataParallel AVERAGES
gradients, assuming a loss that is a mean over a sample axis. Here the
reduction is a SUM over pairs INSIDE a nonlinear ratio, so DDP would silently
deliver 1/world_size of a gradient that is also structurally wrong. This uses
a bare process group and reduces by hand.

THE GRADIENT IS EXACT, AND NEEDS NO DIFFERENTIABLE COLLECTIVES
--------------------------------------------------------------
Hellmann-Feynman already detaches v, so with c_p = (2 - delta_jl) v_j v_l both
quadratic forms are plain sums over pairs:

    num = sum_p c_p B_p,   den = sum_p c_p A_p,   R = k num / den,
    dR/dtheta = sum_ranks [ (k/den) dnum_r - (k num/den^2) dden_r ].

So: all-reduce the two scalar VALUES; backprop the local surrogate with num
and den frozen; all-reduce parameter gradients with op=SUM. That is the
quotient rule, not an approximation. Two scalar all-reduces forward, one
gradient all-reduce backward, plus 2P doubles for the assembly (1 MB at
m = 512). The surrogate's own value is identically zero — it is a directional
derivative — so rayleigh() restores the true R via surr - surr.detach() + R.

THE THREE HAZARDS
-----------------
 (1) COLLECTIVE DIVERGENCE. report() raises ValidationError on gate failures.
     A gate that fires on one rank and not another leaves that rank unwinding
     while the others block at the next all-reduce forever. Dist.agree() makes
     every verdict unanimous. Same for the eigensolve's FloatingPointError.
 (2) EIGENSOLVE DRIFT. Near a degenerate lambda_max the eigenvector is defined
     only up to the degenerate subspace; two ranks picking different
     representatives then optimise different objectives, surfacing much later
     as an inexplicable loss of monotonicity in the polish. Rank 0 solves, v
     is broadcast (m doubles).
 (3) PARAMETER DRIFT. Identical seeds are not enough — grow() draws randn, and
     a warm start may consume a different amount. sync_module() runs after
     init, after every grow, and after the warm-start fit.

CONDITIONING (why sharded and single-process differ in the last digits)
-----------------------------------------------------------------------
Measured at k=10, m=6: the two quotient-rule terms have norms 313.96 and
313.98 against a gradient of norm 0.199 — a cancellation factor of 1.6e3, on
top of ~1e3 cancellation inside v'Bv itself. So ~1e6 amplification of machine
epsilon, and any change of summation order lands at ~1e-10 in the gradient.
v16 has identical conditioning; it merely rounds differently. Both agree with
finite differences to 4e-7 (FD-truncation-limited) with IDENTICAL analytic
values. The optimiser trajectory is chaotic at that level, so a distributed
run will not retrace a single-process one step for step — it converges to the
same place, which is what --dist-selftest verifies directly.

VALIDATION (all measured, world = 2, 3, 4, gloo)
    --dist-selftest      sharded vs replicated: R rel <= 4.1e-13,
                         gradient rel <= 2.5e-09, every rank identical
    end-to-end 1 vs 4    grow() identical; Adam identical to 10 digits;
                         LBFGS to 9; Richardson R* to 7
    replica fallback     world > P (e.g. m=2, P=3 on 4 ranks) auto-replicates
    free/spectral path   sharded identically
    --check-dense        matches v16 exactly; runs on rank 0 only (its Duffy
                         grid is ~2 GB, two orders above the pipeline it
                         checks — replicating it OOMs boxes that run the real
                         workload comfortably)

Usage:
    # single process, exactly as before
    python v17.py --k 100  --m-schedule 16,32 --iters 3000

    # 4 GPUs on one node. NOTE: torchrun's own argv makes "--m" ambiguous
    # against --max-restarts; use --m-schedule instead.
    torchrun --nproc_per_node=4 v17.py --k 1000 --m-schedule 64,128 \
             --iters 3000 --export k1000.npz

    # RUN THIS ONCE PER MACHINE BEFORE TRUSTING A CAMPAIGN.
    # --nproc_per_node must equal the number of VISIBLE GPUs; asking for more
    # is caught at startup with an explicit message rather than a bare
    # "invalid device ordinal" from inside torch.cuda.set_device.
    #    python -c "import torch; print(torch.cuda.device_count())"
    torchrun --nproc_per_node=2 v17.py --k 100 --m-schedule 32 --dist-selftest

    # multi-node
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv-backend=c10d \
             --rdzv-endpoint=HOST:29500 v17.py --k 1000 --m-schedule 128

Windows: NCCL does not exist there. --dist-backend auto falls back to gloo,
which handles CUDA tensors; the payloads are O(m^2) doubles per iteration, so
the penalty is irrelevant at this compute-to-communication ratio.
"""

from __future__ import annotations

import argparse
import builtins
import math
import time
import socket
from copy import deepcopy
from dataclasses import dataclass

import sys

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


# Distributed execution is isolated from the mathematical discovery code.
from .distributed import DIST, PairPlan, print, print_all


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------






















# ---------------------------------------------------------------------------
# Channel functions.  Protocol: parts(x) -> (raw, rates) with
#     g_j(x) = raw_j(x) * exp(-rates_j * x)
# The split is what lets the exponential be factored out of the convolution
# analytically instead of being resolved on a grid.
# ---------------------------------------------------------------------------




def raw_curvature_penalty(g: nn.Module, L: float = 1.0, n: int = 257,
                          span: float = 8.0) -> torch.Tensor:
    """Mean squared curvature of log raw, per channel, in the channel's own
    envelope-scaled coordinate u = rho * x.

    Rationale (v22): the factorisation g = raw * e^{-rho x} assumes ALL
    sharpness lives in the exponential envelope, with raw an O(1)-smooth
    factor -- that assumption is what the psi-power recursion and the fixed
    meshes are built on.  With the rate clamped, the optimizer's remaining
    exploit route is to grow a sub-envelope-scale core in raw itself (the
    k=1000 channel-27 mode: rate 375 but g max/rms = 197, i.e. structure far
    below the 1/rho scale).  Penalising curvature of log raw in u allows
    legitimate envelope-scale variation (curvature O(1) in u) while making
    sub-scale spikes expensive (curvature ~ (rho*w)^{-4} for a width-w spike).

    Grid: per channel, uniform in x on [0, min(L, span/rho_j)], n points --
    uniform in u with u-spacing at most span/(n-1) for every channel, so the
    second-difference curvature has comparable meaning across channels.  The
    grid is built from DETACHED rates (a theta-dependent grid would make the
    gradient inconsistent with the value, same argument as the fixed a_ref
    mesh).  Cost: one net forward on m*n points.

    This is a mitigation that shapes the search away from the exploit; the
    moment gate remains the enforcement backstop.
    """
    p0 = next(g.parameters())
    with torch.no_grad():
        _, rates_d = channel_parts(g, torch.zeros(1, dtype=p0.dtype,
                                                  device=p0.device))
    rates_d = rates_d.detach().clamp_min(1e-6)                         # (m,)
    m = rates_d.shape[0]
    xmax = torch.minimum(torch.full_like(rates_d, L), span / rates_d)  # (m,)
    t = torch.linspace(0.0, 1.0, n, dtype=p0.dtype, device=p0.device)  # (n,)
    pts = xmax.view(m, 1) * t.view(1, n)                               # (m,n)
    lr, _ = channel_log_parts(g, pts.reshape(-1))                      # (m*n, m)
    lr = lr.reshape(m, n, m)
    jj = torch.arange(m, device=p0.device).view(m, 1, 1).expand(m, n, 1)
    lrd = torch.gather(lr, 2, jj).squeeze(-1)                          # (m,n)
    d2 = lrd[:, 2:] - 2.0 * lrd[:, 1:-1] + lrd[:, :-2]                 # (m,n-2)
    h_u = (rates_d * xmax / (n - 1)).clamp_min(1e-12)                  # (m,)
    curv = d2 / (h_u.view(m, 1) ** 2)
    return curv.square().mean()


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
                 rate_cap: float | None = None, positive: bool = True,
                 rate_hard_clamp: bool = True) -> None:
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
        # Hard-clamp the trained rate to rate_cap (v21).  softplus(rho) is
        # otherwise unbounded, but the convolution/outer mesh is built around a
        # FIXED reference a_ref = 2*rate_cap (see _mesh); a channel whose rate
        # runs past rate_cap leaves the regime that mesh resolves, and the
        # psi-power recursion mis-assembles that pair -- an inflation that grows
        # without bound as the rate climbs (the v20 divergence).  Clamping keeps
        # every rate <= rate_cap, so a_jl = rho_j+rho_l <= 2*rate_cap = a_ref
        # always, i.e. inside the mesh's design range.  Lowering rate_cap does
        # NOT substitute for this: without the clamp the rate escapes any cap.
        self.rate_hard_clamp = bool(rate_hard_clamp)
        rates = np.geomspace(0.5, max(cap, 4.0), m)
        self.rho = nn.Parameter(torch.tensor(inv_softplus(rates)))

    def _rates(self) -> torch.Tensor:
        """Channel envelope rates, hard-clamped to rate_cap when enabled.

        clamp_max saturates the gradient above the cap, so a channel that
        reaches rate_cap simply stops sharpening rather than running off into
        the unresolved regime; that is exactly the intended behaviour."""
        r = nn.functional.softplus(self.rho)
        if self.rate_hard_clamp:
            r = r.clamp_max(self.rate_cap)
        return r

    def _net_out(self, x: torch.Tensor):
        feats = torch.stack([x, torch.exp(-self.k * x)], dim=-1)
        return self.net(feats)

    def log_parts(self, x: torch.Tensor):
        """Positive mode: raw_j = exp(net_j), so log raw = net directly — no
        exp/log round trip and no overflow."""
        if not self.positive:
            raise TypeError("log_parts requires positive=True")
        return self._net_out(x), self._rates()

    def parts(self, x: torch.Tensor):
        out = self._net_out(x)
        raw = torch.exp(out.clamp(-700.0, 700.0)) if self.positive else out
        return raw, self._rates()

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


class MaynardProfile(nn.Module):
    """The analytic near-optimal family g_A(t) = 1/(1 + A t), A ~ log k.

    Maynard's explicit choice: a Ritz solve over a few members of this family
    achieves M_k >= log k - O(1), i.e. within O(1) of the rigorous ceiling
    (k/(k-1)) log k.  Three things make it the right preflight (v22):

      (1) it anchors the evaluator in the REGIME THAT MATTERS -- smooth,
          log-k-scale channels, which g==1 / g==x never exercise;
      (2) it sets the target: the trained net must beat this number or the
          discovery is adding nothing over the closed form;
      (3) it is fixed (no parameters), so like the dictionary it cannot
          exploit anything: a crazy score here is an evaluator defect, in
          seconds, before GPU time is spent.

    rates = 0 (all structure is in raw), variation scale 1/A ~ 1/log k --
    comfortably inside every mesh's design range.
    """

    def __init__(self, k: int, m: int = 8) -> None:
        super().__init__()
        self.k, self.m = k, m
        lk = max(math.log(k), 1.0)
        A = np.geomspace(0.5 * lk, 6.0 * lk, m)
        self.register_buffer("A", torch.tensor(A))

    def parts(self, x: torch.Tensor):
        raw = 1.0 / (1.0 + x.unsqueeze(-1) * self.A.view(1, -1))
        rates = torch.zeros(self.m, dtype=x.dtype, device=x.device)
        return raw, rates

    def log_parts(self, x: torch.Tensor):
        lr = -torch.log1p(x.unsqueeze(-1) * self.A.view(1, -1))
        rates = torch.zeros(self.m, dtype=x.dtype, device=x.device)
        return lr, rates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, _ = self.parts(x)
        return raw








# ---------------------------------------------------------------------------
# The separable Maynard problem
# ---------------------------------------------------------------------------


@dataclass
class GramParts:
    """This rank's slice of the Gram assembly.

    Avec/Bvec are DIFFERENTIABLE and cover only the local pairs; they are the
    only objects the gradient ever flows through. logAd is global and
    detached. Nothing here is a full m x m matrix — assembling one is a
    separate, detached step (see SeparableMaynard.assemble)."""
    Avec: torch.Tensor
    Bvec: torch.Tensor
    iu0: torch.Tensor
    iu1: torch.Tensor
    logAd: torch.Tensor
    plan: "PairPlan"
    xi: torch.Tensor | None = None
    # Widest per-channel log span, computed from the channel forward that the
    # Gram assembly ALREADY does. Recomputing it in report() cost a second
    # full MLP pass over pts_all on every validation point -- 82 s of 143 s
    # startup in a k=300 profile, all of it torch.tanh. Never re-evaluate the
    # network for a diagnostic.
    tail_span_local: torch.Tensor | None = None
    xi_hi: torch.Tensor | None = None
    xi_lo: torch.Tensor | None = None
    xi_jump_local: torch.Tensor | None = None


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
    tail_span: float = 0.0


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

    @staticmethod
    def _remark_66_bound(k: int, eps: float) -> float:
        """Polymath8b Remark 6.6: for 1/(1+eps) < a < 1/(1-eps),

            M_{k,eps} <= k/(a(k-1)) * log( k + (a(1+eps)-1)(k-1)/(1-a(1-eps)) )

        Sharper than Prop 6.5 at small eps -- at k=1000, eps=0.004 it gives
        6.9423 against Prop 6.5's 7.6080, i.e. barely above the vanilla 6.9147.
        That is the number which shows the eps enlargement buys almost nothing
        at usable eps, and it is worth PRINTING even though the uniform bound
        is what gets enforced.  a = 1 recovers Prop 6.5; the limit a -> 1/(1+eps)
        recovers Cor 6.4 as eps -> 0."""
        if k < 2:
            return float("inf")
        if eps <= 0.0:
            return (k / (k - 1.0)) * math.log(k)
        lo, hi = 1.0 / (1.0 + eps), 1.0 / (1.0 - eps)
        best = float("inf")
        n = 4001
        for i in range(1, n):
            a = lo + (hi - lo) * i / n
            den = 1.0 - a * (1.0 - eps)
            if den <= 0.0:
                continue
            arg = k + (a * (1.0 + eps) - 1.0) * (k - 1.0) / den
            if arg <= 0.0:
                continue
            v = k / (a * (k - 1.0)) * math.log(arg)
            if v < best:
                best = v
        return best


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
        # v4: number of LOCAL pairs materialised at once in detached assembly
        # and in the differentiable recomputation pass. 0 retains the legacy
        # all-local-pairs path; a small positive value makes peak activation
        # memory essentially independent of m.
        self.stream_pair_chunk = 0
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
        # ---- scalar single-channel cross-check (v19, advisory) ------------
        # The per-pair gate (1b) rejects a diagonal only above the GLOBAL bound
        # (k/(k-1))log k.  At k=3600 that ceiling is ~8.19, so a channel that
        # scores 8.13 by a mis-scaled convolution slips through even though its
        # true single-channel value is ~0.004.  This independent check closes
        # that blind spot: for a single channel k B_jj/A_jj is compared against
        # the direct 1-D moment k (int g)^2/int g^2, computed on a fine graded
        # grid -- a DIFFERENT quadrature from the Gauss-Jacobi convolution chain
        # that produces B_jj -- so a large gap on a channel whose mass sits well
        # inside the simplex localises a scale/Jacobian or inner-resolution bug
        # to that pair.  It is ADVISORY (a loud warning), not a hard gate: the
        # effective L2 width is not exact support, so a hard rejection could
        # false-fire on tail/quadrature effects; the rigorous stops stay with
        # gate 1b and gate 2.  The worst case is recorded in last_scalar_xcheck.
        self.scalar_xcheck = True          # warn on; --no-scalar-xcheck to mute
        self.scalar_xcheck_tol = 5e-2      # relative-gap threshold for the warning
        self.scalar_xcheck_massfrac = 1.0 - 1e-8   # width = where this L2 mass sits
        self.last_scalar_xcheck = None     # (j, R_conv, R_1D, rel) of worst narrow ch
        # Hard-gate backstop (v21): a narrow channel (k*sigma_eff <= 1, closed
        # form near-exact) whose convolution diagonal EXCEEDS its honest 1-D
        # value by more than this multiple is a real inflation, not tail noise,
        # so it is rejected (raised as a ValidationError, handled by the Adam
        # loop's strike policy exactly like a bound violation).  Directional:
        # only fires when R_conv > mult * R_1D, so a legitimately
        # constraint-active channel (where R_conv < R_1D) is never rejected.
        # This catches subtle inflation that sneaks UNDER the global bound gate
        # (the k=3600 8.13 < 8.19 case); gross violations are still caught by
        # gate 1b first.  Set mult high or use --no-scalar-xcheck-hardfail off.
        self.scalar_xcheck_hardfail_mult = 3.0
        # Moment upper-bound gate (v22): reject ANY diagonal that exceeds its
        # own unconstrained moment k (int g)^2/int g^2 by more than this
        # relative tolerance.  None disables.  See _gate_scalar_crosscheck.
        self.moment_gate_tol = 0.10
        # Rigorous upper bound on the variational constant.
        #
        # eps = 0 : Corollary 6.4 of Polymath8b,  M_k <= (k/(k-1)) log k.
        #   Proof is two lines and holds for EVERY square-integrable F on R_k:
        #   with G_i = (k-1)/log k * 1/(1 - sum t + k t_i) one has
        #   int G_i dt_i = 1 exactly, and sum_i 1/G_i = (k/(k-1)) log k
        #   identically on R_k; Cauchy-Schwarz (Lemma 6.1) then gives the
        #   bound.  It is a theorem, not a calibrated threshold.
        #
        # eps > 0 : Proposition 6.5,  M_{k,eps} <= (k/(k-1)) log(2k-1),
        #   UNIFORMLY IN eps.  Earlier versions fell back to the trivial
        #   Cauchy-Schwarz bound R <= k here, which at k=5000 means 5000
        #   instead of 9.21 -- i.e. no gate at all.  That is exactly how an
        #   eps run once reached R = 968 against a true value near 5.7 with
        #   nothing complaining, while the vanilla gate caught the same
        #   configuration at iteration 0.  Prop 6.5 is only ~10% looser than
        #   Cor 6.4 (the gap tends to log 2), so it costs essentially nothing
        #   in legitimate headroom and restores a real gate.
        #
        # Remark 6.6 gives a sharper eps-dependent bound; it is reported as a
        # diagnostic (see R_bound_sharp) but NOT used to reject, since its
        # derivation involves an optimisation over a free parameter and the
        # uniform Prop 6.5 bound is the safer thing to enforce.
        if k < 2:
            self.R_bound = float("inf")
            self.R_bound_name = "none (k < 2)"
        elif epsilon > 0.0:
            self.R_bound = (k / (k - 1.0)) * math.log(2.0 * k - 1.0)
            self.R_bound_name = "Prop 6.5: (k/(k-1)) log(2k-1)"
        else:
            self.R_bound = (k / (k - 1.0)) * math.log(k)
            self.R_bound_name = "Cor 6.4: (k/(k-1)) log k"
        # Remark 6.6, sharp in eps, for reporting only.
        self.R_bound_sharp = self._remark_66_bound(k, epsilon)
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
        # ---- inner-integral exponential-scaling cap (v20) ------------------
        # The inner integral G(Y) = int_0^Y raw(u) e^{-rho u} du is fixed-node
        # Gauss-Legendre in u.  When rho*Y is large, e^{-rho u} is a boundary
        # layer at u=0 that no interior u-node resolves, and the rule collapses
        # to its first weight: G/Y -> w_gl[0], so k B_jj/A_jj -> k*w_gl[0], a
        # channel-INDEPENDENT artifact (this is the k=3600 -> 8.13 and k=3000 ->
        # 6.75 inflation, = k*w_gl[0] at ng=40).  Fix: for rho*Y >= inner_z_cap
        # integrate in z = rho*u instead, on z in [0, Z], Z = min(rho*Y, cap),
        # where e^{-z} is O(1) and a fixed GL rule resolves it exactly.  The
        # discarded tail is e^{-cap}; cap=60 -> ~9e-27, deep in float64 noise.
        # rho*Y < cap keeps the original u-rule verbatim (it is accurate far
        # past cap, and this avoids the 1/rho in the z-form when rho -> 0).
        self.inner_z_cap = 60.0
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
        self._pair_cache: dict = {}
        # When True, plan() hands every rank the full pair set. Used by the
        # --dist-selftest reference pass, and available as an escape hatch.
        self._force_replica: bool = False

        # fixed scan grid for locating the outer-integral mode (detached)
        scan = np.concatenate([np.geomspace(1e-12, 1.0, 400), np.linspace(0.0, 1.0, 200)])
        scan = np.unique(np.clip(scan, 1e-14, 1.0))
        self.scan = torch.tensor(scan, dtype=dtype, device=self.device)

    # ---- pair bookkeeping ---------------------------------------------------
    def _pairs(self, m: int):
        """LOCAL pair index tensors for this rank. Everything downstream
        (_xi_power, _log_inner, the outer integrals) is index-agnostic — it
        only ever sees iu0/iu1 — so sharding is invisible to the kernels."""
        pl = self.plan(m)
        return pl.iu0, pl.iu1

    def plan(self, m: int) -> PairPlan:
        """Round-robin the m(m+1)/2 triu pairs across ranks.

        Round-robin rather than contiguous blocks for two reasons: per-pair
        cost is uniform (n_out, nq, N are all fixed), so any balanced split
        works for time; but a CONTIGUOUS split would give rank 0 all the pairs
        (0,l) and rank W-1 all the pairs (m-1,l), i.e. each rank would see a
        narrow band of envelope rates. Round-robin keeps every rank's outer
        windows spread over the whole range of a = rho_j + rho_l, which keeps
        per-rank timing uniform even if the mesh work ever becomes a-dependent.
        """
        key = ("plan", m, self._force_replica)
        if key in self._pair_cache:
            return self._pair_cache[key]
        iu = torch.triu_indices(m, m, device=self.device)
        P = iu.shape[1]
        world, rank = DIST.world, DIST.rank
        if world == 1 or self._force_replica or P < world:
            mode = "replica"
            idx = torch.arange(P, device=self.device)
        else:
            mode = "shard"
            idx = torch.arange(rank, P, world, device=self.device)
        pl = PairPlan(mode=mode, m=m, P_global=P, idx=idx,
                      iu0=iu[0][idx].contiguous(), iu1=iu[1][idx].contiguous())
        self._pair_cache[key] = pl
        return pl

    def _reduce_pairvec(self, vec_loc: torch.Tensor, pl: PairPlan):
        """Scatter a local (P_loc,) result into a global (P,) vector and
        all-reduce. DETACHED by construction: this path feeds the eigensolve
        and report(), never the gradient (see rayleigh() for the gradient
        route, which never materialises a global differentiable vector)."""
        if pl.mode == "replica":
            return vec_loc.detach()
        full = vec_loc.new_zeros(pl.P_global)
        full[pl.idx] = vec_loc.detach()
        DIST.all_reduce_(full, "sum")
        return full

    def _global_logAd(self, logA_loc: torch.Tensor, pl: PairPlan):
        """The diagonal entries logA_jj, gathered across ranks.

        This is the ONE genuine cross-shard dependency in the assembly: the
        preconditioner D = diag(A)^{-1/2} needs all m diagonal pairs, and pair
        (j,j) lives on whichever rank owns it. It is cheap and detachable —
        the congruence A -> D A D, B -> D B D leaves lambda(B,A) exactly
        invariant, so `half` is detached in the original code too and nothing
        differentiable crosses the wire here.
        """
        m = pl.m
        is_diag = (pl.iu0 == pl.iu1)
        buf = logA_loc.new_zeros(m)
        buf[pl.iu0[is_diag]] = logA_loc.detach()[is_diag]
        if pl.mode == "shard":
            # each (j,j) is owned by exactly one rank, so SUM over a
            # zero-filled buffer is a gather
            DIST.all_reduce_(buf, "sum")
        return buf

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

    def _log_inner(self, g, log_raw, rates, iu0, iu1):
        """log Ghat_j and log Hhat_jl at the representation nodes, in log space,
        with the linear zero factored out (G = Y*Ghat).

            G_j(Y)   = int_0^Y raw_j(u) e^{-rho_j u} du
            H_jl(Y)  = int_0^Y raw_j(u) raw_l(u) e^{-a_jl u} du,  a_jl = rho_j+rho_l

        For rate*Y < inner_z_cap this is the original fixed-node Gauss-Legendre
        rule in u (accurate: the boundary layer 1/rate is wider than the first
        u-node until rate*Y ~ 1/x_gl[0] ~ 2500, far past the cap).  For
        rate*Y >= cap the u-rule collapses to its first weight (the k*w_gl[0]
        inflation); there we substitute z = rate*u and integrate on
        z in [0, Z], Z = min(rate*Y, cap) = cap in this branch, where e^{-z} is
        O(1) and the SAME Gauss-Legendre rule resolves it to machine precision:

            Ghat_j(Y) = (cap/(rho_j Y)) sum_r w_r raw_j(cap*x_r/rho_j) e^{-cap*x_r}.

        rate >= cap in this branch, so 1/rate is safe; rate ~ 0 stays in the
        u-branch.  The channel is evaluated at cap*x_r/rate (in [0, cap/rate]
        subset [0, L]); those points are runtime-dependent, hence the fresh
        net evaluation here rather than a precomputed pts_all block.
        """
        N, ng = self.N, self.ng
        C = self.inner_z_cap
        lr = log_raw[self._sl_inner].reshape(N, ng, -1)      # (N, ng, m)
        pts = self.pts_all[self._sl_inner].reshape(N, ng)    # (N, ng) = Y_i x_gl
        lw = self.w_gl.clamp_min(1e-300).log().view(1, ng, 1)
        Y = self.Xrep                                        # (N,)

        # ---- unsaturated u-rule (verbatim original computation) -----------
        expo = -pts.unsqueeze(-1) * rates.view(1, 1, -1)
        logGhat = torch.logsumexp(lw + lr + expo, dim=1)     # (N, m)
        lh = lr[..., iu0] + lr[..., iu1]
        a = rates[iu0] + rates[iu1]                          # (P,)
        eh = -pts.unsqueeze(-1) * a.view(1, 1, -1)
        logHhat = torch.logsumexp(lw + lh + eh, dim=1)       # (N, P)

        # ---- saturated z-rule, only where rate*Y >= cap -------------------
        # The passed-in log_raw is L2-NORMALISED (log raw - lognorm). Raw
        # evaluated fresh below is UNnormalised, so we recompute the same
        # per-channel lognorm and subtract it -- otherwise the saturated
        # entries would carry a different per-channel scale than the
        # unsaturated ones and the ratio would be corrupted. Recomputed from
        # the net (rather than threaded in) to keep this change local to
        # _log_inner; it matches _normalized_log_channels exactly and only
        # runs when some node actually saturates.
        lw1 = self.w_gl.clamp_min(1e-300).log().view(1, ng)  # (1, ng)
        zc = C * self.x_gl                                   # (ng,) z-nodes
        lzc = (-zc).view(1, ng)                              # log e^{-z}
        logY = Y.clamp_min(1e-300).log()                     # (N,)

        satG = (rates.view(1, -1) * Y.view(-1, 1)) >= C      # (N, m)
        satH = (a.view(1, -1) * Y.view(-1, 1)) >= C          # (N, P)
        if bool(satG.any()) or bool(satH.any()):
            # per-channel L2 normalisation, identical to _normalized_log_channels
            lraw_rep, _ = channel_log_parts(g, self.pts_all[self._sl_rep])
            gr2 = 2.0 * (lraw_rep - self.Xrep.unsqueeze(-1) * rates.unsqueeze(0))
            lognorm = 0.5 * torch.logsumexp(
                self.w_cc.clamp_min(1e-300).log().unsqueeze(-1) + gr2, dim=0)  # (m,)
        if bool(satG.any()):
            rho_s = rates.clamp_min(1e-300)                  # (m,)
            m = rates.shape[0]
            # per-channel points cap*x_r/rho_j, clamped into [0,L] (a no-op for
            # saturated channels, where rho>=cap => cap/rho<=L; guards the net
            # against out-of-domain eval on the unsaturated rows we discard).
            ptsG = (C * self.x_gl.view(1, ng) / rho_s.view(-1, 1)).clamp(0.0, self.L)
            lrG, _ = channel_log_parts(g, ptsG.reshape(-1))  # (m*ng, m)
            lrG = lrG.reshape(m, ng, m)
            # gather channel j at its own points, then normalise (subtract lognorm_j)
            jj = torch.arange(m, device=lrG.device).view(m, 1, 1).expand(m, ng, 1)
            lrGd = torch.gather(lrG, 2, jj).squeeze(-1) - lognorm.view(m, 1)  # (m, ng)
            sumG = torch.logsumexp(lw1 + lrGd + lzc, dim=1)  # (m,)
            logGhat_s = (math.log(C) - rho_s.log().view(1, -1)
                         - logY.view(-1, 1) + sumG.view(1, -1))   # (N, m)
            logGhat = torch.where(satG, logGhat_s, logGhat)

        if bool(satH.any()):
            a_s = a.clamp_min(1e-300)                        # (P,)
            P = a.shape[0]
            ptsH = (C * self.x_gl.view(1, ng) / a_s.view(-1, 1)).clamp(0.0, self.L)
            lrH, _ = channel_log_parts(g, ptsH.reshape(-1))  # (P*ng, m)
            lrH = lrH.reshape(P, ng, -1)
            # gather channels j,l at the pair's points, each normalised by its lognorm
            pj = (torch.gather(lrH, 2, iu0.view(P, 1, 1).expand(P, ng, 1)).squeeze(-1)
                  - lognorm[iu0].view(P, 1))
            pl = (torch.gather(lrH, 2, iu1.view(P, 1, 1).expand(P, ng, 1)).squeeze(-1)
                  - lognorm[iu1].view(P, 1))
            sumH = torch.logsumexp(lw1 + pj + pl + lzc, dim=1)   # (P,)
            logHhat_s = (math.log(C) - a_s.log().view(1, -1)
                         - logY.view(-1, 1) + sumH.view(1, -1))   # (N, P)
            logHhat = torch.where(satH, logHhat_s, logHhat)

        return logGhat, logHhat

    def _normalized_log_channels(self, g: nn.Module):
        """Evaluate and L2-normalise every channel in log space.

        This is deliberately recomputed for every streamed gradient chunk.
        The ChannelNet evaluation is tiny compared with the pair convolution,
        while recomputation lets autograd release the entire chunk graph after
        each ``autograd.grad`` call.
        """
        log_raw, rates = channel_log_parts(g, self.pts_all)
        gr2 = 2.0 * (log_raw[self._sl_rep]
                     - self.Xrep.unsqueeze(-1) * rates.unsqueeze(0))
        lognorm = 0.5 * torch.logsumexp(
            self.w_cc.clamp_min(1e-300).log().unsqueeze(-1) + gr2, dim=0)
        return log_raw - lognorm.unsqueeze(0), rates

    def _gram_log_chunk(self, g, log_raw: torch.Tensor, rates: torch.Tensor,
                        iu0: torch.Tensor, iu1: torch.Tensor):
        """Unpreconditioned ``(logA, logB, xi)`` for one local pair chunk."""
        a = rates[iu0] + rates[iu1]
        xi = self._xi_power(log_raw, iu0, iu1)
        logG, logH = self._log_inner(g, log_raw, rates, iu0, iu1)
        logA = self._outer_integral_log_shared(
            xi, a, self.L, logH, iu0, iu1, True)
        logB = self._outer_integral_log_shared(
            xi, a, self.U, logG, iu0, iu1, False)
        return logA, logB, xi

    def _stream_slices(self, n: int):
        c = self.stream_pair_chunk if self.stream_pair_chunk > 0 else n
        c = max(1, min(c, max(n, 1)))
        for start in range(0, n, c):
            yield slice(start, min(start + c, n))

    @torch.no_grad()
    def _gram_log_parts_streamed_detached(self, g: nn.Module) -> GramParts:
        """Detached first pass, storing only two scalars per local pair.

        No convolution or outer-integral tensor ever contains more than
        ``stream_pair_chunk`` pairs.  The returned vectors are small O(P_loc)
        objects used for assembly/eigensolve/reporting.
        """
        log_raw, rates = self._normalized_log_channels(g)
        m = log_raw.shape[1]
        pl = self.plan(m)
        nloc = pl.idx.numel()
        logA_all = log_raw.new_empty(nloc)
        logB_all = log_raw.new_empty(nloc)
        xi_hi = log_raw.new_tensor(-float("inf"))
        xi_lo = log_raw.new_tensor(float("inf"))
        xi_jp = log_raw.new_zeros(())
        for sl in self._stream_slices(nloc):
            i0, i1 = pl.iu0[sl], pl.iu1[sl]
            la, lb, xi = self._gram_log_chunk(g, log_raw, rates, i0, i1)
            logA_all[sl], logB_all[sl] = la, lb
            xi_hi = torch.maximum(xi_hi, xi.amax())
            xi_lo = torch.minimum(xi_lo, xi.amin())
            if xi.shape[0] > 1:
                xi_jp = torch.maximum(xi_jp,
                                      (xi[1:] - xi[:-1]).abs().amax())
            del la, lb, xi
        logAd = self._global_logAd(logA_all, pl)
        half = 0.5 * (logAd[pl.iu0] + logAd[pl.iu1])
        Avec = torch.exp((logA_all - half).clamp(-700.0, 700.0))
        Bvec = torch.exp((logB_all - half).clamp(-700.0, 700.0))
        return GramParts(Avec=Avec, Bvec=Bvec, iu0=pl.iu0, iu1=pl.iu1,
                         logAd=logAd, plan=pl, xi=None,
                         tail_span_local=self._tail_span(log_raw),
                         xi_hi=xi_hi, xi_lo=xi_lo,
                         xi_jump_local=xi_jp)

    def _tail_span(self, log_raw: torch.Tensor) -> torch.Tensor:
        """max_j [ max_x log g_j - min_x log g_j ] over the rep nodes.

        Free: log_raw is the channel forward the Gram assembly already made."""
        lr = log_raw[self._sl_rep]
        return (lr.amax(dim=0) - lr.amin(dim=0)).amax().detach().reshape(1)

    def _gram_log_parts(self, g: nn.Module):
        """LOCAL, DIFFERENTIABLE (Avec, Bvec) over this rank's pairs.

        The channel forward and its L2 normalisation are per-CHANNEL, not
        per-pair, so every rank computes them redundantly for all m channels.
        That is deliberate: it costs one small MLP evaluation and buys zero
        communication on the channel side, whereas splitting it would need a
        gather of log_raw (n_pts x m) every iteration.
        """
        log_raw, rates = self._normalized_log_channels(g)
        m = log_raw.shape[1]
        pl = self.plan(m)
        iu0, iu1 = pl.iu0, pl.iu1
        a = rates[iu0] + rates[iu1]
        xi = self._xi_power(log_raw, iu0, iu1)
        logG, logH = self._log_inner(g, log_raw, rates, iu0, iu1)

        logA = self._outer_integral_log_shared(xi, a, self.L, logH, iu0, iu1, True)
        logB_ = self._outer_integral_log_shared(xi, a, self.U, logG, iu0, iu1, False)

        logAd = self._global_logAd(logA, pl)                   # (m,) detached
        half = 0.5 * (logAd[iu0] + logAd[iu1])                 # (P_loc,)
        Avec = torch.exp((logA - half).clamp(-700.0, 700.0))
        Bvec = torch.exp((logB_ - half).clamp(-700.0, 700.0))
        return GramParts(Avec=Avec, Bvec=Bvec, iu0=iu0, iu1=iu1,
                         logAd=logAd, plan=pl, xi=xi,
                         tail_span_local=self._tail_span(log_raw))

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
    def gram_parts(self, g: nn.Module) -> GramParts:
        """This rank's differentiable slice of A and B, preconditioned so that
        the GLOBAL diag(A) = 1."""
        if self.channel_sign == "positive":
            if self.stream_pair_chunk > 0 and not torch.is_grad_enabled():
                return self._gram_log_parts_streamed_detached(g)
            return self._gram_log_parts(g)
        return self._gram_free_parts(g)

    def assemble(self, parts: GramParts):
        """Full m x m A, B, DETACHED and bitwise identical on every rank.

        Used by the eigensolve and by report(). Communication is one packed
        all-reduce of 2P doubles (P = m(m+1)/2): 1 MB at m = 512. The
        gradient does NOT come through here — see rayleigh()."""
        pl = parts.plan
        m = pl.m
        if pl.mode == "replica":
            Ag = parts.Avec.detach()
            Bg = parts.Bvec.detach()
        else:
            # One collective for both forms: same 2P-double payload as before,
            # but half the collective latency of two separate all-reduces.
            pairbuf = parts.Avec.new_zeros(2, pl.P_global)
            pairbuf[0, pl.idx] = parts.Avec.detach()
            pairbuf[1, pl.idx] = parts.Bvec.detach()
            DIST.all_reduce_(pairbuf, "sum")
            Ag, Bg = pairbuf[0], pairbuf[1]
        iu = torch.triu_indices(m, m, device=self.device)
        Af = Ag.new_zeros(m, m)
        Bf = Bg.new_zeros(m, m)
        Af[iu[0], iu[1]] = Ag
        Bf[iu[0], iu[1]] = Bg
        A = Af + Af.T - torch.diag_embed(Af.diagonal())
        B = Bf + Bf.T - torch.diag_embed(Bf.diagonal())
        return A, B

    def gram_matrices(self, g: nn.Module):
        """Returns (A, B, logA_diag) with A, B DIAGONALLY PRECONDITIONED so
        diag(A) = 1. The congruence is detached, and lambda(B,A) is exactly
        invariant under it, so both value and gradient are exact.

        NOTE (v17): A and B come back DETACHED. Under sharding no rank holds a
        differentiable global matrix — assembling one would require an
        autograd-aware all-reduce. Every existing caller of this method
        (preflight, report, the polish v-solve, --check-dense) already used it
        under no_grad; the gradient route is rayleigh(), which never
        materialises a global matrix at all."""
        parts = self.gram_parts(g)
        A, B = self.assemble(parts)
        return A, B, parts.logAd

    def _gram_free_parts(self, g: nn.Module) -> GramParts:
        """Sign-changing ('free') channels: linear recursion, diagnostic use
        at small k only. Sharded identically to the log path."""
        k, N, ng, L, U = self.k, self.N, self.ng, self.L, self.U
        raw, rates, env = self._channels(g)
        m = raw.shape[1]
        pl = self.plan(m)
        iu0, iu1 = pl.iu0, pl.iu1
        a = rates[iu0] + rates[iu1]                                # (P_loc,)

        psi, sigma = self._psi_power(raw, iu0, iu1)                # (N,P), (P,)

        # inner integrals at the representation nodes
        gi = (raw * env)[self._sl_inner].reshape(N, ng, -1)        # (N, ng, m)
        Grep = self.Xrep.unsqueeze(-1) * torch.einsum("v,qvj->qj", self.w_gl, gi)
        hi_ = gi[..., iu0] * gi[..., iu1]
        Hrep = self.Xrep.unsqueeze(-1) * torch.einsum("v,qvp->qp", self.w_gl, hi_)

        A_val, A_shift = self._outer_integral(psi, sigma, a, L, Hrep, iu0, iu1, True)
        B_val, B_shift = self._outer_integral(psi, sigma, a, U, Grep, iu0, iu1, False)

        logA = sigma + A_shift + A_val.detach().abs().clamp_min(1e-300).log()
        # diagonal preconditioner from the true log-diagonal of A, gathered
        # across ranks (pair (j,j) may be owned by any of them)
        logAd = self._global_logAd(logA, pl)                       # (m,)
        half = 0.5 * (logAd[iu0] + logAd[iu1])                     # (P_loc,)

        fA = torch.exp((sigma + A_shift - half).clamp(-700.0, 700.0)).detach()
        fB = torch.exp((sigma + B_shift - half).clamp(-700.0, 700.0)).detach()
        return GramParts(tail_span_local=self._tail_span(
                             raw.abs().clamp_min(1e-300).log()),
                         Avec=A_val * fA, Bvec=B_val * fB, iu0=iu0, iu1=iu1,
                         logAd=logAd, plan=pl, xi=None)

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
    def _solve_top_sync(self, Ar: torch.Tensor, Br: torch.Tensor):
        """_solve_top, but the answer comes from rank 0 and is broadcast.

        Letting each rank run its own eigh() on a nominally identical A would
        usually work and occasionally not: near a degenerate lambda_max the
        eigenvector is defined only up to the degenerate subspace, and two
        ranks that pick different representatives then optimise different
        objectives — a divergence that surfaces much later as an inexplicable
        loss of monotonicity in the polish. m doubles per call removes the
        failure mode. The eigensolve can also legitimately FAIL (A numerically
        zero after truncation); that must become the same exception on every
        rank, or the ranks that succeeded hang at the next collective."""
        m = Ar.shape[0]
        v = torch.zeros(m, dtype=self.ritz_dtype, device=self.device)
        aux = torch.zeros(2, dtype=torch.float64, device=self.device)
        err = 0
        if DIST.is_main or not DIST.enabled:
            try:
                lam0, v0, r0 = self._solve_top(Ar, Br)
                v.copy_(v0)
                aux[0], aux[1] = float(lam0), float(r0)
            except FloatingPointError:
                err = 1
        if not DIST.agree(err == 0, self.device):
            raise FloatingPointError(
                "eigensolve failed on rank 0: A has no positive numerical "
                "direction, or the Ritz vector has non-positive A-norm "
                "(verdict synchronised across ranks)")
        DIST.broadcast_(v, 0)
        DIST.broadcast_(aux, 0)
        return aux[0], v, int(aux[1])

    def _gate(self, ok: bool, msg: str) -> None:
        """Raise ValidationError UNANIMOUSLY. A gate that fires on one rank
        and not another is the classic collective deadlock: the failing rank
        unwinds out of report() while the others march on to the next
        all-reduce and block forever."""
        if not DIST.agree(bool(ok), self.device):
            raise ValidationError(msg)

    def solve_v(self, parts: GramParts) -> torch.Tensor:
        A, B = self.assemble(parts)
        _, v, r = self._solve_top_sync(A.to(self.ritz_dtype),
                                       B.to(self.ritz_dtype))
        self.last_rank = r
        return v

    def rayleigh(self, g: nn.Module, ridge: float = 0.0, grad_mode: str = "hf",
                 frozen_v: torch.Tensor | None = None) -> torch.Tensor:
        """R = k lambda_max(B, A) by Hellmann-Feynman / Danskin: solve the
        m x m pencil outside autograd, detach v, return k (v'Bv)/(v'Av).
        Value exactly k*lambda at the current point; gradient exact by the
        envelope theorem; a minorant of k*lambda_max for every theta, hence
        safe under a line search.

        SHARDED FORM. Because v is detached, both quadratic forms are plain
        SUMS over pairs,

            v'Bv = sum_p c_p B_p,   v'Av = sum_p c_p A_p,
            c_p = (2 - delta_{jl}) v_j v_l,

        so each rank contributes num_loc, den_loc; the two VALUES are
        all-reduced; and the local surrogate

            surr_r = (k/den) num_r - (k num/den^2) den_r

        differentiates to this rank's exact share of dR/dtheta by the quotient
        rule, with num and den frozen. Summing the parameter gradients across
        ranks (Dist.sync_grads, op=SUM) therefore gives the EXACT global
        gradient — no approximation, and no autograd-aware collective.

        surr is identically zero in value (it is a directional derivative), so
        the true R is added back detached: correct value, correct gradient.
        """
        if grad_mode == "backprop":
            raise NotImplementedError(
                "the differentiated jittered pencil is inconsistent with the "
                "ridge-free HF route and was removed in v7; use --grad-mode hf")
        parts = self.gram_parts(g)
        v = (self.solve_v(parts) if frozen_v is None else frozen_v).detach()
        v = v.to(dtype=parts.Avec.dtype)

        # (2 - delta_{jl}) because A, B are stored as upper triangles
        w = torch.where(parts.iu0 == parts.iu1, 1.0, 2.0).to(parts.Avec.dtype)
        c = w * v[parts.iu0] * v[parts.iu1]                      # (P_loc,)
        num_loc = (c * parts.Bvec).sum()
        den_loc = (c * parts.Avec).sum()

        if parts.plan.mode == "shard":
            num_g, den_g = DIST.sum_floats([num_loc, den_loc], self.device)
        else:
            num_g, den_g = float(num_loc.detach()), float(den_loc.detach())
        if not math.isfinite(den_g) or den_g == 0.0:
            raise FloatingPointError("non-positive A-norm of the Ritz vector")

        R_val = self.k * num_g / den_g
        surr = (self.k / den_g) * num_loc - (self.k * num_g / den_g**2) * den_loc
        return surr - surr.detach() + R_val

    def rayleigh_backward_streamed(self, g: nn.Module, *, sign: float = 1.0,
                                   frozen_v: torch.Tensor | None = None) -> torch.Tensor:
        """Two-pass exact Hellmann--Feynman value and gradient.

        Pass 1 is detached and streamed over local pairs.  It stores only
        ``logA_p`` and ``logB_p``, assembles the global pencil, solves/broadcasts
        the Ritz vector, and obtains the frozen global numerator/denominator.

        Pass 2 recomputes one pair chunk at a time with autograd enabled and
        immediately calls ``torch.autograd.grad`` on that chunk's exact share
        of the quotient-rule surrogate.  The chunk graph is then released.
        Consequently peak activation memory is O(stream_pair_chunk), not
        O(P_local), while the accumulated gradient is algebraically identical
        to the legacy all-pairs backward.

        ``sign=-1`` writes the gradient of the minimisation loss ``-R``;
        ``sign=+1`` writes the gradient of ``R`` (used by diagnostics).
        Gradient synchronization across ranks remains the caller's job.
        """
        if self.channel_sign != "positive" or self.stream_pair_chunk <= 0:
            out = sign * self.rayleigh(g, frozen_v=frozen_v)
            out.backward()
            return out.detach()

        # ---------- pass 1: detached streamed values and global Ritz vector
        with torch.no_grad():
            parts0 = self._gram_log_parts_streamed_detached(g)
            if frozen_v is None:
                v = self.solve_v(parts0).detach()
            else:
                v = frozen_v.detach().to(self.device)
            v = v.to(dtype=parts0.Avec.dtype)
            w = torch.where(parts0.iu0 == parts0.iu1, 1.0, 2.0).to(v.dtype)
            coeff = w * v[parts0.iu0] * v[parts0.iu1]
            num_loc = (coeff * parts0.Bvec).sum()
            den_loc = (coeff * parts0.Avec).sum()
            if parts0.plan.mode == "shard":
                num_g, den_g = DIST.sum_floats(
                    [num_loc, den_loc], self.device)
            else:
                num_g = float(num_loc)
                den_g = float(den_loc)
            if not math.isfinite(den_g) or den_g <= 0.0:
                raise FloatingPointError(
                    "non-positive A-norm of the Ritz vector")
            R_val = self.k * num_g / den_g
            alpha = sign * self.k / den_g
            beta = sign * self.k * num_g / (den_g ** 2)
            logAd = parts0.logAd.detach()
            pl = parts0.plan
            coeff = coeff.detach()

        # ---------- pass 2: exact differentiable recomputation by pair chunk
        params = [p for p in g.parameters() if p.requires_grad]
        accum = [torch.zeros_like(p) for p in params]
        nloc = pl.idx.numel()
        for sl in self._stream_slices(nloc):
            # Fresh channel graph per chunk: autograd can free everything from
            # this chunk immediately after autograd.grad returns.
            log_raw, rates = self._normalized_log_channels(g)
            i0, i1 = pl.iu0[sl], pl.iu1[sl]
            logA, logB, _ = self._gram_log_chunk(g, log_raw, rates, i0, i1)
            half = 0.5 * (logAd[i0] + logAd[i1])
            Avec = torch.exp((logA - half).clamp(-700.0, 700.0))
            Bvec = torch.exp((logB - half).clamp(-700.0, 700.0))
            cc = coeff[sl]
            chunk_obj = alpha * (cc * Bvec).sum() - beta * (cc * Avec).sum()
            grads = torch.autograd.grad(chunk_obj, params,
                                        allow_unused=True,
                                        retain_graph=False,
                                        create_graph=False)
            with torch.no_grad():
                for a, gg in zip(accum, grads):
                    if gg is not None:
                        a.add_(gg)
            del log_raw, rates, logA, logB, Avec, Bvec, chunk_obj, grads

        with torch.no_grad():
            for p, gg in zip(params, accum):
                if p.grad is None:
                    p.grad = gg
                else:
                    p.grad.add_(gg)
        return torch.tensor(sign * R_val, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    @torch.no_grad()
    def _scalar_channel_moments(self, g: nn.Module):
        """Independent 1-D single-channel scores R_1D_j = k (int g_j)^2/int g_j^2
        and effective supports sigma_j.

        Evaluated on a FINE GRADED grid that resolves the sharpest channel's
        own scale 1/rate, NOT on the coarse rep grid.  This is deliberate: the
        Chebyshev rep grid under-resolves a narrow spike (one or two nodes land
        on it), so a moment read there is unreliable for exactly the narrow
        channels this cross-check targets -- and an unreliable R_1D could
        false-fire the gate on a channel whose convolution is actually correct.
        The graded grid (geometric limb from 1/(200 r_max) plus a uniform limb,
        the same construction the export uses) makes R_1D accurate and
        n_rep-independent, so the comparison against the convolution diagonal
        is trustworthy in both directions.  Cost is one small MLP forward on a
        few thousand points -- negligible next to the pair convolution.
        """
        rates = torch.nn.functional.softplus(g.rho).detach() if hasattr(g, "rho") \
            else None
        if rates is None:
            with torch.no_grad():
                _, rates = channel_parts(
                    g, torch.zeros(1, dtype=self.dtype, device=self.device))
            rates = rates.detach()
        r_max = float(rates.max().clamp_min(1e-30))
        L = self.L
        lo = min(1.0 / (200.0 * r_max), L * 1e-3)
        geo = torch.logspace(math.log10(lo), math.log10(L), 4000,
                             dtype=self.dtype, device=self.device)
        uni = torch.linspace(0.0, L, 2001, dtype=self.dtype, device=self.device)
        xf = torch.unique(torch.cat([geo, uni,
                                     torch.zeros(1, dtype=self.dtype,
                                                 device=self.device)]))
        xf, _ = torch.sort(xf)
        gr = g(xf)                                         # (P, m)
        dx = (xf[1:] - xf[:-1])                            # (P-1,)
        def trap(y):                                       # trapezoid over xf
            return (0.5 * (y[1:] + y[:-1]) * dx.unsqueeze(-1)).sum(dim=0)
        i1 = trap(gr)
        i2 = trap(gr.square()).clamp_min(1e-300)
        R1d = self.k * i1 * i1 / i2
        # effective support: cumulative L2 mass along the (sorted) fine grid
        seg = 0.5 * (gr[1:].square() + gr[:-1].square()) * dx.unsqueeze(-1)
        cum = torch.cumsum(seg, dim=0)                     # (P-1, m)
        tot = cum[-1].clamp_min(1e-300)
        thresh = self.scalar_xcheck_massfrac * tot
        reached = (cum >= thresh.unsqueeze(0)).to(torch.int8)
        idx = torch.argmax(reached, dim=0)                 # first segment past thresh
        sigma = xf[1:][idx]
        return R1d, sigma

    def _gate_scalar_crosscheck(self, g: nn.Module, Ar: torch.Tensor,
                                Br: torch.Tensor) -> None:
        """Cross-check every diagonal against an independent 1-D integral.

        B_jj/A_jj is invariant under the symmetric diagonal preconditioning
        (the scale d_j^2 cancels in the ratio), so k B_jj/A_jj is the true
        single-channel Rayleigh value.  Compared against the direct 1-D moment
        R_1D = k (int g)^2/int g^2 (evaluated on a fine graded grid, a DIFFERENT
        quadrature from the Gauss-Jacobi convolution that builds B_jj), a large
        gap on a channel whose L2 mass is concentrated well inside the simplex
        localises a scale/Jacobian or inner-resolution defect to that pair --
        this is the k=3600 channel-7 failure mode (8.13 vs 0.004).

        This is ADVISORY, not a hard gate.  `sigma` is an effective L2 width,
        not exact support: these channels have a nonzero tail out to x = L, so
        even at k*sigma_eff <= 1 the closed-form identity is only approximate
        and a hard rejection here could false-fire on legitimate tail or
        quadrature effects.  The rigorous rejections stay with the theorems
        that own them -- gate 1b (global per-pair bound) and gate 2
        (R <= (k/(k-1)) log k).  So we WARN loudly and record the worst case in
        `last_scalar_xcheck`, leaving the hard stop to the rigorous gates.
        """
        if not self.scalar_xcheck:
            return
        R1d, sigma = self._scalar_channel_moments(g)
        # module-agnostic rates (ChannelNet has .rho; FixedDictionary / LinearX
        # / ConstantOne do not -- reaching into g.rho AttributeErrors on the
        # dictionary baseline and the g==1 / g==x controls).
        if hasattr(g, "rho"):
            rates = torch.nn.functional.softplus(g.rho).detach()
        else:
            with torch.no_grad():
                _, rates = channel_parts(
                    g, torch.zeros(1, dtype=self.dtype, device=self.device))
            rates = rates.detach()
        # detach: the check only reads values; it must neither retain the pair
        # graph nor emit the requires_grad->scalar warning inside a grad context.
        Rconv = (self.k * Br.detach().diagonal()
                 / Ar.detach().diagonal().clamp_min(1e-300))
        ksig = self.k * sigma
        R1d_c = R1d.to(Rconv.dtype)
        rel = (Rconv - R1d_c).abs() / R1d_c.abs().clamp_min(1e-30)

        # ---- MOMENT UPPER-BOUND GATE (v22, ALL channels) -------------------
        # For a single channel, the unconstrained moment k (int g)^2 / int g^2
        # is an upper bound on the true constrained single-channel value: it is
        # exactly what the diagonal becomes when the simplex constraint is
        # dropped from BOTH quadratic forms, and empirically dominates the
        # constrained value with wide margin across every shape tested
        # (exponentials rho=0.5..40, bumps, two-scale, poly*exp, sharp-core;
        # k=50 reference by scale-stabilised direct convolution; and the exact
        # controls g==1: 2 <= k, g==x: 1 <= 3k/4).  A convolution diagonal
        # ABOVE this moment is therefore an assembly artifact, not a
        # variational value -- this caught the k=1000 channel-27 case
        # (R_conv = 5.737 vs moment 3.356, a wide channel that the
        # narrow-only check could not see).  Unlike the narrow check this
        # needs no support condition, so it covers the full width range.
        # Caveat, stated honestly: the domination is verified empirically and
        # on the controls, not yet proved as a theorem; the tolerance below
        # keeps ordinary quadrature noise on the right side of it, and the
        # strike policy makes a firing recoverable rather than fatal.
        if self.moment_gate_tol is not None:
            mviol = Rconv > R1d_c * (1.0 + self.moment_gate_tol)
            if bool(mviol.any()):
                jm = int(torch.argmax(torch.where(
                    mviol, Rconv / R1d_c.clamp_min(1e-30),
                    torch.zeros_like(Rconv))))
                self._gate(
                    False,
                    f"moment gate: channel {jm} convolution diagonal "
                    f"k B_jj/A_jj = {float(Rconv[jm]):.6f} exceeds its own "
                    f"unconstrained moment k (int g)^2/int g^2 = "
                    f"{float(R1d_c[jm]):.6f} by "
                    f"{float(Rconv[jm]/R1d_c[jm].clamp_min(1e-30)):.2f}x "
                    f"(tol {self.moment_gate_tol:.0%}). The moment is what the "
                    f"diagonal becomes with the simplex constraint dropped, so "
                    f"the constrained value cannot exceed it; this diagonal is "
                    f"an assembly artifact (e.g. sub-grid structure in raw "
                    f"that the psi recursion cannot represent). "
                    f"k*sigma_eff = {float(ksig[jm]):.3f}, "
                    f"rate = {float(rates[jm]) if jm < rates.numel() else float('nan'):.3e}. "
                    f"Lift with --no-moment-gate if intended.")

        # "sigma" is an effective L2 width, not an exact support.
        effective_narrow = ksig <= 1.0
        if bool(effective_narrow.any()):
            # Mask non-narrow channels BEFORE argmax and compare the MASKED
            # value: when every narrow channel agrees (rel ~ 0), a bare argmax
            # would fall on index 0 (possibly a WIDE channel) whose rel is the
            # meaningless unconstrained-moment gap.
            rel_eff = torch.where(effective_narrow, rel, torch.zeros_like(rel))
            jw = int(torch.argmax(rel_eff))
            rel_jw = float(rel_eff[jw])
            self.last_scalar_xcheck = (jw, float(Rconv[jw]),
                                       float(R1d_c[jw]), rel_jw)
            if rel_jw > self.scalar_xcheck_tol and DIST.is_main:
                rate_j = float(rates[jw]) if jw < rates.numel() else float("nan")
                print(
                    f"   [scalar-xcheck] WARNING: channel {jw} has "
                    f"k B_jj/A_jj = {float(Rconv[jw]):.6f}, while the "
                    f"unconstrained 1-D moment gives {float(R1d_c[jw]):.6e}; "
                    f"relative gap {rel_jw:.1%}. Its effective "
                    f"{self.scalar_xcheck_massfrac:.8%} L2 width satisfies "
                    f"k*sigma_eff = {float(ksig[jw]):.3e}. This is not an "
                    f"exact-support theorem because the channel has a nonzero "
                    f"tail, but a large discrepancy is strong evidence of a "
                    f"scale, Jacobian, export, or quadrature inconsistency. "
                    f"rate={rate_j:.3e}.")
            # hard-gate backstop: only in the INFLATION direction (R_conv far
            # above the honest 1-D value), so a constraint-active channel
            # (R_conv < R_1D) can never trip it.
            hm = self.scalar_xcheck_hardfail_mult
            if hm is not None and hm > 0:
                infl = float(Rconv[jw]) > hm * float(R1d_c[jw])
                self._gate(
                    not infl,
                    f"scalar cross-check HARD FAIL on narrow channel {jw}: "
                    f"convolution diagonal k B_jj/A_jj = {float(Rconv[jw]):.6f} "
                    f"exceeds {hm:g}x its independent 1-D value "
                    f"{float(R1d_c[jw]):.6e} (k*sigma_eff = {float(ksig[jw]):.3e} "
                    f"<= 1, so the closed form is near-exact and they must "
                    f"agree). This is a pair-assembly inflation the global "
                    f"bound did not catch -- e.g. a rate past rate_cap leaving "
                    f"the convolution mesh's design range. Enforce the rate "
                    f"clamp, or lift with --no-scalar-xcheck-hardfail.")
        # support-aware Cauchy-Schwarz smell test for the wider channels
        wide_viol = (~effective_narrow) & (Rconv > ksig * (1.0 + self.scalar_xcheck_tol))
        if bool(wide_viol.any()) and DIST.is_main:
            jv = int(torch.argmax(torch.where(wide_viol, Rconv,
                     torch.full_like(Rconv, -1.0))))
            print(f"   [scalar-xcheck] WARNING: channel {jv} diagonal "
                  f"k B_jj/A_jj = {float(Rconv[jv]):.4f} exceeds its "
                  f"support-aware ceiling k*sigma_eff = {float(ksig[jv]):.4f} "
                  f"(sigma_eff = {float(sigma[jv]):.3e}); the pair may be "
                  f"leaning on unresolved sharpness. Independent 1-D value is "
                  f"{float(R1d_c[jv]):.4e}.")

    def report(self, g: nn.Module, ridge: float = 0.0) -> RayleighReport:
        parts = self.gram_parts(g)
        A, B = self.assemble(parts)
        logAd = parts.logAd
        Ar, Br = A.to(self.ritz_dtype), B.to(self.ritz_dtype)
        # --- gate 1: A must be numerically PSD. A_{jl} = <Phi_j, Phi_l>, so
        # any eigenvalue materially below zero means the assembly is not a
        # Gram matrix. Rank truncation can DISCARD small directions; it cannot
        # repair a spectrum with a genuine negative branch.
        wA = torch.linalg.eigvalsh(0.5 * (Ar + Ar.T))
        self._gate(
            not (wA[-1] <= 0 or wA[0] < -self.psd_tol * wA[-1]),
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
        self._gate(
            not (float(dR[j_bad]) > self.R_bound * (1.0 + self.R_tol)),
                f"diagonal pair {j_bad} gives k B_jj/A_jj = "
                f"{float(dR[j_bad]):.6f} > bound {self.R_bound:.6f}. That "
                f"single-channel trial function is admissible, so this is a "
                f"defect in the assembly of THIS pair, not in the pencil.")
        self.last_diag_R = float(dR.max())
        # --- 1b' (advisory): INDEPENDENT scalar cross-check. Gate 1b only
        # rejects a diagonal above the GLOBAL bound; this warns when a diagonal
        # that is locally sane against that ceiling is inconsistent with the
        # direct 1-D integral of its own channel (the k=3600 channel-7 mode).
        # Warning-only by design -- see _gate_scalar_crosscheck.
        self._gate_scalar_crosscheck(g, Ar, Br)
        _, v, rank = self._solve_top_sync(Ar, Br)
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
        self._gate(
            not (rel_gap > self.stability_rtol),
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
        self._gate(
            math.isfinite(float(R)) and float(R) <= self.R_bound * (1.0 + self.R_tol),
                f"R = {float(R):.6f} exceeds the rigorous upper bound "
                f"M_k <= (k/(k-1)) log k = {self.R_bound:.6f} (Polymath8b; "
                f"tol {self.R_tol:.0e}). Every admissible trial function "
                f"satisfies R <= M_k, so the discretisation is being "
                f"exploited, not the variational problem solved. Raise "
                f"--n-rep and rerun.")
        # xi is already in hand from gram_parts on the positive path; only the
        # 'free' path has to redo the recursion. Either way it is LOCAL to
        # this rank's pairs, so the span and the adjacent-jump have to be
        # reduced or each rank would report a different resolution diagnostic.
        if parts.xi_hi is not None:
            xi_hi = parts.xi_hi.reshape(1).clone()
            xi_lo = parts.xi_lo.reshape(1).clone()
            xi_jp = parts.xi_jump_local.reshape(1).clone()
        else:
            if parts.xi is not None:
                xi = parts.xi
            elif self.channel_sign == "positive":
                log_raw, _ = self._normalized_log_channels(g)
                xi = self._xi_power(log_raw, parts.iu0, parts.iu1)
            else:
                raw, rates, _ = self._channels(g)
                psi, _ = self._psi_power(raw, parts.iu0, parts.iu1)
                xi = psi.abs().clamp_min(1e-300).log()
            # In the log-space formulation a large span of xi is EXPECTED.
            xi_hi = xi.amax().reshape(1).clone()
            xi_lo = xi.amin().reshape(1).clone()
            xi_jp = ((xi[1:, :] - xi[:-1, :]).abs().amax().reshape(1).clone()
                     if xi.shape[0] > 1 else xi_hi.new_zeros(1))
        if parts.plan.mode == "shard":
            DIST.all_reduce_(xi_hi, "max")
            DIST.all_reduce_(xi_lo, "min")
            DIST.all_reduce_(xi_jp, "max")
        xi_span = float((xi_hi - xi_lo).detach())
        xi_jump = float(xi_jp.detach())

        # ---- CHANNEL TAIL SPAN -------------------------------------------
        # Separates "narrow because the optimum IS narrow" from "narrow in a
        # way the grid cannot audit". Thm 6.7's optimal g = 1/(c+(k-1)t) spans
        # ~4 NATS over its support -- a power law, not a spike. Measured at
        # k=3000: ~37 nats (16 orders). The excess lives below every
        # quadrature node, which is the freedom the optimiser spends
        # invisibly. NOT xi_span: xi is the k-fold convolution in log space,
        # where a huge span is expected and harmless.
        ts = parts.tail_span_local
        if ts is None:
            ts = torch.zeros(1, device=self.device, dtype=self.dtype)
        else:
            ts = ts.clone()
        if parts.plan.mode == "shard":
            DIST.all_reduce_(ts, "max")
        tail_span = float(ts)

        return RayleighReport(R=float(R), c=v.cpu().numpy(),
                              A=Ar.cpu().numpy(), B=Br.cpu().numpy(),
                              logA_diag=logAd.cpu().numpy(), rank=rank,
                              xi_span=xi_span, xi_jump=xi_jump,
                              tail_span=tail_span)


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
          excursion_factor: float = 3.0, excursion_floor: float = 2.0,
          raw_smooth: float = 0.0):
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
                if dump_violation and strikes == 1 and DIST.is_main:
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
                    if dump_violation and strikes == 1 and DIST.is_main:
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
        loss = problem.rayleigh_backward_streamed(g, sign=-1.0)
        if raw_smooth > 0.0:
            # v22: curvature penalty on log raw in the envelope-scaled
            # coordinate. rayleigh_backward_streamed has ACCUMULATED the exact
            # gradient of -R into .grad; this backward() adds the penalty's
            # gradient on top before the (clipped) step. Value is small for
            # smooth channels and explodes for sub-envelope-scale spikes.
            pen = raw_smooth * raw_curvature_penalty(g, L=problem.L)
            pen.backward()
        # The streamed routine has already accumulated the exact gradient of
        # -R chunk by chunk. Its VALUE is globally reduced, so this verdict is
        # unanimous and cannot split the ranks.
        if not torch.isfinite(loss):
            g.load_state_dict(last_good)
            raise FloatingPointError(
                f"non-finite loss at iter {it}; restored last validated state")
        # SUM this rank's share of dR/dtheta into the global gradient BEFORE
        # clipping — clipping a partial gradient would apply a different scale
        # factor on every rank and the result would not be a gradient of
        # anything.
        DIST.sync_grads(g, problem.plan(g.m).mode)
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
            _, v, _ = problem._solve_top_sync(A.to(problem.ritz_dtype),
                                              B.to(problem.ritz_dtype))
        v_frozen = v.to(dtype=problem.dtype).detach()
        opt = torch.optim.LBFGS(
            g.parameters(), lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn="strong_wolfe")

        # The strong-Wolfe line search calls the closure a data-dependent
        # number of times. That is safe here only because both the returned
        # LOSS and the synced GRADIENT are globally reduced, so every rank
        # sees identical numbers and therefore takes identical line-search
        # decisions. Without the sync inside the closure the ranks would
        # accept different step lengths and deadlock at the next collective.
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = problem.rayleigh_backward_streamed(
                g, sign=-1.0, frozen_v=v_frozen)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite LBFGS loss")
            DIST.sync_grads(g, problem.plan(g.m).mode)
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
def _flat_grad(module: nn.Module) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None
                       else torch.zeros_like(p)).reshape(-1)
                      for p in module.parameters()])


def dist_selftest(problem: SeparableMaynard, g: nn.Module,
                  rtol: float = 1e-9) -> bool:
    """Sharded value+gradient vs. a fully replicated reference.

    This is the load-bearing check for the whole multi-GPU path. It exercises,
    in one shot: the round-robin pair split; the cross-rank gather of
    diag(A) that the preconditioner needs; the two-scalar reduction of num and
    den; the surrogate's value-restoration trick; and the SUM all-reduce of
    the parameter gradients. If the surrogate were wrong by so much as the
    quotient-rule sign, or the diagonal gather picked up a pair twice, this
    disagrees immediately.

    The reference pass sets _force_replica, which hands EVERY rank the full
    pair set — so it computes exactly what a single-GPU run computes, on every
    rank, with no reduction of pair data at all.
    """
    dev = problem.device
    g.zero_grad(set_to_none=True)
    Rs = problem.rayleigh_backward_streamed(g, sign=1.0)
    R_shard = float(Rs)
    DIST.sync_grads(g, problem.plan(g.m).mode)
    mode_shard = problem.plan(g.m).mode
    grad_shard = _flat_grad(g).clone()

    problem._force_replica = True
    try:
        g.zero_grad(set_to_none=True)
        Rr = problem.rayleigh_backward_streamed(g, sign=1.0)
        R_ref = float(Rr)
        DIST.sync_grads(g, problem.plan(g.m).mode)
        grad_ref = _flat_grad(g).clone()
    finally:
        problem._force_replica = False
    g.zero_grad(set_to_none=True)

    dR = abs(R_shard - R_ref) / max(abs(R_ref), 1e-30)
    dg = float((grad_shard - grad_ref).norm()
               / grad_ref.norm().clamp_min(1e-300))
    # every rank must agree on the verdict, and the numbers themselves are a
    # useful cross-rank consistency probe, so print them from every rank
    ok = (dR <= rtol) and (dg <= rtol * 1e3)
    print_all(f"dist self-test  world={DIST.world}  mode={mode_shard}  "
              f"m={g.m}  P={problem.plan(g.m).P_global}  "
              f"P_local={problem.plan(g.m).idx.numel()}")
    print_all(f"  R  sharded {R_shard:.15f}   replicated {R_ref:.15f}   "
              f"rel {dR:.2e}")
    print_all(f"  |grad_shard - grad_ref| / |grad_ref| = {dg:.2e}   "
              f"(gate {rtol*1e3:.0e})")
    ok = DIST.agree(ok, dev)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else "  — the sharded gradient is NOT the true "
                           "gradient; do not run a multi-GPU campaign."))
    return ok


def dist_trajectory_selftest(problem: SeparableMaynard, g: nn.Module,
                             adam_steps: int = 5) -> bool:
    """Exercise optimizer-state synchronization beyond the algebraic check.

    Runs a short sharded Adam trajectory, checks that all ranks remain
    parameter-identical, then executes one deliberately tiny LBFGS/MM step.
    This does not compare against a separate single-process job, but it covers
    the distributed closure count, gradient reductions, optimizer states and
    parameter locking that the pointwise self-test cannot see.
    """
    dev = problem.device
    opt = torch.optim.Adam(g.parameters(), lr=1e-4)
    for _ in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        loss = problem.rayleigh_backward_streamed(g, sign=-1.0)

        DIST.sync_grads(g, problem.plan(g.m).mode)
        opt.step()

    flat = torch.cat([p.detach().reshape(-1) for p in g.parameters()])
    ref = flat.clone()
    DIST.broadcast_(ref, 0)
    adam_rel = float((flat - ref).norm() / ref.norm().clamp_min(1e-300))

    with torch.no_grad():
        parts = problem.gram_parts(g)
        v = problem.solve_v(parts).to(problem.dtype).detach()
    lb = torch.optim.LBFGS(g.parameters(), lr=1e-3, max_iter=2, max_eval=4,
                           history_size=2, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad(set_to_none=True)
        loss = problem.rayleigh_backward_streamed(
            g, sign=-1.0, frozen_v=v)
        DIST.sync_grads(g, problem.plan(g.m).mode)
        return loss

    lb.step(closure)
    flat2 = torch.cat([p.detach().reshape(-1) for p in g.parameters()])
    ref2 = flat2.clone()
    DIST.broadcast_(ref2, 0)
    lbfgs_rel = float((flat2 - ref2).norm() / ref2.norm().clamp_min(1e-300))
    ok = DIST.agree(adam_rel <= 1e-13 and lbfgs_rel <= 1e-13, dev)
    print_all(f"trajectory self-test: Adam rank drift={adam_rel:.2e}, "
              f"LBFGS rank drift={lbfgs_rel:.2e}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Separable-DeepSets Maynard M_k solver, v17/v4 streamed "
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
    p.add_argument("--pair-chunk", type=int, default=0,
                   help="legacy in-recursion pair chunk; does not by itself "
                        "release autograd graphs")
    p.add_argument("--stream-pair-chunk", type=int, default=16,
                   help="v4 two-pass streaming chunk over LOCAL pairs. The "
                        "detached assembly and differentiable recomputation "
                        "never materialise more than this many pairs at once; "
                        "0 disables streamed backward and uses the v3 path")
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
    p.add_argument("--no-scalar-xcheck", action="store_true",
                   help="mute the v19 independent 1-D single-channel "
                        "cross-check (k B_jj/A_jj vs k (int g)^2/int g^2). "
                        "On by default; it is ADVISORY (a warning, not a hard "
                        "gate) and surfaces scale/Jacobian or inner-resolution "
                        "inconsistencies the global R bound misses.")
    p.add_argument("--scalar-xcheck-tol", type=float, default=5e-2,
                   help="relative-gap threshold for the scalar cross-check "
                        "warning on effectively-narrow channels (default 5e-2)")
    p.add_argument("--no-scalar-xcheck-hardfail", action="store_true",
                   help="lift the v21 hard-gate backstop (reject a narrow "
                        "channel whose convolution diagonal exceeds "
                        "--scalar-xcheck-hardfail-mult times its honest 1-D "
                        "value). On by default; directional, inflation only.")
    p.add_argument("--scalar-xcheck-hardfail-mult", type=float, default=3.0,
                   help="hard-fail a narrow channel when k B_jj/A_jj exceeds "
                        "this multiple of its independent 1-D value (default 3)")
    p.add_argument("--no-rate-hard-clamp", action="store_true",
                   help="lift the v21 hard clamp of softplus(rho) to rate_cap. "
                        "On by default; the convolution mesh is built for "
                        "a_ref=2*rate_cap, so an unclamped rate that runs past "
                        "rate_cap leaves the mesh's design range and inflates.")
    p.add_argument("--no-moment-gate", action="store_true",
                   help="lift the v22 moment upper-bound gate (reject any "
                        "diagonal exceeding its own unconstrained moment "
                        "k(int g)^2/int g^2). On by default; this is the "
                        "all-width check that caught the k=1000 channel-27 "
                        "inflation (5.74 vs moment 3.36).")
    p.add_argument("--moment-gate-tol", type=float, default=0.10,
                   help="relative tolerance of the moment gate (default 0.10)")
    p.add_argument("--raw-smooth", type=float, default=1e-3,
                   help="curvature penalty weight on log raw in the "
                        "envelope-scaled coordinate u = rho*x (v22). Legit "
                        "channels have O(1) curvature in u -> penalty ~1e-3; "
                        "sub-envelope-scale spikes (the raw-route exploit) "
                        "have curvature >> 1 -> penalty explodes. 0 disables.")
    p.add_argument("--no-maynard-baseline", action="store_true",
                   help="skip the v22 analytic Maynard-profile preflight "
                        "g_A(t)=1/(1+A t), A ~ log k (known near-optimal "
                        "family; anchors the evaluator in the regime that "
                        "matters and sets the discovery target).")
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
    p.add_argument("--dist-backend", type=str, default="auto",
                   choices=("auto", "nccl", "gloo"),
                   help="process-group backend. auto = nccl on Linux+CUDA, "
                        "gloo otherwise. NCCL does not exist on Windows, so a "
                        "Windows multi-GPU run uses gloo; the payloads here "
                        "are O(m^2) doubles per iteration, so that costs "
                        "essentially nothing.")
    p.add_argument("--dist-timeout-minutes", type=float, default=30.0,
                   help="process-group collective timeout. A real rank/collective "
                        "mismatch then fails instead of hanging indefinitely")
    p.add_argument("--dist-oversubscribe", action="store_true",
                   help="allow more ranks than GPUs by wrapping local_rank "
                        "modulo the device count. This does NOT save VRAM — "
                        "two ranks on one card hold the same total activation "
                        "memory as one rank holding every pair — it exists "
                        "only to smoke-test the NCCL path on a single card.")
    p.add_argument("--dist-trajectory-selftest", action="store_true",
                   help="run the algebraic self-test plus a short replicated-vs-"
                        "sharded Adam trajectory and one LBFGS step; then exit")
    p.add_argument("--dist-selftest", action="store_true",
                   help="verify that the pair-sharded gradient reproduces the "
                        "replicated one to machine precision, then exit. Run "
                        "this ONCE per machine before trusting a multi-GPU "
                        "campaign: it is the only check that the surrogate, "
                        "the diagonal gather and the gradient all-reduce are "
                        "consistent with each other.")
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
    _p.stream_pair_chunk = args.stream_pair_chunk
    _p.trunc_floor = args.trunc_floor
    _p.stability_tol = args.stability_tol
    _p.stability_rtol = args.stability_rtol
    _p.scalar_xcheck = not args.no_scalar_xcheck
    _p.scalar_xcheck_tol = args.scalar_xcheck_tol
    _p.scalar_xcheck_hardfail_mult = (None if args.no_scalar_xcheck_hardfail
                                      else args.scalar_xcheck_hardfail_mult)
    _p.moment_gate_tol = (None if args.no_moment_gate else args.moment_gate_tol)
    return _p  # noqa


def main() -> None:
    args = parse_args()
    DIST.init(backend=args.dist_backend, device_hint=args.device,
              oversubscribe=args.dist_oversubscribe,
              timeout_minutes=args.dist_timeout_minutes)
    device = resolve_device(args.device)
    if DIST.enabled and device.type == "cuda":
        # torchrun gives each rank a LOCAL_RANK; "auto" would otherwise put
        # every rank on cuda:0 and the whole point would be lost.
        device = torch.device(f"cuda:{DIST.local_rank}")
        torch.cuda.set_device(device)
    cdt = parse_torch_dtype(args.dtype)
    rdt = parse_torch_dtype(args.ritz_dtype)
    torch.set_float32_matmul_precision(args.cuda_matmul_precision)
    # IDENTICAL seed on every rank: the model is replicated, not sharded, so
    # the ranks must start from the same parameters. Divergence here would not
    # crash anything — it would quietly optimise m different models.
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
    print(f"   streamed HF    : local pair chunk = {args.stream_pair_chunk} "
          f"({'ON' if args.stream_pair_chunk > 0 else 'OFF / legacy v3'})")
    print(f"   m schedule     : {ms}   adam iters = {its}")
    print("=" * 78)
    print(f" seed={args.seed}  device={device}  eps={eps:g}  rate_cap={rate_cap:g}")
    print(f" dtype={args.dtype}  ritz={args.ritz_dtype}  trunc_tol={args.trunc_tol:.0e}"
          f"  min_rank_frac={args.min_rank_frac:g}")
    if device.type == "cuda":
        pr = torch.cuda.get_device_properties(device)
        print(f" GPU: {pr.name}  VRAM={pr.total_memory/2**30:.1f} GiB")
    if DIST.enabled:
        per = [len(range(r, Pf, DIST.world)) for r in range(DIST.world)]
        print(f" distributed    : world={DIST.world}  backend={DIST.backend}"
              f"  pairs/rank at m={mf}: {min(per)}-{max(per)} of {Pf}"
              f"   (~{max(per)/Pf:.1%} of the activation memory each)")
        # Confirm the ranks really landed on DISTINCT devices. Every rank
        # sitting on cuda:0 is a silent failure -- the run works, produces
        # correct numbers, and delivers none of the memory relief that was
        # the entire point.
        if device.type == "cuda":
            meta = {
                "rank": DIST.rank,
                "host": socket.gethostname(),
                "local_rank": DIST.local_rank,
                "device": int(device.index or 0),
                "name": torch.cuda.get_device_name(device),
            }
            metas = DIST.all_gather_objects(meta)
            metas = sorted(metas, key=lambda z: z["rank"])
            print("                  device map: " + ", ".join(
                f"rank{x['rank']}@{x['host']}->cuda:{x['device']}"
                for x in metas))
            by_host = {}
            for x in metas:
                by_host.setdefault(x["host"], []).append(x)
            collisions = []
            for host, rows in by_host.items():
                seen = {}
                for x in rows:
                    seen.setdefault(x["device"], []).append(x["rank"])
                for dev_idx, ranks in seen.items():
                    if len(ranks) > 1:
                        collisions.append((host, dev_idx, ranks))
            if collisions:
                desc = "; ".join(
                    f"{host} cuda:{dev_idx} used by ranks {ranks}"
                    for host, dev_idx, ranks in collisions)
                print(f"   WARNING: GPU oversubscription detected: {desc}. "
                      f"VRAM per physical card is NOT reduced.")
    else:
        print(f" distributed    : off (single process). Launch with "
              f"`torchrun --nproc_per_node=N` to shard the {Pf} pairs.")

    problem = build_problem(args, k, n_rep, eps, device, cdt, rdt)
    print(f" rigorous ceiling: R <= {problem.R_bound:.6f}   [{problem.R_bound_name}]"
          + (f"   (Rem 6.6, sharp in eps: {problem.R_bound_sharp:.6f})"
             if eps > 0 else ""))
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

    if args.dist_selftest or args.dist_trajectory_selftest:
        gs = ChannelNet(k, m=ms[0], hidden=args.hidden, depth=args.depth,
                        rate_cap=rate_cap,
                        rate_hard_clamp=not args.no_rate_hard_clamp,
                        positive=(args.channel_sign == "positive")).to(
                            device=problem.device, dtype=problem.dtype)
        DIST.sync_module(gs)
        print("\n--- distributed self-test (sharded vs replicated) ---")
        ok = dist_selftest(problem, gs)
        if ok and args.dist_trajectory_selftest:
            print("\n--- distributed trajectory self-test ---")
            ok = dist_trajectory_selftest(problem, gs)
        DIST.shutdown(graceful=True)
        sys.exit(0 if ok else 1)

    if args.grad_check:
        gc = ChannelNet(k, m=min(ms[0], 6), hidden=16, depth=2, rate_cap=rate_cap,
                        positive=(args.channel_sign == "positive")).to(device=problem.device, dtype=problem.dtype)
        DIST.sync_module(gc)
        problem.rayleigh_backward_streamed(gc, sign=1.0)
        DIST.sync_grads(gc, problem.plan(gc.m).mode)
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
            frac = rd.R / problem.R_bound
            print(f"\n dictionary baseline (x^p e^-rx, m = {dic.m}) : "
                  f"R = {rd.R:.10f}   rank {rd.rank}/{dic.m}"
                  f"   = {frac:.1%} of the rigorous ceiling {problem.R_bound:.4f}")
            # ---- QUADRATURE GATE -------------------------------------------
            # The dictionary is a FIXED basis. It cannot optimise, cannot
            # exploit, and is evaluated before a single gradient step. Its
            # value is therefore a pure function of k and the quadrature, and
            # it obeys exactly the same rigorous ceiling as everything else.
            # If it lands above the ceiling, the discretisation is invalid and
            # no amount of training will fix it -- abort now rather than after
            # an hour. (Measured: at k=5000, n_rep=12000 this read 8.9278
            # against a ceiling of 8.5189, and at n_rep=45000 it still read
            # 8.4757 = 99.5%. Both runs were worthless; both were detectable
            # here, in seconds, before any GPU time was spent.)
            if frac > 1.0 + problem.R_tol:
                raise ValidationError(
                    f"the FIXED dictionary basis scores {rd.R:.6f}, above the "
                    f"rigorous ceiling {problem.R_bound:.6f} "
                    f"({problem.R_bound_name}). A fixed basis cannot exploit "
                    f"anything, so this is a defect in the quadrature itself, "
                    f"present before any training. Raise --n-rep, or check "
                    f"--rate-cap: holding rate_cap/k constant matters, the "
                    f"outer mesh is graded on x_star = (k-2)/(2*rate_cap).")
            # ---- QUADRATURE ALARM (soft) -----------------------------------
            # Even below the ceiling, the dictionary's trajectory in k should
            # be SMOOTH: it is a fixed 12-function basis. Measured history at
            # rate_cap/k = 5: 52.4% (k=1000), 63.5% (k=3000). The runs that
            # later proved inflated read 72.7% (k=3500), 74.5% (k=3600),
            # 99.5% (k=5000) -- a 6x acceleration in the growth rate. There is
            # no clean analytic form for the expected value, so this is a
            # printed warning, not a gate.
            if frac > 0.80:
                print(f"   WARNING: the fixed dictionary is at {frac:.1%} of "
                      f"the ceiling. Historically >80% has meant the "
                      f"quadrature is inflating: compare against the same "
                      f"quantity at a smaller k with rate_cap/k held fixed, "
                      f"and treat R from this run as provisional until the "
                      f"certifier confirms it.")
        except FloatingPointError as e:
            print(f"\n dictionary baseline skipped: {e}")

    if not args.no_maynard_baseline:
        # ---- v22: analytic Maynard-profile anchor ---------------------------
        # Known family g_A(t)=1/(1+At): achieves log k - O(1). Its Ritz value
        # here is (a) the target the trained net must beat and (b) an evaluator
        # check in the smooth log-k regime the g==1/g==x controls never reach.
        try:
            mprof = MaynardProfile(k).to(device=problem.device,
                                         dtype=problem.dtype)
            rm = problem.report(mprof)
            lk = math.log(k)
            print(f"\n Maynard-profile baseline 1/(1+At), A~log k (m = "
                  f"{mprof.m}) : R = {rm.R:.10f}   rank {rm.rank}/{mprof.m}")
            print(f"   expected log k - O(1) = {lk:.4f} - O(1); ceiling "
                  f"{problem.R_bound:.4f}. The trained channels must beat "
                  f"{rm.R:.4f} for the discovery to add value.")
            if rm.R > problem.R_bound * (1.0 + problem.R_tol):
                raise ValidationError(
                    f"the FIXED Maynard profile scores {rm.R:.6f} above the "
                    f"ceiling {problem.R_bound:.6f}; like the dictionary gate "
                    f"this is an evaluator defect, present before training.")
            if rm.R < 0.5 * lk:
                print(f"   WARNING: the analytic profile scores only "
                      f"{rm.R:.4f} < 0.5 log k = {0.5*lk:.4f}. The known "
                      f"value is log k - O(1), so the evaluator is likely "
                      f"UNDER-resolving smooth log-k channels (outer mesh / "
                      f"n_rep); discovered values from this run will "
                      f"understate M_k.")
        except (ValidationError, FloatingPointError) as e:
            print(f"\n Maynard-profile baseline: {e}")

    if args.check_dense:
        if k > 4:
            print("\n --check-dense skipped: only feasible for k <= 4")
        else:
            gk = ChannelNet(k, m=min(ms[0], 6), hidden=32, depth=2, rate_cap=rate_cap,
                            positive=(args.channel_sign == "positive")).to(device=problem.device,
                                                  dtype=problem.dtype)
            DIST.sync_module(gk)
            # report() is collective, so EVERY rank must call it. What follows
            # is not: DenseCheck builds a full (k-1)-dimensional Duffy grid,
            # which is by far the most memory-hungry object in the whole
            # script (~2 GB at k=4, n_quad=30 — two orders more than the
            # sharded pipeline it is checking). Replicating that across ranks
            # is pure waste and will OOM a box that runs the real workload
            # comfortably. Rank 0 alone does the cross-check.
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
            if DIST.is_main:
                dense = DenseCheck(k, n_quad=30, epsilon=eps,
                                   device=problem.device, dtype=problem.dtype)
                I_d, J_d = dense.I_and_J(gk, c_raw)
                print(f"\n dense Duffy cross-check (k = {k}, random init model):")
                print(f"   J/I  conv {J_conv/I_conv:.12e}   dense {J_d/I_d:.12e}   "
                      f"rel {abs(J_conv/I_conv - J_d/I_d)/abs(J_d/I_d):.2e}")
                print(f"   (ratios only: v9 stores A, B diagonally preconditioned)")
                del dense
            DIST.barrier()

    if args.skip_neural:
        return

    g = ChannelNet(k, m=ms[0], hidden=args.hidden, depth=args.depth,
                   rate_cap=rate_cap, rate_hard_clamp=not args.no_rate_hard_clamp,
                   positive=(args.channel_sign == "positive")).to(device=problem.device, dtype=problem.dtype)
    DIST.sync_module(g)

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
        DIST.sync_module(g)

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
            # grow() draws randn for the appended rows. Even with identical
            # seeds the RNG streams can drift (a rank that took an extra
            # branch consumed extra randomness), and the failure is silent:
            # the ranks would optimise different models while agreeing on
            # every reduced scalar. Broadcasting costs microseconds.
            DIST.sync_module(g)
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
            excursion_floor=args.excursion_floor,
            raw_smooth=args.raw_smooth)
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
    print(f" channel tail span  = {rep.tail_span:.1f} nats   (widest single "
          f"channel; Thm 6.7's optimal g spans ~4)")
    if rep.tail_span > 12.0:
        print(f"   WARNING: {rep.tail_span:.0f} nats is {rep.tail_span/4.0:.0f}x "
              f"the analytic expectation. Concentration beyond ~12 nats sits "
              f"below the quadrature nodes and is the mechanism behind every "
              f"inflated run measured so far. Certify before quoting R.")
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

    if not args.no_ridge_sweep and DIST.is_main:
        # pure numpy on matrices that are already bitwise identical everywhere
        ridge_sensitivity(rep.A, rep.B, k=k)
    DIST.barrier()

    if args.export and DIST.is_main:
        # ---- channel diagnostics FIRST: x_fine depends on `rates` ---------
        # (this ordering is load-bearing: an earlier revision built x_fine
        # from `rates` four lines before `rates` was assigned, which either
        # raised NameError or, worse, silently picked up a stale binding and
        # wrote a grid graded for the wrong scale.)
        with torch.no_grad():
            rates = torch.nn.functional.softplus(g.rho).cpu().numpy()
            gr = g(problem.Xrep)
            nrm = torch.sqrt(torch.einsum("q,qj->j", problem.w_cc,
                                          gr.square())).cpu().numpy()

        # ---- GRADED EXPORT GRID -------------------------------------------
        # The channels vary on the envelope scale 1/rate. A uniform 2001-point
        # grid has spacing 5.0e-04; at rate_cap 25000 the channel's knee is at
        # 4.0e-05, i.e. 12.5x FINER than the sampling. Measured on the k=3000
        # and k=3500 exports: exactly ONE sample had |g| > max/e -- the whole
        # channel was a single point. Everything downstream (projection, float
        # DP, exact pass) was then fitting a spike-shaped hole, which is why
        # the certifier's float-predicted R sat at 0.46-0.63 against a
        # discovery R of 6.78 and refused to converge with --grid.
        #
        # A geometric limb from 1/(200*r_max) resolves the knee with ~40
        # points; the uniform limb keeps global coverage so the tail is not
        # starved. NOTE the projection must use TRAPEZOID-WEIGHTED least
        # squares once the grid is non-uniform -- unweighted lstsq on a graded
        # grid silently reweights the objective toward wherever the points are
        # densest.
        r_max = float(np.max(rates))
        x_fine = np.unique(np.concatenate([
            np.geomspace(1.0 / (200.0 * r_max), problem.L, 4000),
            np.linspace(0.0, problem.L, 2001),
            np.array([0.0])]))
        with torch.no_grad():
            g_fine = g(torch.tensor(x_fine, dtype=problem.dtype,
                                    device=problem.device)).cpu().numpy()

        # ---- resolution check on what we are about to write ---------------
        # cheap, and it is the check that would have caught the 2001-point
        # export immediately.
        jdom = int(np.argmax(np.abs(rep.c)))
        gj = np.abs(g_fine[:, jdom])
        n_res = int((gj > gj.max() / math.e).sum()) if gj.max() > 0 else 0
        print(f"\n export grid: {len(x_fine)} points, "
              f"min spacing {np.diff(x_fine).min():.3e}, "
              f"knee 1/r_max = {1.0/r_max:.3e}")
        print(f"   dominant channel {jdom}: {n_res} samples above max/e")
        if n_res < 8:
            print(f"   WARNING: only {n_res} samples resolve the dominant "
                  f"channel. The certifier cannot recover a function it was "
                  f"never shown; widen the geometric limb "
                  f"(1/(200*r_max) -> 1/(2000*r_max)) before certifying.")

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
    try:
        main()
    except BaseException:
        # Never enter a fresh barrier while unwinding: another rank may still
        # be blocked in a different collective. Destroy locally and let the
        # configured process-group timeout/torchrun terminate the peers.
        if DIST.enabled:
            builtins.print(f"[rank {DIST.rank}] aborting", file=sys.stderr,
                           flush=True)
        DIST.shutdown(graceful=False)
        raise
    else:
        DIST.shutdown(graceful=True)
